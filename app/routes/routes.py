"""
routes.py
FastAPI router that wires HTTP endpoints to a singleton DataAnalysisService.
All read‑only endpoints fall back to the dataset stored in memory
(via /upload_data/*) when the caller omits an explicit `data` payload.
"""

from typing import List, Dict, Any, Optional
from fastapi.encoders import jsonable_encoder

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Body,
    Query,
    HTTPException,
)
from fastapi.responses import JSONResponse
import pandas as pd

from app.services.analytics_service import (
    DataAnalysisService,
    DataFrameRequest,
    CleanupAction,
    CleanupPlanRequest,
    CleanupExecuteRequest,
    ExecuteCodeWithDataPayload,
    SummarizePayload,
    LLMFreeAnalysisRequest,
)

# ------------------------------------------------------------------ #
#  Singleton service                                                 #
# ------------------------------------------------------------------ #
router = APIRouter()
service = DataAnalysisService()

# ================================================================== #
#  🔺  UPLOAD ENDPOINTS                                              #
# ================================================================== #
@router.post("/upload_data/csv")
async def upload_data_csv(file: UploadFile = File(..., description="CSV file"), 
                          has_header: bool = Query(True, description="True if first row is a header")):
    """Upload a CSV file; the rows are cached + profiled server‑side."""
    return await service.upload_data(file=file, has_header=has_header)


@router.post("/upload_data/json")
async def upload_data_json(
    json_body: List[Dict[str, Any]] = Body(..., description="JSON array (no wrapper)"),
):
    """
    Upload a JSON array of objects, e.g.:

    [
      {"emp_id": 1, "department": "Sales", "salary": 78000},
      {"emp_id": 2, "department": "HR",   "salary": 61000}
    ]
    """
    if not isinstance(json_body, list):
        raise HTTPException(status_code=400, detail="Top‑level JSON must be an array.")
    return await service.upload_data(json_rows=json_body)

@router.post("/cleanup_apply")
def cleanup_apply(actions: List[CleanupAction]):   # body = [... action objects ...]
    """
    Execute the chosen cleanup actions, then immediately
    generate synthetic data and profiles. Returns:

      {
        "message": "...",
        "cleanup": { cleaned_data, applied_actions, … },
        "synthetic": { row_count, column_count, preview }
      }
    """
    return JSONResponse(content=service.apply_cleanup_then_synthesize(actions))

# ================================================================== #
#  🔺  READ‑ONLY GETTERS (data + profiles)                            #
# ================================================================== #
# Helper function to make data JSON serializable
def make_serializable(data: Any) -> Any:
    if isinstance(data, pd.Timestamp):
        # Convert Timestamp to ISO 8601 string format
        # Choose the format that best suits your needs
        return data.isoformat() 
    elif pd.isna(data): 
        # Convert pandas NA or numpy NaN to None
        return None
    # Add checks for other non-serializable types if necessary
    # elif isinstance(data, np.integer):
    #     return int(data)
    # elif isinstance(data, np.floating):
    #     return float(data)
    # elif isinstance(data, np.ndarray):
    #     return data.tolist() 
    return data

@router.get("/data/original")
def get_original_data():
    """Return the last uploaded (and optionally cleaned) dataset, ensuring JSON compatibility."""
    try:
        # Get the raw dataset (list of dicts) which might contain non-serializable types
        original_dataset = service.get_original_data() 
        
        # Convert the dataset to be JSON serializable
        serializable_content = []
        for row_dict in original_dataset:
            serializable_row = {key: make_serializable(value) for key, value in row_dict.items()}
            serializable_content.append(serializable_row)
            
        # Return the serializable content using JSONResponse
        return JSONResponse(content=serializable_content)
        
    except HTTPException as e:
         # Re-raise HTTPException (like 404 Not Found)
         raise e
    except Exception as e:
         # Handle other potential errors during serialization or data retrieval
         # Log the error for debugging
         # logger.error(f"Error retrieving or serializing original data: {e}", exc_info=True) 
         raise HTTPException(status_code=500, detail=f"Failed to retrieve or serialize data: {str(e)}")


@router.get("/data/synthetic")
def get_synthetic_data():
    """Return the most recently generated synthetic dataset."""
    return JSONResponse(content=service.get_synthetic_data())


@router.get("/profiles")
def get_profiles():
    """
    Return both original and synthetic DataProfiler reports (if available):

      {
        "original_profile": {...},
        "synthetic_profile": {...} | null
      }
    """
    return JSONResponse(content=service.get_profiles())


# ================================================================== #
#  🔺  SYNTHETIC‑DATA GENERATOR                                      #
# ================================================================== #
@router.post("/generate_synthetic")
def generate_synthetic(sample_size: Optional[int] = Query(None, ge=1)):
    """
    Fit a Gaussian Copula on the current dataset and create synthetic rows.

    Query params:
      • sample_size – number of rows to generate (defaults to original row‑count)

    Side‑effects:
      • Caches synthetic rows in memory
      • Builds/updates the synthetic profile

    Returns a small preview plus counts.
    """
    payload = service.generate_synthetic_data()
    # Make everything JSON-safe (Timestamps → ISO-8601 strings, NaN → None, NumPy scalars → plain int/float)
    safe_payload = jsonable_encoder(payload)
    return JSONResponse(content=safe_payload)
    
    # return JSONResponse(content=service.generate_synthetic_data(sample_size))
@router.post("/upload_anonymized_data")
def upload_anonymized_data(data: List[Dict[str, Any]] = Body(...)):
    """
    Receives a full, anonymized dataset generated on the client.

    Side-effects:
      • Caches the anonymized dataset in memory.
      • Builds/updates the synthetic profile based on the received data.

    Returns a confirmation message.
    """
    # The 'data' is automatically validated by FastAPI as a list of dictionaries.
    # We are assuming the service object has been updated with the new method.
    payload = service.process_anonymized_data(data)

    return JSONResponse(content=payload)

# ================================================================== #
#  🔺  CLEANUP PIPELINE                                              #
# ================================================================== #
@router.post("/cleanup_plan")
def cleanup_plan(payload: CleanupPlanRequest):
    return service.cleanup_plan(payload)


@router.post("/cleanup_execute")
def cleanup_execute(payload: CleanupExecuteRequest):
    return service.cleanup_execute(payload)


# ================================================================== #
#  🔺  GENERIC CODE EXECUTION & ANALYSIS                             #
# ================================================================== #
@router.post("/execute_code")
async def execute_code(payload: ExecuteCodeWithDataPayload):
    return await service.execute_code(payload)


@router.post("/summary")
def summarize_output(payload: SummarizePayload):
    return service.summarize(payload)


@router.post("/analysis")
def analysis(payload: DataFrameRequest):
    return service.analyze(payload)


@router.post("/full_analysis")
async def full_analysis(payload: DataFrameRequest):
    return await service.full_analysis(payload)


# ================================================================== #
#  🔺  LLM‑FREE ANALYSIS                                             #
# ================================================================== #
@router.post("/llm_free_analysis")
async def llm_free_analysis(payload: LLMFreeAnalysisRequest):
    return await service.llm_free_analysis(payload)
