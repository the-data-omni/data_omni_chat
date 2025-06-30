"""
routes.py
FastAPI router that wires HTTP endpoints to a singleton DataAnalysisService.
All read‑only endpoints fall back to the dataset stored in memory
(via /upload_data/*) when the caller omits an explicit `data` payload.
"""

from typing import List, Dict, Any, Optional

from fastapi import (
    APIRouter,
    Body,
    HTTPException,
    Depends,
    Header
)
from fastapi.responses import JSONResponse
import pandas as pd
from pydantic import BaseModel

from app.services.analytics_service import (
    DataAnalysisService,
    DataFrameRequest,
)


router = APIRouter()
service = DataAnalysisService()


async def get_api_key(authorization: Optional[str] = Header(None)) -> str:
    """Dependency to extract and validate the API key from the Authorization header."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header is missing")
    
    # The standard format is "Bearer <key>"
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization format. Use 'Bearer <key>'")
    
    return parts[1]

class ConnectionCheckRequest(BaseModel):
    model: str

# --- 2. Add the new /verify_connection endpoint ---
@router.post("/verify_connection")
async def verify_connection(payload: ConnectionCheckRequest, api_key: str = Depends(get_api_key)):
    """
    Checks if a connection can be established with the given model and API key.
    Returns a success or failure message.
    """
    try:
        result = await service.check_llm_connection(payload.model, api_key)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def make_serializable(data: Any) -> Any:
    if isinstance(data, pd.Timestamp):
        return data.isoformat() 
    elif pd.isna(data): 
        return None
    return data


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


@router.post("/full_analysis")
async def full_analysis(payload: DataFrameRequest, api_key: str = Depends(get_api_key)):
    return await service.full_analysis(payload, api_key)


