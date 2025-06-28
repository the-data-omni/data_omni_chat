
"""
analysis_service.py
------------------------------------------------------------------
A single class (`DataAnalysisService`) that holds:

• a persistent in‑memory dataset (self.dataset)
• helpers to upload / clean / analyse / summarise that dataset
• OpenAI‑powered code‑generation & correction
• CSV or JSON upload support

All endpoints can work with either:
  1. an explicit data list passed in the request, or
  2. the dataset that was previously uploaded via /upload_data
------------------------------------------------------------------
"""
import base64
import io
import json
import math
import os
import re
import traceback
import uuid
from contextlib import redirect_stdout
from typing import Any, Dict, List, Optional

# Third-party imports
import numpy as np
import openai
import pandas as pd
from dataprofiler import Profiler
from dotenv import load_dotenv
from fastapi import HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from openai import OpenAI

from app.synthetic_generator.generate_fake_data import fake_preview_df
from app.synthetic_generator.light_weight_synth import SimpleTabularSynth

# Local application/library specific imports
try:
    from app.classifier.cleanup import \
        OPENAI_MODEL_DEFAULT  # Assuming these constants are also in cleanup.py or its config
    from app.classifier.cleanup import (  # Other necessary functions like visualize_hidden_chars, etc., if they are not; part of main_analysis_and_cleanup_pipeline's internal workings or are also needed here.; For this example, we assume main_analysis_and_cleanup_pipeline is self-contained; or handles its own dependencies from app.classifier.cleanup.
        HEAD_ROWS_DEFAULT, IMG_PATH_DEFAULT, MAX_ATTEMPTS_DEFAULT,
        main_analysis_and_cleanup_pipeline)
except ImportError as e:
    print(f"Error importing from app.classifier.cleanup: {e}")
    print("Please ensure app/classifier/cleanup.py exists and is in the PYTHONPATH.")
    # Define fallbacks or raise error if critical
    IMG_PATH_DEFAULT = "table_screenshot.png"
    HEAD_ROWS_DEFAULT = 20
    OPENAI_MODEL_DEFAULT = "gpt-4o"
    MAX_ATTEMPTS_DEFAULT = 5
    async def main_analysis_and_cleanup_pipeline(*args, **kwargs): # Fallback dummy
        print("Warning: main_analysis_and_cleanup_pipeline could not be imported. Using dummy function.")
        return "FALLBACK_CLASSIFICATION", "FALLBACK_JSON_SUMMARY", "FALLBACK_CLEANUP_CODE"

from app.cleanup.clean_multivalue import generalize_csv_restructure
from app.cleanup.clean_pivot import restructure_pivoted_csv_v3
from app.models.schemas import (CleanupAction, CleanupExecuteRequest,
                                CleanupPlanRequest, DataFrameRequest,
                                DataProfile, ExecuteCodeWithDataPayload,
                                LLMFreeAnalysisRequest, SummarizePayload, ChatTurn)


def _sanitize_nan(obj):             # paste the helper here
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nan(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


# ------------------------------------------------------------------#
#                    ---------  Service class  ---------            #
# ------------------------------------------------------------------#
class DataAnalysisService:
    """
    Central point for every operation.
    After a successful `/upload_data`, the dataset lives in `self.dataset`
    and is automatically used by subsequent calls when the user omits
    the `data` field.
    """

    def __init__(self) -> None:
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=openai.api_key)
        self.dataset: Optional[List[Dict[str, Any]]] = None  # persistent store
        self.synthetic_dataset: Optional[List[Dict[str, Any]]] = None 
        self.profile_original: Optional[Dict[str, Any]] = None
        self.profile_synthetic: Optional[Dict[str, Any]] = None
        self.csv_path: Optional[str] = None
        self.chat_sessions: Dict[str, List[ChatTurn]] = {}

    # --------------------  Internal helpers  -------------------- #
    def _get_df(self, data: Optional[List[Dict[str, Any]]]) -> pd.DataFrame:
        """
        Convert either the provided list‑of‑dicts OR the stored dataset
        into a Pandas DataFrame. Raises if nothing is available.
        """
        if data is not None:
            return pd.DataFrame(data)
        if self.dataset is not None:
            return pd.DataFrame(self.dataset)
        raise HTTPException(status_code=400, detail="No dataset available. Upload data first.")
    
    def _summarize_execution(self, execution_result: str | dict,
                            code: str,
                            question: str = "") -> Dict[str, Any]:
        """
        Wrapper that calls self.summarize() and returns its JSON,
        making sure execution_result is already JSON-serialisable.
        """
        return self.summarize(
            SummarizePayload(
                execution_result=execution_result,
                code=code,
                question=question,
            )
        )

    async def upload_data(
        
        self,
        *,
        file: Optional[UploadFile] = None, # Changed Any to UploadFile
        json_rows: Optional[List[Dict[str, Any]]] = None,
        drop_null_threshold: float = 0.5,
        has_header: bool = True, 
    ) -> Dict[str, Any]:
        """function that calls cleanup, and updated dataframe in place"""
        df_orig: Optional[pd.DataFrame] = None

        if file:
            if not file.filename: # Basic validation
                raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

            os.makedirs("uploads", exist_ok=True)
            # Use the actual filename from the UploadFile object
            base_filename, file_extension = os.path.splitext(file.filename)
            # Sanitize filename slightly or use a completely random one to avoid issues
            # For simplicity, using uuid with original extension
            fname = f"{uuid.uuid4().hex}{file_extension}"
            path = os.path.join("uploads", fname)

            try:
                # Read contents from the UploadFile object
                contents = await file.read()
                if not contents:
                    raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            except Exception as e:
                # Handle potential errors during file read (though await file.read() itself might raise)
                print(f"Error reading uploaded file content: {e}")
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Could not read uploaded file: {e}")

            try:
                with open(path, "wb") as out:
                    out.write(contents)
            except Exception as e:
                print(f"Error writing uploaded file to disk: {e}")
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Could not save uploaded file: {e}")

            self.csv_path = path
            try:
                # Attempt to read the CSV into a DataFrame
                # keep_default_na=False and na_filter=False might affect how empty strings vs NaNs are handled.
                # Adjust as per your data's nature.
                header_arg = 0 if has_header else None
                print(header_arg, "header")
                df_orig = pd.read_csv(
                    self.csv_path,header=header_arg, keep_default_na=True, na_filter=True # Adjusted for more standard NaN handling
                )
            except pd.errors.EmptyDataError:
                print(f"CSV parsing failed: The file at {self.csv_path} is empty or contains no data.")
                raise HTTPException(status_code=400, detail="CSV file is empty or contains no data.")
            except Exception as exc:
                print(f"CSV parsing failed: {exc}")
                traceback.print_exc()
                raise HTTPException(status_code=400, detail=f"CSV parsing failed: {exc}")

        elif json_rows is not None:
            if not isinstance(json_rows, list) or not all(isinstance(row, dict) for row in json_rows):
                raise HTTPException(status_code=400, detail="JSON input must be a list of objects.")
            if not json_rows:
                 raise HTTPException(status_code=400, detail="JSON input is an empty list.")
            df_orig = pd.DataFrame(json_rows)
            os.makedirs("uploads", exist_ok=True) # Ensure uploads directory exists for JSON case too
            fname = f"{uuid.uuid4().hex}_from_json.csv"
            path = os.path.join("uploads", fname)
            try:
                df_orig.to_csv(path, index=False)
            except Exception as e:
                print(f"Error saving DataFrame from JSON to CSV: {e}")
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Could not process JSON data into CSV: {e}")
            self.csv_path = path # Set csv_path for consistency if needed later
        else:
            raise HTTPException(status_code=400, detail="No data provided. Please upload a CSV file or provide JSON rows.")

        if df_orig is None or df_orig.empty: # Check if DataFrame is empty after loading
             # This might be redundant if pd.read_csv raises EmptyDataError, but good as a fallback.
             print("DataFrame is empty after loading.")
             raise HTTPException(status_code=400, detail="Loaded DataFrame is empty.")

        # --- The rest of your pipeline remains the same ---
        print("\nStarting initial classification and cleanup workflow...")
        # Ensure your constants like IMG_PATH_DEFAULT, HEAD_ROWS_DEFAULT, OPENAI_MODEL_DEFAULT, MAX_ATTEMPTS_DEFAULT are defined
        # and functions like fake_preview_df, main_analysis_and_cleanup_pipeline are correctly imported and working.

        # Example placeholder for fake_preview_df if it's just taking a head:


        classification_code, raw_classification_json, generated_cleanup_code, split_params = await main_analysis_and_cleanup_pipeline(
            df=fake_preview_df(df_orig.copy(), HEAD_ROWS_DEFAULT), # Assuming HEAD_ROWS_DEFAULT is defined
            img_path=IMG_PATH_DEFAULT,       # Assuming IMG_PATH_DEFAULT is defined
            head_rows=HEAD_ROWS_DEFAULT,     # Assuming HEAD_ROWS_DEFAULT is defined
            model=OPENAI_MODEL_DEFAULT,      # Assuming OPENAI_MODEL_DEFAULT is defined
            max_attempts_classification=MAX_ATTEMPTS_DEFAULT, # Assuming MAX_ATTEMPTS_DEFAULT is defined
            max_llm_iterations_cleanup=MAX_ATTEMPTS_DEFAULT   # Assuming MAX_ATTEMPTS_DEFAULT is defined
        )
        
        final_classification_code = classification_code
        final_llm_summary_json = raw_classification_json
        final_generated_code_to_apply = generated_cleanup_code

        if final_classification_code == "PIVOT":
            print("\nInitial classification is PIVOT. Performing pivot cleanup...")
            if not self.csv_path:
                 raise HTTPException(status_code=500, detail="CSV path not available for pivot cleanup.")

            cleaned_df_after_pivot = restructure_pivoted_csv_v3(self.csv_path) # Assuming this function is defined
            if cleaned_df_after_pivot is None or cleaned_df_after_pivot.empty:
                print("Pivot cleanup function returned None or empty. Assuming cleanup failed or was not applicable.")
            else:
                print("Pivot cleanup successful. Updating DataFrame and CSV path.")
                df_orig = cleaned_df_after_pivot
                
                base, ext = os.path.splitext(self.csv_path)
                cleaned_path = f"{base}_pivoted_restructured{ext}"
                df_orig.to_csv(cleaned_path, index=False)
                self.csv_path = cleaned_path
                print(f"Pivoted restructured data saved to: {self.csv_path}")

                print("Re-classifying the unpivoted data...")
                classification_code_after_pivot, raw_json_after_pivot, generated_code_after_pivot, _ = await main_analysis_and_cleanup_pipeline(
                    df=fake_preview_df(df_orig.copy(), HEAD_ROWS_DEFAULT),
                    img_path=IMG_PATH_DEFAULT,
                    head_rows=HEAD_ROWS_DEFAULT,
                    model=OPENAI_MODEL_DEFAULT,
                    max_attempts_classification=MAX_ATTEMPTS_DEFAULT,
                    max_llm_iterations_cleanup=MAX_ATTEMPTS_DEFAULT
                )
                final_classification_code = classification_code_after_pivot
                final_llm_summary_json = raw_json_after_pivot
                final_generated_code_to_apply = generated_code_after_pivot
        
        elif final_classification_code == "MULTI_VALUE_CELLS_SPLIT" and split_params:
            print("\nInitial classification is MULTI_VALUE_CELLS_SPLIT. Performing split cleanup...")
            if not self.csv_path:
                raise HTTPException(status_code=500, detail="CSV path not available for split cleanup.")

            type_mapping = {'float': float, 'int': int, 'str': str}
            parsed_data_types = {}
            if split_params.get('data_types'):
                for col, type_str in split_params['data_types'].items():
                    if type_str in type_mapping:
                        parsed_data_types[col] = type_mapping[type_str]
                    else:
                        print(f"Warning: Unknown data type '{type_str}' for column '{col}'. Defaulting to string.")
                        parsed_data_types[col] = str

            cleaned_df_after_split = generalize_csv_restructure( # Assuming this function is defined
                file_path_or_buffer=self.csv_path,
                id_cols=split_params['id_cols'],
                value_cols_to_explode=split_params['value_cols_to_explode'],
                delimiter=split_params['delimiter'],
                data_types=parsed_data_types
            )

            if cleaned_df_after_split is None or cleaned_df_after_split.empty:
                print("Split cleanup function returned None or empty DataFrame. Assuming cleanup failed or was not applicable.")
            else:
                print("Split cleanup successful. Updating DataFrame and CSV path.")
                df_orig = cleaned_df_after_split
                
                base, ext = os.path.splitext(self.csv_path)
                cleaned_path = f"{base}_split_restructured{ext}"
                df_orig.to_csv(cleaned_path, index=False)
                self.csv_path = cleaned_path
                print(f"Split restructured data saved to: {self.csv_path}")

                print("Re-classifying the split data...")
                classification_code_after_split, raw_json_after_split, generated_code_after_split, _ = await main_analysis_and_cleanup_pipeline(
                    df=fake_preview_df(df_orig.copy(), HEAD_ROWS_DEFAULT),
                    img_path=IMG_PATH_DEFAULT,
                    head_rows=HEAD_ROWS_DEFAULT,
                    model=OPENAI_MODEL_DEFAULT,
                    max_attempts_classification=MAX_ATTEMPTS_DEFAULT,
                    max_llm_iterations_cleanup=MAX_ATTEMPTS_DEFAULT
                )
                final_classification_code = classification_code_after_split
                final_llm_summary_json = raw_json_after_split
                final_generated_code_to_apply = generated_code_after_split
        
        cleanup_applied_successfully = False
        if final_generated_code_to_apply:
            print(f"\nApplying final generated cleanup code to the dataset (current classification: {final_classification_code})...")
            try:
                exec_globals = {'pd': pd, 'df': df_orig.copy()} # Operate on a copy for safety during exec
                # Ensure run_in_threadpool is defined or remove if exec is not significantly blocking
                await run_in_threadpool(exec, final_generated_code_to_apply, exec_globals) # Assuming run_in_threadpool is defined
                df_orig = exec_globals['df']
                print("cleaned_df",df_orig.head(5))
                print("Successfully applied final generated cleanup code to df_orig.")
                cleanup_applied_successfully = True
            except Exception as e:
                print(f"🚨 Error applying final generated cleanup code to df_orig: {e}")
                traceback.print_exc()
                # Decide how to handle df_orig if exec fails; it might be partially modified.
                # For now, we proceed with potentially modified df_orig.
        
        self.dataset = df_orig.to_dict(orient="records")
        print(f"\nFinal self.dataset updated with {len(self.dataset)} records.")

        self.profile_original = self._build_profile(df_orig)
        
        # Assuming CleanupPlanRequest and DataProfile are defined and can be instantiated
        plan_payload = CleanupPlanRequest(
            data_profile=DataProfile(**self.profile_original), 
            drop_null_threshold=drop_null_threshold,
        )
        cleanup_plan_result = self.cleanup_plan(plan_payload)

        if hasattr(self, 'synthetic_dataset'):
            self.synthetic_dataset = None
        if hasattr(self, 'profile_synthetic'):
            self.profile_synthetic = None
        screenshot_b64 = None
        if os.path.exists(IMG_PATH_DEFAULT):
            with open(IMG_PATH_DEFAULT, "rb") as img_f:
                screenshot_b64 = base64.b64encode(img_f.read()).decode("utf-8")

        return {
            "message": "Data uploaded, processed, and profiled.",
            "classification_final": final_classification_code or "NONE",
            "llm_summary_json_final": final_llm_summary_json,
            "cleanup_code_generated_and_applied": cleanup_applied_successfully,
            "row_count": len(df_orig),
            "column_count": len(df_orig.columns),
            "profile": self.profile_original,
            "cleanup_plan": cleanup_plan_result["plan"],
            "table_screenshot_b64": screenshot_b64,        # 👈 NEW FIELD
        }

    # --------------------  CLEANUP + synth in one step ------------- #
    def apply_cleanup_then_synthesize(
        self,
        actions: List[CleanupAction],
    ) -> Dict[str, Any]:
        """
        1. Execute user-approved cleanup actions on self.dataset
        2. Build synthetic data + synthetic profile
        3. Return both cleaned original and synthetic metadata
        """

        exec_payload = CleanupExecuteRequest(actions=actions)
        cleanup_result = self.cleanup_execute(exec_payload)

        # (cleanup_execute already updates self.dataset)
        # synth_meta = self.generate_synthetic_data()
        synth_meta = SimpleTabularSynth(
            seed=42
            ).fit(self.dataset)

        return {
            "message": "Cleanup executed, synthetic data generated & profiled.",
            "cleanup": cleanup_result,
            "synthetic": synth_meta,
        }
# ------------------------------------------------------------------ #
#  2️⃣  SYNTHETIC‑DATA METHOD                                         #
# ------------------------------------------------------------------ #
    def generate_synthetic_data(self) -> Dict[str, Any]:
        """generate synthetic data for the full dataset for llm based analysis"""
        df_real = self._get_df(None)
        synth = SimpleTabularSynth(
                seed=42
                ).fit(df_real)
        df_synth = synth.sample(len(df_real))

            #     # ---- cache + profile ------------------------------------------ #
        self.synthetic_dataset = df_synth.to_dict(orient="records")
        self.profile_synthetic = self._build_profile(df_synth)

        preview = json.loads(
                        df_synth.head().to_json(orient="records", date_format="iso")
                    )
        return {
            "message": "Synthetic data generated and profiled.",
            "row_count": int(len(df_synth)),        # make sure these are plain ints
            "column_count": int(df_synth.shape[1]),
            "preview": preview,
        }
    
    def process_anonymized_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Receives client-generated anonymized data, loads it into a DataFrame,
        and performs backend analysis like profiling.
        """
        # Convert the received JSON data into a pandas DataFrame
        df_anonymized = pd.DataFrame(data)

        # --- Replicate original logic on the new data ---
        # Cache the received dataset and profile it
        self.synthetic_dataset = df_anonymized.to_dict(orient="records")
        self.profile_synthetic = self._build_profile(df_anonymized) # Assumes _build_profile exists

        # Return a success response
        return {
            "message": f"Successfully received and processed {len(df_anonymized)} rows of anonymized data."
        }
     
    def profile_data(self) -> Dict[str, Any]:
        """Method to profile dataset"""

        df = self._get_df(None)
        profile = Profiler(df.to_dict(orient="records"))
        report = profile.report(report_options={"output_format": "compact"})

        # 🔑  new line – deep sanitise NumPy NaN/Inf
        report_clean = _sanitize_nan(report)

        return report_clean
   
    def _build_profile(self, df: pd.DataFrame) -> Dict[str, Any]:

        prof = Profiler(df.to_dict(orient="records"))
        report = prof.report(report_options={"output_format": "compact"})
        return _sanitize_nan(report)
    # --------------------  CLEANUP (plan)  ---------------------- #
    def cleanup_plan(self, payload: CleanupPlanRequest) -> Dict[str, Any]:
        """
        Build a smart cleanup plan.
        """
        df = self._get_df(payload.data)
        row_count = payload.data_profile.global_stats.row_count
        drop_threshold = payload.drop_null_threshold
        plan: List[Dict[str, Any]] = []

        # ---------- helper: full filler choice --------------------------- #
        filler_opts = [
            "fill_with_mean",
            "fill_with_median",
            "fill_with_mode",
            "fill_with_value",
            "interpolate_linear",
            "interpolate_ffill",
            "interpolate_bfill",
        ]
        struct_opts = ["split_and_explode", "melt_wide_to_long", "promote_first_row_header"]

        # ---------- 1. column-level analysis ----------------------------- #
        delimiters = r"[|,;/]"      # quick pattern for multi-value cells

        for col_stat in payload.data_profile.data_stats:
            col = col_stat.column_name
            if col not in df.columns:
                continue

            stats = col_stat.statistics
            null_cnt = stats.get("null_count", 0)
            null_ratio = null_cnt / row_count if row_count else 0.0
            unique_ratio = stats.get("unique_ratio") or 0.0
            data_type = (col_stat.data_type or "").lower()
            # sample_str = col_stat.samples or ""
            raw_sample = stats.get("samples", stats.get("sample_values", ""))
            if isinstance(raw_sample, list):
               sample_str = str(raw_sample[0]) if raw_sample else ""
            else:
               sample_str = str(raw_sample)

            # --------- detect multi-value strings ------------------------ #
            has_delim = bool(re.search(delimiters, sample_str))
            if has_delim and data_type == "string":
                action = "split_and_explode"
                reason = "Values look concatenated (e.g. 'A | B | C')"
            # --------- high nulls → drop -------------------------------- #
            elif null_ratio > drop_threshold:
                action, reason = "drop", f"Null ratio {null_ratio:.1%} > {drop_threshold:.0%}"
            # --------- some nulls → fill -------------------------------- #
            elif null_ratio > 0:
                action = "fill_with_mean" if data_type in ("int", "float") else "fill_with_mode"
                reason = f"{null_cnt} missing values"
            # --------- default ignore ----------------------------------- #
            else:
                action, reason = "ignore", "No immediate issue detected"

            other = list(dict.fromkeys(      # keep order / de-dup
                (struct_opts + filler_opts + ["drop", "ignore"]))
            )
            if action in other:
                other.remove(action)

            plan.append(
                {
                    "column_name": col,
                    "null_ratio": round(null_ratio, 4),
                    "unique_ratio": round(unique_ratio, 4),
                    "data_type": data_type,
                    "suggested_action": action,
                    "reason": reason,
                    "other_options": other,
                }
            )

        # ---------- 2. table-level heuristics --------------------------- #
        wide_numeric_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and df[c].isna().sum() > 0
        ]
        if len(wide_numeric_cols) >= 3:
            plan.append(
                {
                    "column_name": "<table>",
                    "suggested_action": "melt_wide_to_long",
                    "reason": f"Data looks like a pivot table with {len(wide_numeric_cols)} value columns",
                    "other_options": ["ignore", "drop"] + struct_opts,
                }
            )

        if any(col.startswith(("Unnamed", "Column")) for col in df.columns) or df.columns.duplicated().any():
            plan.append(
                {
                    "column_name": "<table>",
                    "suggested_action": "promote_first_row_header",
                    "reason": "Column names appear generic or duplicated",
                    "other_options": ["ignore"] + struct_opts,
                }
            )

        return {"plan": plan, "message": "Enhanced cleanup suggestions generated."}

    # --------------------  CLEANUP (execute) -------------------- #
    def cleanup_execute(self, payload: CleanupExecuteRequest) -> Dict[str, Any]:
        df = self._get_df(payload.data).copy()
        applied: List[Dict[str, Any]] = []

        for item in payload.actions:
            col = item.column_name          # may be "" for whole-table actions
            action = item.action
            extra  = item.fill_value        # overloaded for parameters

            try:
                # ===================================================== DROP
                if action == "drop":
                    if col in df.columns:
                        df.drop(columns=[col], inplace=True)
                    else:
                        raise KeyError("Column not found")
                    msg = "Column dropped"

                # =========================================== CONVERT TO DATE
                elif action == "convert_to_date":
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    msg = "Converted to datetime"

                # =============================================== FILLERS
                # … (unchanged code for fill_with_* and interpolate_*) …

                # ================================================= SPLIT +
                elif action == "split_and_explode":
                    delimiter = extra or "|"        # default delimiter
                    if col not in df.columns:
                        raise KeyError("Column not found")
                    df[col] = (
                        df[col]
                        .astype(str)
                        .str.split(delimiter)
                        .apply(lambda lst: [s.strip() for s in lst])
                    )
                    df = df.explode(col).reset_index(drop=True)
                    msg = f"Split on '{delimiter}' and exploded to rows"

                # ================================================== MELT ==
                elif action == "melt_wide_to_long":
                    # extra can carry {"value_vars": [...], "var_name": "...", "value_name":"..."}
                    params = extra or {}
                    id_vars   = params.get("id_vars")   or [c for c in df.columns if c not in params.get("value_vars", [])]
                    value_vars = params.get("value_vars") or [c for c in df.columns if c not in id_vars]
                    var_name  = params.get("var_name", "variable")
                    value_name = params.get("value_name", "value")

                    df = df.melt(id_vars=id_vars, value_vars=value_vars,
                                var_name=var_name, value_name=value_name)
                    df.dropna(subset=[value_name], inplace=True)
                    df.reset_index(drop=True, inplace=True)
                    msg = f"Melted {len(value_vars)} columns to long format"

                # ====================================== PROMOTE HEADER ====
                elif action == "promote_first_row_header":
                    df.columns = df.iloc[0].astype(str).str.strip()
                    df = df.iloc[1:].reset_index(drop=True)
                    msg = "First row promoted to header"

                # ================================================= IGNORE
                elif action == "ignore":
                    msg = "No change made"

                # ========================================== UNKNOWN
                else:
                    raise ValueError(f"No handler for '{action}'")

                applied.append(
                    {"column_name": col or "<table>", "action": action, "status": "success", "message": msg}
                )

            except Exception as exc:
                applied.append(
                    {"column_name": col or "<table>", "action": action, "status": "failed", "message": str(exc)}
                )

        # persist & return
        self.dataset = df.to_dict(orient="records")
        return {
            "cleaned_data": self.dataset,
            "applied_actions": applied,
            "message": "Cleanup executed.",
        }
    
    def get_original_data(self) -> List[Dict[str, Any]]:
        if self.dataset is None:
            raise HTTPException(status_code=404, detail="No dataset uploaded")
        return self.dataset

    def get_synthetic_data(self) -> List[Dict[str, Any]]:
        if self.synthetic_dataset is None:
            raise HTTPException(status_code=404, detail="No synthetic data available")
        return self.synthetic_dataset

    def get_profiles(self) -> Dict[str, Any]:
        if self.profile_original is None:
            raise HTTPException(status_code=404, detail="No profile available")
        return {
            "original_profile": self.profile_original,
            "synthetic_profile": self.profile_synthetic,
        }
    # --------------------  EXECUTE ARBITRARY CODE -------------------- #
    async def execute_code(self, payload: ExecuteCodeWithDataPayload) -> Dict[str, Any]:
        df = self._get_df(payload.data)
        if not payload.code:
            raise HTTPException(status_code=400, detail="No code provided")

        exec_ns = {"df": df}
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(payload.code, exec_ns)  # nosec
        except Exception as exc:
            return {"result": f"Error executing code: {exc}\n{traceback.format_exc()}"}

        raw = buf.getvalue().strip()
        try:
            return {"result": json.loads(raw)}
        except json.JSONDecodeError:
            return {"result": raw}
        except Exception as exc:
            return {"result": f"Error executing code: {exc}\n{traceback.format_exc()}"}


    def summarize(self, payload: SummarizePayload) -> Dict[str, Any]:
            raw_output = (
                payload.execution_result.strip()
                if isinstance(payload.execution_result, str)
                else json.dumps(payload.execution_result)
            )
            try:
                parsed = json.loads(raw_output)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Execution result is not valid JSON.")

            analysis_data = parsed.get("analysis_data", {})
            chart_data = parsed.get("chart_data") # This chart_data comes from the executed code

            system_msg = {
                "role": "system",
                "content": (
                    "You are a data analysis assistant. You will be given the results of Python code execution, "
                    "which includes analysis data and chart data. Your task is to provide a textual summary "
                    "and, if applicable, an enhanced version of the original Python code. "
                    "The chart_data is already generated and should be used for context, not recreated. "
                    "Return JSON with keys 'summary' and 'enhanced_code' only."
                ),
            }
            user_msg = {
                "role": "user",
                "content": f"""
    Original code:
    {payload.code}

    analysis_data: {analysis_data}
    chart_data: {chart_data}
    User question: {payload.question or ''}

    Return valid JSON with keys summary and enhanced_code.
    Ensure chart_data itself is NOT part of your direct output, but inform your summary.
    """,
            }

            resp = self.client.chat.completions.create(model="gpt-4o", messages=[system_msg, user_msg], temperature=0)
            ai_text = resp.choices[0].message.content
            if ai_text is None:
                raise HTTPException(status_code=500, detail="OpenAI returned no content.")
            ai_text = ai_text.strip().lstrip("```json").rstrip("```").strip()


            try:
                return json.loads(ai_text)
            except json.JSONDecodeError:
                raise HTTPException(status_code=500, detail=f"OpenAI returned invalid JSON:\n{ai_text}")

    # --------------------  GPT‑DRIVEN ANALYSIS (Code Generation) -------------------- #
    def analyze(self, req: DataFrameRequest) -> Dict[str, Any]:
        """
        Prompts an LLM to generate Python code for data analysis based on user's question.
        The generated code should output analysis_data, chart_data (for ECharts),
        and a parameterized_answer.
        This method focuses on generating the code; execution happens elsewhere.
        """
        if req.data is None:
            raise HTTPException(status_code=400, detail="No data provided for analysis. The 'data' field in DataFrameRequest is required.")
        if not req.data: # Handles empty list case
             raise HTTPException(status_code=400, detail="Data provided is empty. Cannot perform analysis.")

        try:
            df = pd.DataFrame(req.data)
            if df.empty:
                sample_data_to_show = "DataFrame is empty based on provided data."
            else:
                sample_data_to_show = df.head(10).to_string() # Show up to 10 rows
        except Exception as e:
            print(f"Error creating DataFrame: {e}")
            raise HTTPException(status_code=400, detail=f"Error creating DataFrame from provided data. Please check data format.")

        cols = ", ".join(df.columns) if not df.empty else "No columns (empty DataFrame)"
        num_rows = len(df)
        num_cols = len(df.columns)


        history_parts = []
        for turn in req.chat_history:
            turn_str = f"User: {turn.question}\nAssistant: {turn.answer}"
            if turn.code:
                # Use markdown code fences for clarity
                turn_str += f"\nAssistant's Code:\n```python\n{turn.code}\n```"
            history_parts.append(turn_str)

        # Join turns with a separator to make the history log clear
        history_str = "\n---\n".join(history_parts)

        prompt = f"""
You are a data analysis assistant. Your task is to generate Python code to analyze a pandas DataFrame named 'df'.
The DataFrame 'df' has {num_rows} rows and {num_cols} columns.
Column names in df: {cols}

Sample Data (up to the first 10 rows of 'df' for schema understanding only):
{sample_data_to_show}
Please note: The number of rows shown in this sample IS NOT an instruction on how many results your Python code should return, unless the user's question specifically refers to this sample size.

IMPORTANT INSTRUCTIONS FOR PYTHON CODE GENERATION:
1.  **Adhere to User-Specified Item Counts:** If the user's question asks for a specific number of items (e.g., "top 10 customers"), your Python code MUST produce exactly that number of items. Use methods like `df.nlargest(10, 'some_column')` or `df.head(10)`. If the dataset has fewer items than requested, return all available items.
2.  **Output Format:** The Python code you generate MUST print a single JSON string to standard output. This JSON string must be the *only* thing printed. The JSON string must contain three top-level keys:
    * `"analysis_data"`: A Python dictionary containing the raw data points for the analysis.
    * `"chart_data"`: A Python dictionary formatted specifically for ECharts.
    * `"parameterized_answer"`: A string that acts as a template for the final textual summary, using placeholders like `{{some_value}}`.
3.  **CRITICAL: Use the NumpyEncoder for JSON Serialization:** To prevent `TypeError` exceptions from numpy data types (like `int64`), your code MUST define and use a custom `NumpyEncoder` class. The final `print()` statement must call `json.dumps` with `cls=NumpyEncoder`. Follow the example below exactly.
4.  **No Direct Plotting:** Do NOT include any Python code that attempts to display or render plots itself (e.g., avoid `matplotlib.pyplot.show()`).
5.  **Available Libraries:** Assume `pandas` (as `pd`), `json`, and `numpy` (as `np`) are available.
6.  **DataFrame Name:** The pandas DataFrame will be available under the variable name `df`.

Example of the required JSON output format from the generated Python script:
```json
{{
  "analysis_data": {{
    "mean_sales": 4550.75,
    "top_product": "Gadget Pro",
    "top_product_sales": 12500,
    "num_products": 15
  }},
  "chart_data": {{
    "title": {{ "text": "Top 5 Products by Sales" }},
    "xAxis": {{ "type": "category", "data": ["Gadget Pro", "Widget Max", "Thingamajig", "Super Gizmo", "Device Plus"] }},
    "yAxis": {{ "type": "value" }},
    "series": [{{ "name": "Sales", "type": "bar", "data": [12500, 11000, 9800, 8500, 7200] }}]
  }},
  "parameterized_answer": "The analysis reviewed {{num_products}} products. The top-selling product is '{{top_product}}' with sales of {{top_product_sales}}. The average sales across all products is {{mean_sales}}."
}}
Your response (as the LLM assistant generating the code) MUST be a raw JSON object containing only one key: "code". The value for this key should be the complete Python script as a string.

Example of YOUR response format:
{{{{
  "code": "import json\\nimport pandas as pd\\nimport numpy as np\\n\\n# df is pre-defined. Example: User asked for 'top 3 countries by population'\\n\\n# CRITICAL: Define a robust JSON encoder to handle numpy types\\nclass NumpyEncoder(json.JSONEncoder):\\n    def default(self, obj):\\n        if isinstance(obj, np.integer):\\n            return int(obj)\\n        elif isinstance(obj, np.floating):\\n            return float(obj)\\n        elif isinstance(obj, np.ndarray):\\n            return obj.tolist()\\n        return super(NumpyEncoder, self).default(obj)\\n\\n# Analysis\\nN = 3\\nresult_df = df.nlargest(N, 'population')\\nmean_pop = df['population'].mean()\\ncountry_count = df['country'].nunique()\\ntop_country_name = result_df.iloc[0]['country']\\ntop_country_pop = result_df.iloc[0]['population']\\n\\n# Data for the final JSON output.\\n# No need for manual int() or float() casting here because the Encoder will handle it.\\nanalysis_data_dict = {{\\n    'top_country': top_country_name,\\n    'top_population': top_country_pop,\\n    'average_population': mean_pop,\\n    'total_countries': country_count,\\n    'top_n_count': N\\n}}\\n\\nparameterized_answer_str = \\"Out of {{total_countries}} countries, the most populous is {{top_country}} with {{top_population}} people. The average population is {{average_population}}. This chart shows the top {{top_n_count}} countries.\\"\\n\\nchart_data_for_echarts = {{\\n    'title': {{'text': f'Top {{N}} Countries by Population'}},\\n    'xAxis': {{'type': 'category', 'data': result_df['country'].tolist()}},\\n    'yAxis': {{'type': 'value'}},\\n    'series': [{{'name': 'Population', 'type': 'bar', 'data': result_df['population'].tolist()}}]\\n}}\\n\\n# Final print statement is crucial and MUST use the custom encoder\\nprint(json.dumps({{\\n    'analysis_data': analysis_data_dict,\\n    'chart_data': chart_data_for_echarts,\\n    'parameterized_answer': parameterized_answer_str\\n}}, cls=NumpyEncoder))"
}}}}

---
Here is the conversation history for context. The user may be asking a follow-up question
based on the summary or the code from a previous turn.

--- CONVERSATION HISTORY START ---
{history_str}
--- CONVERSATION HISTORY END ---

Based on all the instructions and the history provided above, please respond to the following user question.

User question: "{req.question or 'Perform a general descriptive analysis.'}
"""
        sys_msg = {"role": "system", "content": "Respond with raw JSON containing only key 'code'."}
        user_msg = {"role": "user", "content": prompt}

        resp = self.client.chat.completions.create(model="gpt-4o", messages=[sys_msg, user_msg], temperature=0)
        txt = resp.choices[0].message.content
        if txt is None:
            return {"openai_result": {"code": None, "error": "OpenAI returned no content."}}
        txt = txt.strip().lstrip("```json").rstrip("```").strip()
        try:
            return {"openai_result": json.loads(txt)}
        except json.JSONDecodeError:
            return {"openai_result": {"code": None, "error": f"Invalid JSON from OpenAI: {txt}"}}

    # --------------------  CODE CORRECTION -------------------- #
    def correct_code(
        self,
        original_code: str,
        error_message: str,
        df_head: pd.DataFrame, #DataFrame head for context
        row_count: int,
        col_count: int,
        user_question: str,
    ) -> Dict[str, Any]:
        cols = ", ".join(df_head.columns)
        sys_msg = {
            "role": "system",
            "content": "Fix the Python code. Return JSON with key 'code' only, no commentary. The corrected code must perform analysis and print a JSON string to stdout with 'analysis_data' and 'chart_data' keys, similar to the original intention.",
        }
        user_msg = {
            "role": "user",
            "content": f"""
        I will provide an error, the original Python code that caused it, and context about the pandas DataFrame it was running on. Your task is to fix the code and return ONLY a raw JSON object with the single key "code".

        ---
        EXAMPLE
        Error:
        TypeError: can only concatenate str (not "int") to str on column 'A'

        Original code that caused the error:
        print(df['A'].sum())

        DataFrame context: The code will run on a DataFrame 'df' with 10 rows and 2 columns.
        Column names: A, B
        DataFrame head:
        A  B
        0  1  x
        1  2  y
        2  z  z

        User question that the original code was trying to answer: "What is the sum of column A?"

        Expected JSON response:
        {{
        "code": "df['A'] = pd.to_numeric(df['A'], errors='coerce')\\nprint(df['A'].sum())"
        }}
        ---

        Here is the real request.

        Error:
        {error_message}

        Original code that caused the error:
        {original_code}

        DataFrame context: The code will run on a DataFrame 'df' with {row_count} rows and {col_count} columns.
        Column names: {cols}
        DataFrame head:
        {df_head.to_string()}

        User question that the original code was trying to answer: {user_question}

        Return raw JSON: {{ "code": "fixed python code" }}
        The fixed code should print a JSON string to stdout with 'analysis_data' and 'chart_data'.
        """,
        }

        resp = self.client.chat.completions.create(model="gpt-4o", messages=[sys_msg, user_msg], temperature=0)
        txt = resp.choices[0].message.content
        if txt is None:
            return {"openai_result": {"code": None, "error": "OpenAI returned no content during code correction."}}
        txt = txt.strip().lstrip("```json").rstrip("```").strip()
        try:
            return {"openai_result": json.loads(txt)}
        except json.JSONDecodeError:
            return {"openai_result": {"code": None, "error": f"Invalid JSON from OpenAI during code correction: {txt}"}}

    # --------------------  FULL ANALYSIS (on Synthetic Data) -------------------- #
    async def full_analysis(self, req: DataFrameRequest) -> Dict[str, Any]:
            """
            Performs LLM-driven analysis on self.synthetic_dataset, then (optionally)
            executes LLM-suggested enhanced code. Always returns a summary that matches
            the code which ultimately produced the chart & analysis data.
            """
            # ─────────────────────── sanity checks ───────────────────────
            if self.synthetic_dataset is None:
                raise HTTPException(status_code=400, detail="Synthetic dataset not available.")

            # --- SESSION HANDLING LOGIC ---
            if req.conversation_id and req.conversation_id in self.chat_sessions:
                # Existing conversation
                conversation_id = req.conversation_id
                chat_history = self.chat_sessions[conversation_id]
            else:
                # New conversation
                conversation_id = str(uuid.uuid4())
                chat_history = []
                self.chat_sessions[conversation_id] = chat_history
            # --- END SESSION HANDLING ---

            synthetic_req = DataFrameRequest(
                question=req.question,
                data=self.synthetic_dataset,
                # We don't need to pass the ID down, just the retrieved history
                chat_history=chat_history,
            )

            analysis_resp = self.analyze(synthetic_req)
            code = analysis_resp["openai_result"].get("code")
            if not code:
                detail = analysis_resp["openai_result"].get(
                    "error",
                    "GPT did not return valid code for synthetic data.",
                )
                raise HTTPException(status_code=500, detail=detail)

            # ─────────────────── helpers / context ────────────────────────
            df_syn      = pd.DataFrame(self.synthetic_dataset)
            df_head_syn = df_syn.head()
            rows_syn, cols_syn = len(df_syn), len(df_syn.columns)
            current_q   = req.question or ""
            max_attempt = 3

            async def run_with_correction(code_snippet: str, question: str) -> Any:
                """Try code, auto-correct via LLM up to `max_attempt` times."""
                current_code = code_snippet
                for attempt in range(max_attempt):
                    print(f"Full Analysis: attempt {attempt+1}\n{current_code}")
                    exec_resp = await self.execute_code(
                        ExecuteCodeWithDataPayload(
                            code=current_code,
                            data=self.synthetic_dataset,
                        )
                    )
                    result = exec_resp["result"]

                    if isinstance(result, str) and result.startswith("Error executing code:"):
                        if attempt == max_attempt - 1:
                            return result
                        print("Correction round due to:", result)
                        corr = self.correct_code(
                            original_code=current_code,
                            error_message=result,
                            df_head=df_head_syn,
                            row_count=rows_syn,
                            col_count=cols_syn,
                            user_question=question,
                        )
                        corrected_code = corr["openai_result"].get("code")
                        if not corrected_code:
                            return f"Error executing code: correction produced no new code. Last error: {result}"
                        current_code = corrected_code
                        continue
                    return result
                return f"Error executing code: exhausted attempts. Last error: {result}"

            # ───────────── 2) execute initial code (with fixes) ────────────
            exec_res_initial = await run_with_correction(code, current_q)
            if isinstance(exec_res_initial, str) and exec_res_initial.startswith("Error executing code:"):
                raise HTTPException(status_code=500, detail=exec_res_initial)

            # ───────────── 3) summarise initial execution result ───────────
            summary_data = self.summarize(
                SummarizePayload(
                    execution_result=exec_res_initial,
                    code=code,
                    question=current_q,
                )
            )

            final_json_output   = (
                json.loads(exec_res_initial)
                if isinstance(exec_res_initial, str)
                else exec_res_initial
            )
            final_code_executed = code

            # ───────────── 4) try LLM-provided enhanced_code ───────────────
            enhanced_code = summary_data.get("enhanced_code")
            if enhanced_code:
                print("Full Analysis: executing enhanced code")
                exec_res_enhanced = await run_with_correction(enhanced_code, current_q)

                enhanced_ok = not (
                    isinstance(exec_res_enhanced, str)
                    and exec_res_enhanced.startswith("Error executing code:")
                )

                if enhanced_ok:
                    print("Full Analysis: enhanced code succeeded.")
                    summary_data = self.summarize(
                        SummarizePayload(
                            execution_result=exec_res_enhanced,
                            code=enhanced_code,
                            question=current_q,
                        )
                    )
                    final_json_output = (
                        json.loads(exec_res_enhanced)
                        if isinstance(exec_res_enhanced, str)
                        else exec_res_enhanced
                    )
                    final_code_executed = enhanced_code
                else:
                    print("Full Analysis: enhanced code failed, using initial result.")

            # ───────────── 5) build & return response ──────────────────────
            if not isinstance(final_json_output, dict):
                raise HTTPException(
                    status_code=500,
                    detail="Execution result is not a valid JSON object.",
                )
            final_summary = summary_data.get("summary")
            if final_summary:
                self.chat_sessions[conversation_id].append(
                    ChatTurn(
                        question=req.question,
                        answer=final_summary,
                        code=final_code_executed  # Include the code that was executed
                    )
                )

            return {
                "summary":               summary_data.get("summary"),
                "parameterized_summary": final_json_output.get("parameterized_answer"),
                "generated_code":        final_code_executed,
                "chart_data":            final_json_output.get("chart_data"),
                "analysis_data":         final_json_output.get("analysis_data"),
                "conversation_id":       conversation_id,
            }



    # --------------------  LLM‑FREE ANALYSIS (on Original Data) -------------------- #
    async def llm_free_analysis(self, req: LLMFreeAnalysisRequest) -> Dict[str, Any]:
            """
            Executes provided code against a dataset.
            The executed code is expected to print a JSON object containing 'parameterized_answer',
            'chart_data', and 'analysis_data'. This method structures that output.
            """
            data_for_execution: Optional[List[Dict[str, Any]]] = None
            if req.data is not None:
                print("LLM-Free Analysis: Using data provided in the request.")
                data_for_execution = req.data
            elif self.dataset is not None:
                print("LLM-Free Analysis: Using self.dataset (original data).")
                data_for_execution = self.dataset
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Original dataset not available and no data provided in request for llm_free_analysis."
                )

            exec_resp = await self.execute_code(
                ExecuteCodeWithDataPayload(code=req.code, data=data_for_execution)
            )
            result = exec_resp["result"]

            if isinstance(result, str) and result.startswith("Error executing code:"):
                raise HTTPException(status_code=400, detail=f"Error executing provided code in llm_free_analysis: {result}")

            try:
                parsed_result = json.loads(result) if isinstance(result, str) else result
                if not isinstance(parsed_result, dict):
                    raise HTTPException(status_code=500, detail=f"LLM-free execution result is not a valid JSON object: {parsed_result}")
            except json.JSONDecodeError:
                raise HTTPException(status_code=500, detail=f"LLM-free execution result is not valid JSON: {result}")

            return {
                # Use the 'parameterized_answer' from the code output as the primary summary.
                "summary": parsed_result.get("parameterized_answer") or parsed_result.get("parameterized_summary") or parsed_result.get("summary"),
                "chart_data": parsed_result.get("chart_data"),
                # Directly get the 'analysis_data' object.
                "analysis_data": parsed_result.get("analysis_data"),
                "code": req.code, # The code that was executed
            }

    async def execute_code(self, payload: ExecuteCodeWithDataPayload) -> Dict[str, Any]:
        """
        Placeholder for the actual code execution logic.
        This would typically involve a sandboxed environment.
        For this example, it simulates execution and expects the Python code
        to print a JSON string to stdout.
        The 'df' variable should be available in the scope of the executed code.
        """
        # This is a mock execution. In a real scenario, you'd use a secure execution environment.
        # e.g., using restricted_exec, a Docker container, or a dedicated microservice.
        
        # For testing, let's try to actually execute it if pandas is available
        # This is highly simplified and UNSAFE for untrusted code.
        # A proper implementation requires a secure sandbox.

        if payload.data is None:
             # Fallback to self.dataset if payload.data is not provided, common for llm_free_analysis.
             # For full_analysis, data should always be explicitly provided (synthetic_dataset).
            if self.dataset is not None and not payload.data: # Check if service has original data
                 df = pd.DataFrame(self.dataset)
            else: # No data provided and no default original data. Code might not expect df or might fail.
                 df = pd.DataFrame() # Or raise error if df is always expected
                 # print("Warning: execute_code called with no data and no self.dataset available.")
        else:
            df = pd.DataFrame(payload.data)

        # Create a string buffer to capture stdout
        import sys
        from io import StringIO

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        global_vars = {'pd': pd, 'df': df, 'json': json} # Make pandas, df, and json available

        try:
            exec(payload.code, global_vars)
            output = captured_output.getvalue()
            # Try to parse it as JSON, assuming the code prints JSON
            try:
                # If code prints multiple things, only the last JSON is usually desired,
                # or the code must be structured to print only one JSON.
                # This simplistic approach takes the whole output.
                parsed_output = json.loads(output)
                return {"result": parsed_output}
            except json.JSONDecodeError:
                 # If not JSON, return the raw string output (might be an error or just text)
                if not output and payload.code: # Code ran but printed nothing
                    return {"result": "Error executing code: Code executed but produced no JSON output."}
                return {"result": output if output else "Error executing code: Code executed but produced no output."}

        except Exception as e:
            return {"result": f"Error executing code: {type(e).__name__}: {str(e)}"}
        finally:
            sys.stdout = old_stdout # Restore stdo

