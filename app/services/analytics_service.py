"""
A class that holds:
• OpenAI‑powered code‑generation & correction
• CSV or JSON upload support

All endpoints can work with either:
  1. an explicit data list passed in the request, or
  2. the dataset that was previously uploaded via /upload_data
"""
import io
import json
import math
import traceback
import uuid
from contextlib import redirect_stdout
from typing import Any, Dict, List, Optional

# Third-party imports
import numpy as np
import pandas as pd
from app.services.llm_providers import (AnthropicProvider, GoogleProvider,
                                        LLMProvider, OpenAIProvider)

from fastapi import HTTPException

MAX_ATTEMPTS_DEFAULT = 5

from app.models.schemas import (ChatTurn, DataFrameRequest,
                                ExecuteCodeWithDataPayload, SummarizePayload)


def _sanitize_nan(obj):           
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nan(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


class DataAnalysisService:
    """
    Central point for every operation.
    After a successful `/upload_data`, the dataset lives in `self.dataset`
    and is automatically used by subsequent calls when the user omits
    the `data` field.
    """

    def __init__(self) -> None:
        self.dataset: Optional[List[Dict[str, Any]]] = None  # persistent store
        self.synthetic_dataset: Optional[List[Dict[str, Any]]] = None 
        self.profile_original: Optional[Dict[str, Any]] = None
        self.profile_synthetic: Optional[Dict[str, Any]] = None
        self.csv_path: Optional[str] = None
        self.chat_sessions: Dict[str, List[ChatTurn]] = {}
        self.providers: Dict[str, LLMProvider] = {
            "openai": OpenAIProvider(),
            "google": GoogleProvider(),
            "anthropic": AnthropicProvider(),
        }

    def _get_provider_for_model(self, model_name: str) -> LLMProvider:
        """Selects the correct provider based on the model name prefix."""
        if model_name.startswith("gpt"):
            return self.providers["openai"]
        elif model_name.startswith("gemini"):
            return self.providers["google"]
        elif model_name.startswith("claude"):
            return self.providers["anthropic"]
        else:
            # Default to OpenAI or raise an error
            raise ValueError(f"Unsupported model provider for model: {model_name}")

    async def check_llm_connection(self, model_name: str, api_key: str) -> Dict[str, Any]:
        """
        Attempts a low-cost operation with the provider to verify the key and model access.
        """
        print(f"--- Verifying connection for model: {model_name} ---")
        try:
            provider = self._get_provider_for_model(model_name)
            # We'll delegate the actual check to the provider instance
            is_successful, message = await provider.verify_connection(model=model_name, api_key=api_key)

            if is_successful:
                return {"status": "success", "message": message}
            else:
                raise Exception(message)

        except Exception as e:
            print(f"!!! Connection check failed: {e}")
            # Re-raise to be caught by the route handler, which will return a 400 error
            raise e
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
    
    async def _summarize_execution(self, execution_result: str | dict,
                            code: str,
                            question: str = "") -> Dict[str, Any]:
        """
        Wrapper that calls self.summarize() and returns its JSON,
        making sure execution_result is already JSON-serialisable.
        """
        return await self.summarize(
            SummarizePayload(
                execution_result=execution_result,
                code=code,
                question=question,
            )
        )

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

        # Return a success response
        return {
            "message": f"Successfully received and processed {len(df_anonymized)} rows of anonymized data."
        }

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


    async def summarize(self, payload: SummarizePayload, model: str, api_key: str) -> Dict[str, Any]:
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
    """""
            }

            messages = [system_msg, user_msg]

            # --- CHANGE: Use the provider system ---
            provider = self._get_provider_for_model(model)
            try:
                ai_text = await provider.chat_completion(
                    model=model,
                    messages=messages,
                    temperature=0,
                    api_key=api_key
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"LLM provider error during summary: {e}")

            if ai_text is None:
                raise HTTPException(status_code=500, detail="LLM provider returned no content.")
    
            ai_text = ai_text.strip().lstrip("```json").rstrip("```").strip()


            try:
                return json.loads(ai_text)
            except json.JSONDecodeError:
                raise HTTPException(status_code=500, detail=f"OpenAI returned invalid JSON:\n{ai_text}")

    # --------------------  GPT‑DRIVEN ANALYSIS (Code Generation) -------------------- #
    async def analyze(self, req: DataFrameRequest, api_key: str) -> Dict[str, Any]:
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

**CRITICAL RULE: DO NOT, under any circumstances, write any code to read a file (e.g., `pd.read_csv(...)` or `pd.read_json(...)`). The DataFrame `df` is pre-defined and ready to use. Any code that tries to open or read a file will fail.**

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
  "code": "import json\nimport pandas as pd\nimport numpy as np\n\n# df is pre-defined. Example: User asked for 'top 3 countries by population'\n\n# CRITICAL: Define a robust JSON encoder to handle numpy types\nclass NumpyEncoder(json.JSONEncoder):\n    def default(self, obj):\n        if isinstance(obj, np.integer):\n            return int(obj)\n        elif isinstance(obj, np.floating):\n            return float(obj)\n        elif isinstance(obj, np.ndarray):\n            return obj.tolist()\n        return super(NumpyEncoder, self).default(obj)\n\n# Analysis\nN = 3\nresult_df = df.nlargest(N, 'population')\nmean_pop = df['population'].mean()\ncountry_count = df['country'].nunique()\ntop_country_name = result_df.iloc[0]['country']\ntop_country_pop = result_df.iloc[0]['population']\n\n# Data for the final JSON output.\n# No need for manual int() or float() casting here because the Encoder will handle it.\nanalysis_data_dict = {{\n    'top_country': top_country_name,\n    'top_population': top_country_pop,\n    'average_population': mean_pop,\n    'total_countries': country_count,\n    'top_n_count': N\n}}\n\nparameterized_answer_str = \"Out of {{total_countries}} countries, the most populous is {{top_country}} with {{top_population}} people. The average population is {{average_population}}. This chart shows the top {{top_n_count}} countries.\"\n\nchart_data_for_echarts = {{\n    'title': {{'text': f'Top {{N}} Countries by Population'}},\n    'xAxis': {{'type': 'category', 'data': result_df['country'].tolist()}},\n    'yAxis': {{'type': 'value'}},\n    'series': [{{'name': 'Population', 'type': 'bar', 'data': result_df['population'].tolist()}}]\n}}\n\n# Final print statement is crucial and MUST use the custom encoder\nprint(json.dumps({{\n    'analysis_data': analysis_data_dict,\n    'chart_data': chart_data_for_echarts,\n    'parameterized_answer': parameterized_answer_str\n}}, cls=NumpyEncoder))"
}}}}

---
Here is the conversation history for context. The user may be asking a follow-up question
based on the summary or the code from a previous turn.

--- CONVERSATION HISTORY START ---
{history_str}
--- CONVERSATION HISTORY END ---

Based on all the instructions and the history provided above, please respond to the following user question.

User question: "{req.question or 'Perform a general descriptive analysis.'}"
"""
        sys_msg = {"role": "system", "content": "Respond with raw JSON containing only key 'code'."}
        user_msg = {"role": "user", "content": prompt}
        messages = [sys_msg, user_msg]
        
        # --- CHANGE: Use the provider system ---
        provider = self._get_provider_for_model(req.model)
        try:
            print(f"analyze: Calling provider for model '{req.model}'...")
            txt = await provider.chat_completion(
                model=req.model,
                messages=messages,
                temperature=0,
                api_key=api_key
            )
            # This is the log you added, which is great for debugging
            print(f"analyze: Received content from provider:\n{txt}")
            
        except Exception as e:
            print(f"!!! analyze: LLM provider raised an exception: {e}")
            # ALWAYS return a dictionary on failure
            return {"openai_result": {"code": None, "error": f"LLM provider error: {e}"}}

        # --- 3. Handle empty response from provider ---
        if not txt:
            print("!!! analyze: Provider returned an empty or None response.")
            return {"openai_result": {"code": None, "error": "LLM provider returned no content."}}
        
        # --- 4. Clean and parse the response ---
        # This handles the "```json" fences correctly
        cleaned_txt = txt.strip().lstrip("```json").rstrip("```").strip()
        
        try:
            result = {"openai_result": json.loads(cleaned_txt)}
            print("analyze: JSON parsing successful. Returning result.")
            # ALWAYS return a dictionary on success
            return result
        except json.JSONDecodeError:
            print(f"!!! analyze: JSONDecodeError. The LLM returned invalid JSON: {cleaned_txt}")
            # ALWAYS return a dictionary on failure
            return {"openai_result": {"code": None, "error": f"Invalid JSON format from LLM provider."}}        

    # --------------------  CODE CORRECTION -------------------- #
    async def correct_code(
        self,
        original_code: str,
        error_message: str,
        df_head: pd.DataFrame, #DataFrame head for context
        row_count: int,
        col_count: int,
        user_question: str,
        model: str,      # <-- Add model
        api_key: str     # <-- Add api_key
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
        "code": "df['A'] = pd.to_numeric(df['A'], errors='coerce')\nprint(df['A'].sum())"
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
        """""
        }

        messages = [sys_msg, user_msg]

        # --- CHANGE: Use the provider system ---
        provider = self._get_provider_for_model(model)
        try:
            txt = await provider.chat_completion(
                model=model,
                messages=messages,
                temperature=0,
                api_key=api_key
            )
        except Exception as e:
            # Return a dict, don't raise HTTPException from this helper
            return {"openai_result": {"code": None, "error": f"LLM provider error during code correction: {e}"}}

        if txt is None:
            return {"openai_result": {"code": None, "error": "LLM provider returned no content during code correction."}}
        
        txt = txt.strip().lstrip("```json").rstrip("```").strip()
        try:
            return {"openai_result": json.loads(txt)}
        except json.JSONDecodeError:
            return {"openai_result": {"code": None, "error": f"Invalid JSON from OpenAI during code correction: {txt}"}}

    # --------------------  FULL ANALYSIS (on Synthetic Data) -------------------- #
    async def full_analysis(self, req: DataFrameRequest, api_key: str) -> Dict[str, Any]:
        """
        **Efficiently** performs LLM-driven analysis on `self.synthetic_dataset`.

        This revised method streamlines the process to a single primary LLM call
        for code generation, followed by an execution and correction loop. It avoids
        the expensive, chained summarization and enhancement calls of the previous version.

        Workflow:
        1.  **Code Generation:** Call the LLM once to generate the initial analysis code.
        2.  **Execution & Correction Loop:**
            - Execute the generated code.
            - If it fails, enter a correction loop (max 3 attempts) where the LLM
              is asked to fix the code based on the error message.
            - If it succeeds, exit the loop.
        3.  **Summarization:** After successful execution, call the LLM a final time to
           summarize the results.
        4.  **Return:** Send back the complete analysis, including the final code,
           summary, and data.
        """
        # ─────────────────────── sanity checks ───────────────────────
        if self.synthetic_dataset is None:
            raise HTTPException(status_code=400, detail="Synthetic dataset not available.")

        # --- SESSION HANDLING LOGIC ---
        conversation_id = req.conversation_id or str(uuid.uuid4())
        if conversation_id not in self.chat_sessions:
            self.chat_sessions[conversation_id] = []
        chat_history = self.chat_sessions[conversation_id]

        # --- PREPARE REQUEST FOR LLM --- 
        synthetic_req = DataFrameRequest(
            question=req.question,
            data=self.synthetic_dataset,
            chat_history=chat_history,
            model=req.model
        )

        # ───────────── 1) Generate Initial Code ─────────────
        analysis_resp = await self.analyze(synthetic_req, api_key=api_key)
        initial_code = analysis_resp["openai_result"].get("code")
        if not initial_code:
            error_detail = analysis_resp["openai_result"].get("error", "LLM failed to return valid code.")
            raise HTTPException(status_code=500, detail=error_detail)

        # ─────────────────── helpers / context ────────────────────────
        df_syn = pd.DataFrame(self.synthetic_dataset)
        df_head_syn = df_syn.head()
        rows_syn, cols_syn = len(df_syn), len(df_syn.columns)
        current_q = req.question or ""
        max_attempts = 3
        final_code_executed = initial_code
        exec_res = None

        # ───────────── 2) Execute Code w/ Correction Loop ─────────────
        for attempt in range(max_attempts):
            print(f"Full Analysis: attempt {attempt + 1}\n{final_code_executed}")
            exec_resp = await self.execute_code(
                ExecuteCodeWithDataPayload(
                    code=final_code_executed,
                    data=self.synthetic_dataset,
                )
            )
            result = exec_resp["result"]

            # If execution was successful, break the loop
            if not (isinstance(result, str) and result.startswith("Error executing code:")):
                exec_res = result
                break

            # If it was the last attempt, raise an error
            if attempt == max_attempts - 1:
                raise HTTPException(status_code=500, detail=f"Code execution failed after {max_attempts} attempts. Last error: {result}")

            # Otherwise, try to correct the code
            print(f"Correction round due to: {result}")
            correction_resp = await self.correct_code(
                original_code=final_code_executed,
                error_message=result,
                df_head=df_head_syn,
                row_count=rows_syn,
                col_count=cols_syn,
                user_question=current_q,
                model=req.model,
                api_key=api_key
            )
            corrected_code = correction_resp["openai_result"].get("code")
            if not corrected_code:
                raise HTTPException(status_code=500, detail=f"LLM failed to correct the code. Last error: {result}")
            
            final_code_executed = corrected_code

        # ───────────── 3) Summarize Final Result ─────────────
        summary_data = await self.summarize(
            SummarizePayload(
                execution_result=exec_res,
                code=final_code_executed,
                question=current_q,
            ),
            model=req.model,
            api_key=api_key
        )

        # ───────────── 4) Build & Return Response ──────────────────────
        final_json_output = json.loads(exec_res) if isinstance(exec_res, str) else exec_res
        if not isinstance(final_json_output, dict):
            raise HTTPException(status_code=500, detail="Execution result is not a valid JSON object.")

        final_summary = summary_data.get("summary")
        if final_summary:
            self.chat_sessions[conversation_id].append(
                ChatTurn(
                    question=req.question,
                    answer=final_summary,
                    code=final_code_executed
                )
            )

        return {
            "summary": summary_data.get("summary"),
            "parameterized_summary": final_json_output.get("parameterized_answer"),
            "generated_code": final_code_executed,
            "chart_data": final_json_output.get("chart_data"),
            "analysis_data": final_json_output.get("analysis_data"),
            "conversation_id": conversation_id,
        }