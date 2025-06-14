# import base64
# import re
# import traceback
# import unicodedata
# import json
# from typing import Optional, Tuple
# import asyncio
# from fastapi.concurrency import run_in_threadpool
# import pandas as pd
# import dataframe_image as dfi
# from openai import OpenAI

# # ——— DEFAULT CONFIG ———
# IMG_PATH_DEFAULT = "table_screenshot.png"
# HEAD_ROWS_DEFAULT = 20
# OPENAI_MODEL_DEFAULT = "gpt-4.1"
# MAX_ATTEMPTS_DEFAULT = 5


# def visualize_hidden_chars(s: str) -> str:
#     """Convert control/non-standard whitespace chars into visible \\u escapes."""
#     out = []
#     for ch in s:
#         if (ch.isspace() and ord(ch) != 0x20) or unicodedata.category(ch).startswith("C"):
#             out.append(f"\\u{ord(ch):04X}")
#         else:
#             out.append(ch)
#     return "".join(out)


# def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
#     """Expose hidden chars in each column name."""
#     df = df.copy()
#     df.columns = [visualize_hidden_chars(str(c)) for c in df.columns]
#     return df


# def prepare_preview(df: pd.DataFrame, head_rows: int) -> pd.DataFrame:
#     """Take the top rows and expose hidden chars in any string cell."""
#     df_preview = df.head(head_rows).copy()
#     return df_preview.map(lambda x: visualize_hidden_chars(x) if isinstance(x, str) else x)


# def _export_styled_sync(styled, img_path: str):
#     """
#     Blocking call to dataframe_image + Playwright.
#     Must be run off the event loop.
#     """
#     dfi.export(styled, img_path, table_conversion="playwright")
    
# async def style_and_export(df: pd.DataFrame, img_path: str):
#     """
#     Style the DataFrame to preserve whitespace, then offload the
#     blocking Playwright screenshot to a thread so we don’t block the event loop.
#     """
#     styled = (
#         df.style
#           .format(na_rep="")
#           .set_table_styles([
#               {"selector": "th, td", "props": [("white-space", "pre")]}
#           ])
#     )
#     # offload to thread
#     await run_in_threadpool(_export_styled_sync, styled, img_path)


# def encode_image_to_data_uri(img_path: str) -> str:
#     """Read an image file and return a data URI (base64)."""
#     with open(img_path, "rb") as f:
#         b64 = base64.b64encode(f.read()).decode("utf-8")
#     return f"data:image/png;base64,{b64}"


# def extract_json_block(text: str) -> str:
#     """Extract the JSON object from a fenced block or plain text."""
#     m = re.search(r"```json\s*(?P<json>\{.*?\})\s*```", text, re.DOTALL)
#     if m:
#         return m.group("json").strip()
#     m2 = re.search(r"(\{.*\})", text, re.DOTALL)
#     return m2.group(1).strip() if m2 else text.strip()


# def interpret_classification(content: str) -> Optional[str]:
#     """Parse JSON response and return a classification code."""
#     raw = extract_json_block(content)
#     try:
#         data = json.loads(raw)
#     except json.JSONDecodeError:
#         print("Failed to parse LLM JSON. Raw response:")
#         print(raw)
#         return None
#     cls = data.get("Cleanup_Classification", "").strip().upper()
#     if cls == "PIVOTED":
#         return "PIVOT"
#     if cls == "CLEAN":
#         return "CLEAN"
#     if cls == "REQUIRES_CLEANUP":
#         return "CLEANUP"
#     return None


# def build_classification_prompt(data_uri: str) -> dict:
#     """Construct the user message payload with embedded image."""
#     return {
#         "role": "user",
#         "content": [
#             {"type": "text", "text": (
#                     "Summarize what's in this image of a snippet of a dataset. "
#                     "Identify any structural clean up required for the dataset to be analytics ready. "
#                     "Classify required cleanup as either PIVOTED if any multilevel headers are detected, if any PIVOT is detected, prioritize this output"
#                     "CLEAN if no cleanup required, OR REQUIRES_CLEANUP for anything else. "
#                     "Output a JSON with keys Summary and Cleanup_Classification."
#             )},
#             {"type": "image_url", "image_url": {"url": data_uri}}
#         ]
#     }


# async def classify_dataframe(
#     df: pd.DataFrame,
#     img_path: str = IMG_PATH_DEFAULT,
#     head_rows: int = HEAD_ROWS_DEFAULT,
#     model: str = OPENAI_MODEL_DEFAULT,
#     max_attempts: int = MAX_ATTEMPTS_DEFAULT
# ) -> Tuple[Optional[str], Optional[str]]:


#     df_clean = clean_column_names(df)
#     preview = prepare_preview(df_clean, head_rows)
#     # make sure the image is actually written before moving on
#     await style_and_export(preview, img_path)
#     data_uri = encode_image_to_data_uri(img_path)

#     # 2️⃣ LLM call
#     client = OpenAI()
#     messages = [build_classification_prompt(data_uri)]
#     content = None
#     for attempt in range(1, max_attempts + 1):
#         try:
#             resp = client.chat.completions.create(
#                 model=model,
#                 messages=messages,
#                 max_tokens=2000
#             )
#             content = resp.choices[0].message.content.strip()
#             break
#         except Exception as err:
#             print(f"Attempt {attempt} failed: {err}")
#             traceback.print_exc()
#     if content is None:
#         raise RuntimeError(f"All {max_attempts} attempts failed.")

#     # 3️⃣ Interpret and return
#     code = interpret_classification(content)
#     return code, content

# New logic with clean up code generator
import base64
import re
import traceback
import unicodedata
import json
from typing import Optional, Tuple, List, Dict, Any
import asyncio # Retained if other parts of a larger application use it
from fastapi.concurrency import run_in_threadpool # For running blocking IO in async context
import pandas as pd
import dataframe_image as dfi
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam # For type hinting

# ——— DEFAULT CONFIG ———
IMG_PATH_DEFAULT = "table_screenshot.png"
HEAD_ROWS_DEFAULT = 20
OPENAI_MODEL_DEFAULT = "gpt-4.1" # Updated to a vision-capable model
MAX_ATTEMPTS_DEFAULT = 5
MAX_TOKENS_CLASSIFICATION = 2000
MAX_TOKENS_CODE_GENERATION = 4000 # Increased for potentially longer code output


def visualize_hidden_chars(s: str) -> str:
    """Convert control/non-standard whitespace chars into visible \\u escapes."""
    out = []
    for ch in s:
        if (ch.isspace() and ord(ch) != 0x20) or unicodedata.category(ch).startswith("C"):
            out.append(f"\\u{ord(ch):04X}")
        else:
            out.append(ch)
    return "".join(out)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Expose hidden chars in each column name. Returns a new DataFrame."""
    df_copy = df.copy()
    df_copy.columns = [visualize_hidden_chars(str(c)) for c in df_copy.columns]
    return df_copy


def prepare_preview(df: pd.DataFrame, head_rows: int) -> pd.DataFrame:
    """Take the top rows and expose hidden chars in any string cell. Returns a new DataFrame."""
    df_preview = df.head(head_rows).copy()
    # Apply visualize_hidden_chars to all string cells
    for col in df_preview.columns:
        if df_preview[col].dtype == 'object': # Process only object columns (potential strings)
            df_preview[col] = df_preview[col].apply(lambda x: visualize_hidden_chars(x) if isinstance(x, str) else x)
    return df_preview

# ───────────────────────────────────────────────────────────────────────────────
# 1.  DataFrame → nicely styled HTML table
# ───────────────────────────────────────────────────────────────────────────────
def build_excel_like_style(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """
    Return a Styler that looks much closer to what Excel shows on screen:
      • Calibri 11 pt
      • thin grey gridlines
      • no pandas index
    """
    return (
        df.style
        #   .hide(axis="index")                              # drop index col  :contentReference[oaicite:0]{index=0}
          .format(na_rep="")                               # blank for NaN
          .set_table_styles([
              # table-wide defaults
              {"selector": "table",
               "props": [("border-collapse", "collapse"),
                          ("font-family", "Calibri, Arial, sans-serif"),
                          ("font-size", "11pt")]},
              # header row
              {"selector": "th",
               "props": [("border", "1px solid #d0d0d0"),
                          ("background-color", "#f3f3f3"),
                          ("padding", "4px 6px"),
                          ("text-align", "left")]},
              # body cells
              {"selector": "td",
               "props": [("border", "1px solid #d0d0d0"),
                          ("padding", "4px 6px"),
                          ("white-space", "pre")]}
          ])
    )


# ───────────────────────────────────────────────────────────────────────────────
# 2.  Off-thread export (Playwright via dataframe_image)
# ───────────────────────────────────────────────────────────────────────────────
def _export_styled_sync(styler: pd.io.formats.style.Styler,
                        img_path: str) -> None:
    """
    Heavy lifting happens here (blocking), so keep it in a thread.
    """
    dfi.export(styler, img_path, table_conversion="playwright")

def strip_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace any column label that looks like “Unnamed: n” (or is literally None / '')
    with the empty string so the header cell renders blank – exactly what Excel
    shows when a CSV has no header for that field.
    """
    df = df.copy()
    df.columns = [
        "" if (c is None
               or str(c).strip() == ""
               or re.fullmatch(r"Unnamed: \d+", str(c))) else c
        for c in df.columns
    ]
    return df




async def style_and_export(df: pd.DataFrame,
                           img_path: str,
                           head_rows: int = 20) -> None:
    preview = (
        df.head(head_rows)            # first N rows only
          .reset_index(drop=True)     # drop pandas row index
          .pipe(strip_unnamed_columns)   # ← NEW LINE
    )

    styled = build_excel_like_style(preview)
    await run_in_threadpool(_export_styled_sync, styled, img_path)



# def _export_styled_sync(styled_df, img_path: str):
#     """
#     Blocking call to dataframe_image + Playwright.
#     Must be run off the event loop in an async application.
#     """
#     dfi.export(styled_df, img_path, table_conversion="playwright")
    
# async def style_and_export(df: pd.DataFrame, img_path: str):
#     """
#     Style the DataFrame to preserve whitespace, then offload the
#     blocking Playwright screenshot to a thread so we don’t block the event loop.
#     """
#     styled = (
#         df.style
#           .format(na_rep="") # Represent NaN as empty string
#           .set_table_styles([
#               {"selector": "th, td", "props": [("white-space", "pre")]} # Preserve whitespace
#           ])
#     )
#     await run_in_threadpool(_export_styled_sync, styled, img_path)


def encode_image_to_data_uri(img_path: str) -> str:
    """Read an image file and return a data URI (base64)."""
    with open(img_path, "rb") as f:
        b64_bytes = base64.b64encode(f.read())
        b64_string = b64_bytes.decode("utf-8")
    return f"data:image/png;base64,{b64_string}"


# def extract_json_block(text: str) -> str:
#     """Extract the JSON object from a fenced block or plain text."""
#     # Try to find JSON within ```json ... ```
#     match_json_block = re.search(r"```json\s*(?P<json>\{.*?\})\s*```", text, re.DOTALL)
#     if match_json_block:
#         return match_json_block.group("json").strip()
#     # If not found, try to find any JSON object pattern
#     match_any_json = re.search(r"(\{.*\})", text, re.DOTALL)
#     if match_any_json:
#         return match_any_json.group(1).strip()
#     return text.strip() # Fallback to the original text


# def interpret_classification(content: str) -> Optional[str]:
#     """Parse JSON response and return a classification code."""
#     raw_json_str = extract_json_block(content)
#     try:
#         data = json.loads(raw_json_str)
#     except json.JSONDecodeError:
#         print(f"Failed to parse LLM JSON for classification. Raw response part: '{raw_json_str[:200]}...'")
#         return None
    
#     classification = data.get("Cleanup_Classification", "").strip().upper()
    
#     if "PIVOTED" in classification or "PIVOT" in classification : # More robust check
#         return "PIVOT"
#     if "CLEAN" == classification: # Exact match for CLEAN
#         return "CLEAN"
#     if "REQUIRES_CLEANUP" in classification or "CLEANUP" in classification: # More robust check
#         return "CLEANUP"
    
#     print(f"Unknown classification value: {data.get('Cleanup_Classification')}")
#     return None

def extract_json_block(text: str) -> str: # Your provided function
    """Extract the JSON object from a fenced block or plain text."""
    match_json_block = re.search(r"```json\s*(?P<json>\{.*?\})\s*```", text, re.DOTALL)
    if match_json_block:
        return match_json_block.group("json").strip()
    match_any_json = re.search(r"(\{.*\})", text, re.DOTALL)
    if match_any_json:
        return match_any_json.group(1).strip()
    return text.strip()

def interpret_classification(content: str) -> Optional[str]:
    """Parse JSON response and return a classification code."""
    raw_json_str = extract_json_block(content)
    try:
        data = json.loads(raw_json_str)
    except json.JSONDecodeError:
        print(f"Failed to parse LLM JSON for classification. Raw response part: '{raw_json_str[:200]}...'")
        return None

    classification = data.get("Cleanup_Classification", "").strip().upper()

    # Add the new classification check here. Order might be important if terms overlap.
    if "MULTI_VALUE_CELLS_SPLIT" in classification: # Check for the new type first
        return "MULTI_VALUE_CELLS_SPLIT"
    if "PIVOTED" in classification or "PIVOT" in classification :
        return "PIVOT"
    if "CLEAN" == classification: # Exact match for CLEAN
        return "CLEAN"
    # "REQUIRES_CLEANUP" or "CLEANUP" should ideally be more general.
    # If MULTI_VALUE_CELLS_SPLIT is a *type* of cleanup, ensure it's caught by the more specific case above.
    if "REQUIRES_CLEANUP" in classification or "CLEANUP" in classification:
        return "CLEANUP" # This is for general cleanup code generation

    print(f"Unknown classification value: {data.get('Cleanup_Classification')}")
    return None # Or perhaps a default like "UNKNOWN"

def build_classification_prompt(data_uri: str) -> Dict[str, Any]:
    """Construct the user message payload with embedded image for classification."""
    return {
        "role": "user",
        "content": [
            {
                "type": "text", 
                "text": (
                    "Summarize what's in this image of a snippet of a dataset. "
                    "Identify any structural clean up required for the dataset to be analytics ready. "
                    "Classify required cleanup as either:\n"
                    "1. 'PIVOTED': If any multilevel headers are detected, and needs unpivoting. Prioritize this if applicable.\n"
                    "2. 'CLEAN': If the data snippet appears clean and ready for analysis with no obvious structural issues or data type inconsistencies.\n"
                    "3. 'REQUIRES_CLEANUP': For any other issues like , mixed values in columns, messy strings, structural problems not covered by PIVOTED (e.g., multiple tables in one, footer rows,multiple values in one cell etc.).Pay attention to dates, make sure columns that have names that appear to be dates and data that is date like is converted to a date type, reformat dates to DD-MM-YYYY.\n"
                    ''' 4. 'MULTI_VALUE_CELLS_SPLIT': If two or more columns contain multiple distinct values within single cells, separated by a common delimiter, requiring these cells to be split into new rows.
                        Output a JSON object with keys: 'Summary', 'Cleanup_Classification'.
                        If 'Cleanup_Classification' is 'MULTI_VALUE_CELLS_SPLIT', also include a 'Split_Parameters' key with an object containing:
                            'id_cols' (list of strings),
                            'value_cols_to_explode' (list of strings),
                            'delimiter' (string),
                            'data_types' (object, e.g., {"column_name": "float"})
                        '''
                                            "Output a JSON object with two keys: 'Summary' (a brief text summary of the data and findings) and 'Cleanup_Classification' (one of 'PIVOTED', 'CLEAN', or 'REQUIRES_CLEANUP')."
                )
            },
            {
                "type": "image_url", 
                "image_url": {"url": data_uri}
            }
        ]
    }

async def get_llm_classification(
    data_uri: str,
    model: str = OPENAI_MODEL_DEFAULT,
    max_attempts: int = MAX_ATTEMPTS_DEFAULT
) -> Tuple[Optional[str], Optional[str]]:
    """Gets classification from LLM based on the data URI of the DataFrame image."""
    client = OpenAI()
    # Ensure messages is a list of ChatCompletionMessageParam
    messages: List[ChatCompletionMessageParam] = [build_classification_prompt(data_uri)] # type: ignore 
                                                                                      # Casting to ChatCompletionMessageParam for clarity
                                                                                      # as build_classification_prompt returns a dict.
    raw_content = None
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"Attempting classification LLM call (Attempt {attempt}/{max_attempts})...")
            resp = await run_in_threadpool( # Run blocking OpenAI call in threadpool
                client.chat.completions.create,
                model=model,
                messages=messages,
                max_tokens=MAX_TOKENS_CLASSIFICATION,
                temperature=0.2 # Lower temperature for more deterministic classification
            )
            raw_content = resp.choices[0].message.content
            if raw_content:
                 raw_content = raw_content.strip()
                 break # Success
            else:
                print(f"Classification LLM Attempt {attempt} returned empty content.")
        except Exception as err:
            print(f"Classification LLM Attempt {attempt} failed: {err}")
            traceback.print_exc()
            if attempt == max_attempts:
                return None, f"All {max_attempts} classification attempts failed. Last error: {err}"
    
    if not raw_content:
        return None, "LLM returned no content for classification after all attempts."

    classification_code = interpret_classification(raw_content)
    return classification_code, raw_content

def extract_python_block(text: str) -> Optional[str]:
    """Extracts Python code from a markdown fenced block."""
    # Pattern to find ```python ... ``` or ``` ... ``` (non-greedy)
    match = re.search(r"```(?:python\s*)?(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: if the entire text seems to be code (e.g., no ``` found but looks like code)
    # This is risky, so for now, we are strict.
    # If LLM consistently forgets ```, this part might need adjustment or prompt reinforcement.
    return None

async def generate_and_test_cleanup_code(
    original_preview_df: pd.DataFrame,
    original_dtypes: Dict[str, str], # <<< NEW PARAMETER: Dtypes of the original preview
    data_uri_for_llm_vision: str,
    image_preview_columns: List[str], # Column names as they appear in the image
    head_rows: int,
    model: str = OPENAI_MODEL_DEFAULT,
    max_llm_iterations: int = MAX_ATTEMPTS_DEFAULT
) -> Optional[str]:
    """
    Generates and iteratively tests Python code for cleaning a DataFrame,
    providing original dtypes to the LLM.
    """
    client = OpenAI()
    
    # initial_system_message = "You are an expert Python programmer specializing in pandas DataFrame manipulation for data cleaning. Your goal is to write robust and correct Python code to generalized normalized code ready for machine learning and data analytics workflows."
    initial_system_message ="You are an expert Python programmer specializing in pandas DataFrame manipulation for data cleaning and preparation. Your primary goal is to generate robust, correct, and generalized Python code to transform raw CSV data into a normalized format, ready for machine learning and data analytics workflows. A key aspect of this transformation is to ensure data atomicity, where each cell contains a single value, by appropriately expanding multi-value cells into new rows or columns. Before generating code, you will first outline a step-by-step plan for the cleaning process."
    # --- Prepare dtypes string for the prompt ---
    dtypes_string = json.dumps(original_dtypes, indent=2) # Format dtypes as a readable JSON string

    # initial_user_prompt_text = (
    #     f"An image of a pandas DataFrame snippet is provided (see below). This image is for visual context.\n"
    #     f"The character data within the cells of this image has been randomly generated but preserves the original data's structure, length, and character types (e.g., uppercase, lowercase, digits, special characters). Use the *structure and format* of this faked data in the image to help infer data types and identify cleaning needs, but remember your code operates on the original data.\n"
    #     f"The column names *as displayed in the image* are: {image_preview_columns}.\n"
    #     f"String cell values *as displayed in the image* might also have had hidden/control characters visualized (e.g., as \\uXXXX).\n\n"
    #     f"Your task is to write a Python script to clean an *original* pandas DataFrame. "
    #     f"For the script you write, this original DataFrame will be provided to your code as a variable named `df`.\n"
    #     f"- `df` is a pandas DataFrame.\n"
    #     f"- `df` contains the first {head_rows} rows of the full original dataset.\n"
    #     f"- `df.columns` will be the *original, raw* column names (e.g., 'User Name', not 'User\\u0020Name').\n"
    #     f"- String data within `df` also contains raw characters, not the \\uXXXX visualized versions shown in the image.\n"
    #     f"- The pandas dtypes of the columns in the *original* `df` (before your cleaning script runs) are:\n" # <<< DTYPES INFO ADDED
    #     f"```json\n{dtypes_string}\n```\n\n" # <<< DTYPES INFO ADDED
    #     f"Based on your interpretation of the provided image AND the original dtypes, write a Python script to perform necessary cleaning operations on `df`. "
    #     F"separate values like this Binders | Art | Phones | Fasteners | Paper into individual rows each with Art in one row, Plones in the next etc"
    #     f"Focus on tasks like: correcting data types (e.g., `df['col'].astype(int)` - be mindful of the original dtype and potential errors), "
    #     f"string manipulations (e.g., `df['col'].str.strip()`), removing unnecessary rows/columns, standardizing values, etc. "
    #     f"Make sure each data point is in its own cell, when data points are concatenated separate into different rows "
    #     f"When there are mutiple values in one cell, create new rows for each data point, do not make up new columns, only allowed to create explode into more rows"
    #     f"Always make sire the final table is a flat table"
    #     f"Make sure the final table does not have arrays or lists in a cell"
    #     f"When values are split into new rows, this should results in more rows than the original. Map each exploded value to the corresponding value in other columns"
    #     f"The goal is to make `df` analytics-ready.\n\n"
    #     f"IMPORTANT INSTRUCTIONS:\n"
    #     f"1. Provide ONLY the Python code for cleaning. Enclose it in a single markdown code block starting with ```python and ending with ```.\n"
    #     f"2. Do NOT include any explanatory text outside this code block (comments within the Python code are encouraged).\n"
    #     f"3. Assume `import pandas as pd` is already executed. Your code will operate on the DataFrame named `df`.\n"
    #     f"4. Write robust code. For example, before operating on a column, check if it exists (`if 'col_name' in df.columns:`) and handle potential errors during type conversion (e.g., using `pd.to_numeric(errors='coerce')`).\n"
    #     f"5. State assumptions briefly in comments in your code."
    # )
    initial_user_prompt_text = (
        f"You will be provided with information about a pandas DataFrame that needs cleaning. "
        f"Your task is to first devise a detailed plan and then write a Python script to clean `df`.\n\n"

        f"After planning, write the Python script for `df`.\n\n"
        f"**Context for Cleaning `df`:**\n"
        f"- Image snippet (below) shows data structure (faked data, real format).\n"
        f"- Image column names: {image_preview_columns}.\n"
        f"- Visualized chars (\\uXXXX) are not in raw `df` data.\n\n"
        f"**Information about *original* `df`:**\n"
        f"- `df`: pandas DataFrame, first {head_rows} rows.\n"
        f"- `df.columns`: original raw names.\n"
        f"- Original dtypes:\n"
        f"```json\n{dtypes_string}\n```\n\n"
        f"**IMPORTANT SCRIPTING INSTRUCTIONS:**\n"
        f"1.  ONLY Python code in ```python ... ``` blocks.\n"
        f"2.  NO text outside the code block. Comments INSIDE code are essential.\n"
        f"3.  Assume `import pandas as pd` and `import numpy as np`. Operate directly on `df`.\n"
        f"4.  Robust code: check column existence, handle errors gracefully.\n"
        f"5.  State any assumptions in code comments.\n"
        f"5.  If there are cells with multiple values create new columns for each with a count of the number of occurrences.\n"
        f"6.  If there is metadata included, denomalize to each row,find where to split the metadata into column name and value e.g colon. do not drop any information.\n"

    )


    conversation_history: List[ChatCompletionMessageParam] = [
        {"role": "system", "content": initial_system_message}, # type: ignore
        {
            "role": "user", # type: ignore
            "content": [ # type: ignore
                {"type": "text", "text": initial_user_prompt_text},
                {"type": "image_url", "image_url": {"url": data_uri_for_llm_vision}}
            ]
        }
    ]

    for attempt in range(1, max_llm_iterations + 1):
        print(f"Attempting cleanup code generation LLM call (Attempt {attempt}/{max_llm_iterations})...")
        try:
            resp = await run_in_threadpool( # Run blocking OpenAI call in threadpool
                client.chat.completions.create,
                model=model,
                messages=conversation_history,
                max_tokens=MAX_TOKENS_CODE_GENERATION,
                temperature=0
            )
            llm_response_content = resp.choices[0].message.content
            if not llm_response_content:
                print("LLM returned empty content for cleanup code.")
                conversation_history.append({"role": "assistant", "content": ""}) # type: ignore
                conversation_history.append({"role": "user", "content": "You returned an empty response. Please provide the Python code as requested."}) # type: ignore
                continue

            conversation_history.append({"role": "assistant", "content": llm_response_content}) # type: ignore

            extracted_code = extract_python_block(llm_response_content)

            if not extracted_code:
                print("Failed to extract Python code block from LLM response.")
                error_feedback_prompt = (
                    "Your response did not contain a valid Python code block (e.g., ```python ... ```). "
                    "Please provide ONLY the Python code for cleaning, enclosed in a single markdown code block."
                )
                conversation_history.append({"role": "user", "content": error_feedback_prompt}) # type: ignore
                continue

            print(f"\n--- Generated Code (Attempt {attempt}) ---\n{extracted_code}\n------------------------------")
            
            # Test the extracted code using the original preview data
            df_test = original_preview_df.copy()
            exec_globals = {'pd': pd, 'df': df_test}
            
            print("Testing generated code...")
            await run_in_threadpool(exec, extracted_code, exec_globals) # exec is blocking
            print("Generated code executed successfully!")
            return extracted_code # Success!

        except Exception as e:
            error_message = traceback.format_exc()
            print(f"Attempt {attempt} code execution failed:\n{error_message}")
            
            if attempt < max_llm_iterations:
                # Add original dtypes to the error feedback prompt as well for context
                error_feedback_prompt = (
                    f"The previously generated Python code resulted in an error when executed on the DataFrame `df`:\n"
                    f"```traceback\n{error_message}\n```\n\n"
                    f"Remember, the original dtypes of the `df` before cleaning were:\n"
                    f"```json\n{dtypes_string}\n```\n"
                    f"Please analyze this error and the context. Provide a corrected Python script. \n"
                    f"Key reminders for your corrected script:\n"
                    f"- The DataFrame is named `df`.\n"
                    f"- `df.columns` are the original raw column names.\n"
                    f"- The image provided earlier shows a snippet of the data for visual context only.\n"
                    f"- Provide ONLY the Python code in a single markdown code block (```python ... ```).\n"
                    f"- Ensure the corrected code directly addresses the root cause of this error and handles potential data type issues."
                )
                conversation_history.append({"role": "user", "content": error_feedback_prompt}) # type: ignore
            else:
                print("Max LLM iterations reached for code correction.")
                break
                
    print(f"Failed to generate working cleanup code after {max_llm_iterations} attempts.")
    return None


# async def main_analysis_and_cleanup_pipeline(
#     df: pd.DataFrame,
#     img_path: str = IMG_PATH_DEFAULT,
#     head_rows: int = HEAD_ROWS_DEFAULT,
#     model: str = OPENAI_MODEL_DEFAULT,
#     max_attempts_classification: int = MAX_ATTEMPTS_DEFAULT,
#     max_llm_iterations_cleanup: int = MAX_ATTEMPTS_DEFAULT
# ):
#     """Orchestrates the DataFrame analysis, classification, and conditional cleanup code generation."""
#     print("Starting DataFrame analysis and cleanup pipeline...")

#     # 1. Prepare previews
#     # original_preview is a slice of the original df, used for testing the generated cleanup code
#     original_preview_df = df.head(head_rows).copy() 
#     original_dtypes_dict = {col: str(dtype) for col, dtype in original_preview_df.dtypes.items()}
#     print(f"Created original preview with {len(original_preview_df)} rows.")

#     # df_cleaned_cols has column names with hidden characters visualized (e.g., spaces as \u0020)
#     df_cleaned_cols = clean_column_names(df)
    
#     # prepared_preview_for_image has visualized hidden chars in both column names and cell string data
#     # This is what the LLM will "see" in the image.
#     prepared_preview_for_image = prepare_preview(df_cleaned_cols, head_rows)
#     print("Prepared preview for image generation.")

#     # 2. Generate image and data_uri (once)
#     print(f"Exporting styled DataFrame preview to {img_path}...")
#     await style_and_export(prepared_preview_for_image, img_path)
#     data_uri = encode_image_to_data_uri(img_path)
#     print("Encoded image to data URI.")

#     # 3. Classify DataFrame structure
#     print("Classifying DataFrame structure using LLM...")
#     classification_code, raw_classification_content = await get_llm_classification(
#         data_uri, model, max_attempts_classification
#     )

#     print(f"\n--- Classification Result ---")
#     print(f"Code: {classification_code}")
#     if raw_classification_content:
#         try:
#             # Try to pretty print if it's JSON
#             parsed_json = json.loads(extract_json_block(raw_classification_content))
#             print(f"Summary: {parsed_json.get('Summary', 'N/A')}")
#         except json.JSONDecodeError:
#             print(f"Raw Classification Response: {raw_classification_content[:500]}...") # Print snippet
#     print("-----------------------------")


#     generated_cleanup_code = None
#     if classification_code == "CLEANUP":
#         print("\nDataset classified as 'REQUIRES_CLEANUP'. Attempting to generate cleanup code...")
#         generated_cleanup_code = await generate_and_test_cleanup_code(
#             original_preview_df=original_preview_df,
#             original_dtypes=original_dtypes_dict,
#             data_uri_for_llm_vision=data_uri,
#             image_preview_columns=list(prepared_preview_for_image.columns),
#             head_rows=head_rows,
#             model=model,
#             max_llm_iterations=max_llm_iterations_cleanup
#         )
#         if generated_cleanup_code:
#             print("\n✅ Successfully generated and validated cleanup code:")
#             print("----------------------------------------------------")
#             print(generated_cleanup_code)
#             print("----------------------------------------------------")
            
#             print("\nApplying generated code to a copy of the original_preview_df for demonstration:")
#             df_test_final = original_preview_df.copy()
#             try:
#                 # Using run_in_threadpool for exec as it might be long-running for complex scripts
#                 await run_in_threadpool(exec, generated_cleanup_code, {'pd': pd, 'df': df_test_final})
#                 print("\n--- Original Preview DataFrame After Cleanup ---")
#                 # print(df_test_final.head())
#                 print(df_test_final.iloc[:5, :5])
#                 print("---------------------------------------------")
#             except Exception as e:
#                 print(f"🚨 Error applying final generated code to original_preview_df for demonstration: {e}")
#                 traceback.print_exc()
#         else:
#             print("\n❌ Failed to generate working cleanup code after multiple attempts.")
#     elif classification_code == "PIVOT":
#         print("\nDataset classified as 'PIVOTED'. Pivoting often requires complex, specific logic. Manual review and intervention are recommended.")
#     elif classification_code == "CLEAN":
#         print("\nDataset classified as 'CLEAN'. No automated cleanup actions initiated.")
#     else:
#         print("\nUnknown classification or classification failed. No cleanup actions initiated.")
    
#     print("\nDataFrame analysis and cleanup pipeline finished.")
#     return classification_code, raw_classification_content, generated_cleanup_code
async def main_analysis_and_cleanup_pipeline(
    df: pd.DataFrame,
    img_path: str,
    head_rows: int,
    model: str,
    max_attempts_classification: int,
    max_llm_iterations_cleanup: int
):
    """Orchestrates the DataFrame analysis, classification, and conditional cleanup code generation."""
    print("Starting DataFrame analysis and cleanup pipeline...")

    # 1. Prepare previews
    original_preview_df = df.head(head_rows).copy()
    original_dtypes_dict = {col: str(dtype) for col, dtype in original_preview_df.dtypes.items()}
    print(f"Created original preview with {len(original_preview_df)} rows.")

    # Assuming df_cleaned_cols and prepared_preview_for_image are prepared here
    # For example, using your functions:
    # df_cleaned_cols = clean_column_names(df) # If you have this
    # prepared_preview_for_image = prepare_preview(df_cleaned_cols, head_rows) # If you have this
    # For simplicity if these are not the issue, use original_preview_df or df.head()
    prepared_preview_for_image = df.head(head_rows).copy() # Use a copy for styling
    print("Prepared preview for image generation.")

    # --- RESTORED IMAGE GENERATION LOGIC ---
    # 2. Generate image and data_uri (once)
    print(f"Exporting styled DataFrame preview to {img_path}...")
    await style_and_export(prepared_preview_for_image, img_path) # Ensure this function works
    data_uri = encode_image_to_data_uri(img_path) # Ensure this function works and creates a valid URI
    print(f"Encoded image to data URI: {data_uri[:100]}...") # Print a snippet of the URI
    # --- END RESTORED LOGIC ---

    # 3. Classify DataFrame structure
    print("Classifying DataFrame structure using LLM...")
    classification_code, raw_classification_content = await get_llm_classification(
        data_uri, model, max_attempts_classification
    )

    print(f"\n--- Classification Result ---")
    print(f"Code: {classification_code}")
    
    split_params = None

    if raw_classification_content:
        try:
            json_str = extract_json_block(raw_classification_content)
            parsed_json_content = {}
            if json_str:
                parsed_json_content = json.loads(json_str)
            elif raw_classification_content.strip().startswith("{"): # Basic check if raw content itself is JSON
                parsed_json_content = json.loads(raw_classification_content.strip())
            else:
                print("Warning: No valid JSON found in raw_classification_content.")

            if parsed_json_content: # Proceed only if parsing was successful
                print(f"Summary: {parsed_json_content.get('Summary', 'N/A')}")
                if classification_code == "MULTI_VALUE_CELLS_SPLIT":
                    if 'Split_Parameters' in parsed_json_content:
                        split_params = parsed_json_content['Split_Parameters']
                        print(f"Extracted Split_Parameters: {split_params}")
                    else:
                        print("Warning: Classification is MULTI_VALUE_CELLS_SPLIT but Split_Parameters not found in LLM response.")
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse raw_classification_content as JSON: {e}")
            print(f"Raw Classification Response (first 500 chars): {raw_classification_content[:500]}...")
        except Exception as e:
            print(f"Error processing LLM classification response: {e}")
    print("-----------------------------")

    generated_cleanup_code = None
    if classification_code == "CLEANUP":
        print("\nDataset classified as 'REQUIRES_CLEANUP'. Attempting to generate cleanup code...")
        generated_cleanup_code = await generate_and_test_cleanup_code(
            original_preview_df=original_preview_df,
            original_dtypes=original_dtypes_dict,
            data_uri_for_llm_vision=data_uri,
            image_preview_columns=list(prepared_preview_for_image.columns),
            head_rows=head_rows,
            model=model,
            max_llm_iterations=max_llm_iterations_cleanup
        )
        if generated_cleanup_code:
            print("\n✅ Successfully generated and validated cleanup code.")
            # (rest of your existing logging/testing for generated_cleanup_code)
        else:
            print("\n❌ Failed to generate working cleanup code after multiple attempts.")
    elif classification_code == "PIVOT":
        print("\nDataset classified as 'PIVOTED'. Pivoting handled by specialized function.")
    elif classification_code == "MULTI_VALUE_CELLS_SPLIT":
        print("\nDataset classified as 'MULTI_VALUE_CELLS_SPLIT'. Splitting handled by specialized function.")
    elif classification_code == "CLEAN":
        print("\nDataset classified as 'CLEAN'. No automated cleanup actions initiated.")
    else:
        print(f"\nUnknown or unhandled classification ('{classification_code}'). No cleanup actions initiated.")
    
    print("\nDataFrame analysis and cleanup pipeline finished.")
    return classification_code, raw_classification_content, generated_cleanup_code, split_params
# Example Usage (Async context needed to run this)
async def run_example():
    # Create a sample DataFrame that might need cleaning
    data = {
        'Name\\u0020ID': [' Alice  ', 'Bob', ' Charlie ', 'David\\u0000', None, 'Eve'], # Mixed spacing, nulls, hidden char
        ' Score ': [' 50', '85 ', '  N/A  ', '77', '92', '60%'], # Mixed types, spaces, text N/A, percentage
        'Join Date': ['2023-01-01', '2022/05/15', '2024-Mar-10', 'Unknown', '2023-11-20', '2022-07-01'], # Mixed date formats, text
        '   Status   ': ['Active', 'Inactive', 'Active', 'Pending Review', 'Active', ' Inactive '] # Extra spaces in col name and values
    }
    sample_df = pd.DataFrame(data)
    print("--- Original Sample DataFrame ---")
    print(sample_df)
    print("-------------------------------\n")

    # Ensure dataframe_image is configured (if needed, e.g., for browser path)
    # dfi.config.set_matplotlib(False) # Example: if using playwright and want to ensure it
    # dfi.config.set_chrome_path("path/to/your/chrome") # If auto-detection fails

    try:
        classification, raw_class_resp, cleanup_script = await main_analysis_and_cleanup_pipeline(sample_df)
        # You can now do something with the results, e.g., save the script, apply it to the full df, etc.
    except Exception as e:
        print(f"An error occurred during the pipeline: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    # To run the async example:
    # In a Jupyter notebook, you can often just `await run_example()` in a cell.
    # In a .py file, you'd use `asyncio.run()`:
    asyncio.run(run_example())
