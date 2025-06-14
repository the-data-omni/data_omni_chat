# from typing import Optional, Tuple
# import pandas as pd
# import traceback


# def find_header_data_split_df(
#     df: pd.DataFrame,
#     max_check: int = 20,
#     min_id_cols: int = 1
# ) -> Tuple[Optional[int], Optional[int]]:
#     """
#     Attempts to find the index of the ID header row and
#     the start index of the main data block in a DataFrame.

#     Returns:
#         (id_header_row_index, data_start_row_index) or (None, None)
#     """
#     # Work on a copy with string dtype for robust checking
#     df_sample = df.iloc[:max_check].astype(str)
#     id_header_row_index = None

#     # Scan backwards to detect the header/data boundary
#     for i in range(df_sample.shape[0] - 1, 0, -1):
#         current = df_sample.iloc[i]
#         next_row = df_sample.iloc[i + 1] if i + 1 < df_sample.shape[0] else None

#         # find first blank cell (pivot start)
#         pivot_start = next((j for j, v in enumerate(current)
#                             if pd.isna(v) or not str(v).strip()),
#                            len(current))
#         # ensure at least min_id_cols identifier columns
#         if pivot_start >= min_id_cols:
#             id_part = current.iloc[:pivot_start]
#             pivot_part = current.iloc[pivot_start:]
#             # id_part must be non-null, pivot_part must be blank
#             if (not id_part.isnull().any() and not (id_part == '').any() and
#                     (pivot_part.isnull() | (pivot_part == '')).all()):
#                 # next row must be data (id columns non-null)
#                 if next_row is not None:
#                     next_id_part = next_row.iloc[:pivot_start]
#                     if (not next_id_part.isnull().any() and not (next_id_part == '').any()):
#                         id_header_row_index = i
#                         break

#     if id_header_row_index is None:
#         return None, None
#     # data starts immediately after header
#     return id_header_row_index, id_header_row_index + 1


# def restructure_pivoted_dataframe(df: pd.DataFrame) -> Optional[pd.DataFrame]:
#     """
#     Restructures a DataFrame with multiple header rows (pivoted layout)
#     into a long-form table. Behaves identically to the CSV-based version,
#     but operates directly on the passed DataFrame.

#     Returns:
#         Cleaned DataFrame or None if an error occurs.
#     """
#     # copy and stringify
#     df_raw = df.reset_index(drop=True).astype(str)
#     try:
#         # 1. Detect header/data split
#         header_row, data_start = find_header_data_split_df(df_raw)
#         if header_row is None or header_row < 1:
#             print("Error: Could not determine header/data split or no pivot detected.")
#             return None
#         num_pivot_levels = header_row

#         # 2. Extract header block
#         header_block = df_raw.iloc[:num_pivot_levels + 1]
#         id_row = header_block.iloc[num_pivot_levels]
#         # pivot starts at first blank in id_row
#         pivot_start_col = next((j for j, v in enumerate(id_row)
#                                  if pd.isna(v) or not str(v).strip()),
#                                 None)
#         if pivot_start_col is None or pivot_start_col < 1:
#             print("Error: Could not determine pivot start column.")
#             return None
#         # identifier column names
#         id_col_names = [str(x).strip() for x in id_row.iloc[:pivot_start_col]]
#         if any(not name for name in id_col_names):
#             print("Error: Invalid or missing ID column names.")
#             return None

#         # 3. Pivot level names
#         level_src_col = pivot_start_col - 1
#         pivot_level_names = []
#         for lvl in range(num_pivot_levels):
#             val = str(header_block.iloc[lvl, level_src_col]).strip()
#             pivot_level_names.append(val or f"Level_{lvl}")

#         # 4. Build MultiIndex tuples
#         # ID tuples: (id_name, '', '', ...)
#         id_placeholders = [''] * (num_pivot_levels - 1)
#         id_tuples = [(name, *id_placeholders) for name in id_col_names]
#         tuples = list(id_tuples)

#         # pivot header values per level, forward-fill blanks
#         pivot_values_per_level = []
#         for lvl in range(num_pivot_levels):
#             last = ''
#             vals = []
#             for v in header_block.iloc[lvl, pivot_start_col:]:
#                 s = str(v).strip()
#                 if s:
#                     last = s
#                 vals.append(last)
#             pivot_values_per_level.append(vals)

#         # validate equal lengths
#         num_cols = len(pivot_values_per_level[0])
#         if any(len(lst) != num_cols for lst in pivot_values_per_level):
#             print("Error: Pivot header rows have inconsistent column counts.")
#             return None

#         # combine into tuples and extend
#         for combo in zip(*pivot_values_per_level):
#             tuples.append(combo)

#         # create MultiIndex
#         mindex = pd.MultiIndex.from_tuples(tuples, names=pivot_level_names)

#         # 5. Extract data block
#         df_data = df_raw.iloc[data_start:].reset_index(drop=True)
#         # trim or error on mismatch
#         if df_data.shape[1] != len(mindex):
#             if df_data.shape[1] > len(mindex):
#                 df_data = df_data.iloc[:, :len(mindex)]
#             else:
#                 print(f"Error: Column count mismatch ({df_data.shape[1]} vs {len(mindex)}).")
#                 return None
#         df_data.columns = mindex

#         # 6. Melt to long form
#         df_melted = pd.melt(
#             df_data,
#             id_vars=id_tuples,
#             value_name='Value'
#         )
#         # rename ID columns
#         rename_map = {tup: name for tup, name in zip(id_tuples, id_col_names)}
#         df_melted = df_melted.rename(columns=rename_map)

#         # 7. Clean values
#         df_melted['Value'] = pd.to_numeric(
#             df_melted['Value'].replace('', pd.NA),
#             errors='coerce'
#         )
#         df_cleaned = df_melted.dropna(subset=['Value']).reset_index(drop=True)

#         # strip whitespace in ID columns
#         for col in id_col_names:
#             if col in df_cleaned.columns and df_cleaned[col].dtype == 'object':
#                 df_cleaned[col] = df_cleaned[col].str.strip()

#         # 8. Optional: convert date columns
#         for col in id_col_names:
#             if 'date' in col.lower() and col in df_cleaned:
#                 try:
#                     df_cleaned[col] = pd.to_datetime(
#                         df_cleaned[col], format='%d-%b-%y', errors='coerce'
#                     )
#                     if df_cleaned[col].isnull().all():
#                         print(f"Warning: Date format '%d-%b-%y' failed for all in '{col}'. Keeping object.")
#                 except Exception as e:
#                     print(f"Warning: Error parsing dates for '{col}': {e}")

#         return df_cleaned

#     except Exception as e:
#         print(f"Unexpected error during pivot restructuring: {e}")
#         traceback.print_exc()
#         return None
import os
import pandas as pd
import traceback
from typing import Optional, Tuple

def process_file(filepath: str):
    print(f"\n>>> Processing file: {filepath}")
    print("  Attempting restructuring with auto-detection (v3 - Fixed Level Names)…")
    cleaned_df = restructure_pivoted_csv_v3(filepath)

    if cleaned_df is not None:
        base, ext = os.path.splitext(filepath)
        output_filename = f"{base}_restructured_v3_fixed{ext}"
        cleaned_df.to_csv(output_filename, index=False)
        print(f"  Successfully restructured. Saved to: {output_filename}")
        print("  Preview of restructured data (first 5 rows):")
        print(cleaned_df.head().to_markdown(index=False))
    else:
        print(f"  Failed to restructure {filepath} using v3.")
    return cleaned_df

# find_header_data_split function remains the same
def find_header_data_split(filepath, max_check=20, min_id_cols=1):
    """
    Attempts to automatically find the index of the ID header row and
    the start row index of the main data block.
    """
    try:
        df_sample = pd.read_csv(filepath, header=None, nrows=max_check, dtype=str)
        id_header_row_index = None
        pivot_start_col = None
        for i in range(df_sample.shape[0] - 1, 0, -1):
            current_row = df_sample.iloc[i]
            next_row = df_sample.iloc[i + 1] if i + 1 < df_sample.shape[0] else None
            current_pivot_start = -1
            for j, val in enumerate(current_row):
                if pd.isna(val) or str(val).strip() == '':
                    current_pivot_start = j
                    break
            if current_pivot_start == -1: current_pivot_start = len(current_row)
            if current_pivot_start >= min_id_cols:
                 id_part_valid = not current_row.iloc[:current_pivot_start].isnull().any() and not (current_row.iloc[:current_pivot_start] == '').any()
                 pivot_part_blank = (current_row.iloc[current_pivot_start:].isnull() | (current_row.iloc[current_pivot_start:] == '')).all()
                 next_row_is_data = False
                 if next_row is not None:
                     next_row_id_part = next_row.iloc[:current_pivot_start]
                     if not next_row_id_part.isnull().any() and not (next_row_id_part == '').any():
                          next_row_is_data = True
                 if id_part_valid and pivot_part_blank and next_row_is_data:
                     id_header_row_index = i
                     pivot_start_col = current_pivot_start
                     break
        if id_header_row_index is not None:
            data_start_row_index = id_header_row_index + 1
            return id_header_row_index, data_start_row_index
        else:
            return None, None
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None, None
    except Exception as e:
        print(f"An error occurred during header/data split detection for {filepath}: {e}")
        return None, None


def restructure_pivoted_csv_v3(filepath):
    """
    Restructures a pivoted CSV with potentially multiple leading identifier
    columns and multiple pivot levels in the header. Attempts auto-detection.
    Corrected pivot level name detection.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pandas.DataFrame or None: The cleaned DataFrame, or None if an error occurs.
    """
    try:
        # --- 1. Detect Structure ---
        id_header_row_index, data_start_row_index = find_header_data_split(filepath)
        if id_header_row_index is None:
            print("Error: Could not automatically determine file structure (ID header/data start).")
            return None
        num_pivot_levels = id_header_row_index
        if num_pivot_levels < 1:
             print("Error: Detected structure implies no pivot levels.")
             return None

        # --- 2. Read Header Info & Identify Column Roles ---
        header_rows_to_read = id_header_row_index + 1
        df_header_info = pd.read_csv(filepath, header=None, nrows=header_rows_to_read, dtype=str)
        pivot_start_col = -1
        id_header_row = df_header_info.iloc[id_header_row_index]
        for i, val in enumerate(id_header_row):
            if pd.isna(val) or str(val).strip() == '':
                pivot_start_col = i
                break
        if pivot_start_col <= 0:
            print("Error: Could not determine pivot start column from detected ID header row.")
            return None
        id_col_names = [str(name).strip() for name in id_header_row[:pivot_start_col]]
        if not id_col_names or any(not name for name in id_col_names):
             print("Error: Invalid or missing ID column names.")
             return None

        # --- FIX: Get Pivot Level Names (from column before pivot_start_col) ---
        pivot_level_names = []
        level_name_src_col = pivot_start_col - 1 if pivot_start_col > 0 else 0 # Source col index

        for i in range(num_pivot_levels):
            level_name = ''
            # Check if src col index is valid before trying iloc
            if level_name_src_col < df_header_info.shape[1]:
                 val = df_header_info.iloc[i, level_name_src_col]
                 level_name = str(val).strip() if pd.notna(val) else '' # Get value and strip

            pivot_level_names.append(level_name if level_name else f'Level_{i}') # Use default if blank/invalid

        # --- 3. Create N-Level FULL MultiIndex ---
        multi_index_tuples = []
        id_placeholders = [''] * (num_pivot_levels -1) # Correctly handles num_pivot_levels=1
        id_tuples = [(name, *id_placeholders) for name in id_col_names]
        multi_index_tuples.extend(id_tuples)

        pivot_headers_list = []
        for i in range(num_pivot_levels):
             level_values = df_header_info.iloc[i, pivot_start_col:].tolist()
             current_level_val = ''
             filled_level_values = []
             for val in level_values:
                 str_val = str(val).strip()
                 if pd.notna(val) and str_val:
                     current_level_val = str_val
                 filled_level_values.append(current_level_val)
             pivot_headers_list.append(filled_level_values)

        num_pivot_columns = len(pivot_headers_list[0]) if pivot_headers_list else 0
        if any(len(lst) != num_pivot_columns for lst in pivot_headers_list):
             print("Error: Pivot header rows have inconsistent column counts.")
             return None

        pivot_tuples = list(zip(*pivot_headers_list))
        multi_index_tuples.extend(pivot_tuples)

        # Create the full MultiIndex object using CORRECTED level names
        multi_index = pd.MultiIndex.from_tuples(multi_index_tuples, names=pivot_level_names)


        # --- 4. Read data ---
        df_data = pd.read_csv(filepath, skiprows=data_start_row_index, header=None, dtype=str, keep_default_na=False)


        # --- 5. Assign FULL MultiIndex ---
        if df_data.shape[1] != len(multi_index):
             print(f"Error: Column count mismatch between data ({df_data.shape[1]}) and headers ({len(multi_index)}).")
             expected_cols = len(multi_index)
             if df_data.shape[1] > expected_cols:
                  print("Attempting to trim extra columns from data...")
                  df_data = df_data.iloc[:, :expected_cols]
             else: return None
        df_data.columns = multi_index


        # --- 6. Melt ---
        id_var_tuples = id_tuples
        df_melted = pd.melt(
            df_data,
            id_vars = id_var_tuples,
            value_name = 'Value',
        )


        # --- 7. Rename ID columns *after* melting ---
        rename_dict = {id_tuple: id_name for id_tuple, id_name in zip(id_var_tuples, id_col_names)}
        df_melted = df_melted.rename(columns=rename_dict)
        # Pivot level columns should now be named correctly (e.g., 'Ship Mode', 'Segment')


        # --- 8. Clean ---
        df_melted['Value'] = df_melted['Value'].replace('', pd.NA)
        df_melted['Value'] = pd.to_numeric(df_melted['Value'], errors='coerce')
        df_cleaned = df_melted.dropna(subset=['Value']).copy()


        # Optional: Clean ID columns
        for col in id_col_names:
            if col in df_cleaned.columns and df_cleaned[col].dtype == 'object':
                df_cleaned[col] = df_cleaned[col].str.strip()

        # Optional: Convert date column(s)
        for col in id_col_names:
             if "date" in col.lower() and col in df_cleaned.columns:
                  date_format = '%d-%b-%y'
                  try:
                      df_cleaned[col] = pd.to_datetime(df_cleaned[col], format=date_format, errors='coerce')
                      if df_cleaned[col].isnull().all():
                           print(f"Warning: Date format '{date_format}' failed for all values in '{col}'. Reverting to object.")
                           # Find original data to revert - safer to just leave as object if format fails
                           # df_cleaned[col] = df_data[col] # This wouldn't work as df_data is gone
                  except ValueError:
                      print(f"Warning: Could not parse dates in column '{col}' with format '{date_format}'. Keeping as object.")
                  except Exception as date_err:
                       print(f"Warning: Error converting date column '{col}': {date_err}. Keeping as object.")


        df_cleaned = df_cleaned.reset_index(drop=True)
        return df_cleaned

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An unexpected {type(e).__name__} occurred while processing {filepath}: {e}")
        traceback.print_exc()
        return None


