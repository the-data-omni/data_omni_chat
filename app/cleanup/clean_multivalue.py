import pandas as pd
import io

def generalize_csv_restructure(file_path_or_buffer, id_cols, value_cols_to_explode, delimiter, data_types=None):
    """
    Cleans and restructures a CSV file where specified columns contain multiple
    delimited values, creating new rows for each value set.

    Args:
        file_path_or_buffer (str or file-like object): Path to the CSV file or a buffer.
        id_cols (list of str): List of column names to be used as identifiers and
                               repeated for each new row.
        value_cols_to_explode (list of str): List of column names that contain delimited
                                          strings to be split. The order in this list
                                          matters as it defines how the split values
                                          are paired.
        delimiter (str): The delimiter string used in the value_cols_to_explode.
        data_types (dict, optional): A dictionary where keys are column names (from
                                     value_cols_to_explode) and values are the desired
                                     data types (e.g., {'Amount': float}).
                                     Defaults to None, meaning no explicit type conversion
                                     beyond string.

    Returns:
        pandas.DataFrame: A new DataFrame with restructured data.
    """
    if data_types is None:
        data_types = {}

    try:
        df = pd.read_csv(file_path_or_buffer)
    except FileNotFoundError:
        print(f"Error: The file '{file_path_or_buffer}' was not found.")
        return None
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return None

    new_rows = []

    for index, row in df.iterrows():
        # Extract identifier column values
        ids = {id_col: row[id_col] for id_col in id_cols}

        # Split the value columns
        split_values_list = []
        min_len = float('inf') # To handle cases where lists might not be of same length after split

        for col_name in value_cols_to_explode:
            if col_name not in row:
                print(f"Warning: Column '{col_name}' not found in the CSV. Skipping this column for exploding.")
                continue
            if pd.isna(row[col_name]): # Handle missing values in columns to be exploded
                split_col_values = []
            else:
                split_col_values = [item.strip() for item in str(row[col_name]).split(delimiter)]
            split_values_list.append(split_col_values)
            if len(split_col_values) < min_len:
                min_len = len(split_col_values)
        
        # Check if all columns to explode have the same number of items after splitting
        # (This version will take the minimum common length)
        # If you require strict matching, an error or warning could be raised here.
        num_items = min_len if split_values_list else 0
        if not split_values_list: # if no columns to explode were found or valid
            # Add the original row (or just id_cols if no other logic applies)
            # This behavior can be customized based on requirements
            if not df.empty:
                new_rows.append(row.to_dict()) # Keeps original row if no explode happens
            continue


        for i in range(num_items):
            new_row_data = ids.copy()
            all_items_present = True
            for j, col_name in enumerate(value_cols_to_explode):
                # Ensure the column was actually processed and has enough items
                if j < len(split_values_list) and i < len(split_values_list[j]):
                    value = split_values_list[j][i]
                    # Apply data type conversion if specified
                    if col_name in data_types:
                        try:
                            value = data_types[col_name](value)
                        except ValueError:
                            print(f"Warning: Could not convert '{value}' to {data_types[col_name]} for column '{col_name}'. Keeping as string.")
                    new_row_data[col_name] = value
                else:
                    # This case should ideally be handled by min_len logic or more robust error checking
                    # For now, we can add a placeholder or skip if an item is missing.
                    new_row_data[col_name] = None # Or some other placeholder
                    all_items_present = False # Mark that some data was missing for this set

            # Only add the row if all expected items were present, or adjust as needed
            # if all_items_present: # Uncomment if you want to skip rows with mismatched item counts
            new_rows.append(new_row_data)


    return pd.DataFrame(new_rows)