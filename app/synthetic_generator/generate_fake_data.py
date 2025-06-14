# import re
# import random
# import pandas as pd
# from faker import Faker

# fake = Faker()

# _num_re  = re.compile(r"^[\d,.\-]+$")                # crude “looks like a number”
# _date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$"
#                       r"|^\d{2}/\d{2}/\d{4}$")       # 2025-04-30  or  04/30/2025

# # ✨ NEW — cache original → fake so identical inputs stay identical
# _fake_cache: dict = {}

# def _fake_like(val):
#     """Return a fake value that preserves *shape* and is repeatable per input."""
#     # NaN / None should stay as-is
#     if pd.isna(val):
#         return val

#     # if we've already seen this exact value, reuse the fake
#     if val in _fake_cache:
#         return _fake_cache[val]

#     # —— real numerics ——
#     if isinstance(val, float):
#         new_val = round(random.uniform(1, 100), 4)   # keep it float-ish
#     elif isinstance(val, int):
#         new_val = random.randint(1, 100)

#     # —— strings that *look* numeric ——
#     elif isinstance(val, str) and _num_re.match(val):
#         digits_only = re.sub(r"\D", "", val) or "0"
#         fake_digits = "".join(random.choice("0123456789") for _ in digits_only)
#         fake_iter   = iter(fake_digits)
#         new_val     = re.sub(r"\d", lambda _: next(fake_iter), val)

#     # —— strings that look like dates ——
#     elif isinstance(val, str) and _date_re.match(val):
#         new_val = fake.date_between("-10y", "today").strftime("%Y-%m-%d")

#     # —— other strings ——
#     elif isinstance(val, str):
#         word = fake.word()
#         if len(word) < len(val):                      # pad if too short
#             word = (word * ((len(val) // len(word)) + 1))
#         new_val = word[:len(val)]

#     # —— anything else ——
#     else:
#         new_val = val                                 # leave untouched

#     # remember the mapping so next identical value gets the same fake
#     _fake_cache[val] = new_val
#     return new_val


# def fake_preview_df(df: pd.DataFrame, rows: int = 25, reset_cache: bool = True) -> pd.DataFrame:
#     """
#     Return a synthetic preview that keeps schema + rough shape
#     but never exposes real data. Identical source values map to identical fakes.
#     """
#     if reset_cache:
#         _fake_cache.clear()

#     preview = df.head(rows).copy()
#     for col in preview.columns:
#         preview[col] = preview[col].apply(_fake_like)
#     return preview
import random
import pandas as pd
from app.synthetic_generator.run_synth import overwrite_strings 

# Cache to store mappings of original values to their fake counterparts
# This ensures that identical input values always produce the same fake output.
_fake_cache: dict = {}

# Global character maps to ensure consistent character replacement across all strings
# processed since the last cache reset.
_global_char_map_lower: dict = {}
_global_char_map_upper: dict = {}
_global_char_map_digit: dict = {}

# Define character sets for replacements
LOWERCASE_LETTERS = 'abcdefghijklmnopqrstuvwxyz'
UPPERCASE_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
DIGITS = '0123456789'

def _generate_random_digits(length: int, allow_leading_zero: bool = False) -> str:
    """Helper function to generate a string of random digits of a given length."""
    if length == 0:
        return ""
    if length == 1:
        return random.choice(DIGITS)
    
    first_digit_options = DIGITS if allow_leading_zero else DIGITS[1:] # Avoid leading zero unless allowed or length is 1 (handled above)
    if not first_digit_options: # Handles case where length > 1 but only '0' would be allowed (e.g. if DIGITS was just '0')
        first_digit_options = DIGITS

    res = [random.choice(first_digit_options)]
    for _ in range(length - 1):
        res.append(random.choice(DIGITS))
    return "".join(res)

def _fake_like(val):
    """
    Return a fake value that preserves the *shape* of the original value.
    - Identical input values will produce identical faked output values (due to _fake_cache).
    - Identical characters (e.g., all 'E's, all '1's) across different input strings
      will map to the same respective faked character (due to _global_char_map_*).
    - Integers and Floats will retain their original number of digits and decimal structure.
    """
    # Handle NaN/None values: they should remain as is
    if pd.isna(val):
        return val

    # If this exact value has been faked before, return its cached fake version
    if val in _fake_cache:
        return _fake_cache[val]

    new_val = None # Initialize new_val

    # --- Handle floating-point numbers ---
    if isinstance(val, float):
        s_val = str(val)
        is_negative = s_val.startswith('-')
        if is_negative:
            s_val = s_val[1:]

        if '.' in s_val:
            int_part_str, frac_part_str = s_val.split('.', 1)
            
            num_int_digits = len(int_part_str)
            # Allow leading zero for int part if original is like "0.xxx"
            fake_int_part = _generate_random_digits(num_int_digits, allow_leading_zero=(num_int_digits == 1 and int_part_str == '0'))
            
            num_frac_digits = len(frac_part_str)
            fake_frac_part = _generate_random_digits(num_frac_digits, allow_leading_zero=True) # Fractional part can have leading/trailing zeros
            
            new_s_val = f"{fake_int_part}.{fake_frac_part}"
        else: # Float without a decimal point in its string representation (e.g. from int)
            num_digits = len(s_val)
            new_s_val = _generate_random_digits(num_digits, allow_leading_zero=(num_digits == 1 and s_val == '0'))
            # Ensure it's still treated as a float, e.g. by adding .0 if it was an integer-like float
            if not '.' in str(float(new_s_val)): # Check if it became an int string
                 new_s_val += ".0"


        new_val = float(new_s_val)
        if is_negative:
            new_val = -new_val
            
    # --- Handle integers ---
    elif isinstance(val, int):
        s_val = str(val)
        is_negative = s_val.startswith('-')
        if is_negative:
            s_val = s_val[1:]
            
        num_digits = len(s_val)
        # Allow leading zero only if the number itself is 0
        fake_digits_str = _generate_random_digits(num_digits, allow_leading_zero=(num_digits == 1 and s_val == '0'))
        
        new_val = int(fake_digits_str)
        if is_negative:
            new_val = -new_val

    # --- Handle strings (unified character-by-character replacement using global maps) ---
    elif isinstance(val, str):
        faked_chars = []
        for char_original in val:
            if 'a' <= char_original <= 'z': # Lowercase letter
                if char_original not in _global_char_map_lower:
                    _global_char_map_lower[char_original] = random.choice(LOWERCASE_LETTERS)
                faked_chars.append(_global_char_map_lower[char_original])
            elif 'A' <= char_original <= 'Z': # Uppercase letter
                if char_original not in _global_char_map_upper:
                    _global_char_map_upper[char_original] = random.choice(UPPERCASE_LETTERS)
                faked_chars.append(_global_char_map_upper[char_original])
            elif '0' <= char_original <= '9': # Digit
                if char_original not in _global_char_map_digit:
                    _global_char_map_digit[char_original] = random.choice(DIGITS)
                faked_chars.append(_global_char_map_digit[char_original])
            else:
                # Keep any other characters (spaces, hyphens, symbols, etc.) as they are
                faked_chars.append(char_original)
        new_val = "".join(faked_chars)
        
    # --- Handle any other data types ---
    else:
        # If the value is not NaN, float, int, or string, leave it untouched
        new_val = val

    # Store the original -> fake mapping in the cache for entire values
    _fake_cache[val] = new_val
    return new_val


def fake_preview_df(df: pd.DataFrame, rows: int = 25, reset_cache: bool = True) -> pd.DataFrame:
    """
    Return a synthetic preview of a DataFrame that keeps the schema and rough shape
    of the data but never exposes real data. Identical source values map to identical fakes,
    and identical characters across strings map to identical fake characters. Numeric values
    retain their original digit structure.

    Args:
        df: The input pandas DataFrame.
        rows: The number of rows to include in the preview.
        reset_cache: If True, clears the cache of faked values and global character maps
                     before processing. Set to False if you want consistent faking across
                     multiple calls for the same values/characters within a session.

    Returns:
        A pandas DataFrame with faked data.
    """
    if reset_cache:
        _fake_cache.clear()
        _global_char_map_lower.clear()
        _global_char_map_upper.clear()
        _global_char_map_digit.clear()

    # Take a head of the DataFrame to create the preview
    preview = df.head(rows).copy()
    # Apply the _fake_like function to each cell in the preview DataFrame
    for col in preview.columns:
        preview[col] = preview[col].apply(_fake_like)
    return preview

# def fake_preview_df(                               # ← replacement
#     df: pd.DataFrame,
#     rows: int = 25,
#     *_,
#     **__,
# ) -> pd.DataFrame:
#     """
#     Return a synthetic preview that:
#       • Keeps schema / null pattern
#       • Replaces every string token with a real English word
#       • Leaves identifier words (name, age, gender, …) untouched
#     """
#     preview = df.head(rows).copy()
#     return overwrite_strings(preview) 