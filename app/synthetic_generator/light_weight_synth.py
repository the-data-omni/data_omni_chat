# # from __future__ import annotations

# # from typing import Any, Dict, Callable, Optional, List
# # import pandas as pd
# # import numpy as np
# # from scipy.stats import gaussian_kde, norm
# # from faker import Faker
# # import re # Import regex for sanitizing column names

# # # --- Gaussian Copula Synthesizer (Unchanged) ---
# # # (Keep the previous version of GaussianCopulaSynth here)
# # class GaussianCopulaSynth:
# #     """Gaussian Copula synthesizer for numeric columns to preserve joint distributions."""
# #     def __init__(self, seed: Optional[int] = None):
# #         self.rng = np.random.default_rng(seed)
# #         self.names: list[str] = []
# #         self.cov: Optional[np.ndarray] = None
# #         self.sorted_vals: Dict[str, np.ndarray] = {}

# #     def fit(self, df: pd.DataFrame) -> "GaussianCopulaSynth":
# #         self.names = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
# #         if not self.names:
# #             print("Warning: No numeric columns found for GaussianCopulaSynth.")
# #             return self
# #         df_num = df[self.names].dropna()
# #         if len(df_num) < 2:
# #              print(f"Warning: Not enough non-null numeric data ({len(df_num)} rows) to fit Gaussian Copula. Skipping.")
# #              self.names = []
# #              return self
# #         try:
# #             U = df_num.rank(method="average") / (len(df_num) + 1)
# #             U = np.clip(U, 1e-9, 1 - 1e-9)
# #             Z = norm.ppf(U)
# #             self.cov = np.cov(Z, rowvar=False)
# #             if np.isnan(self.cov).any():
# #                  print("Warning: NaN found in covariance matrix. Gaussian Copula might not work as expected.")
# #             for col in self.names:
# #                 vals = df_num[col].to_numpy()
# #                 self.sorted_vals[col] = np.sort(vals)
# #         except Exception as e:
# #             print(f"Error during GaussianCopulaSynth fit: {e}")
# #             self.names = []
# #         return self

# #     def sample(self, n: int) -> pd.DataFrame:
# #         if not self.names or self.cov is None or np.isnan(self.cov).any():
# #             print("Skipping GaussianCopulaSynth sampling due to lack of fitted columns or invalid covariance.")
# #             return pd.DataFrame(columns=self.names)
# #         d = len(self.names)
# #         try:
# #             min_eig = np.min(np.real(np.linalg.eigvals(self.cov)))
# #             current_cov = self.cov
# #             if min_eig < -1e-9:
# #                  print("Warning: Covariance matrix is not positive semi-definite. Adding jitter.")
# #                  current_cov = self.cov + np.eye(d) * 1e-6
# #             z = self.rng.multivariate_normal(np.zeros(d), current_cov, size=n)
# #             u = norm.cdf(z)
# #             data: Dict[str, np.ndarray] = {}
# #             for j, col in enumerate(self.names):
# #                 vals = self.sorted_vals.get(col, np.array([])) # Use get for safety
# #                 if len(vals) > 0 :
# #                      data[col] = np.interp(u[:, j], np.linspace(0, 1, len(vals)), vals)
# #                 else:
# #                      data[col] = np.full(n, np.nan)
# #         except np.linalg.LinAlgError as e:
# #             print(f"Error sampling from multivariate normal (likely covariance issue): {e}")
# #             return pd.DataFrame(columns=self.names)
# #         except Exception as e:
# #             print(f"Error during GaussianCopulaSynth sampling: {e}")
# #             return pd.DataFrame(columns=self.names)
# #         return pd.DataFrame(data)


# # # --- Simple Tabular Synthesizer (Simplified ID Replacement Version) ---
# # class SimpleTabularSynth:
# #     """
# #     Simplified synthesizer focusing on basic column types and replacing
# #     string values with consistent identifiers (column_name_number).
# #     """

# #     def __init__(self, seed: int | None = None):
# #         """
# #         Initializes the synthesizer.
# #         Args:
# #             seed: Random seed for reproducibility.
# #         """
# #         self.rng = np.random.default_rng(seed)
# #         self.col_meta: Dict[str, Dict[str, Any]] = {}
# #         self.fake = Faker() # Keep Faker for potential future use or other column types


# #     def _sanitize_col_name(self, col_name: str) -> str:
# #         """Removes problematic characters for use in generated IDs."""
# #         # Remove non-alphanumeric characters, replace spaces/hyphens with underscore
# #         s = re.sub(r'[^\w\s-]', '', col_name).strip()
# #         s = re.sub(r'[-\s]+', '_', s)
# #         return s if s else "col" # Fallback if name becomes empty


# #     # --- KDE Sampler (Unchanged) ---
# #     def _kde_sampler(self, vals: np.ndarray, is_int: bool) -> Callable[[int], np.ndarray]:
# #         if len(vals) == 0: return lambda m: np.full(m, np.nan)
# #         # Ensure vals has variance for KDE
# #         if np.std(vals) < 1e-9:
# #              const = vals[0] if len(vals) > 0 else 0
# #              return lambda m, v=const: np.repeat(v, m)
# #         try:
# #             kde = gaussian_kde(vals)
# #             def _draw(m: int) -> np.ndarray:
# #                 out = kde.resample(m).flatten()
# #                 return np.round(out) if is_int else out
# #             return _draw
# #         except Exception as e:
# #              print(f"Warning: KDE fitting failed: {e}. Falling back to uniform sampling.")
# #              min_val, max_val = vals.min(), vals.max()
# #              if min_val == max_val:
# #                    return lambda m, v=min_val: np.repeat(v, m)
# #              if is_int:
# #                   return lambda m: self.rng.integers(int(round(min_val)), int(round(max_val)) + 1, size=m)
# #              else:
# #                   return lambda m: self.rng.uniform(min_val, max_val, size=m)


# #     # --- Fit Method (Simplified for ID Replacement) ---
# #     def fit(self, df: pd.DataFrame) -> "SimpleTabularSynth":
# #         print("Fitting SimpleTabularSynth (ID Replacement Mode)...")
# #         for col in df.columns:
# #             print(f"  Processing column: {col}")
# #             ser = df[col]
# #             meta: Dict[str, Any] = { "dtype": ser.dtype, "null_ratio": ser.isna().mean()}
# #             ser_notna = ser.dropna()
# #             n_notna = len(ser_notna)

# #             if n_notna == 0:
# #                  print(f"    Column '{col}' is all NULLs.")
# #                  self.col_meta[col] = meta
# #                  continue

# #             # --- Numeric ---
# #             if pd.api.types.is_numeric_dtype(ser):
# #                  print(f"    Type: Numeric")
# #                  vals = ser_notna.to_numpy()
# #                  meta["is_int"] = pd.api.types.is_integer_dtype(ser)
# #                  unique_vals = np.unique(vals)
# #                  if len(unique_vals) > 0:
# #                       meta["positive_only"] = float(vals.min()) >= 0
# #                       if len(unique_vals) > 1 and np.std(vals) > 1e-9:
# #                           base_sampler = self._kde_sampler(vals, meta["is_int"])
# #                       else: # Constant value case
# #                           const = float(unique_vals[0])
# #                           base_sampler = lambda m, v=const: np.repeat(v, m)

# #                       # Positive wrapper (simplified)
# #                       if meta.get("positive_only", False):
# #                           def positive_wrapper(m: int, bs=base_sampler) -> np.ndarray:
# #                               out = bs(m)
# #                               invalid_mask = out <= 0
# #                               attempts = 0
# #                               while np.any(invalid_mask) and attempts < 10:
# #                                    num_invalid = np.sum(invalid_mask)
# #                                    out[invalid_mask] = bs(num_invalid)
# #                                    invalid_mask = out <= 0
# #                                    attempts += 1
# #                               if np.any(invalid_mask):
# #                                    out[invalid_mask] = np.abs(out[invalid_mask]) + 1e-6
# #                               return out
# #                           meta["sampler"] = positive_wrapper
# #                       else:
# #                            meta["sampler"] = base_sampler
# #                  else: # Should not happen if n_notna > 0
# #                        meta["sampler"] = lambda m: np.full(m, np.nan)


# #             # --- Datetime ---
# #             elif pd.api.types.is_datetime64_any_dtype(ser):
# #                 print(f"    Type: Datetime")
# #                 ints = ser_notna.astype("int64")
# #                 meta["min"], meta["max"] = ints.min(), ints.max()

# #             # --- String / Object Columns -> ID Replacement ---
# #             # Includes boolean, category, object, string dtypes
# #             else:
# #                 print(f"    Type: {ser.dtype} -> Replacing with ID")
# #                 # Ensure we work with string representations for mapping
# #                 ser_notna_str = ser_notna.astype(str)
# #                 unique_originals = ser_notna_str.unique().tolist()

# #                 if unique_originals:
# #                     # Sanitize column name for use in ID
# #                     sanitized_col_name = self._sanitize_col_name(str(col))

# #                     # Create mapping and list of IDs
# #                     value_to_id_map = {}
# #                     id_list = []
# #                     for i, original_val in enumerate(unique_originals):
# #                          # Use index + 1 for human-readable 1-based numbering
# #                          generated_id = f"{sanitized_col_name}_{i+1}"
# #                          value_to_id_map[original_val] = generated_id
# #                          id_list.append(generated_id)

# #                     # Calculate probabilities based on original value frequencies
# #                     counts = ser_notna_str.value_counts()
# #                     id_probs = [counts[original_val] / n_notna for original_val in unique_originals]

# #                     # Store the necessary info for sampling
# #                     meta["id_list"] = id_list
# #                     meta["id_probs"] = id_probs
# #                     # meta["value_to_id_map"] = value_to_id_map # Keep if needed for inspection
# #                 else:
# #                      print(f"    Warning: Column {col} has no valid unique values after astype(str).")
# #                      meta["id_list"] = [] # Flag that no IDs can be generated
# #                      meta["id_probs"] = []

# #             self.col_meta[col] = meta
# #         print("Fitting complete.")
# #         return self


# #     # --- Sample Method (Simplified for ID Replacement) ---
# #     def sample(self, n: int) -> pd.DataFrame:
# #         """Generates n rows of synthetic data."""
# #         print(f"Generating {n} synthetic rows (ID Replacement Mode)...")
# #         rows: list[Dict[str, Any]] = []

# #         for i in range(n):
# #             # Progress indicator (optional)
# #             # if (i + 1) % 100 == 0: print(f"  Generated {i+1}/{n} rows...")

# #             row: Dict[str, Any] = {}
# #             for col, meta in self.col_meta.items():
# #                 # 1) Nulls
# #                 if self.rng.random() < meta.get("null_ratio", 0):
# #                     row[col] = pd.NA
# #                     continue

# #                 dtype = meta.get("dtype", None)

# #                 # 2) Numeric
# #                 if "sampler" in meta:
# #                      try:
# #                          sampler_func = meta["sampler"]
# #                          sampled_array = sampler_func(1)
# #                          val = sampled_array[0]
# #                          if meta.get("is_int"):
# #                              row[col] = int(round(val)) if not pd.isna(val) else pd.NA
# #                          else:
# #                              row[col] = float(val) if not pd.isna(val) else pd.NA
# #                      except Exception as e:
# #                          print(f"Warning: Numeric sampling error for {col}: {e}")
# #                          row[col] = pd.NA

# #                 # 3) Datetime
# #                 elif pd.api.types.is_datetime64_any_dtype(dtype) and "min" in meta:
# #                       try:
# #                           ts_int = self.rng.integers(meta["min"], meta["max"], endpoint=True)
# #                           row[col] = pd.to_datetime(ts_int)
# #                       except Exception as e:
# #                           print(f"Warning: Datetime sampling error for {col}: {e}")
# #                           row[col] = pd.NaT

# #                 # 4) ID Replacement Columns (Sample from generated IDs)
# #                 elif "id_list" in meta:
# #                      if meta["id_list"]: # Check if list is not empty
# #                           # Sample an ID based on original frequencies
# #                           row[col] = self.rng.choice(meta["id_list"], p=meta["id_probs"])
# #                      else:
# #                           # Fallback if fit found no unique values
# #                           row[col] = pd.NA


# #                 # 5) Final Fallback (Shouldn't be reached with current logic)
# #                 else:
# #                      print(f"Warning: No sampling method determined for column '{col}'. Setting to NaN.")
# #                      row[col] = pd.NA

# #             rows.append(row)

# #         print("Synthetic data generation complete.")
# #         # Attempt to infer best dtypes for final dataframe (IDs will be objects/strings)
# #         return pd.DataFrame(rows).infer_objects()


# # # --- Example Usage ---
# # if __name__ == '__main__':

# #     print("Example Usage of Simplified ID Replacement Synthesizer:")

# #     # Sample Data (using the example from your run_synth.py comment)
# #     df_real = pd.DataFrame({
# #         "ints":       [1, 2, 3, np.nan, 5],
# #         "floats":     [0.1, 0.2, 0.3, 0.4, np.nan],
# #         "dates":      ["2020-01-01", None, "2020-01-03", "2020-01-04", "2020-01-05"],
# #         "cats":       ["a","b","a","b","c"],
# #         "long_text":  ["tanaka pfupajena", None, "rachel pfupajena", "simon pfupajena", "bar baz qux"],
# #         "Segment>>":  ["Consumer | edga |simon", "Corporate| edga | simon", "Consumer| edga |simon", "Home Office| edga |simon", "Consumer| edga |simon"], # Added example
# #         "Empty Col": [None, None, None, None, None]
# #     })

# #     print("\n--- Original Data ---")
# #     print(df_real)
# #     print(df_real.info())

# #     # --- Using SimpleTabularSynth (ID Replacement Mode) ---
# #     print("\n--- SimpleTabularSynth (ID Replacement Mode) ---")
# #     # No need for NLP model parameters anymore
# #     simple_synth = SimpleTabularSynth(seed=42)
# #     simple_synth.fit(df_real)
# #     df_simple_fake = simple_synth.sample(5) # Generate 5 rows

# #     print("\n--- Generated Data (SimpleTabularSynth - ID Replacement) ---")
# #     print(df_simple_fake)
# #     print(df_simple_fake.info())

# #     # Expected output for string columns:
# #     # cats: values like "cats_1", "cats_2", "cats_3" sampled based on original frequencies of "a", "b", "c"
# #     # long_text: values like "long_text_1", "long_text_2", etc.
# #     # Segment>>: values like "Segment_1", "Segment_2", etc. (Note sanitized name)
# #     # Empty Col: Should be all <NA> or based on null_ratio (which is 1.0 here)

# from __future__ import annotations

# from typing import Any, Dict, Callable, Optional, List
# import pandas as pd
# import numpy as np
# from scipy.stats import gaussian_kde, norm
# from faker import Faker
# import re

# # --- Gaussian Copula Synthesizer (Unchanged) ---
# # This part of the code remains the same.
# class GaussianCopulaSynth:
#     """Gaussian Copula synthesizer for numeric columns to preserve joint distributions."""
#     def __init__(self, seed: Optional[int] = None):
#         self.rng = np.random.default_rng(seed)
#         self.names: list[str] = []
#         self.cov: Optional[np.ndarray] = None
#         self.sorted_vals: Dict[str, np.ndarray] = {}

#     def fit(self, df: pd.DataFrame) -> "GaussianCopulaSynth":
#         self.names = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
#         if not self.names:
#             # print("Warning: No numeric columns found for GaussianCopulaSynth.")
#             return self
#         df_num = df[self.names].dropna()
#         if len(df_num) < 2:
#              # print(f"Warning: Not enough non-null numeric data ({len(df_num)} rows) to fit Gaussian Copula. Skipping.")
#              self.names = []
#              return self
#         try:
#             U = df_num.rank(method="average") / (len(df_num) + 1)
#             U = np.clip(U, 1e-9, 1 - 1e-9)
#             Z = norm.ppf(U)
#             self.cov = np.cov(Z, rowvar=False)
#             if np.isnan(self.cov).any():
#                  # print("Warning: NaN found in covariance matrix. Gaussian Copula might not work as expected.")
#                  pass
#             for col in self.names:
#                 vals = df_num[col].to_numpy()
#                 self.sorted_vals[col] = np.sort(vals)
#         except Exception as e:
#             # print(f"Error during GaussianCopulaSynth fit: {e}")
#             self.names = []
#         return self

#     def sample(self, n: int) -> pd.DataFrame:
#         if not self.names or self.cov is None or np.isnan(self.cov).any():
#             # print("Skipping GaussianCopulaSynth sampling due to lack of fitted columns or invalid covariance.")
#             return pd.DataFrame(columns=self.names)
#         d = len(self.names)
#         try:
#             min_eig = np.min(np.real(np.linalg.eigvals(self.cov)))
#             current_cov = self.cov
#             if min_eig < -1e-9:
#                  # print("Warning: Covariance matrix is not positive semi-definite. Adding jitter.")
#                  current_cov = self.cov + np.eye(d) * 1e-6
#             z = self.rng.multivariate_normal(np.zeros(d), current_cov, size=n)
#             u = norm.cdf(z)
#             data: Dict[str, np.ndarray] = {}
#             for j, col in enumerate(self.names):
#                 vals = self.sorted_vals.get(col, np.array([]))
#                 if len(vals) > 0 :
#                      data[col] = np.interp(u[:, j], np.linspace(0, 1, len(vals)), vals)
#                 else:
#                      data[col] = np.full(n, np.nan)
#         except np.linalg.LinAlgError as e:
#             # print(f"Error sampling from multivariate normal (likely covariance issue): {e}")
#             return pd.DataFrame(columns=self.names)
#         except Exception as e:
#             # print(f"Error during GaussianCopulaSynth sampling: {e}")
#             return pd.DataFrame(columns=self.names)
#         return pd.DataFrame(data)

# # --- Simple Tabular Synthesizer (MODIFIED) ---
# class SimpleTabularSynth:
#     """
#     Synthesizer that replaces string/object/categorical values with consistent,
#     realistically-shaped fakes, and models other basic column types.
#     """
#     LOWERCASE_LETTERS = 'abcdefghijklmnopqrstuvwxyz'
#     UPPERCASE_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
#     DIGITS = '0123456789'

#     def __init__(self, seed: int | None = None):
#         """Initializes the synthesizer."""
#         self.rng = np.random.default_rng(seed)
#         self.col_meta: Dict[str, Dict[str, Any]] = {}
#         self.fake = Faker()
#         # Caches for the faking mechanism, ensuring consistency.
#         self._value_map: Dict[Any, Any] = {}
#         self._char_map_lower: Dict[str, str] = {}
#         self._char_map_upper: Dict[str, str] = {}
#         self._char_map_digit: Dict[str, str] = {}

#     def _generate_random_digits(self, length: int, allow_leading_zero: bool = False) -> str:
#         """Helper to generate a string of random digits using the seeded RNG."""
#         if length == 0: return ""
#         if length == 1: return self.rng.choice(list(self.DIGITS))
        
#         first_digit_options = self.DIGITS if allow_leading_zero else self.DIGITS[1:]
#         res = [self.rng.choice(list(first_digit_options))]
#         res.extend(self.rng.choice(list(self.DIGITS), size=length - 1))
#         return "".join(res)

#     def _get_or_create_fake_value(self, val: Any) -> Any:
#         """
#         Generates a fake value that preserves the shape of the original.
#         This is an adapted version of the `_fake_like` logic, using the class's
#         internal state (RNG and caches) for reproducibility.
#         """
#         if pd.isna(val): return val
#         if val in self._value_map: return self._value_map[val]

#         new_val = None
#         if isinstance(val, float):
#             s_val = str(val)
#             is_negative = s_val.startswith('-')
#             if is_negative: s_val = s_val[1:]
#             if '.' in s_val:
#                 int_part, frac_part = s_val.split('.', 1)
#                 fake_int = self._generate_random_digits(len(int_part), allow_leading_zero=(len(int_part) == 1 and int_part == '0'))
#                 fake_frac = self._generate_random_digits(len(frac_part), allow_leading_zero=True)
#                 new_s_val = f"{fake_int}.{fake_frac}"
#             else:
#                 new_s_val = self._generate_random_digits(len(s_val), allow_leading_zero=(len(s_val) == 1 and s_val == '0'))
#                 if '.' not in str(float(new_s_val)): new_s_val += ".0"
#             new_val = float(new_s_val)
#             if is_negative: new_val = -new_val
#         elif isinstance(val, int):
#             s_val = str(val)
#             is_negative = s_val.startswith('-')
#             if is_negative: s_val = s_val[1:]
#             fake_digits = self._generate_random_digits(len(s_val), allow_leading_zero=(len(s_val) == 1 and s_val == '0'))
#             new_val = int(fake_digits)
#             if is_negative: new_val = -new_val
#         elif isinstance(val, str):
#             faked_chars = []
#             for char in val:
#                 if 'a' <= char <= 'z':
#                     if char not in self._char_map_lower: self._char_map_lower[char] = self.rng.choice(list(self.LOWERCASE_LETTERS))
#                     faked_chars.append(self._char_map_lower[char])
#                 elif 'A' <= char <= 'Z':
#                     if char not in self._char_map_upper: self._char_map_upper[char] = self.rng.choice(list(self.UPPERCASE_LETTERS))
#                     faked_chars.append(self._char_map_upper[char])
#                 elif '0' <= char <= '9':
#                     if char not in self._char_map_digit: self._char_map_digit[char] = self.rng.choice(list(self.DIGITS))
#                     faked_chars.append(self._char_map_digit[char])
#                 else:
#                     faked_chars.append(char)
#             new_val = "".join(faked_chars)
#         else:
#             new_val = val

#         self._value_map[val] = new_val
#         return new_val

#     def _kde_sampler(self, vals: np.ndarray, is_int: bool) -> Callable[[int], np.ndarray]:
#         # This private method is unchanged.
#         if len(vals) == 0: return lambda m: np.full(m, np.nan)
#         if np.std(vals) < 1e-9:
#              const = vals[0] if len(vals) > 0 else 0
#              return lambda m, v=const: np.repeat(v, m)
#         try:
#             kde = gaussian_kde(vals)
#             def _draw(m: int) -> np.ndarray:
#                 out = kde.resample(m).flatten()
#                 return np.round(out) if is_int else out
#             return _draw
#         except Exception as e:
#              min_val, max_val = vals.min(), vals.max()
#              if min_val == max_val: return lambda m, v=min_val: np.repeat(v, m)
#              if is_int: return lambda m: self.rng.integers(int(round(min_val)), int(round(max_val)) + 1, size=m)
#              else: return lambda m: self.rng.uniform(min_val, max_val, size=m)

#     def fit(self, df: pd.DataFrame) -> "SimpleTabularSynth":
#         # print("Fitting SimpleTabularSynth (Advanced Fake Mode)...")
#         for col in df.columns:
#             # print(f"  Processing column: {col}")
#             ser = df[col]
#             meta: Dict[str, Any] = {"dtype": ser.dtype, "null_ratio": ser.isna().mean()}
#             ser_notna = ser.dropna()
#             n_notna = len(ser_notna)

#             if n_notna == 0:
#                 self.col_meta[col] = meta
#                 continue

#             if pd.api.types.is_numeric_dtype(ser):
#                 # This logic is unchanged
#                 vals = ser_notna.to_numpy()
#                 meta["is_int"] = pd.api.types.is_integer_dtype(ser)
#                 unique_vals = np.unique(vals)
#                 if len(unique_vals) > 0:
#                     meta["positive_only"] = float(vals.min()) >= 0
#                     if len(unique_vals) > 1 and np.std(vals) > 1e-9:
#                         base_sampler = self._kde_sampler(vals, meta["is_int"])
#                     else:
#                         const = float(unique_vals[0])
#                         base_sampler = lambda m, v=const: np.repeat(v, m)
#                     if meta.get("positive_only", False):
#                         def positive_wrapper(m: int, bs=base_sampler) -> np.ndarray:
#                             out = bs(m)
#                             invalid_mask = out <= 0; attempts = 0
#                             while np.any(invalid_mask) and attempts < 10:
#                                 out[invalid_mask] = bs(np.sum(invalid_mask))
#                                 invalid_mask = out <= 0; attempts += 1
#                             if np.any(invalid_mask): out[invalid_mask] = np.abs(out[invalid_mask]) + 1e-6
#                             return out
#                         meta["sampler"] = positive_wrapper
#                     else:
#                         meta["sampler"] = base_sampler
#                 else:
#                     meta["sampler"] = lambda m: np.full(m, np.nan)
#             elif pd.api.types.is_datetime64_any_dtype(ser):
#                 # This logic is unchanged
#                 ints = ser_notna.astype("int64")
#                 meta["min"], meta["max"] = ints.min(), ints.max()
#             else:
#                 # --- MODIFIED: Replace ID logic with advanced faking ---
#                 # print(f"    Type: {ser.dtype} -> Creating fake value mappings")
#                 unique_originals = ser_notna.unique()
#                 if len(unique_originals) > 0:
#                     # Pre-generate fakes for all unique values to populate the map
#                     for val in unique_originals:
#                         self._get_or_create_fake_value(val)
                    
#                     # Store original unique values and their probabilities for sampling
#                     counts = ser_notna.value_counts()
#                     probs = counts / n_notna
#                     meta["unique_originals"] = probs.index.tolist()
#                     meta["probabilities"] = probs.values.tolist()
#                 else:
#                     meta["unique_originals"] = []
#                     meta["probabilities"] = []

#             self.col_meta[col] = meta
#         # print("Fitting complete.")
#         return self

#     def sample(self, n: int) -> pd.DataFrame:
#         # print(f"Generating {n} synthetic rows (Advanced Fake Mode)...")
#         rows: list[Dict[str, Any]] = []
#         for _ in range(n):
#             row: Dict[str, Any] = {}
#             for col, meta in self.col_meta.items():
#                 if self.rng.random() < meta.get("null_ratio", 0):
#                     row[col] = pd.NA
#                     continue

#                 dtype = meta.get("dtype", None)

#                 if "sampler" in meta:
#                     # Numeric sampling (unchanged)
#                     try:
#                         val = meta["sampler"](1)[0]
#                         row[col] = int(round(val)) if meta.get("is_int") and not pd.isna(val) else float(val)
#                     except Exception: row[col] = pd.NA
#                 elif pd.api.types.is_datetime64_any_dtype(dtype) and "min" in meta:
#                     # Datetime sampling (unchanged)
#                     try:
#                         ts_int = self.rng.integers(meta["min"], meta["max"], endpoint=True)
#                         row[col] = pd.to_datetime(ts_int)
#                     except Exception: row[col] = pd.NaT
#                 elif "unique_originals" in meta:
#                     # --- MODIFIED: Sample from pre-generated fake values ---
#                     if meta["unique_originals"]:
#                         # 1. Sample an ORIGINAL value based on learned frequencies
#                         original_val = self.rng.choice(meta["unique_originals"], p=meta["probabilities"])
#                         # 2. Look up its corresponding FAKE value in the map
#                         row[col] = self._value_map.get(original_val, pd.NA)
#                     else:
#                         row[col] = pd.NA # Fallback if no unique values were found
#                 else:
#                     row[col] = pd.NA

#             rows.append(row)
        
#         # print("Synthetic data generation complete.")
#         return pd.DataFrame(rows).infer_objects()
# --- (Keep the GaussianCopulaSynth class as it is) ---
from __future__ import annotations

from typing import Any, Dict, Callable, Optional, List
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde, norm
from faker import Faker
import re

class GaussianCopulaSynth:
    """Gaussian Copula synthesizer for numeric columns to preserve joint distributions."""
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.names: list[str] = []
        self.cov: Optional[np.ndarray] = None
        self.sorted_vals: Dict[str, np.ndarray] = {}

    def fit(self, df: pd.DataFrame) -> "GaussianCopulaSynth":
        self.names = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not self.names:
            # print("Warning: No numeric columns found for GaussianCopulaSynth.")
            return self
        df_num = df[self.names].dropna()
        if len(df_num) < 2:
             # print(f"Warning: Not enough non-null numeric data ({len(df_num)} rows) to fit Gaussian Copula. Skipping.")
             self.names = []
             return self
        try:
            U = df_num.rank(method="average") / (len(df_num) + 1)
            U = np.clip(U, 1e-9, 1 - 1e-9)
            Z = norm.ppf(U)
            self.cov = np.cov(Z, rowvar=False)
            if np.isnan(self.cov).any():
                 # print("Warning: NaN found in covariance matrix. Gaussian Copula might not work as expected.")
                 pass
            for col in self.names:
                vals = df_num[col].to_numpy()
                self.sorted_vals[col] = np.sort(vals)
        except Exception as e:
            # print(f"Error during GaussianCopulaSynth fit: {e}")
            self.names = []
        return self

    def sample(self, n: int) -> pd.DataFrame:
        if not self.names or self.cov is None or np.isnan(self.cov).any():
            # print("Skipping GaussianCopulaSynth sampling due to lack of fitted columns or invalid covariance.")
            return pd.DataFrame(columns=self.names)
        d = len(self.names)
        try:
            min_eig = np.min(np.real(np.linalg.eigvals(self.cov)))
            current_cov = self.cov
            if min_eig < -1e-9:
                 # print("Warning: Covariance matrix is not positive semi-definite. Adding jitter.")
                 current_cov = self.cov + np.eye(d) * 1e-6
            z = self.rng.multivariate_normal(np.zeros(d), current_cov, size=n)
            u = norm.cdf(z)
            data: Dict[str, np.ndarray] = {}
            for j, col in enumerate(self.names):
                vals = self.sorted_vals.get(col, np.array([]))
                if len(vals) > 0 :
                     data[col] = np.interp(u[:, j], np.linspace(0, 1, len(vals)), vals)
                else:
                     data[col] = np.full(n, np.nan)
        except np.linalg.LinAlgError as e:
            # print(f"Error sampling from multivariate normal (likely covariance issue): {e}")
            return pd.DataFrame(columns=self.names)
        except Exception as e:
            # print(f"Error during GaussianCopulaSynth sampling: {e}")
            return pd.DataFrame(columns=self.names)
        return pd.DataFrame(data)


# --- Simple Tabular Synthesizer (MODIFIED with Type Inference) ---
class SimpleTabularSynth:
    """
    Synthesizer that replaces string/object/categorical values with consistent,
    realistically-shaped fakes, and models other basic column types.
    Includes a pre-processing step to correctly identify date-like columns.
    """
    LOWERCASE_LETTERS = 'abcdefghijklmnopqrstuvwxyz'
    UPPERCASE_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    DIGITS = '0123456789'

    def __init__(self, seed: int | None = None):
        """Initializes the synthesizer."""
        self.rng = np.random.default_rng(seed)
        self.col_meta: Dict[str, Dict[str, Any]] = {}
        self.fake = Faker()
        # Caches for the faking mechanism, ensuring consistency.
        self._value_map: Dict[Any, Any] = {}
        self._char_map_lower: Dict[str, str] = {}
        self._char_map_upper: Dict[str, str] = {}
        self._char_map_digit: Dict[str, str] = {}

    def _infer_and_convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-processes the DataFrame to find object columns that should be datetimes.
        This prevents date-like strings from being scrambled by the string faker.
        """
        df_processed = df.copy()
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                # Skip columns that are all null
                if df_processed[col].isnull().all():
                    continue
                
                # Attempt to convert to datetime
                try:
                    coerced_series = pd.to_datetime(df_processed[col], errors='coerce', dayfirst=False, yearfirst=False)
                    
                    # Heuristic: If we successfully converted at least one non-null value,
                    # and didn't just create all NaTs, treat it as a datetime column.
                    original_nulls = df_processed[col].isnull().sum()
                    coerced_nulls = coerced_series.isnull().sum()
                    
                    if coerced_nulls < len(df_processed[col]) and coerced_nulls >= original_nulls:
                        # print(f"  INFO: Column '{col}' detected as date-like and converted to datetime.")
                        df_processed[col] = coerced_series
                except Exception:
                    # If pd.to_datetime fails for any reason, just move on.
                    continue
        return df_processed

    # --- (No changes to _generate_random_digits, _get_or_create_fake_value, _kde_sampler) ---
    def _generate_random_digits(self, length: int, allow_leading_zero: bool = False) -> str:
        if length == 0: return ""
        if length == 1: return self.rng.choice(list(self.DIGITS))
        first_digit_options = self.DIGITS if allow_leading_zero else self.DIGITS[1:]
        res = [self.rng.choice(list(first_digit_options))]
        res.extend(self.rng.choice(list(self.DIGITS), size=length - 1))
        return "".join(res)

    def _get_or_create_fake_value(self, val: Any) -> Any:
        if pd.isna(val): return val
        if val in self._value_map: return self._value_map[val]
        new_val = None
        if isinstance(val, float):
            s_val = str(val); is_negative = s_val.startswith('-');
            if is_negative: s_val = s_val[1:]
            if '.' in s_val:
                int_part, frac_part = s_val.split('.', 1)
                fake_int = self._generate_random_digits(len(int_part), allow_leading_zero=(len(int_part) == 1 and int_part == '0'))
                fake_frac = self._generate_random_digits(len(frac_part), allow_leading_zero=True)
                new_s_val = f"{fake_int}.{fake_frac}"
            else:
                new_s_val = self._generate_random_digits(len(s_val), allow_leading_zero=(len(s_val) == 1 and s_val == '0'))
                if '.' not in str(float(new_s_val)): new_s_val += ".0"
            new_val = float(new_s_val)
            if is_negative: new_val = -new_val
        elif isinstance(val, int):
            s_val = str(val); is_negative = s_val.startswith('-');
            if is_negative: s_val = s_val[1:]
            fake_digits = self._generate_random_digits(len(s_val), allow_leading_zero=(len(s_val) == 1 and s_val == '0'))
            new_val = int(fake_digits)
            if is_negative: new_val = -new_val
        elif isinstance(val, str):
            faked_chars = []
            for char in val:
                if 'a' <= char <= 'z':
                    if char not in self._char_map_lower: self._char_map_lower[char] = self.rng.choice(list(self.LOWERCASE_LETTERS))
                    faked_chars.append(self._char_map_lower[char])
                elif 'A' <= char <= 'Z':
                    if char not in self._char_map_upper: self._char_map_upper[char] = self.rng.choice(list(self.UPPERCASE_LETTERS))
                    faked_chars.append(self._char_map_upper[char])
                elif '0' <= char <= '9':
                    if char not in self._char_map_digit: self._char_map_digit[char] = self.rng.choice(list(self.DIGITS))
                    faked_chars.append(self._char_map_digit[char])
                else: faked_chars.append(char)
            new_val = "".join(faked_chars)
        else: new_val = val
        self._value_map[val] = new_val
        return new_val

    def _kde_sampler(self, vals: np.ndarray, is_int: bool) -> Callable[[int], np.ndarray]:
        if len(vals) == 0: return lambda m: np.full(m, np.nan)
        if np.std(vals) < 1e-9:
             const = vals[0] if len(vals) > 0 else 0
             return lambda m, v=const: np.repeat(v, m)
        try:
            kde = gaussian_kde(vals)
            def _draw(m: int) -> np.ndarray:
                out = kde.resample(m).flatten()
                return np.round(out) if is_int else out
            return _draw
        except Exception:
             min_val, max_val = vals.min(), vals.max()
             if min_val == max_val: return lambda m, v=min_val: np.repeat(v, m)
             if is_int: return lambda m: self.rng.integers(int(round(min_val)), int(round(max_val)) + 1, size=m)
             else: return lambda m: self.rng.uniform(min_val, max_val, size=m)

    def fit(self, df: pd.DataFrame) -> "SimpleTabularSynth":
        """
        Fits the synthesizer to the data, including a new pre-processing step
        to correctly identify and convert data types like dates.
        """
        # print("Fitting SimpleTabularSynth...")
        # --- NEW: Pre-process DataFrame to fix data types ---
        df_fitted = self._infer_and_convert_types(df)
        
        for col in df_fitted.columns:
            # print(f"  Processing column: {col} (dtype: {df_fitted[col].dtype})")
            ser = df_fitted[col]
            meta: Dict[str, Any] = {"dtype": ser.dtype, "null_ratio": ser.isna().mean()}
            ser_notna = ser.dropna()
            n_notna = len(ser_notna)

            if n_notna == 0:
                self.col_meta[col] = meta
                continue

            # The logic now correctly handles columns converted to datetime
            if pd.api.types.is_numeric_dtype(ser):
                vals = ser_notna.to_numpy()
                meta["is_int"] = pd.api.types.is_integer_dtype(ser)
                unique_vals = np.unique(vals)
                if len(unique_vals) > 0:
                    meta["positive_only"] = float(vals.min()) >= 0
                    base_sampler = self._kde_sampler(vals, meta["is_int"])
                    if meta.get("positive_only", False):
                        def positive_wrapper(m: int, bs=base_sampler) -> np.ndarray:
                            out = bs(m); invalid_mask = out <= 0; attempts = 0
                            while np.any(invalid_mask) and attempts < 10:
                                out[invalid_mask] = bs(np.sum(invalid_mask))
                                invalid_mask = out <= 0; attempts += 1
                            if np.any(invalid_mask): out[invalid_mask] = np.abs(out[invalid_mask]) + 1e-6
                            return out
                        meta["sampler"] = positive_wrapper
                    else: meta["sampler"] = base_sampler
                else: meta["sampler"] = lambda m: np.full(m, np.nan)
            
            elif pd.api.types.is_datetime64_any_dtype(ser):
                ints = ser_notna.astype("int64")
                meta["min"], meta["max"] = ints.min(), ints.max()

            else: # String / Object / Categorical
                unique_originals = ser_notna.unique()
                if len(unique_originals) > 0:
                    for val in unique_originals:
                        self._get_or_create_fake_value(val)
                    counts = ser_notna.value_counts()
                    probs = counts / n_notna
                    meta["unique_originals"] = probs.index.tolist()
                    meta["probabilities"] = probs.values.tolist()
                else:
                    meta["unique_originals"], meta["probabilities"] = [], []

            self.col_meta[col] = meta
        # print("Fitting complete.")
        return self

    # --- (No changes to the sample method) ---
    def sample(self, n: int) -> pd.DataFrame:
        # print(f"Generating {n} synthetic rows...")
        rows: list[Dict[str, Any]] = []
        for _ in range(n):
            row: Dict[str, Any] = {}
            for col, meta in self.col_meta.items():
                if self.rng.random() < meta.get("null_ratio", 0):
                    row[col] = pd.NA
                    continue

                dtype = meta.get("dtype", None)

                if "sampler" in meta:
                    try:
                        val = meta["sampler"](1)[0]
                        row[col] = int(round(val)) if meta.get("is_int") and not pd.isna(val) else float(val)
                    except Exception: row[col] = pd.NA
                elif pd.api.types.is_datetime64_any_dtype(dtype) and "min" in meta:
                    try:
                        ts_int = self.rng.integers(meta["min"], meta["max"], endpoint=True)
                        row[col] = pd.to_datetime(ts_int)
                    except Exception: row[col] = pd.NaT
                elif "unique_originals" in meta:
                    if meta["unique_originals"]:
                        original_val = self.rng.choice(meta["unique_originals"], p=meta["probabilities"])
                        row[col] = self._value_map.get(original_val, pd.NA)
                    else: row[col] = pd.NA
                else: row[col] = pd.NA
            rows.append(row)
        
        # print("Synthetic data generation complete.")
        return pd.DataFrame(rows).infer_objects()