from __future__ import annotations

from typing import Any, Dict, Callable, Optional, List
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde, norm
from faker import Faker
import re # Import regex for sanitizing column names

# --- Gaussian Copula Synthesizer (Unchanged) ---
# (Keep the previous version of GaussianCopulaSynth here)
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
            print("Warning: No numeric columns found for GaussianCopulaSynth.")
            return self
        df_num = df[self.names].dropna()
        if len(df_num) < 2:
             print(f"Warning: Not enough non-null numeric data ({len(df_num)} rows) to fit Gaussian Copula. Skipping.")
             self.names = []
             return self
        try:
            U = df_num.rank(method="average") / (len(df_num) + 1)
            U = np.clip(U, 1e-9, 1 - 1e-9)
            Z = norm.ppf(U)
            self.cov = np.cov(Z, rowvar=False)
            if np.isnan(self.cov).any():
                 print("Warning: NaN found in covariance matrix. Gaussian Copula might not work as expected.")
            for col in self.names:
                vals = df_num[col].to_numpy()
                self.sorted_vals[col] = np.sort(vals)
        except Exception as e:
            print(f"Error during GaussianCopulaSynth fit: {e}")
            self.names = []
        return self

    def sample(self, n: int) -> pd.DataFrame:
        if not self.names or self.cov is None or np.isnan(self.cov).any():
            print("Skipping GaussianCopulaSynth sampling due to lack of fitted columns or invalid covariance.")
            return pd.DataFrame(columns=self.names)
        d = len(self.names)
        try:
            min_eig = np.min(np.real(np.linalg.eigvals(self.cov)))
            current_cov = self.cov
            if min_eig < -1e-9:
                 print("Warning: Covariance matrix is not positive semi-definite. Adding jitter.")
                 current_cov = self.cov + np.eye(d) * 1e-6
            z = self.rng.multivariate_normal(np.zeros(d), current_cov, size=n)
            u = norm.cdf(z)
            data: Dict[str, np.ndarray] = {}
            for j, col in enumerate(self.names):
                vals = self.sorted_vals.get(col, np.array([])) # Use get for safety
                if len(vals) > 0 :
                     data[col] = np.interp(u[:, j], np.linspace(0, 1, len(vals)), vals)
                else:
                     data[col] = np.full(n, np.nan)
        except np.linalg.LinAlgError as e:
            print(f"Error sampling from multivariate normal (likely covariance issue): {e}")
            return pd.DataFrame(columns=self.names)
        except Exception as e:
            print(f"Error during GaussianCopulaSynth sampling: {e}")
            return pd.DataFrame(columns=self.names)
        return pd.DataFrame(data)


# --- Simple Tabular Synthesizer (Simplified ID Replacement Version) ---
class SimpleTabularSynth:
    """
    Simplified synthesizer focusing on basic column types and replacing
    string values with consistent identifiers (column_name_number).
    """

    def __init__(self, seed: int | None = None):
        """
        Initializes the synthesizer.
        Args:
            seed: Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)
        self.col_meta: Dict[str, Dict[str, Any]] = {}
        self.fake = Faker() # Keep Faker for potential future use or other column types


    def _sanitize_col_name(self, col_name: str) -> str:
        """Removes problematic characters for use in generated IDs."""
        # Remove non-alphanumeric characters, replace spaces/hyphens with underscore
        s = re.sub(r'[^\w\s-]', '', col_name).strip()
        s = re.sub(r'[-\s]+', '_', s)
        return s if s else "col" # Fallback if name becomes empty


    # --- KDE Sampler (Unchanged) ---
    def _kde_sampler(self, vals: np.ndarray, is_int: bool) -> Callable[[int], np.ndarray]:
        if len(vals) == 0: return lambda m: np.full(m, np.nan)
        # Ensure vals has variance for KDE
        if np.std(vals) < 1e-9:
             const = vals[0] if len(vals) > 0 else 0
             return lambda m, v=const: np.repeat(v, m)
        try:
            kde = gaussian_kde(vals)
            def _draw(m: int) -> np.ndarray:
                out = kde.resample(m).flatten()
                return np.round(out) if is_int else out
            return _draw
        except Exception as e:
             print(f"Warning: KDE fitting failed: {e}. Falling back to uniform sampling.")
             min_val, max_val = vals.min(), vals.max()
             if min_val == max_val:
                   return lambda m, v=min_val: np.repeat(v, m)
             if is_int:
                  return lambda m: self.rng.integers(int(round(min_val)), int(round(max_val)) + 1, size=m)
             else:
                  return lambda m: self.rng.uniform(min_val, max_val, size=m)


    # --- Fit Method (Simplified for ID Replacement) ---
    def fit(self, df: pd.DataFrame) -> "SimpleTabularSynth":
        print("Fitting SimpleTabularSynth (ID Replacement Mode)...")
        for col in df.columns:
            print(f"  Processing column: {col}")
            ser = df[col]
            meta: Dict[str, Any] = { "dtype": ser.dtype, "null_ratio": ser.isna().mean()}
            ser_notna = ser.dropna()
            n_notna = len(ser_notna)

            if n_notna == 0:
                 print(f"    Column '{col}' is all NULLs.")
                 self.col_meta[col] = meta
                 continue

            # --- Numeric ---
            if pd.api.types.is_numeric_dtype(ser):
                 print(f"    Type: Numeric")
                 vals = ser_notna.to_numpy()
                 meta["is_int"] = pd.api.types.is_integer_dtype(ser)
                 unique_vals = np.unique(vals)
                 if len(unique_vals) > 0:
                      meta["positive_only"] = float(vals.min()) >= 0
                      if len(unique_vals) > 1 and np.std(vals) > 1e-9:
                          base_sampler = self._kde_sampler(vals, meta["is_int"])
                      else: # Constant value case
                          const = float(unique_vals[0])
                          base_sampler = lambda m, v=const: np.repeat(v, m)

                      # Positive wrapper (simplified)
                      if meta.get("positive_only", False):
                          def positive_wrapper(m: int, bs=base_sampler) -> np.ndarray:
                              out = bs(m)
                              invalid_mask = out <= 0
                              attempts = 0
                              while np.any(invalid_mask) and attempts < 10:
                                   num_invalid = np.sum(invalid_mask)
                                   out[invalid_mask] = bs(num_invalid)
                                   invalid_mask = out <= 0
                                   attempts += 1
                              if np.any(invalid_mask):
                                   out[invalid_mask] = np.abs(out[invalid_mask]) + 1e-6
                              return out
                          meta["sampler"] = positive_wrapper
                      else:
                           meta["sampler"] = base_sampler
                 else: # Should not happen if n_notna > 0
                       meta["sampler"] = lambda m: np.full(m, np.nan)


            # --- Datetime ---
            elif pd.api.types.is_datetime64_any_dtype(ser):
                print(f"    Type: Datetime")
                ints = ser_notna.astype("int64")
                meta["min"], meta["max"] = ints.min(), ints.max()

            # --- String / Object Columns -> ID Replacement ---
            # Includes boolean, category, object, string dtypes
            else:
                print(f"    Type: {ser.dtype} -> Replacing with ID")
                # Ensure we work with string representations for mapping
                ser_notna_str = ser_notna.astype(str)
                unique_originals = ser_notna_str.unique().tolist()

                if unique_originals:
                    # Sanitize column name for use in ID
                    sanitized_col_name = self._sanitize_col_name(str(col))

                    # Create mapping and list of IDs
                    value_to_id_map = {}
                    id_list = []
                    for i, original_val in enumerate(unique_originals):
                         # Use index + 1 for human-readable 1-based numbering
                         generated_id = f"{sanitized_col_name}_{i+1}"
                         value_to_id_map[original_val] = generated_id
                         id_list.append(generated_id)

                    # Calculate probabilities based on original value frequencies
                    counts = ser_notna_str.value_counts()
                    id_probs = [counts[original_val] / n_notna for original_val in unique_originals]

                    # Store the necessary info for sampling
                    meta["id_list"] = id_list
                    meta["id_probs"] = id_probs
                    # meta["value_to_id_map"] = value_to_id_map # Keep if needed for inspection
                else:
                     print(f"    Warning: Column {col} has no valid unique values after astype(str).")
                     meta["id_list"] = [] # Flag that no IDs can be generated
                     meta["id_probs"] = []

            self.col_meta[col] = meta
        print("Fitting complete.")
        return self


    # --- Sample Method (Simplified for ID Replacement) ---
    def sample(self, n: int) -> pd.DataFrame:
        """Generates n rows of synthetic data."""
        print(f"Generating {n} synthetic rows (ID Replacement Mode)...")
        rows: list[Dict[str, Any]] = []

        for i in range(n):
            # Progress indicator (optional)
            # if (i + 1) % 100 == 0: print(f"  Generated {i+1}/{n} rows...")

            row: Dict[str, Any] = {}
            for col, meta in self.col_meta.items():
                # 1) Nulls
                if self.rng.random() < meta.get("null_ratio", 0):
                    row[col] = pd.NA
                    continue

                dtype = meta.get("dtype", None)

                # 2) Numeric
                if "sampler" in meta:
                     try:
                         sampler_func = meta["sampler"]
                         sampled_array = sampler_func(1)
                         val = sampled_array[0]
                         if meta.get("is_int"):
                             row[col] = int(round(val)) if not pd.isna(val) else pd.NA
                         else:
                             row[col] = float(val) if not pd.isna(val) else pd.NA
                     except Exception as e:
                         print(f"Warning: Numeric sampling error for {col}: {e}")
                         row[col] = pd.NA

                # 3) Datetime
                elif pd.api.types.is_datetime64_any_dtype(dtype) and "min" in meta:
                      try:
                          ts_int = self.rng.integers(meta["min"], meta["max"], endpoint=True)
                          row[col] = pd.to_datetime(ts_int)
                      except Exception as e:
                          print(f"Warning: Datetime sampling error for {col}: {e}")
                          row[col] = pd.NaT

                # 4) ID Replacement Columns (Sample from generated IDs)
                elif "id_list" in meta:
                     if meta["id_list"]: # Check if list is not empty
                          # Sample an ID based on original frequencies
                          row[col] = self.rng.choice(meta["id_list"], p=meta["id_probs"])
                     else:
                          # Fallback if fit found no unique values
                          row[col] = pd.NA


                # 5) Final Fallback (Shouldn't be reached with current logic)
                else:
                     print(f"Warning: No sampling method determined for column '{col}'. Setting to NaN.")
                     row[col] = pd.NA

            rows.append(row)

        print("Synthetic data generation complete.")
        # Attempt to infer best dtypes for final dataframe (IDs will be objects/strings)
        return pd.DataFrame(rows).infer_objects()


# --- Example Usage ---
if __name__ == '__main__':

    print("Example Usage of Simplified ID Replacement Synthesizer:")

    # Sample Data (using the example from your run_synth.py comment)
    df_real = pd.DataFrame({
        "ints":       [1, 2, 3, np.nan, 5],
        "floats":     [0.1, 0.2, 0.3, 0.4, np.nan],
        "dates":      ["2020-01-01", None, "2020-01-03", "2020-01-04", "2020-01-05"],
        "cats":       ["a","b","a","b","c"],
        "long_text":  ["tanaka pfupajena", None, "rachel pfupajena", "simon pfupajena", "bar baz qux"],
        "Segment>>":  ["Consumer | edga |simon", "Corporate| edga | simon", "Consumer| edga |simon", "Home Office| edga |simon", "Consumer| edga |simon"], # Added example
        "Empty Col": [None, None, None, None, None]
    })

    print("\n--- Original Data ---")
    print(df_real)
    print(df_real.info())

    # --- Using SimpleTabularSynth (ID Replacement Mode) ---
    print("\n--- SimpleTabularSynth (ID Replacement Mode) ---")
    # No need for NLP model parameters anymore
    simple_synth = SimpleTabularSynth(seed=42)
    simple_synth.fit(df_real)
    df_simple_fake = simple_synth.sample(5) # Generate 5 rows

    print("\n--- Generated Data (SimpleTabularSynth - ID Replacement) ---")
    print(df_simple_fake)
    print(df_simple_fake.info())

    # Expected output for string columns:
    # cats: values like "cats_1", "cats_2", "cats_3" sampled based on original frequencies of "a", "b", "c"
    # long_text: values like "long_text_1", "long_text_2", etc.
    # Segment>>: values like "Segment_1", "Segment_2", etc. (Note sanitized name)
    # Empty Col: Should be all <NA> or based on null_ratio (which is 1.0 here)