"use client";

import * as React from "react";
import { useState, useEffect, useRef } from "react";
import Modal from "@mui/material/Modal";
import Box from "@mui/material/Box";
import Divider from "@mui/material/Divider";
import IconButton from "@mui/material/IconButton";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import FormControl from "@mui/material/FormControl";
import FormControlLabel from "@mui/material/FormControlLabel";
import FormLabel from "@mui/material/FormLabel";
import Switch from "@mui/material/Switch";
import TextField from "@mui/material/TextField";
import Tooltip from "@mui/material/Tooltip";
import Button from "@mui/material/Button";
import { Paperclip as PaperclipIcon, X as XIcon } from "@phosphor-icons/react/dist/ssr";
import CircularProgress from '@mui/material/CircularProgress';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import Alert from "@mui/material/Alert";
import AlertTitle from "@mui/material/AlertTitle";
import { ChatContext } from '@/components/chat/chat-context';

// Props for the modal component
export interface Modal1Props {
  onClose?: (anonymizedDataUploaded: boolean) => void;
}

// Data structure for the anonymized data preview from the worker
interface AnonymizedData {
    message: string;
    rowCount: number;
    columnCount: number;
    preview: Record<string, string | number>[];
}

// Data structure for the original data preview
interface OriginalDataPreview {
  headers: string[];
  rows: (string | number)[][];
}

// The Python code is stored as a string constant to be sent to the worker.
const ANONYMIZATION_PYTHON_CODE = `
from __future__ import annotations
from typing import Any, Dict, Callable, Optional, List
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde, norm
from faker import Faker
import re

class GaussianCopulaSynth:
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.names: list[str] = []
        self.cov: Optional[np.ndarray] = None
        self.sorted_vals: Dict[str, np.ndarray] = {}
    def fit(self, df: pd.DataFrame) -> "GaussianCopulaSynth":
        self.names = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not self.names: return self
        df_num = df[self.names].dropna()
        if len(df_num) < 2: self.names = []; return self
        try:
            U = df_num.rank(method="average") / (len(df_num) + 1)
            U = np.clip(U, 1e-9, 1 - 1e-9)
            Z = norm.ppf(U)
            self.cov = np.cov(Z, rowvar=False)
            if np.isnan(self.cov).any(): pass
            for col in self.names: self.sorted_vals[col] = np.sort(df_num[col].to_numpy())
        except Exception as e: self.names = []
        return self
    def sample(self, n: int) -> pd.DataFrame:
        if not self.names or self.cov is None or np.isnan(self.cov).any(): return pd.DataFrame(columns=self.names)
        d = len(self.names)
        try:
            current_cov = self.cov + np.eye(d) * 1e-6 if np.min(np.real(np.linalg.eigvals(self.cov))) < -1e-9 else self.cov
            z = self.rng.multivariate_normal(np.zeros(d), current_cov, size=n)
            u = norm.cdf(z)
            data: Dict[str, np.ndarray] = {}
            for j, col in enumerate(self.names):
                vals = self.sorted_vals.get(col, np.array([]))
                data[col] = np.interp(u[:, j], np.linspace(0, 1, len(vals)), vals) if len(vals) > 0 else np.full(n, np.nan)
        except Exception as e: return pd.DataFrame(columns=self.names)
        return pd.DataFrame(data)

class SimpleTabularSynth:
    LOWERCASE_LETTERS, UPPERCASE_LETTERS, DIGITS = 'abcdefghijklmnopqrstuvwxyz', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', '0123456789'
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.col_meta: Dict[str, Dict[str, Any]] = {}
        self.fake = Faker()
        self._value_map: Dict[Any, Any] = {}
        self._char_map_lower: Dict[str, str] = {}
        self._char_map_upper: Dict[str, str] = {}
        self._char_map_digit: Dict[str, str] = {}
    def _infer_and_convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object' and not df_processed[col].isnull().all():
                try:
                    coerced_series = pd.to_datetime(df_processed[col], errors='coerce')
                    if coerced_series.isnull().sum() < len(df_processed[col]): df_processed[col] = coerced_series
                except Exception: continue
        return df_processed
    def _generate_random_digits(self, length: int, allow_leading_zero: bool = False) -> str:
        if length == 0: return ""
        first = self.rng.choice(list(self.DIGITS if allow_leading_zero else self.DIGITS[1:]))
        rest = self.rng.choice(list(self.DIGITS), size=length - 1)
        return "".join([first, *rest])
    def _get_or_create_fake_value(self, val: Any) -> Any:
        if pd.isna(val) or val in self._value_map: return self._value_map.get(val, val)
        new_val = val
        if isinstance(val, str):
            faked_chars = []
            for char in val:
                if 'a' <= char <= 'z': self._char_map_lower.setdefault(char, self.rng.choice(list(self.LOWERCASE_LETTERS))); faked_chars.append(self._char_map_lower[char])
                elif 'A' <= char <= 'Z': self._char_map_upper.setdefault(char, self.rng.choice(list(self.UPPERCASE_LETTERS))); faked_chars.append(self._char_map_upper[char])
                elif '0' <= char <= '9': self._char_map_digit.setdefault(char, self.rng.choice(list(self.DIGITS))); faked_chars.append(self._char_map_digit[char])
                else: faked_chars.append(char)
            new_val = "".join(faked_chars)
        self._value_map[val] = new_val
        return new_val
    def _kde_sampler(self, vals: np.ndarray, is_int: bool) -> Callable[[int], np.ndarray]:
        if len(vals) == 0: return lambda m: np.full(m, np.nan)
        if np.std(vals) < 1e-9: const = vals[0]; return lambda m, v=const: np.repeat(v, m)
        try:
            kde = gaussian_kde(vals)
            def _draw(m: int) -> np.ndarray: return np.round(kde.resample(m).flatten()) if is_int else kde.resample(m).flatten()
            return _draw
        except Exception:
             min_v, max_v = vals.min(), vals.max()
             if min_v == max_v: return lambda m, v=min_v: np.repeat(v, m)
             return lambda m: self.rng.integers(int(round(min_v)), int(round(max_v)) + 1, size=m) if is_int else self.rng.uniform(min_v, max_v, size=m)
    def fit(self, df: pd.DataFrame) -> "SimpleTabularSynth":
        df_fitted = self._infer_and_convert_types(df)
        for col in df_fitted.columns:
            ser, meta = df_fitted[col], {"dtype": df_fitted[col].dtype, "null_ratio": df_fitted[col].isna().mean()}
            ser_notna = ser.dropna()
            if len(ser_notna) == 0: self.col_meta[col] = meta; continue
            if pd.api.types.is_numeric_dtype(ser): meta["is_int"] = pd.api.types.is_integer_dtype(ser); meta["sampler"] = self._kde_sampler(ser_notna.to_numpy(), meta["is_int"])
            elif pd.api.types.is_datetime64_any_dtype(ser): ints = ser_notna.astype("int64"); meta["min"], meta["max"] = ints.min(), ints.max()
            else:
                unique_originals = ser_notna.unique()
                for val in unique_originals: self._get_or_create_fake_value(val)
                counts = ser_notna.value_counts(normalize=True)
                meta["unique_originals"], meta["probabilities"] = counts.index.tolist(), counts.values.tolist()
            self.col_meta[col] = meta
        return self
    def sample(self, n: int) -> pd.DataFrame:
        rows = []
        for _ in range(n):
            row: Dict[str, Any] = {}
            for col, meta in self.col_meta.items():
                if self.rng.random() < meta.get("null_ratio", 0): row[col] = pd.NA; continue
                dtype = meta.get("dtype")
                if "sampler" in meta: val = meta["sampler"](1)[0]; row[col] = int(round(val)) if meta.get("is_int") and not pd.isna(val) else float(val)
                elif pd.api.types.is_datetime64_any_dtype(dtype) and "min" in meta: row[col] = pd.to_datetime(self.rng.integers(meta["min"], meta["max"], endpoint=True))
                elif "unique_originals" in meta and meta["unique_originals"]: row[col] = self._value_map.get(self.rng.choice(meta["unique_originals"], p=meta["probabilities"]), pd.NA)
                else: row[col] = pd.NA
            rows.append(row)
        return pd.DataFrame(rows).infer_objects()
`;

function saveFileToIndexedDB(file: File): Promise<void> {
  return new Promise((resolve, reject) => {
    // 1. Open a connection to our database
    const dbName = "OriginalFileDB";
    const storeName = "originalFiles";
    const request = indexedDB.open(dbName, 1);

    // 2. Create the schema if it doesn't exist
    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(storeName)) {
        db.createObjectStore(storeName);
      }
    };

    request.onerror = (event) => {
      console.error("IndexedDB error:", (event.target as IDBOpenDBRequest).error);
      reject("IndexedDB error");
    };

    // 3. Save the file when the connection is successful
    request.onsuccess = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      const transaction = db.transaction(storeName, "readwrite");
      const store = transaction.objectStore(storeName);
      
      // We use the file name as the key and the File object as the value
      const putRequest = store.put(file, file.name);

      putRequest.onsuccess = () => {
        console.log(`File '${file.name}' saved to IndexedDB.`);
        resolve();
      };

      putRequest.onerror = () => {
        console.error("Error saving file to IndexedDB:", putRequest.error);
        reject("Error saving file");
      };
      
      transaction.oncomplete = () => {
        db.close();
      };
    };
  });
}

function getFileFromIndexedDB(fileName: string): Promise<File> {
  return new Promise((resolve, reject) => {
    const dbName = "OriginalFileDB";
    const storeName = "originalFiles";
    const request = indexedDB.open(dbName, 1);

    request.onerror = (event) => reject("IndexedDB error");

    request.onsuccess = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      const transaction = db.transaction(storeName, "readonly");
      const store = transaction.objectStore(storeName);
      const getRequest = store.get(fileName);

      getRequest.onsuccess = () => {
        if (getRequest.result) {
          resolve(getRequest.result);
        } else {
          reject(`File '${fileName}' not found in IndexedDB.`);
        }
      };
      getRequest.onerror = () => reject("Error getting file from IndexedDB");
      transaction.oncomplete = () => db.close();
    };
  });
}

export default function Modal1({ onClose }: Modal1Props): React.JSX.Element {
  // --- State Management ---
  const [isFront, setIsFront] = useState(true);
  const [file, setFile] = useState<File | null>(null);
  const [hasHeaders, setHasHeaders] = useState(true);
  
  // Worker & Anonymization State
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingMessage, setProcessingMessage] = useState('');
  const [anonymizedData, setAnonymizedData] = useState<AnonymizedData | null>(null);
  const [fullAnonymizedData, setFullAnonymizedData] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  // Upload State
  const [isUploading, setIsUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<{ success: boolean; message: string } | null>(null);
  const { setActiveFileName, analysisWorker } = React.useContext(ChatContext);
  const [originalDataPreview, setOriginalDataPreview] = useState<OriginalDataPreview | null>(null);
  
  React.useEffect(() => {
    if (!analysisWorker) {
      return;
    }

    
      const handleMessage = (event: MessageEvent) => {
      const { status, ...data } = event.data;

      // Since the worker is shared, we should check if the message is for us.
      // A simple way is to check for a key that only our response has, like 'fullData'.
      if ('fullData' in data) {
          switch (status) {
          case 'processing': {
              setProcessingMessage(data.message);
          
          break;
          }
          case 'complete': {
            setAnonymizedData(data);
            setFullAnonymizedData(data.fullData);
            setError(null);
            setIsProcessing(false);
          
          break;
          }
          case 'error': {
            setError(data.error || 'An unknown worker error occurred.');
            setIsProcessing(false);
          
          break;
          }
          // No default
          }
      }
    };
    
    analysisWorker.addEventListener('message', handleMessage);
    
    // Cleanup: remove the listener when the modal unmounts
    return () => {
      analysisWorker.removeEventListener('message', handleMessage);
    };

  }, [analysisWorker, setAnonymizedData, setFullAnonymizedData, setError, setIsProcessing, setProcessingMessage]);



  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0] ?? null;
    setFile(selectedFile);
    setActiveFileName(selectedFile ? selectedFile.name : null);

    if (selectedFile) {
      saveFileToIndexedDB(selectedFile).catch(error_ => {
        console.error("Failed to save original file:", error_);
      });

      // --- Preview Logic ---
      const reader = new FileReader();
      reader.onload = (event) => {
        const text = event.target?.result as string;
        const lines = text.split('\n').filter(line => line.trim() !== '');
        if (lines.length > 0) {
          const headers = hasHeaders ? lines[0].split(',') : lines[0].split(',').map((_, i) => `Column ${i + 1}`);
          const rows = (hasHeaders ? lines.slice(1, 6) : lines.slice(0, 5)).map(line => line.split(','));
          setOriginalDataPreview({ headers, rows });
        }
      };
      reader.readAsText(selectedFile);
      // --- End Preview Logic ---
    }

    setAnonymizedData(null);
    setFullAnonymizedData(null);
    setError(null);
    setUploadResult(null);
    setProcessingMessage('');
    setIsProcessing(false);
    setIsUploading(false);
    if (!isFront) setIsFront(true);
  };
  
  const handleAnonymize = () => {
    // This function now uses the shared worker.
    if (!file || !analysisWorker) return;

    setError(null);
    setUploadResult(null);
    setIsProcessing(true);
    setProcessingMessage('Sending file to worker...');
    setIsFront(false);
    
    analysisWorker.postMessage({
      type: 'anonymize',
      file,
      fileName: file.name,
      pythonCode: ANONYMIZATION_PYTHON_CODE,
    });
  };
  
  const handleUploadAnonymizedData = async () => {
    if (!fullAnonymizedData) {
        setUploadResult({ success: false, message: 'No anonymized data available to upload.' });
        return;
    }
    setIsUploading(true);
    setUploadResult(null);
    try {
        const response = await fetch("http://127.0.0.1:8000/upload_anonymized_data", {
            method: "POST",
            headers: { 'Content-Type': 'application/json' },
            body: fullAnonymizedData,
        });
        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.detail || 'An unknown API error occurred.');
        }
        setUploadResult({ success: true, message: result.message });
    } catch (error_: any) {
        setUploadResult({ success: false, message: error_.message });
    } finally {
        setIsUploading(false);
    }
  };

  const handleClose = () => {
    onClose?.(!!uploadResult?.success);
  };
  
  // --- Rendering Logic ---
  const renderFront = () => (
    <Box sx={{ backfaceVisibility: "hidden", display: "flex", flexDirection: "column", height: "100%" }}>
        <Typography variant="h4" gutterBottom>Upload CSV or JSON for Anonymization</Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 2}}>
            Your data will be processed and anonymized entirely within your browser. Nothing is sent to a server until you approve the upload.
        </Typography>
        <Divider sx={{ my: 2 }} />
        <FormControl component="fieldset">
            <FormLabel sx={{ mb: 1 }}>Does the first row contain headers? (For CSV files)</FormLabel>
            <FormControlLabel control={<Switch checked={hasHeaders} onChange={(e) => setHasHeaders(e.target.checked)} />} label={hasHeaders ? "Yes" : "No"} />
        </FormControl>
        {!hasHeaders && <TextField label="Header names (comma separated)" fullWidth sx={{mt: 2}} />}
        <Divider sx={{ my: 3 }} />
        <Stack direction="row" spacing={2} alignItems="center">
            <Tooltip title="Attach CSV or JSON file">
                <IconButton component="label" sx={{ p: 2, border: "1px dashed", borderRadius: 2 }}>
                    <input type="file" accept=".csv,.json" hidden onChange={handleFileChange} />
                    <PaperclipIcon size={28} />
                </IconButton>
            </Tooltip>
            {file && <Typography sx={{ fontWeight: "bold" }}>{file.name}</Typography>}
            <Box sx={{ flex: 1 }} />
            <Button variant="contained" size="large" onClick={handleAnonymize} disabled={!file || isProcessing} sx={{ px: 4 }}>
                {isProcessing ? 'Anonymizing...' : 'Anonymize Data'}
            </Button>
        </Stack>
        {originalDataPreview && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="h6" gutterBottom>Data Preview</Typography>
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    {originalDataPreview.headers.map(header => (
                      <TableCell key={header} sx={{ fontWeight: 'bold' }}>{header}</TableCell>
                    ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {originalDataPreview.rows.map((row, index) => (
                    <TableRow key={index}>
                      {row.map((cell, cellIndex) => (
                        <TableCell key={cellIndex}>{cell}</TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Box>
        )}
    </Box>
  );

  const renderBack = () => (
    <Box sx={{ backfaceVisibility: "hidden", transform: "rotateY(180deg)", position: "absolute", inset: 0, display: "flex", flexDirection: "column", p:2 }}>
      <Typography variant="h4" gutterBottom textAlign="center">Anonymization Result</Typography>
      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', gap: 2, overflowY: 'auto', py: 2 }}>
          {isProcessing && (
              <>
                  <CircularProgress />
                  <Typography>{processingMessage || 'Initializing...'}</Typography>
              </>
          )}
          {error && (
              <Alert severity="error" sx={{width: '100%'}}>
                  <AlertTitle>Anonymization Failed</AlertTitle>
                  {error}
              </Alert>
          )}
          {anonymizedData && (
              <Box sx={{ width: '100%' }}>
                  <Alert severity="success" sx={{ mb: 2 }}>{anonymizedData.message}</Alert>
                  <Typography variant="body2" sx={{ mb: 2 }}>
                      Preview of generated data ({anonymizedData.preview.length} of {anonymizedData.rowCount} total rows):
                  </Typography>
                  <TableContainer component={Paper} variant="outlined">
                      <Table size="small">
                          <TableHead>
                              <TableRow>
                                  {anonymizedData.preview.length > 0 && Object.keys(anonymizedData.preview[0]).map(header => (
                                      <TableCell key={header} sx={{ fontWeight: 'bold' }}>{header}</TableCell>
                                  ))}
                              </TableRow>
                          </TableHead>
                          <TableBody>
                              {anonymizedData.preview.map((row, index) => (
                                  <TableRow key={index}>
                                      {Object.values(row).map((value, cellIndex) => (
                                          <TableCell key={cellIndex}>{String(value)}</TableCell>
                                      ))}
                                  </TableRow>
                              ))}
                          </TableBody>
                      </Table>
                  </TableContainer>
                  <Stack spacing={2} sx={{mt: 3, alignItems: 'center'}}>
                      <Button 
                        variant="contained" 
                        color="secondary"
                        size="large"
                        onClick={handleUploadAnonymizedData} 
                        disabled={isUploading || !!uploadResult?.success}
                      >
                          {isUploading ? <CircularProgress size={24} color="inherit"/> : 'Upload Anonymized Data'}
                      </Button>
                      {uploadResult && (
                          <Alert severity={uploadResult.success ? 'success' : 'error'} sx={{width: '100%', maxWidth: 500}}>
                              <AlertTitle>{uploadResult.success ? 'Success' : 'Error'}</AlertTitle>
                              {uploadResult.message}
                          </Alert>
                      )}
                  </Stack>
              </Box>
          )}
          {!isProcessing && !anonymizedData && error && (
             <Button sx={{mt: 3}} onClick={() => setIsFront(true)}>Back to Upload</Button>
          )}
      </Box>
    </Box>
  );

  return (
    <Modal open onClose={handleClose}>
        <Box sx={{ position: "absolute", top: "50%", left: "50%", transform: "translate(-50%, -50%)", width: "80vw", maxWidth: 1000, maxHeight: "90vh", bgcolor: "background.paper", boxShadow: 24, borderRadius: 4, p: 4, perspective: 1200 }}>
            <IconButton onClick={handleClose} sx={{ position: "absolute", top: 16, right: 16, zIndex: 10 }}>
                <XIcon size={24} />
            </IconButton>
            <Box sx={{ position: "relative", width: "100%", height: "100%", transformStyle: "preserve-3d", transition: "transform 0.8s", transform: isFront ? "rotateY(0)" : "rotateY(180deg)"}}>
                {renderFront()}
                {renderBack()}
            </Box>
        </Box>
    </Modal>
  );
}