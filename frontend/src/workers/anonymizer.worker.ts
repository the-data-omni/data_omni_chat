// // declare const self: DedicatedWorkerGlobalScope;

// // declare global {
// //   const loadPyodide: (config: any) => Promise<any>;
// // }

// // interface IncomingMessage {
// //   type: 'anonymize' | 'executeCode';
// //   file: File;
// //   fileName?: string;
// //   pythonCode?: string; // For anonymization
// //   codeToExecute?: string; // For analysis
// // }

// // type OutgoingMessage =
// //   | { status: 'processing'; message: string }
// //   | { status: 'complete'; message: string; rowCount: number; columnCount: number; preview: Record<string, any>[] }
// //   | { status: 'error'; error: string };

// // // --- Worker Logic ---
// // let pyodide: any;

// // async function initPyodide(): Promise<any> {
// //     if (pyodide) return pyodide;
    
// //     // Use dynamic import() for module workers instead of importScripts()
// //     await import("https://cdn.jsdelivr.net/pyodide/v0.25.1/full/pyodide.js");

// //     pyodide = await loadPyodide({});
    
// //     // --- THE FIX: Use loadPackage for built-in scientific packages ---
// //     console.log("Loading core scientific packages...");
// //     await pyodide.loadPackage(['pandas', 'numpy', 'scipy']);
    
// //     // --- Use micropip only for pure python packages from PyPI ---
// //     console.log("Loading additional packages...");
// //     await pyodide.loadPackage("micropip");
// //     const micropip = pyodide.pyimport("micropip");
// //     await micropip.install('faker');
    
// //     console.log("✅ Pyodide and packages are ready.");
// //     return pyodide;
// // }

// // const pyodideReadyPromise = initPyodide();

// // self.onmessage = async (event: MessageEvent<IncomingMessage>) => {
// //     const { type, file, pythonCode, codeToExecute } = event.data;

// //     // --- FIX: Handle the 'init' message type first and exit ---
// //     if (type === 'init') {
// //         // This just ensures the init function is called and warms up the worker.
// //         await pyodideReadyPromise;
// //         return; // Stop processing for this message type.
// //     }

// //     try {
// //         const pyodideInstance = await pyodideReadyPromise;

// //         // Write the provided file to the virtual filesystem
// //         const fileData = await file.arrayBuffer();
// //         pyodideInstance.FS.writeFile("input_data", new Uint8Array(fileData), { encoding: "binary" });

// //         // Route to the correct function based on the message type
// //         if (type === 'anonymize' && pythonCode) {
// //             await handleAnonymize(pyodideInstance, pythonCode, file.name);
// //         } else if (type === 'executeCode' && codeToExecute) {
// //             await handleExecuteCode(pyodideInstance, codeToExecute, file.name);
// //         } else {
// //             throw new Error(`Invalid worker command type: ${type}`);
// //         }

// //     } catch (error: any) {
// //         console.error("An error occurred in the worker:", error);
// //         self.postMessage({ status: "error", error: error.message });
// //     }
// // };
// // async function handleAnonymize(pyodideInstance: any, pythonCode: string, fileName: string) {
// //     pyodideInstance.globals.set("js_file_name", fileName);
// //     pyodideInstance.runPython(pythonCode);
// //     const anonymizationScript = `
// // import pandas as pd
// // import numpy as np
// // import json

// // response = {}
// // try:
// //     # ... (file reading and anonymization logic is unchanged) ...
// //     file_name = js_file_name
// //     if file_name.lower().endswith('.csv'):
// //         df = pd.read_csv("input_data")
// //     elif file_name.lower().endswith('.json'):
// //         df = pd.read_json("input_data", orient='records')
// //     else:
// //         raise ValueError(f"Unsupported file type: {file_name}. Please upload a CSV or JSON file.")

// //     original_row_count = len(df)
// //     simple_synth = SimpleTabularSynth(seed=42).fit(df)
// //     copula_synth = GaussianCopulaSynth(seed=42).fit(df)
// //     df_simple_sampled = simple_synth.sample(n=original_row_count)
// //     df_copula_sampled = copula_synth.sample(n=original_row_count)
// //     anonymized_df = df_simple_sampled
// //     for col in copula_synth.names:
// //         if col in anonymized_df.columns:
// //             anonymized_df[col] = df_copula_sampled[col]

// //     # Use pandas' .to_json() which handles all special types correctly.
// //     preview_json_string = anonymized_df.head(10).to_json(orient='records', date_format='iso')
// //     preview_records = json.loads(preview_json_string)

// //     # =========================================================================
// //     # NEW: Also create a JSON string of the FULL dataset for upload.
// //     full_data_json_string = anonymized_df.to_json(orient='records', date_format='iso')
// //     # =========================================================================

// //     response = {
// //         "status": "complete", "message": "Data anonymization successful!",
// //         "rowCount": len(anonymized_df), "columnCount": len(anonymized_df.columns),
// //         "preview": preview_records,
// //         "fullData": full_data_json_string  # <-- ADD THIS
// //     }
// // except Exception as e:
// //     response = {"status": "error", "error": f"A Python error occurred: {str(e)}"}

// // json.dumps(response)
// // `;
// //         const resultJson = pyodideInstance.runPython(anonymizationScript);
// //     self.postMessage(JSON.parse(resultJson));
// // }

// // async function handleExecuteCode(pyodideInstance: any, codeToExecute: string, fileName: string) {
// //     // This logic is directly adapted from your backend's 'execute_code' function
    
// //     // =========================================================================
// //     // THE FIX: Pass the code string from JavaScript into the Python global scope.
// //     pyodideInstance.globals.set("js_code_to_execute", codeToExecute);
// //     // =========================================================================

// //     const setupScript = `
// // import pandas as pd
// // import numpy as np
// // import json
// // import io
// // from contextlib import redirect_stdout

// // # Load the dataframe from the file in the virtual system
// // if "${fileName}".lower().endswith('.csv'):
// //     df = pd.read_csv("input_data")
// // elif "${fileName}".lower().endswith('.json'):
// //     df = pd.read_json("input_data", orient='records')
// // else:
// //     raise ValueError(f"Unsupported file type: ${fileName}")

// // # Capture the output of the user's code
// // exec_ns = {"df": df}
// // buf = io.StringIO()
// // try:
// //     with redirect_stdout(buf):
// //         # Now we execute the Python variable that we set from JavaScript
// //         exec(js_code_to_execute, exec_ns)
    
// //     raw_result = buf.getvalue().strip()
// //     # The executed code is expected to print a JSON string
// //     parsed_result = json.loads(raw_result)
// //     response = {"status": "complete", "result": parsed_result}
// // except Exception as e:
// //     response = {"status": "error", "error": f"Error executing Python code: {str(e)}"}

// // # Return the final result
// // json.dumps(response)
// // `;

// //     const resultJson = pyodideInstance.runPython(setupScript);
// //     self.postMessage(JSON.parse(resultJson));
// // }

// export { };

// type WorkerCmd = 'init' | 'anonymize' | 'executeCode';

//  interface IncomingMessage {
//    type: WorkerCmd;
//    file: File;
//    fileName?: string;
//    pythonCode?: string; // For anonymization
//    codeToExecute?: string; // For analysis
//  }

// // --- Worker Logic ---
// let pyodide: any;

// async function initPyodide(): Promise<any> {
//     if (pyodide) return pyodide;
    
//     // Use dynamic import() for module workers instead of importScripts()
//     /* Dynamic import of the ESM build – @vite-ignore stops Vite from bundling it. */
//     const { loadPyodide } = await import(
//       /* @vite-ignore */ "https://cdn.jsdelivr.net/pyodide/v0.25.1/full/pyodide.js"
//     ) as any;


//     pyodide = await loadPyodide({});
    
//     // --- THE FIX: Use loadPackage for built-in scientific packages ---
//     console.log("Loading core scientific packages...");
//     await pyodide.loadPackage(['pandas', 'numpy', 'scipy']);
    
//     // --- Use micropip only for pure python packages from PyPI ---
//     console.log("Loading additional packages...");
//     await pyodide.loadPackage("micropip");
//     const micropip = pyodide.pyimport("micropip");
//     await micropip.install('faker');
    
//     console.log("✅ Pyodide and packages are ready.");
//     return pyodide;
// }

// const pyodideReadyPromise = initPyodide();

// self.addEventListener('message', async (event: MessageEvent<IncomingMessage>) => {
//     const { type, file, pythonCode, codeToExecute } = event.data;

//     // --- FIX: Handle the 'init' message type first and exit ---
//     if (type === 'init') {
//         // This just ensures the init function is called and warms up the worker.
//         await pyodideReadyPromise;
//         self.postMessage({ status: "processing", message: "Pyodide ready" });
//         return; // Stop processing for this message type.
//     }

//     try {
//         const pyodideInstance = await pyodideReadyPromise;

//         // Write the provided file to the virtual filesystem
//         const fileData = await file.arrayBuffer();
//         pyodideInstance.FS.writeFile("input_data", new Uint8Array(fileData), { encoding: "binary" });

//         // Route to the correct function based on the message type
//         if (type === 'anonymize' && pythonCode) {
//             await handleAnonymize(pyodideInstance, pythonCode, file.name);
//         } else if (type === 'executeCode' && codeToExecute) {
//             await handleExecuteCode(pyodideInstance, codeToExecute, file.name);
//         } else {
//             throw new Error(`Invalid worker command type: ${type}`);
//         }

//     } catch (error: any) {
//         console.error("An error occurred in the worker:", error);
//         self.postMessage({ status: "error", error: error.message });
//     }
// });
// async function handleAnonymize(pyodideInstance: any, pythonCode: string, fileName: string) {
//     pyodideInstance.globals.set("js_file_name", fileName);
//     pyodideInstance.runPython(pythonCode);
//     const anonymizationScript = `
// import pandas as pd
// import numpy as np
// import json

// response = {}
// try:
//     # ... (file reading and anonymization logic is unchanged) ...
//     file_name = js_file_name
//     if file_name.lower().endswith('.csv'):
//         df = pd.read_csv("input_data")
//     elif file_name.lower().endswith('.json'):
//         df = pd.read_json("input_data", orient='records')
//     else:
//         raise ValueError(f"Unsupported file type: {file_name}. Please upload a CSV or JSON file.")

//     original_row_count = len(df)
//     simple_synth = SimpleTabularSynth(seed=42).fit(df)
//     copula_synth = GaussianCopulaSynth(seed=42).fit(df)
//     df_simple_sampled = simple_synth.sample(n=original_row_count)
//     df_copula_sampled = copula_synth.sample(n=original_row_count)
//     anonymized_df = df_simple_sampled
//     for col in copula_synth.names:
//         if col in anonymized_df.columns:
//             anonymized_df[col] = df_copula_sampled[col]

//     # Use pandas' .to_json() which handles all special types correctly.
//     preview_json_string = anonymized_df.head(10).to_json(orient='records', date_format='iso')
//     preview_records = json.loads(preview_json_string)

//     # =========================================================================
//     # NEW: Also create a JSON string of the FULL dataset for upload.
//     full_data_json_string = anonymized_df.to_json(orient='records', date_format='iso')
//     # =========================================================================

//     response = {
//         "status": "complete", "message": "Data anonymization successful!",
//         "rowCount": len(anonymized_df), "columnCount": len(anonymized_df.columns),
//         "preview": preview_records,
//         "fullData": full_data_json_string  # <-- ADD THIS
//     }
// except Exception as e:
//     response = {"status": "error", "error": f"A Python error occurred: {str(e)}"}

// json.dumps(response)
// `;
//         const resultJson = pyodideInstance.runPython(anonymizationScript);
//     self.postMessage(JSON.parse(resultJson));
// }

// async function handleExecuteCode(pyodideInstance: any, codeToExecute: string, fileName: string) {
//     // This logic is directly adapted from your backend's 'execute_code' function
    
//     // =========================================================================
//     // THE FIX: Pass the code string from JavaScript into the Python global scope.
//     pyodideInstance.globals.set("js_code_to_execute", codeToExecute);
//     // =========================================================================

//     const setupScript = `
// import pandas as pd
// import numpy as np
// import json
// import io
// from contextlib import redirect_stdout

// # Load the dataframe from the file in the virtual system
// if "${fileName}".lower().endswith('.csv'):
//     df = pd.read_csv("input_data")
// elif "${fileName}".lower().endswith('.json'):
//     df = pd.read_json("input_data", orient='records')
// else:
//     raise ValueError(f"Unsupported file type: ${fileName}")

// # Capture the output of the user's code
// exec_ns = {"df": df}
// buf = io.StringIO()
// try:
//     with redirect_stdout(buf):
//         # Now we execute the Python variable that we set from JavaScript
//         exec(js_code_to_execute, exec_ns)
    
//     raw_result = buf.getvalue().strip()
//     # The executed code is expected to print a JSON string
//     parsed_result = json.loads(raw_result)
//     response = {"status": "complete", "result": parsed_result}
// except Exception as e:
//     response = {"status": "error", "error": f"Error executing Python code: {str(e)}"}

// # Return the final result
// json.dumps(response)
// `;

//     const resultJson = pyodideInstance.runPython(setupScript);
//     self.postMessage(JSON.parse(resultJson));
// }
/* eslint-disable @typescript-eslint/no‑explicit‑any */

/** ------------------------------------------------------------------
 *  This file is treated as a module so that global augmentations are
 *  legal and the `declare global {}` rules are respected.
 *  ----------------------------------------------------------------- */
export {};

/* ------------------------------------------------------------------ *
 *  Types
 * ------------------------------------------------------------------ */

type WorkerCmd = 'init' | 'anonymize' | 'executeCode';

interface IncomingMessage {
  type: WorkerCmd;
  file: File;
  fileName?: string;
  pythonCode?: string;   // Sent when type === 'anonymize'
  codeToExecute?: string; // Sent when type === 'executeCode'
}

type OutgoingMessage =
  | { status: 'processing'; message: string }
  | {
      status: 'complete';
      message: string;
      rowCount: number;
      columnCount: number;
      preview: Record<string, any>[];
      fullData?: string;
    }
  | { status: 'error'; error: string };

/* ------------------------------------------------------------------ *
 *  Pyodide bootstrap
 * ------------------------------------------------------------------ */

let pyodide: any;

/** Lazily load Pyodide and the required packages. */
async function initPyodide(): Promise<any> {
  if (pyodide) return pyodide;

  /* Import the ESM build so that loadPyodide is actually exported. */
  const pyodideModule = await import(
    /* @vite-ignore */ "https://cdn.jsdelivr.net/pyodide/v0.25.1/full/pyodide.mjs"
  );

  // loadPyodide might be a named export or the default export
  const loader =
    typeof pyodideModule.loadPyodide === "function"
      ? pyodideModule.loadPyodide
      : (pyodideModule as any).default;

  pyodide = await loader({});

  console.log("Loading core scientific packages…");
  await pyodide.loadPackage(["pandas", "numpy", "scipy"]);

  console.log("Loading additional packages…");
  await pyodide.loadPackage("micropip");
  const micropip = pyodide.pyimport("micropip");
  await micropip.install("faker");

  console.log("✅ Pyodide and packages are ready.");
  return pyodide;
}


const pyodideReadyPromise = initPyodide();

/* ------------------------------------------------------------------ *
 *  Worker message handler
 * ------------------------------------------------------------------ */

self.addEventListener('message', async (event: MessageEvent<IncomingMessage>) => {
  const { type, file, pythonCode, codeToExecute } = event.data;

  /* Warm‑up message – lets the UI know when Pyodide is ready. */
  if (type === 'init') {
    await pyodideReadyPromise;
    self.postMessage({
      status: 'processing',
      message: 'Pyodide ready',
    } satisfies OutgoingMessage);
    return;
  }

  try {
    const pyodideInstance = await pyodideReadyPromise;

    /* Write the uploaded file into Pyodide’s virtual FS. */
    const fileBuffer = await file.arrayBuffer();
    pyodideInstance.FS.writeFile('input_data', new Uint8Array(fileBuffer), {
      encoding: 'binary',
    });

    if (type === 'anonymize' && pythonCode) {
      await handleAnonymize(pyodideInstance, pythonCode, file.name);
    } else if (type === 'executeCode' && codeToExecute) {
      await handleExecuteCode(pyodideInstance, codeToExecute, file.name);
    } else {
      throw new Error(`Invalid or incomplete worker command: ${type}`);
    }
  } catch (err: any) {
    console.error('Worker error:', err);
    const outgoing: OutgoingMessage = { status: 'error', error: err.message };
    self.postMessage(outgoing);
  }
});

/* ------------------------------------------------------------------ *
 *  Helpers
 * ------------------------------------------------------------------ */

/** Run the user‑supplied anonymisation script. */
async function handleAnonymize(
  pyodideInstance: any,
  pythonCode: string,
  fileName: string,
) {
  /* Pass filename and helper code into the Python scope. */
  pyodideInstance.globals.set('js_file_name', fileName);
  pyodideInstance.runPython(pythonCode); // user’s helper definitions

  const anonymizationScript = `
import pandas as pd
import json

response = {}
try:
    file_name = js_file_name
    if file_name.lower().endswith('.csv'):
        df = pd.read_csv("input_data")
    elif file_name.lower().endswith('.json'):
        df = pd.read_json("input_data", orient='records')
    else:
        raise ValueError(f"Unsupported file type: {file_name}. Please upload a CSV or JSON file.")

    original_row_count = len(df)
    simple_synth = SimpleTabularSynth(seed=42).fit(df)
    copula_synth = GaussianCopulaSynth(seed=42).fit(df)
    df_simple = simple_synth.sample(n=original_row_count)
    df_copula = copula_synth.sample(n=original_row_count)

    anonymized_df = df_simple
    for col in copula_synth.names:
        if col in anonymized_df.columns:
            anonymized_df[col] = df_copula[col]

    preview_json = anonymized_df.head(10).to_json(orient='records', date_format='iso')
    full_json    = anonymized_df.to_json(orient='records', date_format='iso')

    response = {
        "status": "complete",
        "message": "Data anonymization successful!",
        "rowCount": len(anonymized_df),
        "columnCount": len(anonymized_df.columns),
        "preview": json.loads(preview_json),
        "fullData": full_json
    }
except Exception as e:
    response = {"status": "error", "error": f"A Python error occurred: {str(e)}"}

json.dumps(response)
`;
  const resultJson = pyodideInstance.runPython(anonymizationScript);
  self.postMessage(JSON.parse(resultJson) as OutgoingMessage);
}

/** Execute arbitrary analysis code provided by the user. */
async function handleExecuteCode(
  pyodideInstance: any,
  codeToExecute: string,
  fileName: string,
) {
  pyodideInstance.globals.set('js_code_to_execute', codeToExecute);

  const script = `
import pandas as pd, json, io
from contextlib import redirect_stdout

if "${fileName}".lower().endswith('.csv'):
    df = pd.read_csv("input_data")
elif "${fileName}".lower().endswith('.json'):
    df = pd.read_json("input_data", orient='records')
else:
    raise ValueError(f"Unsupported file type: ${fileName}")

buf = io.StringIO()
exec_ns = {"df": df}
try:
    with redirect_stdout(buf):
        exec(js_code_to_execute, exec_ns)

    out = buf.getvalue().strip()
    parsed = json.loads(out)
    response = {"status": "complete", "result": parsed}
except Exception as e:
    response = {"status": "error", "error": f"Error executing Python code: {str(e)}"}

json.dumps(response)
`;
  const resultJson = pyodideInstance.runPython(script);
  self.postMessage(JSON.parse(resultJson) as OutgoingMessage);
}
