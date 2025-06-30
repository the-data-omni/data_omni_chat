(function(){"use strict";let n;async function l(){if(n)return n;const e=await import("https://cdn.jsdelivr.net/pyodide/v0.25.1/full/pyodide.mjs");return n=await(typeof e.loadPyodide=="function"?e.loadPyodide:e.default)({}),console.log("Loading core scientific packages…"),await n.loadPackage(["pandas","numpy","scipy"]),console.log("Loading additional packages…"),await n.loadPackage("micropip"),await n.pyimport("micropip").install("faker"),console.log("✅ Pyodide and packages are ready."),n}const d=l();self.addEventListener("message",async e=>{const{type:s,file:o,pythonCode:r,codeToExecute:t}=e.data;if(s==="init"){await d,self.postMessage({status:"processing",message:"Pyodide ready"});return}try{const a=await d,i=await o.arrayBuffer();if(a.FS.writeFile("input_data",new Uint8Array(i),{encoding:"binary"}),s==="anonymize"&&r)await c(a,r,o.name);else if(s==="executeCode"&&t)await p(a,t,o.name);else throw new Error(`Invalid or incomplete worker command: ${s}`)}catch(a){console.error("Worker error:",a);const i={status:"error",error:a.message};self.postMessage(i)}});async function c(e,s,o){e.globals.set("js_file_name",o),e.runPython(s);const t=e.runPython(`
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
`);self.postMessage(JSON.parse(t))}async function p(e,s,o){e.globals.set("js_code_to_execute",s);const r=`
import pandas as pd, json, io
from contextlib import redirect_stdout

if "${o}".lower().endswith('.csv'):
    df = pd.read_csv("input_data")
elif "${o}".lower().endswith('.json'):
    df = pd.read_json("input_data", orient='records')
else:
    raise ValueError(f"Unsupported file type: ${o}")

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
`,t=e.runPython(r);self.postMessage(JSON.parse(t))}})();
