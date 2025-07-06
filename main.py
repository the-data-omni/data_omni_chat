from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

import logging
from fastapi.middleware.cors import CORSMiddleware

from app.routes.routes import router as analytics_router
from app.routes.google_slides_route   import router as google_slides_router


logging.basicConfig(level=logging.INFO) # Set default level for your app's logs
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

description = """
The Data Beast API helps you do awesome stuff. 🚀

## Items

You can **read items**.

## End To End Build Your Own SQL/LLM System

You will be able to:

* **Evaluate Your Schema** 🚀.
* **Improve Your Schema** 🚀.
* **Set Up Bigquery AI Features for your data** 🚀.
* **Build Question/Query Pair Examples** 🚀.
* **Build Embeddings for RAG** 🚀.
* **Ask Questions** 🚀.
"""

app = FastAPI(
    title="The Data Omni",
    description=description,
    summary="Use an LLM where your data is",
    version="0.0.1",
    contact={
        "name":"Tanaka Pfupajena",
        "email":"Tanaka.Pfupajena@gmail.com"
    },
    license_info={
        "name": "Apache 2.0",
        "identifier": "MIT",
    },

)

build_dir = "dist"

# Path to the assets directory within the build directory
assets_dir = os.path.join(build_dir, "assets")

if not os.path.exists(build_dir):
    raise RuntimeError("Build directory not found. Make sure to build the React app and place the 'build' folder in the backend directory.")

app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


# Allow requests from your frontend
origins = [
    "http://localhost:8000",
    "http://localhost:3000",
    # Add other origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,         # Allows specific origins
    allow_credentials=True,
    allow_methods=["*"],           # Allows all HTTP methods
    allow_headers=["*"],           # Allows all headers
)
# Catch-all route to serve the React app's index.html
@app.get("/{catchall:path}", response_class=FileResponse)
def serve_react_app(catchall: str):
    # This logic serves 'index.html' for any path that isn't a static file
    index_path = os.path.join(build_dir, "index.html")
    return FileResponse(index_path)

# Include the router under a common prefix if desired
app.include_router(analytics_router)
app.include_router(google_slides_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)