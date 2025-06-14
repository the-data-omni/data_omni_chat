from fastapi import FastAPI
import logging
# from app.services.analytics_service import router as analytics_router
from fastapi.middleware.cors import CORSMiddleware

from app.routes.bigquery_routes import router as bigquery_router
from app.routes.embeddings import (rag_embeddings_router, rag_execution_router,
                                   rag_llm_router)
from app.routes.embeddings import rag_setup_router as embeddings_router
from app.routes.github_routes import (github_qq_router, github_repo_router,
                                      github_schema_router)
from app.routes.queries_routes import router as queries_router
from app.routes.routes import router as analytics_router
from app.routes.descriptions_scraper import scrape_router

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

# Allow requests from your frontend
origins = [
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


# Include the router under a common prefix if desired
app.include_router(bigquery_router, prefix="/api")
app.include_router(queries_router, prefix="/api")
app.include_router(embeddings_router, prefix="/api")
app.include_router(rag_embeddings_router, prefix="/api")
app.include_router(rag_execution_router, prefix="/api")
app.include_router(rag_llm_router, prefix="/api")
app.include_router(analytics_router)
app.include_router(github_schema_router,prefix="/api")
app.include_router(github_qq_router,prefix="/api")
app.include_router(github_repo_router,prefix="/api")
app.include_router(scrape_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)