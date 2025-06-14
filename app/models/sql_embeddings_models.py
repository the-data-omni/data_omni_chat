from pydantic import BaseModel, Field, field_validator, ValidationInfo 
from typing import List, Dict, Any, Optional

class PerCallConfig(BaseModel):
    gcp_project_id: str = Field(..., description="Google Cloud Project ID for this operation.")
    dataset_id: str = Field(..., description="BigQuery Dataset ID where RAG resources (models, tables, procedures) are/will be stored.")
    region: str = Field(..., description="Google Cloud region for BigQuery dataset and Vertex connection (e.g., 'US', 'europe-west1').")
    vertex_connection_name: str = Field(..., description="BigQuery connection name to Vertex AI (e.g., 'us-central1.vertex-ai'). Must be fully qualified with region if connection is regional.")
    embedding_model_endpoint: str = Field(..., description="Vertex AI endpoint for the default embedding model created during initialization (e.g., 'text-embedding-004').")

class FieldDetail(BaseModel):
    field_name: str
    field_type: str
    field_description: str

class SchemaInputPayload(BaseModel): 
    config: PerCallConfig
    dataset_name: str 
    table_name: str   
    table_description: Optional[str] = None
    fields: List[FieldDetail]
    join_relationships: List[str] = Field(default_factory=list) 
    document_id: Optional[str] = None

class SingleQQPairInput(BaseModel): # Renamed for clarity, used by the processing service
    config: PerCallConfig
    question_text: str
    sql_query_text: str
    qq_id: Optional[str] = None # As seen in your endpoint logic

class QuestionQueryItem(BaseModel):
    question_text: str
    sql_query_text: str

class QQDataStructure(BaseModel):
    qq_data: List[QuestionQueryItem]

class BulkUploadRequestBody(BaseModel):
    pairs: QQDataStructure
    config: PerCallConfig


class QuestionQueryInputPayload(BaseModel): 
    config: PerCallConfig
    question_text: str
    sql_query_text: str
    qq_id: Optional[str] = None

class InitializeResourcesRequestPayload(BaseModel): 
    config: PerCallConfig
    # Optionally add default generator model endpoint if you want it configurable during init
    # default_generator_model_endpoint: str = Field("gemini-1.0-pro", description="Vertex AI endpoint for the default generator model.")


class CreateRemoteModelRequestPayload(BaseModel):
    config: PerCallConfig
    model_name: str = Field(..., description="The desired name for the BQML model object in your BigQuery RAG dataset.")
    vertex_ai_endpoint: str = Field(..., description="The Vertex AI endpoint this BQML model will point to (e.g., 'text-embedding-004', 'gemini-1.0-pro').")


class CreateVectorIndexRequestPayload(BaseModel): 
    config: PerCallConfig
    index_name: str = Field(..., description="Name for the vector index.")
    embeddings_table_name: str = Field(..., description="Name of the BigQuery table containing the embeddings (e.g., 'schema_embeddings_table').")
    embedding_column_name: str = Field(..., description="Name of the column containing the ARRAY<FLOAT64> embeddings.")
    distance_type: str = Field("COSINE", description="Distance type for vector search.")
    index_type: str = Field("IVF", description="Index type (IVF, TREE_AH, or BRUTE_FORCE for default behavior).") 
    ivf_num_lists: Optional[int] = Field(default=100, description="Number of lists for IVF index if type is IVF.", gt=0)


class LLMOptionsPayload(BaseModel): 
    temperature: Optional[float] = Field(0.2, ge=0.0, le=2.0) # Vertex AI allows up to 2.0 for some models
    max_output_tokens: Optional[int] = Field(1024, gt=0)
    top_p: Optional[float] = Field(0.95, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(40, ge=0)
    flatten_json_output: Optional[bool] = True

class CreateRAGProcedureRequestPayload(BaseModel): 
    config: PerCallConfig 
    procedure_name: str = Field(...)
    embedding_model_name: str = Field(...) 
    generator_model_name: str = Field(...) 
    schema_corpus_table_name: str = Field(...)
    schema_embeddings_table_name: str = Field(...)
    schema_embedding_column_name: str = Field("embedding")
    schema_vector_top_k: int = Field(5, gt=0)
    qq_corpus_table_name: str = Field(...)
    qq_embeddings_table_name: str = Field(...)
    qq_embedding_column_name: str = Field("embedding")
    qq_vector_top_k: int = Field(3, gt=0)
    llm_options: Optional[LLMOptionsPayload] = Field(default_factory=LLMOptionsPayload)

class ExecuteRAGProcedureRequestPayload(BaseModel): 
    config: PerCallConfig 
    procedure_name: str = Field(...)
    user_question: str = Field(...)

class ExecuteRAGProcedureResponse(BaseModel):
    generated_sql: Optional[str] = None
    retrieved_schema_info: Optional[str] = None
    retrieved_qq_examples: Optional[str] = None
    error: Optional[str] = None

class DryRunRequestPayload(BaseModel):
    config: PerCallConfig # To specify which project to run the dry run against
    query: str

class DryRunResponse(BaseModel):
    status: str # "SUCCESS" or "ERROR"
    total_bytes_processed: Optional[int] = None
    formatted_bytes_processed: Optional[str] = None
    error_message: Optional[str] = None
    job_id: Optional[str] = None

class RunQueryWithLimitRequestPayload(BaseModel):
    config: PerCallConfig
    query: str
    limit: int = Field(100, gt=0, le=1000) # Default limit, with bounds

class RunQueryResponse(BaseModel):
    rows: List[Dict[str, Any]]
    columns: List[str]
    error: Optional[str] = None
    query_executed: Optional[str] = None # The actual query run (with limit)

class AskLLMDirectlyRequestPayload(BaseModel):
    config: PerCallConfig
    generator_model_name: str # BQML model name in RAG dataset
    user_question: str
    existing_sql: Optional[str] = None
    llm_options: Optional[LLMOptionsPayload] = None

class AskLLMDirectlyResponse(BaseModel):
    generated_sql: Optional[str] = None
    prompt_used: Optional[str] = None
    error: Optional[str] = None
