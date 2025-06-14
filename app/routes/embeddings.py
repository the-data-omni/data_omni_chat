
from fastapi import APIRouter, Body, Depends, HTTPException, status
from google.cloud import bigquery 
from google.oauth2 import credentials as google_oauth2_credentials 
from google.cloud.exceptions import BadRequest
import re
import uuid 
import logging

# If these were actual separate files, you'd import like this:
from app.models.sql_embeddings_models import (
   CreateRemoteModelRequestPayload, PerCallConfig, InitializeResourcesRequestPayload, SchemaInputPayload, QuestionQueryInputPayload,
   CreateVectorIndexRequestPayload, CreateRAGProcedureRequestPayload, ExecuteRAGProcedureRequestPayload, ExecuteRAGProcedureResponse, LLMOptionsPayload,
   DryRunRequestPayload, DryRunResponse, RunQueryWithLimitRequestPayload, RunQueryResponse, AskLLMDirectlyRequestPayload, AskLLMDirectlyResponse
)
from app.services.sql_embeddings_service import (
    BigQueryRAGService, get_user_credentials 
)

rag_setup_router = APIRouter( tags=["RAG Setup & Management (Stateless)"])
rag_execution_router = APIRouter(tags=["RAG Execution (Stateless)"])
rag_embeddings_router = APIRouter( tags=["RAG Embeddings Management (Stateless)"])
rag_llm_router = APIRouter( tags=["RAG Direct LLM Interaction"]) # New router for direct LLM


def _get_service_instance(config: PerCallConfig, creds: google_oauth2_credentials.Credentials) -> BigQueryRAGService:
    try:
        client = bigquery.Client(project=config.gcp_project_id, credentials=creds)
    except Exception as e:
        logging.error(f"Failed to create BigQuery client for project {config.gcp_project_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not initialize BigQuery client: {e}")
    
    vertex_connection = config.vertex_connection_name
    if '.' not in vertex_connection and config.region: # Auto-qualify if not already
         vertex_connection = f"{config.region.lower()}.{vertex_connection.lstrip('.')}"

    return BigQueryRAGService(
        client=client,
        project_id=config.gcp_project_id,
        dataset_id=config.dataset_id,
        region=config.region,
        vertex_connection=vertex_connection,
        embedding_model_endpoint_name=config.embedding_model_endpoint
    )

@rag_setup_router.post("/initialize-resources", summary="Create Core RAG BQ Resources")
async def initialize_rag_bigquery_resources_endpoint(
    request_data: InitializeResourcesRequestPayload = Body(...),
    creds: google_oauth2_credentials.Credentials = Depends(get_user_credentials)
):
    rag_service = _get_service_instance(request_data.config, creds)
    messages = []
    
    dataset_fqn = f"{rag_service.project_id}.{rag_service.dataset_id}"
    dataset_obj = bigquery.Dataset(dataset_fqn)
    dataset_obj.location = rag_service.region
    try:
        rag_service.client.create_dataset(dataset_obj, exists_ok=True)
        messages.append(f"Dataset '{dataset_fqn}' ensured.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create dataset '{dataset_fqn}': {e}")

    embedding_model_fqn = rag_service.get_text_embedding_model_fqn() 
    model_query_embed = f"""
    CREATE OR REPLACE MODEL `{embedding_model_fqn}`
    REMOTE WITH CONNECTION `{rag_service.vertex_connection}` 
    OPTIONS(ENDPOINT = '{rag_service.embedding_model_endpoint_name}');
    """ 
    rag_service.execute_bq_query(model_query_embed)
    messages.append(f"Default Embedding model '{embedding_model_fqn}' ensured using endpoint '{rag_service.embedding_model_endpoint_name}'.")
    
    # Example: Create a default generator model (user needs to ensure 'gemini-1.0-pro' is a valid endpoint for their connection)
    # You might want to make the generator model endpoint configurable too via InitializeResourcesRequestPayload
    default_generator_model_name = "gemini_pro_text_model" # Or make this configurable
    generator_model_fqn = f"{rag_service.project_id}.{rag_service.dataset_id}.{default_generator_model_name}"
    model_query_gen = f"""
    CREATE OR REPLACE MODEL `{generator_model_fqn}`
    REMOTE WITH CONNECTION `{rag_service.vertex_connection}`
    OPTIONS (ENDPOINT = 'gemini-2.0-flash'); 
    """ # Assuming gemini-1.0-pro, could be another like gemini-pro-vision etc.
    rag_service.execute_bq_query(model_query_gen)
    messages.append(f"Default Generator model '{generator_model_fqn}' ensured (using gemini-1.0-pro endpoint).")


    table_definitions = [
        (rag_service.get_schema_corpus_table_fqn(), f"""CREATE TABLE IF NOT EXISTS `{rag_service.get_schema_corpus_table_fqn()}` (document_id STRING NOT NULL, table_name STRING, document_text STRING) OPTIONS (description="Stores RAG textual descriptions of database schemas.");"""),
        (rag_service.get_schema_embeddings_table_fqn(), f"""CREATE TABLE IF NOT EXISTS `{rag_service.get_schema_embeddings_table_fqn()}` (document_id STRING NOT NULL, embedding ARRAY<FLOAT64>) OPTIONS (description="Stores RAG embeddings for schema documents.");"""),
        (rag_service.get_qq_corpus_table_fqn(), f"""CREATE TABLE IF NOT EXISTS `{rag_service.get_qq_corpus_table_fqn()}` (qq_id STRING NOT NULL, question_text STRING, sql_query_text STRING) OPTIONS (description="Stores RAG example questions and their corresponding SQL queries.");"""),
        (rag_service.get_qq_embeddings_table_fqn(), f"""CREATE TABLE IF NOT EXISTS `{rag_service.get_qq_embeddings_table_fqn()}` (qq_id STRING NOT NULL, embedding ARRAY<FLOAT64>) OPTIONS (description="Stores RAG embeddings for example questions.");""")
    ]
    for table_fqn, query in table_definitions:
        rag_service.execute_bq_query(query)
        messages.append(f"Table '{table_fqn}' ensured.")
    return {"status": "RAG BigQuery core resources initialization process completed.", "details": messages, "config_used": request_data.config.model_dump()}

@rag_setup_router.post("/remote-model", summary="Create or Replace a BQML Remote Model")
async def create_remote_model_endpoint(
    request_data: CreateRemoteModelRequestPayload = Body(...),
    creds: google_oauth2_credentials.Credentials = Depends(get_user_credentials)
):
    rag_service = _get_service_instance(request_data.config, creds)
    return rag_service.create_remote_model(
        model_name=request_data.model_name,
        vertex_ai_endpoint=request_data.vertex_ai_endpoint
    )

@rag_setup_router.post("/vector-index", summary="Create or Replace a Vector Index")
async def create_vector_index_endpoint(
    request_data: CreateVectorIndexRequestPayload = Body(...),
    creds: google_oauth2_credentials.Credentials = Depends(get_user_credentials)
):
    rag_service = _get_service_instance(request_data.config, creds)
    return rag_service.create_vector_index(
        index_name=request_data.index_name,
        embeddings_table_name=request_data.embeddings_table_name, 
        embedding_column_name=request_data.embedding_column_name,
        distance_type=request_data.distance_type,
        index_type=request_data.index_type,
        ivf_num_lists=request_data.ivf_num_lists
    )

@rag_setup_router.post("/rag-procedure", summary="Create or Replace the Text-to-SQL RAG Stored Procedure")
async def create_rag_procedure_endpoint(
    request_data: CreateRAGProcedureRequestPayload = Body(...),
    creds: google_oauth2_credentials.Credentials = Depends(get_user_credentials)
):
    rag_service = _get_service_instance(request_data.config, creds)
    return rag_service.create_text_to_sql_rag_procedure(request_data)


@rag_embeddings_router.post("/schema", status_code=status.HTTP_201_CREATED)
async def add_or_update_rag_schema_embedding_endpoint(
    request_data: SchemaInputPayload = Body(...), 
    creds: google_oauth2_credentials.Credentials = Depends(get_user_credentials)
):
    rag_service = _get_service_instance(request_data.config, creds)
    doc_id = request_data.document_id or str(uuid.uuid4())
    field_texts = [f"- Field Name: {f.field_name}, Field Type: {f.field_type}, Field Description: {f.field_description}" for f in request_data.fields]
    # Join relationships are now directly from payload
    join_texts = [f"- {join}" for join in request_data.join_relationships or []] 
    
    document_text = f"""Dataset Name: {request_data.dataset_name}, Table Name: {request_data.table_name}
Table Description: {request_data.table_description or 'N/A'}
Field Details:\n{chr(10).join(field_texts)}
Integrated Join Relationships:\n{chr(10).join(join_texts) if join_texts else 'N/A'}"""

    # The embedding model used here is the one specified in the PerCallConfig for the RAG dataset
    embedding_model_fqn_for_rag_dataset = f"{request_data.config.gcp_project_id}.{request_data.config.dataset_id}.{request_data.config.embedding_model_endpoint}"
    embedding = rag_service.generate_bqml_embedding(document_text, embedding_model_fqn_override=embedding_model_fqn_for_rag_dataset)
    
    if not embedding:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate schema embedding.")
    
    rag_service.store_document_and_embedding(
        doc_id, document_text, embedding,
        rag_service.get_schema_corpus_table_fqn(), 
        rag_service.get_schema_embeddings_table_fqn(),
        additional_corpus_fields={"table_name": request_data.table_name}
    )
    return {"message": "RAG Schema document and embedding processed.", "document_id": doc_id, "embedding_preview": embedding[:5], "config_used": request_data.config.model_dump()}

@rag_embeddings_router.post("/question-query", status_code=status.HTTP_201_CREATED)
async def add_or_update_rag_question_query_embedding_endpoint(
    request_data: QuestionQueryInputPayload = Body(...), 
    creds: google_oauth2_credentials.Credentials = Depends(get_user_credentials)
):
    rag_service = _get_service_instance(request_data.config, creds)
    qq_id = request_data.qq_id or str(uuid.uuid4())
    question_text_for_embedding = f"Question: {request_data.question_text}" 
    
    embedding_model_fqn_for_rag_dataset = f"{request_data.config.gcp_project_id}.{request_data.config.dataset_id}.{request_data.config.embedding_model_endpoint}"
    embedding = rag_service.generate_bqml_embedding(question_text_for_embedding, embedding_model_fqn_override=embedding_model_fqn_for_rag_dataset)

    if not embedding:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate question embedding.")
    
    rag_service.store_document_and_embedding(
        qq_id, request_data.question_text, embedding, 
        rag_service.get_qq_corpus_table_fqn(), 
        rag_service.get_qq_embeddings_table_fqn(),
        additional_corpus_fields={"sql_query_text": request_data.sql_query_text}
    )
    return {"message": "RAG Question-query pair and embedding processed.", "qq_id": qq_id, "embedding_preview": embedding[:5], "config_used": request_data.config.model_dump()}

@rag_execution_router.post("/generate-sql", response_model=ExecuteRAGProcedureResponse, summary="Generate SQL using the RAG Stored Procedure")
async def execute_rag_procedure_endpoint(
    request_data: ExecuteRAGProcedureRequestPayload = Body(...),
    creds: google_oauth2_credentials.Credentials = Depends(get_user_credentials)
):
    rag_service = _get_service_instance(request_data.config, creds)
    try:
        result = rag_service.execute_text_to_sql_rag_procedure(
            procedure_name=request_data.procedure_name, 
            user_question=request_data.user_question
        )
        return result
    except HTTPException as http_exc: 
        raise http_exc
    except Exception as e:
        logging.error(f"Error executing RAG procedure '{request_data.procedure_name}': {e}")
        return ExecuteRAGProcedureResponse(error=f"Failed to execute RAG procedure: {e}")
    
@rag_execution_router.post("/run-query-with-limit", response_model=RunQueryResponse, summary="Execute a BigQuery SQL query with a limit")
async def run_query_with_limit_endpoint(
    request_data: RunQueryWithLimitRequestPayload = Body(...),
    creds: google_oauth2_credentials.Credentials = Depends(get_user_credentials)
):
    rag_service = _get_service_instance(request_data.config, creds)
    try:
        return rag_service.run_query_with_limit(
            sql_query=request_data.query,
            limit=request_data.limit
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logging.error(f"Error running query with limit: {e}")
        return RunQueryResponse(rows=[], columns=[], error=str(e), query_executed=request_data.query)

@rag_llm_router.post("/ask-directly", response_model=AskLLMDirectlyResponse, summary="Ask LLM directly to generate/refine SQL")
async def ask_llm_directly_endpoint(
    request_data: AskLLMDirectlyRequestPayload = Body(...),
    creds: google_oauth2_credentials.Credentials = Depends(get_user_credentials)
):
    rag_service = _get_service_instance(request_data.config, creds)
    try:
        return rag_service.ask_llm_directly(
            generator_model_name=request_data.generator_model_name,
            user_question=request_data.user_question,
            existing_sql=request_data.existing_sql,
            llm_options_payload=request_data.llm_options
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logging.error(f"Error asking LLM directly: {e}")
        return AskLLMDirectlyResponse(error=str(e))
        
@rag_execution_router.post("/dry_run", response_model=DryRunResponse, summary="Validate a BigQuery SQL query (Dry Run)")
async def dry_run_query_endpoint(
    request_data: DryRunRequestPayload = Body(...),
    creds: google_oauth2_credentials.Credentials = Depends(get_user_credentials)
):
    try:
        client = bigquery.Client(project=request_data.config.gcp_project_id, credentials=creds)
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        query_job = client.query(request_data.query, job_config=job_config)  # Dry run
        
        # If dry run is successful, there are no errors.
        # total_bytes_processed will be available on the job object.
        return DryRunResponse(
            status="SUCCESS",
            total_bytes_processed=query_job.total_bytes_processed,
            formatted_bytes_processed=f"{query_job.total_bytes_processed / (1024*1024):.2f} MB" if query_job.total_bytes_processed else "0 MB",
            job_id=query_job.job_id
        )
    except BadRequest as e: # Catch specific BQ errors for syntax issues
        logging.error(f"Dry run failed for query: {request_data.query[:100]}... Error: {e}")
        return DryRunResponse(status="ERROR", error_message=str(e), job_id=e.job_id if hasattr(e, 'job_id') else None)
    except Exception as e:
        logging.error(f"Unexpected error during dry run: {e}")
        return DryRunResponse(status="ERROR", error_message=f"Unexpected error: {str(e)}")