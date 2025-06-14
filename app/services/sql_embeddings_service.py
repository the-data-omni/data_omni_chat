
import re
from fastapi import HTTPException, Depends, status 
from fastapi.security import OAuth2PasswordBearer
from google.cloud import bigquery
from google.cloud.exceptions import NotFound, Conflict
from google.oauth2 import credentials as google_oauth2_credentials
from typing import List, Dict, Any, Optional
import os 
import logging
from app.models.sql_embeddings_models import CreateRAGProcedureRequestPayload, LLMOptionsPayload # Example if CreateRAGProcedureRequest was still named that

# --- Application-wide Constants (not state) ---
TEXT_EMBEDDING_MODEL_NAME_BASE = "text_embedding_model" 
SCHEMA_CORPUS_TABLE_NAME_BASE = "schema_corpus_table"
SCHEMA_EMBEDDINGS_TABLE_NAME_BASE = "schema_embeddings_table"
QQ_CORPUS_TABLE_NAME_BASE = "qq_corpus_table"
QQ_EMBEDDINGS_TABLE_NAME_BASE = "qq_embeddings_table"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class BigQueryRAGService:
    def __init__(self,
                 client: bigquery.Client, 
                 project_id: str, 
                 dataset_id: str,
                 region: str, 
                 vertex_connection: str, 
                 embedding_model_endpoint_name: str): 
        self.client = client
        self.project_id = project_id
        self.dataset_id = dataset_id 
        self.region = region
        self.vertex_connection = vertex_connection
        self.embedding_model_endpoint_name = embedding_model_endpoint_name 
        
        self.text_embedding_model_name_base = TEXT_EMBEDDING_MODEL_NAME_BASE
        self.schema_corpus_table_name_base = SCHEMA_CORPUS_TABLE_NAME_BASE
        self.schema_embeddings_table_name_base = SCHEMA_EMBEDDINGS_TABLE_NAME_BASE
        self.qq_corpus_table_name_base = QQ_CORPUS_TABLE_NAME_BASE
        self.qq_embeddings_table_name_base = QQ_EMBEDDINGS_TABLE_NAME_BASE

    def get_rag_resource_fqn(self, resource_name_base: str) -> str:
        return f"{self.project_id}.{self.dataset_id}.{resource_name_base}"

    def get_text_embedding_model_fqn(self) -> str: 
        return self.get_rag_resource_fqn(self.text_embedding_model_name_base)

    def get_schema_corpus_table_fqn(self) -> str:
        return self.get_rag_resource_fqn(self.schema_corpus_table_name_base)

    def get_schema_embeddings_table_fqn(self) -> str:
        return self.get_rag_resource_fqn(self.schema_embeddings_table_name_base)

    def get_qq_corpus_table_fqn(self) -> str:
        return self.get_rag_resource_fqn(self.qq_corpus_table_name_base)

    def get_qq_embeddings_table_fqn(self) -> str:
        return self.get_rag_resource_fqn(self.qq_embeddings_table_name_base)

    # def execute_bq_query(self, query: str, job_config: Optional[bigquery.QueryJobConfig] = None):
    #     try:
    #         logging.info(f"Executing BQ Query (first 300 chars): {query[:300]} using project {self.client.project}")
    #         query_job = self.client.query(query, job_config=job_config)
    #         query_job.result() 
    #         logging.info(f"Query executed successfully: {query_job.job_id}")
    #         if query_job.errors:
    #             logging.error(f"Query job {query_job.job_id} had errors: {query_job.errors}")
    #             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"BigQuery job failed: {query_job.errors}")
    #         return query_job
    #     except Conflict: 
    #         logging.warning(f"Conflict during query execution (possibly resource already exists): {query[:100]}")
    #         return None
    #     except Exception as e:
    #         logging.error(f"Error executing BigQuery query: {e}")
    #         error_detail = f"BigQuery query execution failed: {e}"
    #         if hasattr(e, 'errors') and e.errors and isinstance(e.errors, list) and len(e.errors) > 0:
    #             bq_error = e.errors[0]
    #             error_detail = f"BigQuery query execution failed: {bq_error.get('message', str(e))} (Reason: {bq_error.get('reason')}, Location: {bq_error.get('location')})"
    #         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_detail)

    def execute_bq_query(self, query: str, job_config: Optional[bigquery.QueryJobConfig] = None, allow_results: bool = False):
        try:
            logging.info(f"Executing BQ Query (first 300 chars): {query[:300]} using project {self.client.project}")
            query_job = self.client.query(query, job_config=job_config)
            
            if allow_results:
                results = query_job.result() # Waits for the job to complete and fetches results
                if query_job.errors:
                    logging.error(f"Query job {query_job.job_id} had errors: {query_job.errors}")
                    job_id_info = f"Job ID: {query_job.job_id}" if hasattr(query_job, 'job_id') else "Job ID not available."
                    error_detail = f"BigQuery job failed: {query_job.errors}. {job_id_info}"
                    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_detail)
                logging.info(f"Query executed successfully with results: {query_job.job_id}")
                return results # Return iterator
            else: # DDL or statements not expected to return rows
                query_job.result() # Waits for the job to complete.
                logging.info(f"Query executed successfully: {query_job.job_id}")
                if query_job.errors:
                    logging.error(f"Query job {query_job.job_id} had errors: {query_job.errors}")
                    job_id_info = f"Job ID: {query_job.job_id}" if hasattr(query_job, 'job_id') else "Job ID not available."
                    error_detail = f"BigQuery job failed: {query_job.errors}. {job_id_info}"
                    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_detail)
                return query_job # Return job object
        except Conflict: 
            logging.warning(f"Conflict during query execution (possibly resource already exists): {query[:100]}")
            return None
        except Exception as e:
            logging.error(f"Error executing BigQuery query: {e}")
            error_detail = f"BigQuery query execution failed: {e}"
            job_id_info = "Job ID not available for this failure." 
            if 'query_job' in locals() and hasattr(query_job, 'job_id'):
                 job_id_info = f"Job ID: {query_job.job_id}"
            elif hasattr(e, 'job_id'): 
                 job_id_info = f"Job ID: {e.job_id}"

            if hasattr(e, 'errors') and e.errors and isinstance(e.errors, list) and len(e.errors) > 0:
                bq_error = e.errors[0]
                error_detail = f"BigQuery query execution failed: {bq_error.get('message', str(e))} (Reason: {bq_error.get('reason')}, Location: {bq_error.get('location')}). {job_id_info}"
            else:
                error_detail = f"BigQuery query execution failed: {str(e)}. {job_id_info}"
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_detail)

    def generate_bqml_embedding(self, document_text: str, embedding_model_fqn_override: Optional[str] = None) -> List[float] | None:
        model_fqn = embedding_model_fqn_override or self.get_text_embedding_model_fqn()
        
        query = f"""
        SELECT ml_generate_embedding_result
        FROM ML.GENERATE_EMBEDDING(
            MODEL `{model_fqn}`,
            (SELECT @doc_text AS content)
        );
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("doc_text", "STRING", document_text)]
        )
        try:
            logging.debug(f"Executing BQML Embedding Query (model: {model_fqn}) with param (first 100 chars): {document_text[:100]}...")
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()
            for row in results:
                if row[0] and isinstance(row[0], list): return [float(val) for val in row[0]]
            logging.warning(f"BQML embedding did not return expected list structure for model {model_fqn}.")
            return None
        except Exception as e:
            logging.error(f"Error generating BQML embedding with model {model_fqn}. Param: {document_text[:100]}... Error: {e}")
            error_detail = f"BigQuery ML embedding generation failed: {e}"
            if hasattr(e, 'errors') and e.errors and isinstance(e.errors, list) and len(e.errors) > 0:
                bq_error = e.errors[0]
                error_detail = f"BigQuery ML embedding failed: {bq_error.get('message', str(e))} (Location: {bq_error.get('location')}, Reason: {bq_error.get('reason')})"
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_detail)

    def store_document_and_embedding(
        self, document_id: str, document_text: str, embedding: List[float],
        corpus_table_fqn_str: str, embedding_table_fqn_str: str,
        additional_corpus_fields: Dict[str, Any] = None
    ):
        pk_column = "document_id" if self.schema_corpus_table_name_base in corpus_table_fqn_str else "qq_id"
        corpus_data = {pk_column: document_id}
        if self.schema_corpus_table_name_base in corpus_table_fqn_str: 
            corpus_data["table_name"] = additional_corpus_fields.get("table_name", "")
            corpus_data["document_text"] = document_text
        elif self.qq_corpus_table_name_base in corpus_table_fqn_str: 
            corpus_data["question_text"] = document_text
            corpus_data["sql_query_text"] = additional_corpus_fields.get("sql_query_text", "")
        
        corpus_fields = list(corpus_data.keys())
        corpus_placeholders = ", ".join([f"@{f}" for f in corpus_fields])
        corpus_update_setters = ", ".join([f"Target.{f} = Source.{f}" for f in corpus_fields if f != pk_column])

        merge_corpus_query = f"""
        MERGE `{corpus_table_fqn_str}` AS Target
        USING (SELECT {", ".join([f"@{f} AS {f}" for f in corpus_fields])}) AS Source
        ON Target.{pk_column} = Source.{pk_column}
        WHEN MATCHED THEN UPDATE SET {corpus_update_setters}
        WHEN NOT MATCHED THEN INSERT ({", ".join(corpus_fields)}) VALUES ({corpus_placeholders});
        """
        query_params_corpus = [bigquery.ScalarQueryParameter(name, "STRING", str(value)) for name, value in corpus_data.items()]
        self.execute_bq_query(merge_corpus_query, job_config=bigquery.QueryJobConfig(query_parameters=query_params_corpus))

        merge_embedding_query = f"""
        MERGE `{embedding_table_fqn_str}` AS Target
        USING (SELECT @{pk_column} AS {pk_column}, @embedding AS embedding) AS Source
        ON Target.{pk_column} = Source.{pk_column}
        WHEN MATCHED THEN UPDATE SET Target.embedding = Source.embedding
        WHEN NOT MATCHED THEN INSERT ({pk_column}, embedding) VALUES (@{pk_column}, @embedding);
        """
        query_params_embedding = [
            bigquery.ScalarQueryParameter(pk_column, "STRING", document_id),
            bigquery.ArrayQueryParameter("embedding", "FLOAT64", embedding)
        ]
        self.execute_bq_query(merge_embedding_query, job_config=bigquery.QueryJobConfig(query_parameters=query_params_embedding))

    def create_remote_model(self, model_name: str, vertex_ai_endpoint: str):
        model_fqn = f"{self.project_id}.{self.dataset_id}.{model_name}"
        query = f"""
        CREATE OR REPLACE MODEL `{model_fqn}`
        REMOTE WITH CONNECTION `{self.vertex_connection}`
        OPTIONS (ENDPOINT = '{vertex_ai_endpoint}');
        """
        logging.info(f"Attempting to create remote model: {query}")
        self.execute_bq_query(query)
        logging.info(f"Remote model '{model_fqn}' ensured, pointing to endpoint '{vertex_ai_endpoint}'.")
        return {"message": f"Remote model '{model_fqn}' created/re-created successfully.", "model_fqn": model_fqn}

    def create_vector_index(self, index_name: str, embeddings_table_name: str, embedding_column_name: str, distance_type: str, index_type: str, ivf_num_lists: Optional[int]):
        index_fqn = f"{self.project_id}.{self.dataset_id}.{index_name}"
        table_fqn = f"{self.project_id}.{self.dataset_id}.{embeddings_table_name}"
        
        options_list = [f"distance_type='{distance_type.upper()}'"]

        # Only add index_type if it's a specifically supported one like IVF or TREE_AH
        # If client sends "BRUTE_FORCE" or an empty/other string for index_type,
        # we omit explicitly setting index_type, allowing BigQuery to default (usually to brute-force).
        if index_type.upper() == "IVF":
            options_list.append("index_type='IVF'") 
            effective_num_lists = ivf_num_lists if ivf_num_lists is not None and ivf_num_lists > 0 else 100
            ivf_options_json_string = f'{{"num_lists": {effective_num_lists}}}'
            # Other IVF options like 'uses_brute_force' or 'probe_count' could be added here too if needed.
            options_list.append(f"ivf_options='{ivf_options_json_string}'")
        elif index_type.upper() == "TREE_AH":
            options_list.append("index_type='TREE_AH'")
            # Add TREE_AH specific options if any, e.g.,
            # options_list.append("tree_ah_options='{\"leaf_nodes_to_search_percent\": 10}'") 
            # For now, just setting the type.
        # For "BRUTE_FORCE" or other/empty index_type from client, no specific index_type option is added.
        
        options_string = ", ".join(options_list)
        
        # The OPTIONS() clause is always included, even if it only contains distance_type.
        query = f"""
        CREATE OR REPLACE VECTOR INDEX `{index_fqn}` 
        ON `{table_fqn}`({embedding_column_name})
        OPTIONS({options_string});
        """
        
        logging.info(f"Attempting to execute DDL for vector index: {query}")
        self.execute_bq_query(query)
        logging.info(f"Vector index '{index_fqn}' on table '{table_fqn}' column '{embedding_column_name}' ensured.")
        return {"message": f"Vector index '{index_fqn}' created/replaced successfully.", "index_fqn": index_fqn}

    def create_text_to_sql_rag_procedure(self, req: 'CreateRAGProcedureRequestPayload'):
        procedure_fqn = f"{self.project_id}.{self.dataset_id}.{req.procedure_name}"
        # Models are referenced by their fully qualified names within the procedure DDL
        embedding_model_ref_in_proc = f"`{self.project_id}.{self.dataset_id}.{req.embedding_model_name}`"
        generator_model_ref_in_proc = f"`{self.project_id}.{self.dataset_id}.{req.generator_model_name}`"
        
        schema_corpus_table_fqn = f"{self.project_id}.{self.dataset_id}.{req.schema_corpus_table_name}"
        schema_embeddings_table_fqn = f"{self.project_id}.{self.dataset_id}.{req.schema_embeddings_table_name}"
        qq_corpus_table_fqn = f"{self.project_id}.{self.dataset_id}.{req.qq_corpus_table_name}"
        qq_embeddings_table_fqn = f"{self.project_id}.{self.dataset_id}.{req.qq_embeddings_table_name}"

        llm_opts = req.llm_options if req.llm_options else LLMOptionsPayload() 
        opts_parts = []
        if llm_opts.temperature is not None: opts_parts.append(f"{llm_opts.temperature} AS temperature")
        if llm_opts.max_output_tokens is not None: opts_parts.append(f"{llm_opts.max_output_tokens} AS max_output_tokens")
        if llm_opts.top_p is not None: opts_parts.append(f"{llm_opts.top_p} AS top_p")
        if llm_opts.top_k is not None: opts_parts.append(f"{llm_opts.top_k} AS top_k")
        if llm_opts.flatten_json_output is not None: opts_parts.append(f"{str(llm_opts.flatten_json_output).upper()} AS flatten_json_output")
        llm_options_str = f"STRUCT({', '.join(opts_parts)})" if opts_parts else "STRUCT(0.2 AS temperature, 1024 AS max_output_tokens, TRUE AS flatten_json_output)"

        prompt_intro_part1 = 'You are an expert BigQuery SQL generation assistant. Given the following database schema context and some example question/query pairs, your task is to translate the user'
        prompt_intro_part2 = 's natural language question into an accurate and executable BigQuery SQL query. Only output the SQL query and nothing else.\\n\\n'
        
        procedure_ddl = f"""
        CREATE OR REPLACE PROCEDURE `{procedure_fqn}`(
            IN user_question STRING, 
            OUT generated_sql STRING,
            OUT o_retrieved_schema_info STRING,    -- New OUT parameter
            OUT o_retrieved_qq_examples STRING     -- New OUT parameter
        )
        BEGIN
            DECLARE user_question_embedding ARRAY<FLOAT64>;
            -- Removed DECLARE for retrieved_schema_info and retrieved_qq_examples
            -- as they will be directly assigned to OUT parameters.
            DECLARE final_prompt STRING;

            SET user_question_embedding = (
                SELECT ml_generate_embedding_result
                FROM ML.GENERATE_EMBEDDING(MODEL {embedding_model_ref_in_proc}, (SELECT user_question AS content))
            );

            SET o_retrieved_schema_info = (
                SELECT IFNULL(STRING_AGG(S_CORPUS.document_text, '\\n---\\n' ORDER BY VS_SCHEMA.distance LIMIT {req.schema_vector_top_k}), 'No relevant schema snippets found.')
                FROM 
                    VECTOR_SEARCH(
                        TABLE `{schema_embeddings_table_fqn}`, 
                        '{req.schema_embedding_column_name}',
                        (SELECT user_question_embedding AS query_embedding), 
                        top_k => {req.schema_vector_top_k},
                        distance_type => 'COSINE',
                        options => '{{"use_brute_force":true}}' 
                    ) AS VS_SCHEMA 
                JOIN `{schema_corpus_table_fqn}` S_CORPUS
                  ON VS_SCHEMA.base.document_id = S_CORPUS.document_id 
            ); 

            SET o_retrieved_qq_examples = (
                SELECT IFNULL(STRING_AGG(CONCAT('Question: ', QQ_CORPUS.question_text, '\\nSQL: ', QQ_CORPUS.sql_query_text), '\\n---\\n' ORDER BY VS_QQ.distance LIMIT {req.qq_vector_top_k}), 'No relevant Q/Q examples found.')
                FROM 
                    VECTOR_SEARCH(
                        TABLE `{qq_embeddings_table_fqn}`, 
                        '{req.qq_embedding_column_name}',
                        (SELECT user_question_embedding AS query_embedding), 
                        top_k => {req.qq_vector_top_k},
                        distance_type => 'COSINE',
                        options => '{{"use_brute_force":true}}'
                    ) AS VS_QQ
                JOIN `{qq_corpus_table_fqn}` QQ_CORPUS
                  ON VS_QQ.base.qq_id = QQ_CORPUS.qq_id 
            );

            SET final_prompt = CONCAT(
                '{prompt_intro_part1}', CHR(39), '{prompt_intro_part2}',
                'User Question:\\n', user_question,
                '\\n\\nRelevant Database Schema Information (table descriptions, column names, types, and JOIN instructions):\\n==============================\\n', o_retrieved_schema_info, -- Use OUT param directly
                '\\n==============================\\n\\nExample Question/SQL Pairs:\\n==============================\\n', o_retrieved_qq_examples, -- Use OUT param directly
                '\\n==============================\\n\\nGenerated BigQuery SQL Query (do not include any explanation or markdown formatting, just the SQL):'
            );

            SET generated_sql = (
                SELECT ml_generate_text_llm_result
                FROM ML.GENERATE_TEXT(MODEL {generator_model_ref_in_proc}, (SELECT final_prompt AS prompt), {llm_options_str})
            );
        END;
        """
        self.execute_bq_query(procedure_ddl)
        logging.info(f"RAG Procedure '{procedure_fqn}' created/replaced successfully.")
        return {"message": f"RAG Stored Procedure '{procedure_fqn}' created/replaced successfully.", "procedure_fqn": procedure_fqn}

    def execute_text_to_sql_rag_procedure(self, procedure_name: str, user_question: str) -> Dict[str, Any]:
        procedure_fqn = f"{self.project_id}.{self.dataset_id}.{procedure_name}"
        query = f"""
        DECLARE generated_sql_output STRING;
        DECLARE schema_info_output STRING;
        DECLARE qq_examples_output STRING;
        CALL `{procedure_fqn}`(@user_question_param, generated_sql_output, schema_info_output, qq_examples_output);
        SELECT generated_sql_output, schema_info_output, qq_examples_output;
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("user_question_param", "STRING", user_question)]
        )
        query_job = self.client.query(query, job_config=job_config) 
        results = query_job.result() 
        for row in results:
            return {
                "generated_sql": row[0],
                "retrieved_schema_info": row[1],
                "retrieved_qq_examples": row[2]
            } 
        return {"generated_sql": None, "retrieved_schema_info": None, "retrieved_qq_examples": None, "error": "Procedure call did not return results or failed."}

    def run_query_with_limit(self, sql_query: str, limit: int) -> Dict[str, Any]:
        # Basic check to see if LIMIT is already present (case-insensitive)
        # This is a simple check and might not cover all edge cases of existing LIMIT clauses.
        query_to_run = sql_query.strip()
        if query_to_run.endswith(";"):
            query_to_run = query_to_run[:-1]

        if not re.search(r'\bLIMIT\s+\d+\s*$', query_to_run, re.IGNORECASE):
            query_to_run = f"{query_to_run} LIMIT {limit}"
        
        logging.info(f"Executing limited query: {query_to_run}")
        try:
            results_iterator = self.execute_bq_query(query_to_run, allow_results=True)
            if results_iterator:
                rows = [dict(row) for row in results_iterator]
                columns = [field.name for field in results_iterator.schema] if results_iterator.schema else []
                return {"rows": rows, "columns": columns, "query_executed": query_to_run}
            else: # Should not happen if execute_bq_query raises on error for result-bearing queries
                return {"rows": [], "columns": [], "error": "Query execution returned no results object.", "query_executed": query_to_run}
        except Exception as e: # Catch exceptions from execute_bq_query
            logging.error(f"Error in run_query_with_limit: {e}")
            return {"rows": [], "columns": [], "error": str(e), "query_executed": query_to_run}


    def ask_llm_directly(self, generator_model_name: str, user_question: str, existing_sql: Optional[str], llm_options_payload: Optional[LLMOptionsPayload]) -> Dict[str, Any]:
        generator_model_fqn = f"{self.project_id}.{self.dataset_id}.{generator_model_name}"
        
        prompt_parts = [
            "You are an expert BigQuery SQL generation assistant.",
            f"The user's request is: '{user_question}'.",
        ]
        if existing_sql and existing_sql.strip():
            prompt_parts.append(f"Refine or use the following existing SQL query as context if relevant:\n```sql\n{existing_sql}\n```")
        else:
            prompt_parts.append("Generate a new BigQuery SQL query based on the user's request.")
        prompt_parts.append("Only output the SQL query and nothing else. Do not include any explanation or markdown formatting.")
        
        final_prompt = "\n\n".join(prompt_parts)

        llm_opts = llm_options_payload if llm_options_payload else LLMOptionsPayload()
        opts_parts = []
        if llm_opts.temperature is not None: opts_parts.append(f"{llm_opts.temperature} AS temperature")
        if llm_opts.max_output_tokens is not None: opts_parts.append(f"{llm_opts.max_output_tokens} AS max_output_tokens")
        # Add top_p, top_k if needed from llm_opts
        if llm_opts.flatten_json_output is not None: opts_parts.append(f"{str(llm_opts.flatten_json_output).upper()} AS flatten_json_output")
        llm_options_str = f"STRUCT({', '.join(opts_parts)})" if opts_parts else "STRUCT(0.2 AS temperature, 1024 AS max_output_tokens, TRUE AS flatten_json_output)"

        query = f"""
        SELECT ml_generate_text_llm_result
        FROM ML.GENERATE_TEXT(
            MODEL `{generator_model_fqn}`,
            (SELECT @prompt_text AS prompt),
            {llm_options_str}
        );
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("prompt_text", "STRING", final_prompt)]
        )
        try:
            results = self.execute_bq_query(query, job_config=job_config, allow_results=True)
            if results:
                for row in results:
                    return {"generated_sql": row[0], "prompt_used": final_prompt}
            return {"generated_sql": None, "prompt_used": final_prompt, "error": "LLM did not return text."}
        except Exception as e:
             return {"generated_sql": None, "prompt_used": final_prompt, "error": str(e)}



async def get_user_credentials(token: str = Depends(oauth2_scheme)) -> google_oauth2_credentials.Credentials:
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No authorization token provided", headers={"WWW-Authenticate": "Bearer"})
    try:
        return google_oauth2_credentials.Credentials(token=token)
    except Exception as e:
        logging.error(f"Failed to create credentials from token: {e}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid authentication credentials: {e}", headers={"WWW-Authenticate": "Bearer"})
