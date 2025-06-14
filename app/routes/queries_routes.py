# # app/routes/queries_routes.py
# from fastapi import APIRouter, Depends, Query
# from typing import List
# import time
# from app.services.bigquery_service import BigQueryService
# # Assuming your dependencies are set up as previously discussed
# from app.routes.bigquery_routes import get_bigquery_service 
# from app.services.openai_service import generate_natural_language_question
# from app.models.query_models import QueryWithStatsItem, QuestionQueryWithStatsItem, QueryStats

# router = APIRouter()

# @router.get("/queries", response_model=List[QueryWithStatsItem], tags=["Queries"])
# async def get_queries_route(
#     time_interval: str = Query("90 day", description="Time interval, e.g., '90 day', '1 month'"),
#     bq_service: BigQueryService = Depends(get_bigquery_service)
# ):
#     """
#     Route to get queries and their execution statistics.
#     """
#     query_stats_dict = bq_service.get_queries(time_interval) # Assumes get_queries is now async or run in threadpool
    
#     response_list = []
#     for query_text, stats_data in query_stats_dict.items():
#         response_list.append(QuestionQueryWithStatsItem(query=query_text,question="question", stats=QueryStats(**stats_data)))
#     return response_list

# @router.get("/questions", response_model=List[QuestionQueryWithStatsItem], tags=["Queries"])
# async def get_questions_route(
#     time_interval: str = Query("90 day", description="Time interval, e.g., '90 day', '1 month'"),
#     bq_service: BigQueryService = Depends(get_bigquery_service)
# ):
#     """
#     Route to get query corresponding natural language questions and stats.
#     """
#     query_stats_dict = bq_service.get_queries(time_interval)
#     response_list = []
#     for query_text, stats_data in query_stats_dict.items():
#         # Assuming generate_natural_language_question is synchronous, FastAPI runs it in a threadpool
#         question = generate_natural_language_question(query_text) 
#         response_list.append(QuestionQueryWithStatsItem(
#             question=question, 
#             query=query_text, 
#             stats=QueryStats(**stats_data)
#         ))
#     return response_list

# @router.get("/queries_and_questions", response_model=List[QuestionQueryWithStatsItem], tags=["Queries"])
# async def get_queries_and_questions_route(
#     time_interval: str = Query("90 day", description="Time interval, e.g., '90 day', '1 month'"),
#     bq_service: BigQueryService = Depends(get_bigquery_service)
# ):
#     """
#     Route to get question-query pairs along with their execution statistics.
#     (This is functionally identical to /questions, just a different endpoint name)
#     """
#     query_stats_dict = bq_service.get_queries(time_interval)
#     response_list = []
#     for query_text, stats_data in query_stats_dict.items():
#         print(f"Attempting to generate question for query: {query_text[:100]}...")
#         start_time = time.time()
        
#         question = generate_natural_language_question(query_text) 
        
#         end_time = time.time()
#         print(f"Generated question in {end_time - start_time:.2f} seconds for query: {query_text[:100]}")
#         print(f"Question: {question}")

#         response_list.append(QuestionQueryWithStatsItem(
#             question=question, 
#             query=query_text, 
#             stats=QueryStats(**stats_data)
#         ))
#     return response_list
# app/routes/queries_routes.py
from fastapi import APIRouter, Depends, Query
from typing import List
import time

from app.services.bigquery_service import BigQueryService
# MODIFIED: Import the dependency that reads project_id from the path
from app.routes.bigquery_routes import get_project_service 
from app.services.openai_service import generate_natural_language_question
from app.models.query_models import QueryWithStatsItem, QuestionQueryWithStatsItem, QueryStats

router = APIRouter()

# NOTE: The get_project_service dependency from bigquery_routes.py should look like this:
#
# from fastapi import Path, Depends
# from google.auth.credentials import Credentials as AuthCredentials
# from .auth_routes import get_bq_user_credentials # Or wherever your auth dependency is
# from app.services.bigquery_service import BigQueryService
#
# def get_project_service(
#     project_id: str = Path(..., title="Project ID", description="The Google Cloud Project ID."),
#     creds: AuthCredentials = Depends(get_bq_user_credentials)
# ) -> BigQueryService:
#     """
#     A dependency that creates a BigQueryService with the project_id from the path.
#     """
#     return BigQueryService(credentials=creds, project_id=project_id)
#


# MODIFIED: Added /{project_id} to the route and updated the dependency
@router.get("/{project_id}/queries", response_model=List[QueryWithStatsItem], tags=["Queries"])
async def get_queries_route(
    time_interval: str = Query("90 day", description="Time interval, e.g., '90 day', '1 month'"),
    bq_service: BigQueryService = Depends(get_project_service) # Use the project-aware dependency
):
    """
    Route to get queries and their execution statistics for a specific project.
    """
    # The bq_service is now correctly initialized with the project_id from the URL
    query_stats_dict = bq_service.get_queries(time_interval)
    
    response_list = []
    for query_text, stats_data in query_stats_dict.items():
        response_list.append(QuestionQueryWithStatsItem(query=query_text,question="question", stats=QueryStats(**stats_data)))
    return response_list

# MODIFIED: Added /{project_id} to the route and updated the dependency
@router.get("/{project_id}/questions", response_model=List[QuestionQueryWithStatsItem], tags=["Queries"])
async def get_questions_route(
    time_interval: str = Query("90 day", description="Time interval, e.g., '90 day', '1 month'"),
    bq_service: BigQueryService = Depends(get_project_service) # Use the project-aware dependency
):
    """
    Route to get query corresponding natural language questions and stats for a specific project.
    """
    query_stats_dict = bq_service.get_queries(time_interval)
    response_list = []
    for query_text, stats_data in query_stats_dict.items():
        question = generate_natural_language_question(query_text) 
        response_list.append(QuestionQueryWithStatsItem(
            question=question, 
            query=query_text, 
            stats=QueryStats(**stats_data)
        ))
    return response_list

# MODIFIED: Added /{project_id} to the route and updated the dependency
@router.get("/{project_id}/queries_and_questions", response_model=List[QuestionQueryWithStatsItem], tags=["Queries"])
async def get_queries_and_questions_route(
    time_interval: str = Query("90 day", description="Time interval, e.g., '90 day', '1 month'"),
    bq_service: BigQueryService = Depends(get_project_service) # Use the project-aware dependency
):
    """
    Route to get question-query pairs and their stats for a specific project.
    """
    query_stats_dict = bq_service.get_queries(time_interval)
    response_list = []
    for query_text, stats_data in query_stats_dict.items():
        print(f"Attempting to generate question for query: {query_text[:100]}...")
        start_time = time.time()
        
        question = generate_natural_language_question(query_text) 
        
        end_time = time.time()
        print(f"Generated question in {end_time - start_time:.2f} seconds for query: {query_text[:100]}")
        print(f"Question: {question}")

        response_list.append(QuestionQueryWithStatsItem(
            question=question, 
            query=query_text, 
            stats=QueryStats(**stats_data)
        ))
    return response_list