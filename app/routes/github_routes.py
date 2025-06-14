from fastapi import APIRouter, Body, Depends, HTTPException, status
import logging

# If these were actual separate files, you'd import like this:
from app.models.github_models import (
    SaveSchemaToGitRequest, LoadSchemaFromGitResponse, GitOperationResponse,
    CreateGitHubRepoRequest, CreateGitHubRepoResponse,
    SaveQQToGitRequest, LoadQQFromGitResponse, QuestionQueryPairItem, SchemaItem,
    GitRepoTargetConfig, LoadSchemaFromGitRequest, LoadQQFromGitRequest
)
from app.services.github_service import GitHubService, get_github_service_dependency

github_schema_router = APIRouter(prefix="/git/schema", tags=["GitHub Schema Management"])
github_qq_router = APIRouter(prefix="/git/qq", tags=["GitHub Q/Q Pair Management"]) 
github_repo_router = APIRouter(prefix="/git/repository", tags=["GitHub Repository Management"])


@github_schema_router.post("/save", response_model=GitOperationResponse)
async def save_schema_to_github_endpoint(
    request_data: SaveSchemaToGitRequest = Body(...),
    github_service: GitHubService = Depends(get_github_service_dependency) 
):
    try:
        # Log received data summary instead of full data
        logging.info(f"Received request to save schema data to Git. Target file: {request_data.target_file_name}, Items: {len(request_data.schema_data)}")
        return github_service.save_schema_data_to_repo(
            git_config=request_data.git_config,
            target_file_name=request_data.target_file_name,
            schema_items=request_data.schema_data,
            commit_message=request_data.commit_message
        )
    except RuntimeError as e: 
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except HTTPException as e: 
        raise e
    except Exception as e:
        logging.error(f"Unexpected error in save_schema_to_github_endpoint: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {str(e)}")

@github_schema_router.post("/load", response_model=LoadSchemaFromGitResponse) 
async def load_schema_from_github_endpoint(
    request_data: LoadSchemaFromGitRequest = Body(...), 
    github_service: GitHubService = Depends(get_github_service_dependency)
):
    try:
        logging.info(f"Received request to load schema data from Git. Target file: {request_data.target_file_name}")
        response = github_service.load_schema_data_from_repo(
            git_config=request_data.git_config, 
            target_file_name=request_data.target_file_name
        )
        logging.info(f"Returning response: {response}")
        return LoadSchemaFromGitResponse(**response)  # <-- This enforces schema
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Unexpected error in load_schema_from_github_endpoint: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {str(e)}")



@github_qq_router.post("/save", response_model=GitOperationResponse)
async def save_qq_to_github_endpoint(
    request_data: SaveQQToGitRequest = Body(...),
    github_service: GitHubService = Depends(get_github_service_dependency)
):
    try:
        logging.info(f"Received request to save Q/Q data to Git. Target file: {request_data.target_file_name}, Items: {len(request_data.qq_data)}")
        return github_service.save_qq_data_to_repo(
            git_config=request_data.git_config,
            target_file_name=request_data.target_file_name,
            qq_items=request_data.qq_data,
            commit_message=request_data.commit_message
        )
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Unexpected error in save_qq_to_github_endpoint: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {str(e)}")

@github_qq_router.post("/load", response_model=LoadQQFromGitResponse)
async def load_qq_from_github_endpoint(
    request_data: LoadQQFromGitRequest = Body(...),
    github_service: GitHubService = Depends(get_github_service_dependency)
):
    try:
        logging.info(f"Received request to load Q/Q data from Git. Target file: {request_data.target_file_name}")

        # The github_service.load_qq_data_from_repo method should return a dictionary
        # whose keys match the fields of the LoadQQFromGitResponse Pydantic model.
        # For example: {"success": True, "qq_data": [...], "message": "Loaded successfully"}
        # or {"success": False, "error": "File not found"}
        response_dict = github_service.load_qq_data_from_repo(
            git_config=request_data.git_config,
            target_file_name=request_data.target_file_name
        )

        logging.info(f"Service call to load_qq_data_from_repo completed. Preparing response.")
        # Pass the dictionary to the Pydantic model for validation and structuring
        return LoadQQFromGitResponse(**response_dict)

    except RuntimeError as e:
        # Specific internal errors from the service layer
        logging.error(f"RuntimeError in load_qq_from_github_endpoint: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except HTTPException as e:
        # Re-raise HTTPExceptions if they are intentionally thrown (e.g., 404 Not Found from service)
        raise e
    except Exception as e:
        # Catch-all for other unexpected errors
        logging.error(f"Unexpected error in load_qq_from_github_endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred while loading Q/Q data.")


@github_repo_router.post("/create", response_model=CreateGitHubRepoResponse)
async def create_github_repository_endpoint(
    request_data: CreateGitHubRepoRequest = Body(...),
    github_service: GitHubService = Depends(get_github_service_dependency) 
):
    try:
        return await github_service.create_repository_on_github(
            user_provided_pat=request_data.github_pat, 
            repo_name=request_data.name,
            description=request_data.description,
            private=request_data.private
        )
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except HTTPException as e: 
        raise e
    except Exception as e:
        logging.error(f"Unexpected error in create_github_repository_endpoint: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred while creating repository: {str(e)}")
