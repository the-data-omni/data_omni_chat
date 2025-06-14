# from fastapi import APIRouter, Depends, HTTPException, status
# from fastapi.responses import JSONResponse
# from fastapi.security import OAuth2PasswordBearer
# from typing import Dict, List, Any
# from app.models.bigquery_models import (
#     BigQuerySchemaEntry,
#     UpdateDescriptionsRequest,
#     UpdateDescriptionsResponse,
#     TableDefinition
# )
# from app.services.bigquery_service import BigQueryService
# from google.api_core.exceptions import BadRequest, GoogleAPICallError

# # --- Google / BigQuery Imports ---
# from google.oauth2 import credentials
# from google.auth.credentials import Credentials as AuthCredentials
# from google.api_core.exceptions import BadRequest, GoogleAPICallError
# import logging

# # This scheme expects an "Authorization: Bearer <token>" header in the request
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token") # tokenUrl is not used, but required


# router = APIRouter()

# # 1) Load the JSON from file into .scraped_descriptions
# # bq_service.load_scraped_descriptions_from_file("scraped_fivetran_descriptions.json")

# def get_bq_user_credentials(token: str = Depends(oauth2_scheme)) -> AuthCredentials:
#     """
#     FastAPI dependency that takes the bearer token from the request header
#     and creates a Google Auth Credentials object from it.
#     """
#     if not token:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="No authorization token provided",
#         )
#     try:
#         user_creds = credentials.Credentials(token=token)
#         return user_creds
#     except Exception as e:
#         logging.error(f"Failed to create credentials from token: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Invalid authentication credentials",
#         )

# def get_bigquery_service(creds: AuthCredentials = Depends(get_bq_user_credentials)) -> BigQueryService:
#     """
#     FastAPI dependency that creates an instance of BigQueryService
#     using the user's credentials obtained from the token for each request.
#     """
#     target_project_id = "foreign-connect-48db5"
#     # Note: The logic for `load_descriptions` from the old __init__ would go here
#     # if it's needed on a per-request basis.
#     return BigQueryService(credentials=creds, project_id=target_project_id, load_descriptions=False)

# @router.get("/bigquery_info", response_model=List[BigQuerySchemaEntry])
# def bigquery_info(bq_service: BigQueryService = Depends(get_bigquery_service)):
#     """
#     Gets BigQuery info using the authenticated user's credentials.
#     The bq_service instance is created per-request.
#     """
#     try:
#         project_info = bq_service.get_bigquery_info()
#         flattened_schema = bq_service.flatten_bq_schema(project_info)
#         return JSONResponse(content={"schema": flattened_schema})
#         # return flattened_schema
#     except Exception as exc:
#         raise HTTPException(status_code=500, detail=str(exc))


# @router.post("/update_descriptions", response_model=UpdateDescriptionsResponse)
# def update_descriptions(
#     request_data: UpdateDescriptionsRequest,
#     bq_service: BigQueryService = Depends(get_bigquery_service)
# ):
#     """
#     Updates field descriptions using the authenticated user's credentials.
#     The bq_service instance is injected by FastAPI for this specific request.
#     """
#     # The internal logic remains the same, but now uses the user-specific bq_service
#     message, status_code = bq_service.update_field_descriptions(request_data.dict())

#     if status_code == 200:
#         # Assuming UpdateDescriptionsResponse has a "message" field,
#         # you can return a dictionary and FastAPI handles the rest.
#         return {"message": message}
#     else:
#         # If the service method indicates an error, raise a proper HTTPException
#         raise HTTPException(status_code=status_code, detail=message)

# @router.post("/build_update_payload_for_tables")
# def build_update_payload_for_tables_route(
#     table_defs: List[TableDefinition],
#     bq_service: BigQueryService = Depends(get_bigquery_service)
# ):
#     """
#     Builds an update payload using the injected, user-authenticated bq_service.
#     """
#     try:
#         table_defs_dicts = [td.dict() for td in table_defs]

#         result_payload = bq_service.build_update_payload_for_tables(
#             table_defs_dicts, default_project=None
#         )

#         # It's more idiomatic in FastAPI to return the dict/list directly.
#         # FastAPI will automatically convert it to a JSON response with a 200 status code.
#         return result_payload
#     except ValueError as ve:
#         raise HTTPException(status_code=400, detail=str(ve))
#     except Exception as exc:
#         raise HTTPException(status_code=500, detail=str(exc))
from fastapi import APIRouter, Depends, HTTPException, status, Path, Query
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from typing import Dict, List, Any
from app.models.bigquery_models import (
    BigQuerySchemaEntry,
    UpdateDescriptionsRequest,
    UpdateDescriptionsResponse,
    TableDefinition
)
from app.services.bigquery_service import BigQueryService
from google.api_core.exceptions import BadRequest, GoogleAPICallError

# --- Google / BigQuery Imports ---
from google.oauth2 import credentials
from google.auth.credentials import Credentials as AuthCredentials
from google.api_core.exceptions import BadRequest, GoogleAPICallError
import logging

# This scheme expects an "Authorization: Bearer <token>" header in the request
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token") # tokenUrl is not used, but required

router = APIRouter()

def get_bq_user_credentials(token: str = Depends(oauth2_scheme)) -> AuthCredentials:
    """
    FastAPI dependency that takes the bearer token from the request header
    and creates a Google Auth Credentials object from it.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No authorization token provided",
        )
    try:
        user_creds = credentials.Credentials(token=token)
        return user_creds
    except Exception as e:
        logging.error(f"Failed to create credentials from token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )

# MODIFIED: This function now takes project_id as an argument
def get_bigquery_service(
    project_id: str, # The project_id is now a required parameter
    creds: AuthCredentials = Depends(get_bq_user_credentials)
) -> BigQueryService:
    """
    FastAPI dependency that creates an instance of BigQueryService
    using the user's credentials and the provided project_id.
    """
    return BigQueryService(credentials=creds, project_id=project_id, load_descriptions=False)

# This is a new dependency to make project_id reusable
def get_project_service(
    project_id: str = Path(..., title="Project ID", description="The Google Cloud Project ID."),
    creds: AuthCredentials = Depends(get_bq_user_credentials)
) -> BigQueryService:
    """
    A dependency that creates a BigQueryService with the project_id from the path.
    """
    return get_bigquery_service(project_id=project_id, creds=creds)


@router.get("/{project_id}/bigquery_info", response_model=List[BigQuerySchemaEntry])
def bigquery_info(
    bq_service: BigQueryService = Depends(get_project_service)
):
    """
    Gets BigQuery info for a specific project using the authenticated user's credentials.
    """
    try:
        project_info = bq_service.get_bigquery_info()
        flattened_schema = bq_service.flatten_bq_schema(project_info)
        return JSONResponse(content={"schema": flattened_schema})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/{project_id}/update_descriptions", response_model=UpdateDescriptionsResponse)
def update_descriptions(
    request_data: UpdateDescriptionsRequest,
    bq_service: BigQueryService = Depends(get_project_service)
):
    """
    Updates field descriptions for a specific project using the authenticated user's credentials.
    """
    message, status_code = bq_service.update_field_descriptions(request_data.dict())

    if status_code == 200:
        return {"message": message}
    else:
        raise HTTPException(status_code=status_code, detail=message)


@router.post("/{project_id}/build_update_payload_for_tables")
def build_update_payload_for_tables_route(
    table_defs: List[TableDefinition],
    bq_service: BigQueryService = Depends(get_project_service)
):
    """
    Builds an update payload for a specific project using the injected, user-authenticated bq_service.
    """
    try:
        table_defs_dicts = [td.dict() for td in table_defs]
        result_payload = bq_service.build_update_payload_for_tables(
            table_defs_dicts, default_project=bq_service.project_id
        )
        return result_payload
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))