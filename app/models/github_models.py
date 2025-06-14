

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class SchemaItem(BaseModel):
    table_catalog: str
    table_schema: str
    table_name: str
    column_name: str
    field_path: str 
    data_type: str
    description: Optional[str] = None
    collation_name: Optional[str] = None
    rounding_mode: Optional[str] = None
    primary_key: Optional[bool] = False 
    foreign_key: Optional[bool] = False 
    field_mode: Optional[str] = None    
    access_instructions: Optional[Dict[str, str]] = None

class QuestionQueryPairItem(BaseModel):
    question_text: str
    sql_query_text: str

class GitRepoTargetConfig(BaseModel):
    repo_url: str = Field(..., description="HTTPS URL of the target GitHub repository.")
    branch: str = Field("main", description="Target branch in the repository.")
    github_pat: str = Field(..., description="GitHub Personal Access Token for authentication.")

class SaveSchemaToGitRequest(BaseModel):
    git_config: GitRepoTargetConfig
    target_file_name: str = Field("schema_descriptions.json", description="Filename for schema data in the repo.")
    schema_data: List[SchemaItem]
    commit_message: str = Field("Update schema descriptions via API", min_length=1)

class LoadSchemaFromGitRequest(BaseModel): 
    git_config: GitRepoTargetConfig 
    target_file_name: str = Field("schema_descriptions.json", description="Filename for schema data in the repo.")

class SaveQQToGitRequest(BaseModel):
    git_config: GitRepoTargetConfig
    target_file_name: str = Field("question_query_pairs.json", description="Filename for Q/Q data in the repo.")
    qq_data: List[QuestionQueryPairItem]
    commit_message: str = Field("Update Q/Q pairs via API", min_length=1)

class LoadQQFromGitRequest(BaseModel): 
    git_config: GitRepoTargetConfig
    target_file_name: str = Field("question_query_pairs.json", description="Filename for Q/Q data in the repo.")

class LoadSchemaFromGitResponse(BaseModel):
    schema_data: List[SchemaItem]
    commit_hash: Optional[str] = None
    last_modified: Optional[str] = None 
    message: str
    error: Optional[str] = None
    target_file_name: str 

class LoadQQFromGitResponse(BaseModel):
    qq_data: List[QuestionQueryPairItem]
    commit_hash: Optional[str] = None
    last_modified: Optional[str] = None
    message: str
    error: Optional[str] = None

class GitOperationResponse(BaseModel): 
    message: str
    commit_hash: Optional[str] = None
    error: Optional[str] = None

class CreateGitHubRepoRequest(BaseModel):
    github_pat: str 
    name: str 
    description: Optional[str] = None
    private: bool = False

class CreateGitHubRepoResponse(BaseModel):
    name: str
    html_url: str
    ssh_url: str
    clone_url: str 
    message: str
    error: Optional[str] = None