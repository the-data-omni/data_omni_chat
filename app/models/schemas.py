"""models for analytics service"""
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

class ChatTurn(BaseModel):
    question: str
    answer: str
    code: Optional[str] = None # Add the code field

class DataFrameRequest(BaseModel):
    data: Optional[List[Dict[str, Any]]] = None
    question: str
    conversation_id: Optional[str] = None # Frontend sends this back
    chat_history: Optional[List[ChatTurn]] = []

class ExecuteCodeWithDataPayload(BaseModel):
    """
    Defines the payload for requests that execute provided Python code, possibly using input data.
    """
    code: str 
    data: Optional[List[Dict[str, Any]]] = None 

class AnalysisResponse(BaseModel):
    summary: Optional[str] = None
    parameterized_summary: Optional[str] = None
    generated_code: Optional[str] = None
    chart_data: Optional[Dict[str, Any]] = None
    analysis_data: Optional[Dict[str, Any]] = None
    conversation_id: str # Backend always returns this


class SummarizePayload(BaseModel):
    """
    Defines the payload for requests aimed at summarizing the result of a code execution.
    Provides context like the original code and question alongside the result.
    """
    execution_result: Union[str, Dict[str, Any]]
    code: str
    question: Optional[str] = None


class DataProfileColumnStats(BaseModel):
    """
    Represents statistics and metadata for a single column derived from a data profiling process.
    """
    column_name: str
    data_type: Optional[str] = None
    data_label: Optional[str] = None
    statistics: Dict[str, Any]


class GlobalStats(BaseModel):
    """
    Represents global statistics for the entire dataset within a data profile.
    """
    row_count: int #: The total number of rows in the dataset.


class DataProfile(BaseModel):
    """
    Represents a complete data profile report, including global dataset statistics
    and detailed statistics for each individual column.
    """
    global_stats: GlobalStats #: Global statistics applying to the entire dataset.
    data_stats: List[DataProfileColumnStats] #: A list containing statistics objects for each column.


class CleanupPlanRequest(BaseModel):
    """
    Defines the request payload for generating a data cleanup plan.
    Requires a data profile and allows optional sample data and configuration for plan generation.
    """
    data_profile: DataProfile #: The data profile report for the dataset to be cleaned.
    data: Optional[List[Dict[str, Any]]] = None #: Optional sample data (list of dicts) that might assist in generating a more accurate cleanup plan. Optional.
    drop_null_threshold: float = Field(
        0.5,
        ge=0.0, # Ensure value is greater than or equal to 0.0
        le=1.0, # Ensure value is less than or equal to 1.0
        description="Threshold (proportion, 0.0 to 1.0) for suggesting column drops based on the percentage of null/missing values. Default is 0.5 (50%)."
    )


class CleanupAction(BaseModel):
    """
    Represents a single proposed or executed action within a data cleanup process,
    targeting a specific column.
    """
    column_name: str #: The name of the column the action applies to.
    action: str #: The type of cleanup action (e.g., 'drop_column', 'fill_nulls', 'change_type', 'remove_duplicates').
    fill_value: Optional[Any] = None #: The value to use for 'fill_nulls' actions, if applicable. Optional.


class CleanupExecuteRequest(BaseModel):
    """
    Defines the request payload for executing a series of specified cleanup actions on data.
    """
    data: Optional[List[Dict[str, Any]]] = None #: The data (list of dicts) to perform cleanup actions on. Can be optional if data is already loaded server-side.
    actions: List[CleanupAction] #: A list of cleanup actions (defined by CleanupAction model) to be executed, typically in the order provided.


class LLMFreeAnalysisRequest(BaseModel):
    """
    Defines the request payload for performing data analysis tasks that do *not* directly
    require interaction with a Large Language Model (LLM). This might involve executing
    predefined code or using parameterized summary templates.
    """
    code: str #: Code snippet intended for execution during the analysis (e.g., pandas operations).
    parameterized_summary: Optional[str] = None #: An optional template or format string for generating the summary of the analysis results. Optional.
    data: Optional[List[Dict[str, Any]]] = None #: Optional data (list of dicts) to be used in the analysis execution. Optional.