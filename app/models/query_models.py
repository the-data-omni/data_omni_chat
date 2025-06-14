# app/models/query_models.py (Revised based on BigQueryService.get_queries)
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime # Assuming creation_time is datetime

class QueryStats(BaseModel):
    job_type: Optional[str] = None
    statement_type: Optional[str] = None
    creation_time: Optional[datetime] = None
    avg_total_bytes_processed: Optional[float] = None
    avg_execution_time: Optional[float] = None
    # The 'count' key in your service's get_queries was query_count
    query_execution_count: Optional[int] = Field(None, alias="count") # Map 'count' from your service to this

class QueryWithStatsItem(BaseModel):
    query: str
    stats: QueryStats

class QuestionQueryWithStatsItem(BaseModel):
    question: str
    query: str
    stats: QueryStats