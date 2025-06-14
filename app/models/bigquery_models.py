# app/models/bigquery_models.py

from pydantic import BaseModel
from typing import Optional, Dict, List, Any

class BigQuerySchemaEntry(BaseModel):
    table_catalog: str
    table_schema: str
    table_name: str
    column_name: str
    field_path: str
    data_type: str
    description: Optional[str] = None
    collation_name: str
    rounding_mode: Optional[str] = None
    primary_key: bool
    foreign_key: bool
    field_mode: Optional[str] = None
    access_instructions: Dict[str, Any]

# New models for "update descriptions"
class TableUpdate(BaseModel):
    table_id: str
    updated_descriptions: Dict[str, str]  # Maps "field_path" -> "new description"


class DatasetUpdate(BaseModel):
    dataset_id: str
    tables: List[TableUpdate]


class UpdateDescriptionsRequest(BaseModel):
    project_id: str
    datasets: List[DatasetUpdate]

# -------- Models for update_descriptions response --------

class FieldNotFound(BaseModel):
    table: str
    field_path: str

class SkippedInfo(BaseModel):
    datasets_not_found: List[str]
    tables_not_found: List[str]
    fields_not_found: List[FieldNotFound]

class UpdatedInfo(BaseModel):
    # For each table, a list of updated field paths
    updated_tables: Dict[str, List[str]]

class UpdateDescriptionsResponse(BaseModel):
    skipped: SkippedInfo
    updated: UpdatedInfo
    # If an error occurs, the method might add this key
    error: Optional[str] = None


class ColumnDefinition(BaseModel):
    column_name: str
    description: Optional[str] = None


# This model describes the overall table definition
class TableDefinition(BaseModel):
    table_name: str
    table_link: Optional[str] = None
    description: Optional[str] = None
    relation: Optional[str] = None
    table_description: Optional[str] = None
    columns: List[ColumnDefinition] = []
