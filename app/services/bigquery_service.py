import os
import json
import re
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from google.cloud import bigquery
from google.oauth2 import service_account
from google.api_core.exceptions import NotFound
from google.auth.credentials import Credentials as AuthCredentials
from google.api_core.exceptions import NotFound

class BigQueryService:
    def __init__(self, credentials: AuthCredentials,project_id: str, load_descriptions: bool = True):
        """
        Initialize with a generic Google Auth credentials object.
        
        This object can be from a service account OR a user's OAuth token.
        The class is now decoupled from the credential creation process.
        """

        if not credentials:
            raise ValueError("A valid credentials object is required.")
        if not project_id:
            raise ValueError("A valid project_id is required.")

        self.creds = credentials
        # The bigquery.Client is initialized with the provided credentials.
        # We explicitly pass the project_id from the credentials if it exists.
        self.client = bigquery.Client(credentials=self.creds, project=project_id)
        self.flattened_schema = None
        self.scraped_descriptions: Optional[List[Dict[str, Any]]] = None
        self.new_descriptions: Optional[Dict[str, Any]] = None
        logging.info(f"BigQueryService initialized for project: {self.client.project}")



    def load_scraped_descriptions_from_file(self, filename: str = "scraped_fivetran_descriptions.json") -> None:
        """
        Loads the raw scraped descriptions from a JSON file into self.scraped_descriptions.
        The file is expected to contain a list of table definition objects, e.g.:

        [
          {
            "table_name": "recharge__billing_history",
            "table_link": "...",
            "description": "...",
            "relation": "...",
            "table_description": "...",
            "columns": [
                {"column_name": "order_id", "description": "Unique identifier..."},
                ...
            ]
          },
          ...
        ]
        """
        script_dir = os.path.dirname(__file__)
        json_path = os.path.join(script_dir, filename)

        if not os.path.exists(json_path):
            logging.warning(f"{json_path} does not exist; cannot load scraped descriptions.")
            return

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                logging.warning(f"Expected a list of table definitions, got {type(data)} instead.")
                return

            self.scraped_descriptions = data
            logging.info(f"Successfully loaded scraped descriptions from {json_path}.")
        except Exception as e:
            logging.error(f"Failed to load {json_path}: {str(e)}")
 
    def get_queries(self, time_interval: str = "10 day") -> Dict[str, Any]:
        query = f"""
            SELECT
              query,
              ANY_VALUE(job_type) AS job_type,
              ANY_VALUE(statement_type) AS statement_type,
              MAX(creation_time) AS creation_time,
              AVG(total_bytes_processed) AS avg_total_bytes_processed,
              AVG(query_info.performance_insights.avg_previous_execution_ms) AS avg_execution_time,
              COUNT(*) AS query_count
            FROM region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT
            WHERE
              creation_time > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {time_interval})
              AND state = 'DONE'
              AND job_type = 'QUERY'
              AND statement_type ='SELECT'
              --AND user_email ="pfupajenadev@gmail.com"
              AND NOT REGEXP_CONTAINS(LOWER(query), r'\\binformation_schema\\b')
            GROUP BY
              query
        """
        query_job = self.client.query(query)
        query_data = {}
        for row in query_job.result():
            query_text = row.query
            query_data[query_text] = {
                "job_type": row.job_type,
                "statement_type": row.statement_type,
                "creation_time": row.creation_time,
                "avg_total_bytes_processed": row.avg_total_bytes_processed,
                "avg_execution_time": row.avg_execution_time,
                "count": row.query_count,
            }
        return query_data

    def get_bigquery_info(self) -> Dict[str, Any]:
        try:
            project_id = self.client.project
            project_info = {"project_id": project_id, "datasets": []}
            datasets = list(self.client.list_datasets())
            if not datasets:
                return project_info

            # Optionally fetch constraints for each dataset (omitted here for brevity)

            shard_pattern = re.compile(r"^(.*)_\d{8}$")
            for dataset in datasets:
                dataset_id = dataset.dataset_id
                dataset_info = {"dataset_id": dataset_id, "tables": []}
                tables = self.client.list_tables(dataset_id)
                seen_normalized = set()
                for table in tables:
                    original_table_id = table.table_id
                    match = shard_pattern.match(original_table_id)
                    normalized_table_id = f"{match.group(1)}_*" if match else original_table_id
                    if normalized_table_id in seen_normalized:
                        continue
                    seen_normalized.add(normalized_table_id)
                    table_ref = self.client.dataset(dataset_id).table(original_table_id)
                    table_obj = self.client.get_table(table_ref)
                    table_field_info = self.get_field_info(table_obj.schema)
                    dataset_info["tables"].append({
                        "table_id": normalized_table_id,
                        "fields": table_field_info,
                    })
                project_info["datasets"].append(dataset_info)
            return project_info
        except Exception as exc:
            logging.error("Error processing BigQuery info: %s", str(exc))
            raise Exception(f"Error processing BigQuery info: {exc}")

    def get_field_info(self, fields, parent_field_name: str = "") -> List[Dict[str, Any]]:
        field_info_list = []
        for field in fields:
            full_field_name = f"{parent_field_name}.{field.name}" if parent_field_name else field.name
            field_info_list.append({
                "field_path": full_field_name,
                "data_type": field.field_type,
                "description": field.description or None,
                "mode": field.mode
            })
            if field.field_type == "RECORD":
                nested_field_info = self.get_field_info(field.fields, full_field_name)
                field_info_list.extend(nested_field_info)
        return field_info_list

    def human_readable_size(self, num_bytes: float) -> str:
        for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
            if num_bytes < 1024.0:
                return f"{num_bytes:0.2f} {unit}"
            num_bytes /= 1024.0
        return f"{num_bytes:0.2f} PB"

    def dry_run_query(self, sql_query: str) -> Dict[str, Any]:
        try:
            job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
            query_job = self.client.query(sql_query, job_config=job_config)
            raw_bytes = query_job.total_bytes_processed
            formatted = self.human_readable_size(raw_bytes)
            return {
                "total_bytes_processed": raw_bytes,
                "formatted_bytes_processed": formatted,
            }
        except Exception as exc:
            raise Exception(f"Error during dry-run: {exc}")

    def build_access_instructions(
            self,
            field_path: str,
            field_lookup: dict,
            project_id: str,
            dataset_id: str,
            table_id: str
        ) -> dict:
            """
            Given a field_path (e.g. 'hits.promotion.promoId'), plus a lookup of
            all fields in the table, build a from_clause + select_expr that shows
            how to query this nested field.
            """
            if not field_path:
                return {"from_clause": "", "select_expr": ""}

            # Build the full table reference
            full_table_ref = f"`{project_id}.{dataset_id}.{table_id}`"

            # Derive the table alias:
            alias_for_table = table_id
            if alias_for_table.endswith("*"):
                alias_for_table = alias_for_table.replace("_*", "")

            base_alias = alias_for_table
            segments = field_path.split(".")  # e.g. ["hits", "promotion", "promoId"]
            join_lines = []
            current_alias = base_alias
            partial_path = ""

            for i, seg in enumerate(segments):
                partial_path = seg if i == 0 else f"{partial_path}.{seg}"
                field_meta = field_lookup.get(partial_path)
                if not field_meta:
                    continue

                is_repeated = (field_meta.get("mode") == "REPEATED")
                if is_repeated:
                    join_line = f"LEFT JOIN UNNEST({current_alias}.{seg}) AS {seg}"
                    join_lines.append(join_line)
                    current_alias = seg  # now reference 'seg' as the alias
                else:
                    current_alias = f"{current_alias}.{seg}"

            from_clause_lines = [f"{full_table_ref} AS {base_alias}"] + join_lines
            from_clause = "\n".join(from_clause_lines)

            last_field_meta = field_lookup.get(field_path, {})
            last_is_repeated = (last_field_meta.get("mode") == "REPEATED")
            last_type = last_field_meta.get("data_type", "")

            if last_is_repeated and last_type != "RECORD":
                select_expr = current_alias
            else:
                select_expr = current_alias

            return {
                "from_clause": from_clause,
                "select_expr": select_expr
            }
    
    def flatten_bq_schema(self, project_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        flattened = []
        top_level_project = project_info.get("project_id", "unknown_project")
        for dataset in project_info.get("datasets", []):
            ds_project_id = dataset.get("project_id", top_level_project)
            ds_id = dataset.get("dataset_id", "")
            for table in dataset.get("tables", []):
                tbl_id = table.get("table_id", "")
                field_lookup = {f["field_path"]: f for f in table.get("fields", [])}
                for field in table.get("fields", []):
                    access_instructions = self.build_access_instructions(
                        field_path=field.get("field_path", ""),
                        field_lookup=field_lookup,
                        project_id=ds_project_id,
                        dataset_id=ds_id,
                        table_id=tbl_id
                    )
                    flattened.append({
                        "table_catalog": ds_project_id,
                        "table_schema": ds_id,
                        "table_name": tbl_id,
                        "column_name": field.get("field_path", ""),
                        "field_path": field.get("field_path", ""),
                        "data_type": field.get("data_type", ""),
                        "description": field.get("description"),
                        "collation_name": "NULL",
                        "rounding_mode": None,
                        "primary_key": field.get("is_primary_key", False),
                        "foreign_key": field.get("is_foreign_key", False),
                        "field_mode": field.get("mode"),
                        "access_instructions": access_instructions
                    })
        self.flattened_schema = flattened
        return flattened
    
    def update_field_descriptions(self, request_data: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        
            """
            Updates field descriptions in BigQuery tables dynamically from the request_data.
            Supports both single tables and sharded tables (e.g. 'events_*').

            Expected structure of request_data:
            {
                "project_id": "my-gcp-project",
                "datasets": [
                    {
                        "dataset_id": "my_dataset",
                        "tables": [
                            {
                                "table_id": "my_table" OR "events_*",
                                "updated_descriptions": {
                                    "fieldA": "New description for fieldA",
                                    "nestedField.subField": "Description for a nested field"
                                }
                            }
                        ]
                    }
                ]
            }

            Returns a tuple of (result_dict, status_code), where result_dict includes
            "skipped" and "updated" info.
            """
            if not request_data:
                if not self.new_descriptions:
                    return (
                        {"error": "No descriptions provided, and no stored payload available."},
                        400
                    )
                request_data = self.new_descriptions
            project_id = request_data.get("project_id")
            datasets = request_data.get("datasets")
            if not project_id or not datasets:
                return (
                    {
                        "error": "Missing required parameters: 'project_id' and/or 'datasets'.",
                        "skipped": {},
                        "updated": {}
                    },
                    400
                )

            # Decide whether to reuse self.client or create a new one for the specified project
            if project_id == self.client.project:
                client = self.client
            else:
                client = bigquery.Client(credentials=self.creds, project=project_id)

            # Data structures to store what's been skipped and what's been updated
            skipped_info = {
                "datasets_not_found": [],   # a list of dataset_ids that don't exist
                "tables_not_found": [],     # a list of table_id patterns that don't exist
                "fields_not_found": []      # a list of {table: ..., field_path: ...} for unknown fields
            }
            updated_info = {
                # map of table name/pattern to a list of updated field paths
                # e.g. "my_table" -> ["fieldA", "nestedField.subField"]
                "updated_tables": {}
            }

            def gather_existing_field_paths(fields: List[bigquery.SchemaField], parent_path: str = "") -> List[str]:
                """
                Return a flat list of all possible field paths (including nested) for the given schema.
                Example of returned paths: ["fieldA", "nestedField", "nestedField.subField", ...]
                """
                results = []
                for f in fields:
                    full_path = f"{parent_path}.{f.name}".lstrip(".")  # remove leading dot
                    results.append(full_path)
                    if f.field_type == "RECORD":
                        # Recurse into nested fields
                        results.extend(gather_existing_field_paths(f.fields, full_path))
                return results

            def update_schema_fields(
                fields: List[bigquery.SchemaField],
                updated_descriptions: Dict[str, str],
                parent_field_name: str = ""
            ) -> Tuple[List[bigquery.SchemaField], List[str]]:
                """
                Recursively update the BigQuery schema fields (including nested RECORD types).
                `updated_descriptions` is a dict mapping full field paths
                (e.g. "event_params.value.string_value") to a new description string.

                Returns:
                    (updated_fields, updated_field_names): a tuple of:
                    - updated_fields: the new schema (with updated descriptions)
                    - updated_field_names: which field paths were changed
                """
                updated_fields = []
                updated_field_names = []

                for field in fields:
                    full_field_name = f"{parent_field_name}.{field.name}".lstrip(".")

                    if field.field_type == "RECORD":
                        # Recursively update nested fields
                        subfields, subfield_names = update_schema_fields(
                            field.fields, updated_descriptions, full_field_name
                        )
                        updated_fields.append(
                            bigquery.SchemaField(
                                name=field.name,
                                field_type=field.field_type,
                                mode=field.mode,
                                description=updated_descriptions.get(full_field_name, field.description),
                                fields=subfields
                            )
                        )
                        updated_field_names.extend(subfield_names)
                    else:
                        # For a scalar field, apply update if it exists in updated_descriptions
                        updated_description = updated_descriptions.get(full_field_name, field.description)
                        if updated_description != field.description:
                            updated_field_names.append(full_field_name)

                        updated_fields.append(
                            bigquery.SchemaField(
                                name=field.name,
                                field_type=field.field_type,
                                mode=field.mode,
                                description=updated_description
                            )
                        )

                return updated_fields, updated_field_names

            def process_table(
                dataset_id: str,
                table_id: str,
                updated_descriptions: Dict[str, str]
            ) -> List[str]:
                """
                Retrieve the table schema, prune non-existent field paths, update descriptions,
                and apply changes. Returns a list of updated field paths.
                """
                table_ref = client.dataset(dataset_id).table(table_id)

                try:
                    table = client.get_table(table_ref)
                except NotFound:
                    # Table doesn't exist; mark as skipped
                    skipped_info["tables_not_found"].append(table_id)
                    return []

                # Gather all existing fields for the schema
                existing_schema = table.schema
                existing_paths = set(gather_existing_field_paths(existing_schema))

                # Identify and remove any field paths from updated_descriptions that don't exist
                invalid_paths = [fp for fp in updated_descriptions.keys() if fp not in existing_paths]
                for fp in invalid_paths:
                    # Record that we skipped this particular field
                    skipped_info["fields_not_found"].append({"table": table_id, "field_path": fp})
                    del updated_descriptions[fp]

                # If nothing remains to update, return early
                if not updated_descriptions:
                    return []

                # Update the schema for existing fields
                new_schema, updated_field_names = update_schema_fields(
                    existing_schema, updated_descriptions
                )

                if updated_field_names:
                    table.schema = new_schema
                    table = client.update_table(table, ["schema"])

                return updated_field_names

            # -- MAIN LOGIC --
            try:
                for dataset_info in datasets:
                    dataset_id = dataset_info.get("dataset_id")
                    tables_info = dataset_info.get("tables", [])
                    if not dataset_id or not tables_info:
                        continue

                    # First check if the dataset actually exists
                    try:
                        client.get_dataset(dataset_id)
                    except NotFound:
                        # Dataset doesn't exist; skip it entirely
                        skipped_info["datasets_not_found"].append(dataset_id)
                        continue

                    for table_info in tables_info:
                        table_id_pattern = table_info.get("table_id")
                        updated_descriptions = dict(table_info.get("updated_descriptions", {}))  # copy
                        if not table_id_pattern or not updated_descriptions:
                            continue

                        # If table_id_pattern ends with '*', handle all sharded tables
                        if table_id_pattern.endswith("*"):
                            prefix = table_id_pattern[:-1]  # remove the '*'
                            full_dataset_ref = f"{project_id}.{dataset_id}"
                            any_updated = False

                            for table_item in client.list_tables(full_dataset_ref):
                                # If the table name starts with e.g. "events_"
                                if table_item.table_id.startswith(prefix):
                                    updated_fields = process_table(
                                        dataset_id, table_item.table_id, dict(updated_descriptions)
                                    )
                                    if updated_fields:
                                        any_updated = True
                                        # Record that these fields were updated
                                        updated_info["updated_tables"].setdefault(table_item.table_id, [])
                                        updated_info["updated_tables"][table_item.table_id].extend(updated_fields)

                            # If we found no matching tables at all, treat that pattern as not found
                            if not any_updated:
                                # Possibly the dataset has no tables that match the prefix
                                # We'll still treat that as 'no actual updates happened'
                                pass
                        else:
                            # Single table
                            updated_fields = process_table(
                                dataset_id, table_id_pattern, updated_descriptions
                            )
                            if updated_fields:
                                updated_info["updated_tables"].setdefault(table_id_pattern, [])
                                updated_info["updated_tables"][table_id_pattern].extend(updated_fields)

                final_result = {
                    "skipped": skipped_info,
                    "updated": updated_info
                }
                return final_result, 200

            except Exception as e:
                logging.error(f"Error updating field descriptions: {str(e)}")
                error_result = {
                    "error": f"Unexpected error: {str(e)}",
                    "skipped": skipped_info,
                    "updated": updated_info
                }
                return error_result, 500

    def set_new_descriptions(self, payload: Dict[str, Any]) -> None:
        """
        Allows storing a payload globally in the service instance.
        This can be used later if update_field_descriptions is called
        without explicit request_data.
        """
        self.new_descriptions = payload

    def build_update_payload_for_table(
        self,
        table_def: Dict[str, Any],
        default_project: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Given a single 'table definition' object like:
            {
            "table_name": "recharge__billing_history",
            "table_link": "...",          # (optional)
            "description": "...",         # (optional)
            "relation": "...",            # (optional)
            "table_description": "...",   # (optional)
            "columns": [
                {
                    "column_name": "order_id",
                    "description": "Unique identifier of the order."
                },
                ...
            ]
            }

        1. Searches `self.flattened_schema` to find which dataset this table_name
        belongs to (and which project), if available.
        2. Builds an `updated_descriptions` dict mapping each `column_name` to its new
        description.
        3. Returns a dictionary in the format consumed by `update_field_descriptions`, e.g.:

            {
            "project_id": "my-project",
            "datasets": [
                {
                "dataset_id": "my_dataset",
                "tables": [
                    {
                    "table_id": "recharge__billing_history",
                    "updated_descriptions": {
                        "order_id": "Unique identifier of the order.",
                        ...
                    }
                    }
                ]
                }
            ]
            }

        If no match is found in `self.flattened_schema`, we fall back to `default_project`
        (if provided) or the client's project, and no dataset_id will be found (or will
        be set to 'UNKNOWN_DATASET').
        """

        # 1. Collect the table_name and columns from the input
        table_name = table_def.get("table_name")
        columns = table_def.get("columns", [])

        if not table_name or not columns:
            # Return a minimal structure if the input is incomplete
            return {
                "project_id": default_project or self.client.project,
                "datasets": [
                    {
                        "dataset_id": "UNKNOWN_DATASET",
                        "tables": [
                            {
                                "table_id": table_name or "UNKNOWN_TABLE",
                                "updated_descriptions": {}
                            }
                        ]
                    }
                ]
            }

        # 2. Search self.flattened_schema for a matching row
        #    that has table_name == table_def["table_name"].
        #    (If you have date-sharded or wildcard tables, you might need
        #     to compare differently, e.g. removing "_*" suffix.)
        matched_rows = [
            row for row in (self.flattened_schema or [])
            if row.get("table_name") == table_name
        ]

        # 3. If found, extract project/dataset from first match.
        #    Otherwise, fallback to the default or to self.client.project.
        if matched_rows:
            first_match = matched_rows[0]
            project_id = first_match.get("table_catalog", default_project or self.client.project)
            dataset_id = first_match.get("table_schema", "UNKNOWN_DATASET")
        else:
            project_id = default_project or self.client.project
            dataset_id = "UNKNOWN_DATASET"

        # 4. Build the updated_descriptions from columns
        updated_descriptions = {}
        for col in columns:
            col_name = col.get("column_name")
            col_desc = col.get("description")
            if col_name and col_desc:
                updated_descriptions[col_name] = col_desc

        # 5. Construct the final payload
        payload = {
            "project_id": project_id,
            "datasets": [
                {
                    "dataset_id": dataset_id,
                    "tables": [
                        {
                            "table_id": table_name,
                            "updated_descriptions": updated_descriptions
                        }
                    ]
                }
            ]
        }

        return payload

    def build_update_payload_for_tables(
        self,
        table_defs: List[Dict[str, Any]],
        default_project: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        For each item in table_defs, build an update payload and merge them
        into a single request structure. 
        """
        # This dictionary will accumulate all tables grouped by dataset
        # and each dataset grouped by project_id.
        combined_payload = {}
        
        for table_def in table_defs:
            partial_payload = self.build_update_payload_for_table(table_def, default_project)
            proj_id = partial_payload["project_id"]
            ds_list = partial_payload["datasets"]  # typically just one item

            if proj_id not in combined_payload:
                combined_payload[proj_id] = {}

            for ds in ds_list:
                ds_id = ds["dataset_id"]
                if ds_id not in combined_payload[proj_id]:
                    combined_payload[proj_id][ds_id] = []

                # Each ds has `tables` array
                for tbl in ds["tables"]:
                    combined_payload[proj_id][ds_id].append(tbl)

        if len(combined_payload) > 1:
            raise ValueError("Multiple projects found. Logic needed to handle multi-project merges.")

        project_id = list(combined_payload.keys())[0]
        project_datasets = combined_payload[project_id]

        final_datasets = []
        for ds_id, tables in project_datasets.items():
            final_datasets.append({
                "dataset_id": ds_id,
                "tables": tables
            })

        return {
            "project_id": project_id,
            "datasets": final_datasets
        }
