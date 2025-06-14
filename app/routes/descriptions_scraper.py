import base64
import asyncio
import json
import os
import uuid # Kept if any other part might need unique IDs, though not for session/Redis here.
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl

from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import google.generativeai as genai
from markdownify import markdownify as md

# Assuming crawl4ai is installed and its components are importable
# If BrowserConfig, CacheMode, CrawlerRunConfig are directly from crawl4ai, ensure they are.
# from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
# Mocking these for now if crawl4ai is not a standard package to make the code runnable for syntax checking.
# Replace with actual imports from crawl4ai
class MockBrowserConfig:
    def __init__(self, headless=True, viewport_width=1280, viewport_height=720, extra_args=None):
        pass

class MockCacheMode:
    BYPASS = "bypass"

class MockCrawlerRunConfig:
    def __init__(self, wait_until="networkidle", cache_mode=MockCacheMode.BYPASS):
        pass

class MockCrawlResult:
    def __init__(self, markdown_content):
        self.markdown = markdown_content

class AsyncWebCrawler: # Mock
    def __init__(self, config):
        self.config = config
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    async def arun(self, url, config):
        # This is a mock implementation.
        # In a real scenario, this would use Playwright or similar to fetch and process.
        print(f"Mock crawling URL: {url}")
        if "fivetran.github.io/dbt_salesforce" in url : # specific mock for /scrape
             return MockCrawlResult(markdown_content="""
                ### Table: SomeTable
                Link: [docs/some_table.html](./docs/some_table.html)
                Description: This is a test table.
                #### Columns
                | Column      | Description          |
                |-------------|----------------------|
                | id          | The unique identifier|
                | name        | The name of the item |
                | value       | The value of the item|
                | nested.field| A nested field       |
             """)
        return MockCrawlResult(markdown_content="# Mock Content\nThis is mock markdown for " + url)

BrowserConfig = MockBrowserConfig
CacheMode = MockCacheMode
CrawlerRunConfig = MockCrawlerRunConfig
# End of Mocks for crawl4ai


# --- Pydantic Models for Request and Response ---
class UrlRequest(BaseModel):
    url: HttpUrl

class ScreenshotResponse(BaseModel):
    screenshot: str

class ScrapedLink(BaseModel):
    source_name: str
    table_name: str
    link: Optional[HttpUrl] = None

class ColumnDetail(BaseModel):
    column_name: str
    description: str

class ScrapedTableDetail(BaseModel):
    url: HttpUrl
    relation: Optional[str] = None
    table_description: Optional[str] = None
    columns: List[ColumnDetail]

class MergedData(BaseModel):
    source_name: str
    table_name: str
    link: Optional[HttpUrl] = None
    relation: Optional[str] = None
    table_description: Optional[str] = None
    columns: List[ColumnDetail]

class AugmentedDataItem(BaseModel):
    table_schema: str
    table_name: str
    column_name: str
    description: Optional[str] = None
    # Allow other fields to pass through without validation
    class Config:
        extra = "allow"


class LocalScrapedItem(BaseModel):
    tableName: str
    columnName: str
    columnDesc: str

class GetDescriptionsRequest(BaseModel):
    augmentedData: List[AugmentedDataItem]
    localScraped: Optional[List[LocalScrapedItem]] = []

class GetDescriptionsResponse(BaseModel):
    updatedData: List[AugmentedDataItem]
    descFoundCount: int
    matchedTables: List[List[str]] # List of [table_schema, table_name]
    fivetranDatasets: List[str]
    gaDatasets: List[str]


# --- FastAPI Setup ---
# app = FastAPI(title="Scraping Service")
scrape_router = APIRouter()

# --- Helper Functions (Async Playwright Operations) ---
async def take_screenshot_async(url: str) -> str:
    """
    Open the given URL (headless) and take a screenshot.
    Return it as a BASE64-encoded string (no prefix).
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until='networkidle', timeout=30000) # Increased timeout
            png_bytes = await page.screenshot(full_page=True)
        except Exception as e:
            print(f"Error taking screenshot for {url}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to take screenshot: {str(e)}")
        finally:
            await browser.close()
    return base64.b64encode(png_bytes).decode("utf-8")

async def scrape_source_links_async(url: str) -> List[ScrapedLink]:
    """
    Looks for a table with columns: SOURCE, TABLE, LINK
    and extracts the hyperlink under the LINK column.
    Returns a list of dicts with source_name, table_name, link.
    """
    results = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until='networkidle', timeout=30000)
            await page.wait_for_selector("table:has(th:has-text('LINK'))", timeout=30000)
            html = await page.content()
            await browser.close()
    except Exception as e:
        print(f"Playwright error in scrape_source_links for {url}: {e}")
        # Depending on requirements, you might return an empty list or raise
        # For now, let it proceed to BeautifulSoup which will find nothing or error.
        html = "" # Ensure html is defined

    if not html: # If playwright failed and returned empty html
        return results

    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all("table")
    for table in tables:
        headers_tags = table.find_all("th")
        if not headers_tags:
            continue
        headers = [th.get_text(strip=True).lower() for th in headers_tags]
        if {"source", "table", "link"}.issubset(set(headers)):
            try:
                src_idx = headers.index("source")
                tbl_idx = headers.index("table")
                link_idx = headers.index("link")
            except ValueError: # header not found even if subset check passed (e.g. duplicate headers)
                continue

            tbody = table.find("tbody")
            if not tbody:
                tbody = table # Sometimes tables lack explicit tbody but rows are direct children
            
            rows = tbody.find_all("tr", recursive=False)
            if not rows and tbody is table: # if table has no tbody, try finding rows in table directly
                rows = [r for r in table.children if r.name == 'tr']


            for row in rows:
                tds = row.find_all("td", recursive=False)
                if len(tds) > max(src_idx, tbl_idx, link_idx): # Ensure all indices are valid
                    source_name = tds[src_idx].get_text(strip=True)
                    table_name = tds[tbl_idx].get_text(strip=True)
                    link_tag = tds[link_idx].find("a")
                    link_url_str = None
                    if link_tag and link_tag.has_attr("href"):
                        link_url_str = urljoin(url, link_tag["href"])

                    results.append(ScrapedLink(
                        source_name=source_name,
                        table_name=table_name,
                        link=link_url_str
                    ))
    return results

async def scrape_columns_async(url: str) -> ScrapedTableDetail:
    """
    Follows the 'view docs' link to a detail page that lists columns
    in a table with COLUMN and DESCRIPTION headers, plus 'relation' etc.
    """
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until='networkidle', timeout=30000)
            await page.wait_for_selector("table:has(th:has-text('COLUMN'))", timeout=15000)
            html = await page.content()
            await browser.close()
    except Exception as e:
        print(f"Playwright error in scrape_columns_async for {url}: {e}")
        # Return a default/empty structure or raise
        return ScrapedTableDetail(url=url, relation=None, table_description=None, columns=[])


    soup = BeautifulSoup(html, 'html.parser')
    relation = None
    detail_items = soup.find_all('dl', class_='detail')
    for item in detail_items:
        dt = item.find('dt')
        dd = item.find('dd')
        if dt and dd and dt.get_text(strip=True).lower() == 'relation':
            relation = dd.get_text(strip=True)
            break

    table_description = None
    description_header = soup.find('h6', string='Description')
    if description_header:
        desc_el = description_header.find_next(['p', 'div', 'span'])
        if desc_el:
            table_description = desc_el.get_text(strip=True)

    columns_data = []
    tables = soup.find_all("table")
    for tbl in tables:
        headers_tags = tbl.find_all("th")
        if not headers_tags:
            continue
        headers = [th.get_text(strip=True).lower() for th in headers_tags]
        if "column" in headers and "description" in headers:
            try:
                col_idx = headers.index("column")
                desc_idx = headers.index("description")
            except ValueError:
                continue

            tbody = tbl.find("tbody")
            if not tbody:
                tbody = tbl

            rows = tbody.find_all("tr", recursive=False)
            if not rows and tbody is tbl:
                 rows = [r for r in tbl.children if r.name == 'tr']


            for row in rows:
                tds = row.find_all("td", recursive=False)
                if len(tds) > max(col_idx, desc_idx):
                    column_name = tds[col_idx].get_text(strip=True)
                    description = tds[desc_idx].get_text(strip=True)
                    if column_name: # Ensure column name is not empty
                        columns_data.append(ColumnDetail(
                            column_name=column_name,
                            description=description
                        ))
    return ScrapedTableDetail(
        url=url,
        relation=relation,
        table_description=table_description,
        columns=columns_data
    )

async def gather_source_data_async(url: str) -> List[MergedData]:
    source_rows = await scrape_source_links_async(url)
    merged_data_list = []

    async def process_row(row_data: ScrapedLink):
        if row_data.link:
            try:
                # Ensure link is a string for scrape_columns_async
                detail = await scrape_columns_async(str(row_data.link))
                return MergedData(
                    source_name=row_data.source_name,
                    table_name=row_data.table_name,
                    link=row_data.link,
                    relation=detail.relation,
                    table_description=detail.table_description,
                    columns=detail.columns
                )
            except Exception as e:
                print(f"Error scraping columns for {row_data.link}: {e}")
                # Fallback if detail scraping fails
                return MergedData(
                    source_name=row_data.source_name,
                    table_name=row_data.table_name,
                    link=row_data.link,
                    relation=None,
                    table_description=None,
                    columns=[]
                )
        else:
            return MergedData(
                source_name=row_data.source_name,
                table_name=row_data.table_name,
                link=None,
                relation=None,
                table_description=None,
                columns=[]
            )

    tasks = [process_row(row) for row in source_rows]
    merged_data_list = await asyncio.gather(*tasks)
    return merged_data_list


async def scrape_and_convert_async(url: str) -> str:
    browser_conf = BrowserConfig(
        headless=True,
        viewport_width=1280,
        viewport_height=720,
        extra_args=['--disable-web-security'] # Be cautious with this in production
    )
    config = CrawlerRunConfig(
        wait_until="networkidle",
        cache_mode=CacheMode.BYPASS
    )
    try:
        async with AsyncWebCrawler(config=browser_conf) as crawler:
            result = await crawler.arun(url=url, config=config)
            return result.markdown
    except Exception as e:
        print(f"Error in crawl4ai for {url}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to crawl and convert URL: {str(e)}")


# --- API Routes ---
@scrape_router.post("/scrape_screenshot", response_model=ScreenshotResponse)
async def scrape_screenshot_route(payload: UrlRequest):
    """
    Expects JSON body: { "url": "https://..." }
    Returns { "screenshot": "data:image/png;base64,..." }
    """
    try:
        base64_img = await take_screenshot_async(str(payload.url))
        return ScreenshotResponse(screenshot=f"data:image/png;base64,{base64_img}")
    except HTTPException as e:
        raise e # Re-raise if it's an HTTPException from helper
    except Exception as e:
        print(f"Error in /scrape_screenshot route for {payload.url}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@scrape_router.post("/scrape", response_model=List[MergedData])
async def scrape_data_route(payload: UrlRequest):
    """
    POST body: { "url": "https://fivetran.github.io/dbt_salesforce/..." }
    Returns an array of table objects.
    """
    try:
        print(f"Scraping data for URL: {payload.url}")
        result = await gather_source_data_async(str(payload.url))
        print(result)
        return result
        
    except Exception as e:
        print(f"Error in /scrape route for {payload.url}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to scrape data: {str(e)}")


@scrape_router.post("/crawl") # Define a Pydantic model for the response if its structure is consistent
async def crawl_route(payload: UrlRequest):
    """
    Crawls a given URL, extracts schema information using an LLM.
    Expects a JSON payload with a "url" key.
    """
    try:
        markdown_content = await scrape_and_convert_async(str(payload.url))
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')

        prompt = f"""
        You are a helpful assistant that extracts schema information from markdown documentation. The markdown may describe multiple tables. For each table, extract its name, a link to the table definition (if available), a general description of the table, and a list of its columns. For each column, extract its name and a description, make sure to include any nested columns if displayed.

        Return the output as a JSON array in the following format:

        ```json
        [
            {{
                "table_name": "table_name_1",
                "table_link": "url_to_table_1",
                "table_description": "Description of table 1",
                "columns": [
                    {{"column_name": "column_1", "description": "Description of column 1"}},
                    {{"column_name": "column_2", "description": "Description of column 2"}},
                    {{"column_name": "column_2.nested_column", "description": "Description of a nested column in column 2"}},
                    {{"column_name": "column_2.nested_column.subnested_column", "description": "Description of a column subnested in column 2"}}
                ]
            }}
        ]
        ```

        Markdown Schema:
        ```markdown
        {markdown_content}
        ```
        """
        response = await model.generate_content_async(prompt) # Use async version if available
        
        json_output = response.text
        # Clean potential markdown code fences
        if json_output.startswith("```json"):
            json_output = json_output[7:]
        if json_output.endswith("```"):
            json_output = json_output[:-3]
        json_output = json_output.strip()

        try:
            parsed_output = json.loads(json_output)
            return parsed_output # FastAPI will handle JSONResponse
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError from LLM output: {e}. Output was: {json_output}")
            raise HTTPException(status_code=500, detail="Failed to parse LLM response as JSON.")

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error in /crawl route for {payload.url}: {e}")
        # Consider if you want to return empty list or an error
        # return [] # Original code returned empty list on some errors
        raise HTTPException(status_code=500, detail=f"An error occurred during crawl: {str(e)}")


FIVETRAN_COLUMNS = [
    "_fivetran_synced",
    "_fivetran_id",
    "_fivetran_start",
    "_fivetran_active",
    "_fivetran_end"
]

@scrape_router.post("/get_descriptions", response_model=GetDescriptionsResponse) # Use GetDescriptionsResponse if returning the full structure
# async def get_descriptions_route(payload: GetDescriptionsRequest) -> List[AugmentedDataItem]: # If only returning updatedData
async def get_descriptions_route(payload: GetDescriptionsRequest) -> GetDescriptionsResponse:
    """
    Receive field info, detect Fivetran/GA, merge descriptions, and respond.
    """
    augmented_data = payload.augmentedData
    local_scraped = payload.localScraped if payload.localScraped else []

    # Load the server-side schema (synchronous file I/O)
    # For FastAPI, if this were a very slow operation, use asyncio.to_thread
    server_schema = []
    try:
        # Ensure the path is correct relative to where the FastAPI app runs
        with open("data/my-schema.json", "r") as f:
            server_schema = json.load(f)
            print("Server schema loaded.")
    except FileNotFoundError:
        print("data/my-schema.json not found. Proceeding with empty server_schema.")
    except json.JSONDecodeError:
        print("Error decoding data/my-schema.json. Proceeding with empty server_schema.")
    except Exception as e:
        print(f"An error occurred loading data/my-schema.json: {e}. Proceeding with empty server_schema.")

    local_map = {}
    for item in local_scraped:
        key = (item.tableName.lower(), item.columnName.lower())
        local_map[key] = item.columnDesc

    fivetran_found = set()
    ga_found = set()
    for row in augmented_data:
        col_lower = row.column_name.lower()
        table_lower = row.table_name.lower()
        if col_lower in FIVETRAN_COLUMNS:
            fivetran_found.add(row.table_schema)
        if table_lower.startswith("events_"): # GA tables often start with events_
            ga_found.add(row.table_schema)

    schema_map = {}
    for tbl_info in server_schema:
        tbl_name = tbl_info.get("table_name", "").lower()
        columns = tbl_info.get("columns", [])
        schema_map[tbl_name] = columns

    descFoundCount = len(server_schema)
    matched_tables_set = set() # Use a set of tuples for unique [schema, table] pairs

    updated_data_list = []
    for row_model in augmented_data:
        # Convert Pydantic model back to dict if downstream code expects dicts,
        # or operate on the model directly. Here, we update the model.
        current_desc = row_model.description or ""
        if not current_desc:
            t_lower = row_model.table_name.lower()
            c_lower = row_model.column_name.lower()

            local_key = (t_lower, c_lower)
            if local_key in local_map:
                row_model.description = local_map[local_key]
                matched_tables_set.add((row_model.table_schema, row_model.table_name))
            elif t_lower in schema_map:
                for col_schema in schema_map[t_lower]:
                    if col_schema.get("column_name", "").lower() == c_lower:
                        col_desc = col_schema.get("description")
                        if col_desc:
                            row_model.description = col_desc
                            matched_tables_set.add((row_model.table_schema, row_model.table_name))
                        break
        updated_data_list.append(row_model)

    return GetDescriptionsResponse(
        updatedData=updated_data_list,
        descFoundCount=descFoundCount,
        matchedTables=[list(pair) for pair in matched_tables_set], # Convert set of tuples to list of lists
        fivetranDatasets=list(fivetran_found),
        gaDatasets=list(ga_found),
    )
    # If you only want to return the updated data as per original Flask jsonify(updated_data)
    # return updated_data_list


# app.include_router(scrape_router, prefix="/api") # You can add a prefix if desired

# To run this (save as main.py or similar):
# uvicorn main:app --reload
#
# Then you can access the API at http://127.0.0.1:8000/api/...
# And docs at http://127.0.0.1:8000/docs