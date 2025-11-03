# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import uuid
import os
import io
import json
import subprocess
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import storage, bigquery
import vertexai
from vertexai.generative_models import GenerativeModel
import logging
from monitoring import (
    monitor_endpoint, monitor_llm_call, monitor_code_execution, 
    monitor_chart_generation, structured_logger, get_metrics,
    log_security_event
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Botlytics API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

# Environment variables
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCS_BUCKET = os.environ.get("GCS_BUCKET")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")

# Initialize Vertex AI
if GCP_PROJECT_ID:
    vertexai.init(project=GCP_PROJECT_ID, location=LOCATION)

@monitor_llm_call("gemini-1.5-pro")
def call_gemini(prompt: str) -> str:
    """Call Gemini using Vertex AI SDK"""
    try:
        model = GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        
        # Log interaction
        structured_logger.log_llm_interaction(
            model="gemini-1.5-pro",
            prompt_length=len(prompt),
            response_length=len(response.text) if response.text else 0,
            duration=0  # Duration tracked by decorator
        )
        
        return response.text
    except Exception as e:
        logger.error(f"Error calling Gemini: {e}")
        raise HTTPException(500, f"Gemini API error: {str(e)}")

def build_prompt_for_pandas(columns: list, user_query: str) -> str:
    """Build prompt for pandas code generation using two-step pattern"""
    return f"""
You are an expert data analyst. The user will give you a dataset description and a question. 
Respond with a JSON object with these keys:
- "mode": "pandas" (always for CSV data)
- "code": sanitized Pandas code operating on `df` — only transformations, no imports or I/O
- "chart": {{"type": "line|bar|pie|scatter|table", "x": "<column>", "y": "<column>", "title": "<title>"}}
- "explain": short description (<= 60 words) of what the output will show

Dataset columns: {columns}
Question: {user_query}

CRITICAL RULES for code generation:
- ONLY use pandas operations on variable 'df'
- NO imports, file operations, or system calls
- NO exec, eval, __import__, or dangerous functions
- ONLY these pandas methods allowed: groupby, sum, mean, count, head, tail, sort_values, reset_index, set_index, resample, agg, merge, concat, drop, fillna, dropna, to_datetime, select_dtypes
- Code must assign final result to variable 'result'
- Use .head(20) to limit large results
- For time series, convert date columns with pd.to_datetime first

Example response:
{{
    "mode": "pandas",
    "code": "df['date'] = pd.to_datetime(df['date']); result = df.groupby('product')['sales'].sum().reset_index().sort_values('sales', ascending=False).head(10)",
    "chart": {{"type": "bar", "x": "product", "y": "sales", "title": "Top Products by Sales"}},
    "explain": "Shows top 10 products ranked by total sales amount."
}}
"""

def sanitize_pandas_code(code: str, allowed_columns: list) -> str:
    """Enhanced sanitization of pandas code with strict validation"""
    import re
    
    # Dangerous patterns - comprehensive list
    dangerous_patterns = [
        'import', 'exec', 'eval', 'open', 'file', '__', 'subprocess', 'os.',
        'sys.', 'builtins', 'globals', 'locals', 'vars', 'dir', 'getattr',
        'setattr', 'delattr', 'hasattr', 'compile', 'input', 'raw_input',
        'reload', 'help', 'quit', 'exit', 'copyright', 'credits', 'license',
        'print', 'input', 'breakpoint', 'memoryview', 'bytearray', 'bytes'
    ]
    
    code_lower = code.lower()
    for pattern in dangerous_patterns:
        if pattern in code_lower:
            log_security_event("unsafe_code_detected", {
                "pattern": pattern,
                "code_snippet": code[:100]  # First 100 chars for context
            })
            raise HTTPException(400, f"Unsafe operation detected: {pattern}")
    
    # Check for system calls or shell commands
    if any(char in code for char in ['!', '$', '`']):
        raise HTTPException(400, "Shell commands not allowed")
    
    # Allowed pandas methods only
    allowed_methods = {
        'groupby', 'sum', 'mean', 'count', 'head', 'tail', 'sort_values', 
        'reset_index', 'set_index', 'resample', 'agg', 'merge', 'concat', 
        'drop', 'fillna', 'dropna', 'to_datetime', 'select_dtypes', 'describe',
        'value_counts', 'unique', 'nunique', 'min', 'max', 'std', 'var',
        'median', 'quantile', 'corr', 'cov', 'rolling', 'expanding',
        'pivot_table', 'melt', 'stack', 'unstack', 'transpose', 'T',
        'iloc', 'loc', 'at', 'iat', 'query', 'assign', 'pipe', 'apply',
        'map', 'applymap', 'transform', 'filter', 'where', 'mask',
        'isin', 'between', 'isna', 'notna', 'duplicated', 'drop_duplicates'
    }
    
    # Extract method calls and validate
    method_pattern = r'\.(\w+)\s*\('
    methods_used = re.findall(method_pattern, code)
    
    for method in methods_used:
        if method not in allowed_methods and method not in ['dt', 'str', 'cat']:  # Allow accessor methods
            raise HTTPException(400, f"Method '{method}' not allowed")
    
    # Validate column references
    column_pattern = r'[\'\"](.*?)[\'\"]'
    referenced_columns = re.findall(column_pattern, code)
    
    for col in referenced_columns:
        if col not in allowed_columns and col not in ['date', 'datetime']:  # Allow common date formats
            logger.warning(f"Column '{col}' not in dataset, but allowing for flexibility")
    
    # Ensure code assigns to 'result' variable
    if 'result =' not in code and 'result=' not in code:
        raise HTTPException(400, "Code must assign final output to 'result' variable")
    
    # Check for reasonable length
    if len(code) > 1000:
        raise HTTPException(400, "Code too long - keep transformations simple")
    
    return code

@monitor_code_execution("pandas")
def run_pandas_code_safe(df: pd.DataFrame, code: str) -> dict:
    """Execute pandas code safely with resource limits and timeout"""
    import signal
    import resource
    import sys
    from contextlib import contextmanager
    
    @contextmanager
    def timeout_context(seconds):
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Code execution timed out after {seconds} seconds")
        
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    try:
        # Set memory limit (100MB)
        resource.setrlimit(resource.RLIMIT_AS, (100 * 1024 * 1024, 100 * 1024 * 1024))
        
        # Create restricted execution environment
        safe_builtins = {
            'len': len, 'str': str, 'int': int, 'float': float, 'bool': bool,
            'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
            'min': min, 'max': max, 'sum': sum, 'abs': abs, 'round': round,
            'sorted': sorted, 'reversed': reversed, 'enumerate': enumerate,
            'zip': zip, 'range': range, 'type': type, 'isinstance': isinstance
        }
        
        # Safe globals with only pandas and necessary functions
        safe_globals = {
            "__builtins__": safe_builtins,
            "pd": pd,
            "df": df.copy(),  # Work on a copy to prevent modification
        }
        
        local_vars = {}
        
        # Execute with timeout (30 seconds max)
        with timeout_context(30):
            exec(code, safe_globals, local_vars)
        
        # Get the result
        result = local_vars.get('result')
        if result is None:
            raise ValueError("Code must assign result to 'result' variable")
        
        # Limit result size
        if hasattr(result, '__len__') and len(result) > 1000:
            result = result.head(1000) if hasattr(result, 'head') else result[:1000]
        
        # Convert to serializable format
        if hasattr(result, 'to_dict'):
            preview = result.head(20).to_dict('records')
            result_summary = {
                'type': 'DataFrame',
                'shape': result.shape,
                'columns': result.columns.tolist() if hasattr(result, 'columns') else None
            }
        elif hasattr(result, 'tolist'):
            preview = result.tolist()[:20]
            result_summary = {'type': 'Series', 'length': len(result)}
        else:
            preview = str(result)[:1000]  # Limit string length
            result_summary = {'type': type(result).__name__}
        
        return {
            "success": True,
            "result": result,
            "preview": preview,
            "summary": result_summary
        }
        
    except TimeoutError as e:
        logger.error(f"Code execution timeout: {e}")
        return {
            "success": False,
            "error": "Code execution timed out (30s limit)",
            "preview": "Execution timeout"
        }
    except MemoryError as e:
        logger.error(f"Code execution memory error: {e}")
        return {
            "success": False,
            "error": "Code execution exceeded memory limit (100MB)",
            "preview": "Memory limit exceeded"
        }
    except Exception as e:
        logger.error(f"Code execution error: {e}")
        return {
            "success": False,
            "error": str(e),
            "preview": "Error executing code"
        }
    finally:
        # Reset resource limits
        try:
            resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        except:
            pass

def create_chart_and_upload(data, chart_spec: dict, dataset_id: str) -> str:
    """Create chart and upload to GCS"""
    if not chart_spec or not isinstance(data, pd.DataFrame):
        return None
    
    try:
        plt.figure(figsize=(10, 6))
        
        chart_type = chart_spec.get('type', 'bar')
        x_col = chart_spec.get('x_col')
        y_col = chart_spec.get('y_col')
        
        if chart_type == 'bar' and x_col and y_col:
            plt.bar(data[x_col], data[y_col])
        elif chart_type == 'line' and x_col and y_col:
            plt.plot(data[x_col], data[y_col])
        elif chart_type == 'scatter' and x_col and y_col:
            plt.scatter(data[x_col], data[y_col])
        else:
            # Default: simple bar chart of first numeric column
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                data[numeric_cols[0]].plot(kind='bar')
        
        plt.title('Data Visualization')
        plt.tight_layout()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            plt.savefig(tmp.name, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # Upload to GCS
            client = storage.Client()
            bucket = client.bucket(GCS_BUCKET)
            blob_name = f"charts/{dataset_id}_chart.png"
            blob = bucket.blob(blob_name)
            
            with open(tmp.name, 'rb') as f:
                blob.upload_from_file(f, content_type='image/png')
            
            os.unlink(tmp.name)
            return f"gs://{GCS_BUCKET}/{blob_name}"
            
    except Exception as e:
        logger.error(f"Chart creation error: {e}")
        return None

@app.get("/")
async def root():
    return {"message": "Botlytics API", "status": "running", "version": "1.0.0"}

@app.get("/api/v1/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {"status": "healthy", "checks": {}}
    
    # Check GCS connectivity
    try:
        if GCS_BUCKET:
            client = storage.Client()
            bucket = client.bucket(GCS_BUCKET)
            bucket.exists()
            health_status["checks"]["gcs"] = "ok"
        else:
            health_status["checks"]["gcs"] = "not_configured"
    except Exception as e:
        health_status["checks"]["gcs"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Vertex AI
    try:
        if GCP_PROJECT_ID:
            # Simple test call
            model = GenerativeModel("gemini-1.5-pro")
            health_status["checks"]["vertex_ai"] = "ok"
        else:
            health_status["checks"]["vertex_ai"] = "not_configured"
    except Exception as e:
        health_status["checks"]["vertex_ai"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check BigQuery (if configured)
    try:
        if GCP_PROJECT_ID:
            bq_client = bigquery.Client(project=GCP_PROJECT_ID)
            list(bq_client.list_datasets(max_results=1))
            health_status["checks"]["bigquery"] = "ok"
        else:
            health_status["checks"]["bigquery"] = "not_configured"
    except Exception as e:
        health_status["checks"]["bigquery"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(get_metrics(), media_type="text/plain")

@app.post("/api/v1/upload")
@monitor_endpoint
async def upload(file: UploadFile = File(...)):
    """Upload CSV file and return dataset_id"""
    try:
        dataset_id = str(uuid.uuid4())
        filename = f"{dataset_id}.csv"
        contents = await file.read()
        
        # Validate CSV
        try:
            df = pd.read_csv(io.BytesIO(contents))
            if df.empty:
                raise HTTPException(400, "CSV file is empty")
        except Exception as e:
            raise HTTPException(400, f"Invalid CSV file: {str(e)}")
        
        # Upload to GCS
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(filename)
        blob.upload_from_string(contents, content_type='text/csv')
        
        return {
            "dataset_id": dataset_id,
            "gcs_path": f"gs://{GCS_BUCKET}/{filename}",
            "columns": df.columns.tolist(),
            "rows": len(df),
            "preview": df.head().to_dict('records')
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(500, f"Upload failed: {str(e)}")

class QueryRequest(BaseModel):
    dataset_id: str = None
    bigquery_table: str = None
    query: str
    chart_type: str = None

def build_prompt_for_bigquery(table_schema: dict, user_query: str) -> str:
    """Build prompt for BigQuery SQL generation"""
    columns = list(table_schema.keys())
    column_info = ", ".join([f"{col} ({dtype})" for col, dtype in table_schema.items()])
    
    return f"""
You are an expert data analyst. Generate a BigQuery SQL query for the user's question.
Respond with a JSON object with these keys:
- "mode": "sql"
- "code": valid BigQuery SQL query
- "chart": {{"type": "line|bar|pie|scatter|table", "x": "<column>", "y": "<column>", "title": "<title>"}}
- "explain": short description (<= 60 words) of what the query returns

Table schema: {column_info}
Question: {user_query}

CRITICAL RULES for SQL generation:
- Use only SELECT statements (no INSERT, UPDATE, DELETE, DROP, CREATE)
- Reference only columns that exist in the schema
- Use proper BigQuery syntax and functions
- Limit results with LIMIT clause (max 1000 rows)
- Use appropriate aggregations and GROUP BY when needed
- For time series, use DATE functions properly

Example response:
{{
    "mode": "sql",
    "code": "SELECT product, SUM(sales) as total_sales FROM `project.dataset.table` GROUP BY product ORDER BY total_sales DESC LIMIT 10",
    "chart": {{"type": "bar", "x": "product", "y": "total_sales", "title": "Top Products by Sales"}},
    "explain": "Shows top 10 products ranked by total sales amount."
}}
"""

def sanitize_sql_query(sql: str, allowed_tables: list) -> str:
    """Sanitize BigQuery SQL with strict validation"""
    import re
    
    sql_upper = sql.upper()
    
    # Only allow SELECT statements
    if not sql_upper.strip().startswith('SELECT'):
        raise HTTPException(400, "Only SELECT queries are allowed")
    
    # Dangerous SQL patterns
    dangerous_patterns = [
        'DROP', 'DELETE', 'INSERT', 'UPDATE', 'CREATE', 'ALTER', 'TRUNCATE',
        'EXEC', 'EXECUTE', 'DECLARE', 'CURSOR', 'PROCEDURE', 'FUNCTION',
        'GRANT', 'REVOKE', 'COMMIT', 'ROLLBACK', 'SAVEPOINT',
        'INFORMATION_SCHEMA', 'SYSTEM', 'ADMIN', 'USER', 'PASSWORD'
    ]
    
    for pattern in dangerous_patterns:
        if pattern in sql_upper:
            raise HTTPException(400, f"SQL operation '{pattern}' not allowed")
    
    # Ensure LIMIT clause exists and is reasonable
    if 'LIMIT' not in sql_upper:
        sql += ' LIMIT 1000'
    else:
        # Extract limit value and validate
        limit_match = re.search(r'LIMIT\s+(\d+)', sql_upper)
        if limit_match:
            limit_val = int(limit_match.group(1))
            if limit_val > 10000:
                sql = re.sub(r'LIMIT\s+\d+', 'LIMIT 1000', sql, flags=re.IGNORECASE)
    
    # Validate table references
    for table in allowed_tables:
        if table not in sql:
            logger.warning(f"Query doesn't reference expected table: {table}")
    
    return sql

@monitor_code_execution("bigquery")
def run_bigquery_safe(sql: str, project_id: str) -> dict:
    """Execute BigQuery SQL safely"""
    try:
        client = bigquery.Client(project=project_id)
        
        # Configure job with limits
        job_config = bigquery.QueryJobConfig(
            maximum_bytes_billed=100 * 1024 * 1024,  # 100MB limit
            use_query_cache=True,
            dry_run=False
        )
        
        # Run query
        query_job = client.query(sql, job_config=job_config)
        results = query_job.result(timeout=60)  # 60 second timeout
        
        # Convert to DataFrame for consistent handling
        df = results.to_dataframe()
        
        # Limit result size
        if len(df) > 1000:
            df = df.head(1000)
        
        return {
            "success": True,
            "result": df,
            "preview": df.head(20).to_dict('records'),
            "summary": {
                'type': 'DataFrame',
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'bytes_processed': query_job.total_bytes_processed,
                'bytes_billed': query_job.total_bytes_billed
            }
        }
        
    except Exception as e:
        logger.error(f"BigQuery execution error: {e}")
        return {
            "success": False,
            "error": str(e),
            "preview": "BigQuery execution failed"
        }

@app.post("/api/v1/query")
@monitor_endpoint
async def query_endpoint(body: QueryRequest):
    """Process natural language query using two-step LLM pattern"""
    if not body.dataset_id and not body.bigquery_table:
        raise HTTPException(400, "Provide dataset_id or bigquery_table")
    
    try:
        # STEP A: Intent & Plan Generation
        if body.dataset_id:
            # CSV/Pandas flow
            filename = f"{body.dataset_id}.csv"
            client = storage.Client()
            bucket = client.bucket(GCS_BUCKET)
            blob = bucket.blob(filename)
            
            if not blob.exists():
                raise HTTPException(404, "Dataset not found")
            
            csv_bytes = blob.download_as_bytes()
            df = pd.read_csv(io.BytesIO(csv_bytes))
            
            # Generate execution plan
            prompt = build_prompt_for_pandas(df.columns.tolist(), body.query)
            llm_response = call_gemini(prompt)
            
            # Parse plan
            plan = parse_llm_response(llm_response)
            
            # Validate and sanitize
            if plan.get('mode') != 'pandas':
                raise HTTPException(400, "Invalid analysis mode")
            
            safe_code = sanitize_pandas_code(plan['code'], allowed_columns=df.columns.tolist())
            
            # Execute code
            exec_result = run_pandas_code_safe(df, safe_code)
            
            if not exec_result['success']:
                raise HTTPException(500, f"Code execution failed: {exec_result.get('error')}")
            
            execution_mode = "pandas"
            data_result = exec_result['result']
            
        elif body.bigquery_table:
            # BigQuery flow
            # Get table schema first
            bq_client = bigquery.Client(project=GCP_PROJECT_ID)
            table_ref = bq_client.get_table(body.bigquery_table)
            schema = {field.name: field.field_type for field in table_ref.schema}
            
            # Generate SQL plan
            prompt = build_prompt_for_bigquery(schema, body.query)
            llm_response = call_gemini(prompt)
            
            # Parse plan
            plan = parse_llm_response(llm_response)
            
            # Validate and sanitize
            if plan.get('mode') != 'sql':
                raise HTTPException(400, "Invalid analysis mode for BigQuery")
            
            safe_sql = sanitize_sql_query(plan['code'], [body.bigquery_table])
            
            # Execute SQL
            exec_result = run_bigquery_safe(safe_sql, GCP_PROJECT_ID)
            
            if not exec_result['success']:
                raise HTTPException(500, f"BigQuery execution failed: {exec_result.get('error')}")
            
            execution_mode = "bigquery"
            data_result = exec_result['result']
        
        # Create visualization
        chart_url = None
        chart_spec = plan.get('chart')
        if chart_spec and data_result is not None:
            dataset_ref = body.dataset_id or body.bigquery_table.replace('.', '_')
            chart_url = create_chart_and_upload(data_result, chart_spec, dataset_ref)
        
        # STEP B: Final Explanation & Summary
        summary_prompt = f"""
        Based on the data analysis results below, provide a clear business summary.
        
        Original question: "{body.query}"
        Analysis method: {execution_mode}
        Data preview: {exec_result['preview']}
        Chart created: {chart_spec.get('title', 'N/A') if chart_spec else 'None'}
        
        Provide:
        1. A 2-3 sentence summary of key findings
        2. One actionable insight or recommendation
        3. Any data quality observations if relevant
        
        Keep response concise and business-focused.
        """
        
        final_summary = call_gemini(summary_prompt)
        
        # Prepare response
        response = {
            "success": True,
            "summary": final_summary,
            "chart_url": chart_url,
            "chart_spec": chart_spec,
            "table_preview": exec_result['preview'],
            "explanation": plan.get('explain', ''),
            "execution_mode": execution_mode,
            "data_summary": exec_result.get('summary', {}),
            "code_executed": safe_code if execution_mode == "pandas" else safe_sql
        }
        
        # Add BigQuery specific metrics
        if execution_mode == "bigquery" and 'bytes_processed' in exec_result.get('summary', {}):
            response['bigquery_metrics'] = {
                'bytes_processed': exec_result['summary']['bytes_processed'],
                'bytes_billed': exec_result['summary']['bytes_billed']
            }
        
        return response
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(500, f"Query failed: {str(e)}")

def parse_llm_response(llm_response: str) -> dict:
    """Parse LLM response with fallback strategies"""
    try:
        # Try direct JSON parsing
        return json.loads(llm_response.strip())
    except json.JSONDecodeError:
        # Fallback: extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Last resort: try to extract key components
        logger.warning(f"Failed to parse LLM response as JSON: {llm_response[:200]}...")
        
        # Extract code block if present
        code_match = re.search(r'```(?:python|sql)?\n?(.*?)\n?```', llm_response, re.DOTALL)
        code = code_match.group(1) if code_match else ""
        
        # Basic fallback structure
        return {
            "mode": "pandas" if "df" in llm_response else "sql",
            "code": code or "result = df.head(10)",  # Safe fallback
            "chart": {"type": "table", "title": "Data Preview"},
            "explain": "Analysis of the provided data."
        }

@app.get("/api/v1/datasets/{dataset_id}")
async def get_dataset_info(dataset_id: str):
    """Get information about an uploaded dataset"""
    try:
        filename = f"{dataset_id}.csv"
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(filename)
        
        if not blob.exists():
            raise HTTPException(404, "Dataset not found")
        
        # Load and analyze dataset
        csv_bytes = blob.download_as_bytes()
        df = pd.read_csv(io.BytesIO(csv_bytes))
        
        return {
            "dataset_id": dataset_id,
            "columns": df.columns.tolist(),
            "rows": len(df),
            "dtypes": df.dtypes.to_dict(),
            "preview": df.head().to_dict('records'),
            "summary": df.describe().to_dict() if len(df.select_dtypes(include=['number']).columns) > 0 else {}
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset info error: {e}")
        raise HTTPException(500, f"Failed to get dataset info: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
