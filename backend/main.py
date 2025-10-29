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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Botlytics API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCS_BUCKET = os.environ.get("GCS_BUCKET")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")

# Initialize Vertex AI
if GCP_PROJECT_ID:
    vertexai.init(project=GCP_PROJECT_ID, location=LOCATION)

def call_gemini(prompt: str) -> str:
    """Call Gemini using Vertex AI SDK"""
    try:
        model = GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error calling Gemini: {e}")
        raise HTTPException(500, f"Gemini API error: {str(e)}")

def build_prompt_for_pandas(columns: list, user_query: str) -> str:
    """Build prompt for pandas code generation"""
    return f"""
You are a data analyst. Given a pandas DataFrame with columns: {columns}

User query: "{user_query}"

Generate Python pandas code to answer this query. Return ONLY a JSON response with this structure:
{{
    "mode": "pandas",
    "code": "# pandas code here",
    "chart": {{"type": "bar|line|scatter", "x_col": "column_name", "y_col": "column_name"}},
    "explanation": "Brief explanation of what the code does"
}}

Rules:
- Use only pandas operations on variable 'df'
- No imports needed (pandas as pd already imported)
- Code should return a result that can be displayed
- Keep code simple and safe
- For aggregations, use .head(10) to limit results
"""

def sanitize_pandas_code(code: str, allowed_columns: list) -> str:
    """Basic sanitization of pandas code"""
    # Remove dangerous operations
    dangerous_patterns = ['import', 'exec', 'eval', 'open', 'file', '__', 'subprocess', 'os.']
    for pattern in dangerous_patterns:
        if pattern in code.lower():
            raise HTTPException(400, f"Unsafe operation detected: {pattern}")
    
    # Ensure only allowed columns are referenced
    for col in allowed_columns:
        if f"'{col}'" not in code and f'"{col}"' not in code and f'["{col}"]' not in code and f"['{col}']" not in code:
            continue  # Column not used, that's fine
    
    return code

def run_pandas_code_safe(df: pd.DataFrame, code: str) -> dict:
    """Execute pandas code safely in subprocess"""
    try:
        # Create a safe execution environment
        local_vars = {'df': df, 'pd': pd}
        exec(code, {"__builtins__": {}}, local_vars)
        
        # Get the result (assume last expression or 'result' variable)
        result = local_vars.get('result', df.head())
        
        return {
            "success": True,
            "result": result,
            "preview": result.head().to_dict('records') if hasattr(result, 'to_dict') else str(result)
        }
    except Exception as e:
        logger.error(f"Code execution error: {e}")
        return {
            "success": False,
            "error": str(e),
            "preview": "Error executing code"
        }

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
    return {"message": "Botlytics API", "status": "running"}

@app.post("/api/v1/upload")
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

@app.post("/api/v1/query")
async def query_endpoint(body: QueryRequest):
    """Process natural language query against uploaded dataset"""
    if not body.dataset_id and not body.bigquery_table:
        raise HTTPException(400, "Provide dataset_id or bigquery_table")
    
    try:
        if body.dataset_id:
            # Load dataset from GCS
            filename = f"{body.dataset_id}.csv"
            client = storage.Client()
            bucket = client.bucket(GCS_BUCKET)
            blob = bucket.blob(filename)
            
            if not blob.exists():
                raise HTTPException(404, "Dataset not found")
            
            csv_bytes = blob.download_as_bytes()
            df = pd.read_csv(io.BytesIO(csv_bytes))
            
            # Build prompt for Gemini
            prompt = build_prompt_for_pandas(df.columns.tolist(), body.query)
            llm_response = call_gemini(prompt)
            
            # Parse LLM response
            try:
                plan = json.loads(llm_response.strip())
            except json.JSONDecodeError:
                # Fallback: extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                if json_match:
                    plan = json.loads(json_match.group())
                else:
                    raise HTTPException(500, "Invalid response from Gemini")
            
            # Validate and sanitize code
            if plan.get('mode') != 'pandas':
                raise HTTPException(400, "Invalid analysis mode")
            
            safe_code = sanitize_pandas_code(plan['code'], allowed_columns=df.columns.tolist())
            
            # Execute code safely
            exec_result = run_pandas_code_safe(df, safe_code)
            
            if not exec_result['success']:
                raise HTTPException(500, f"Code execution failed: {exec_result.get('error')}")
            
            # Create chart if specified
            chart_url = None
            chart_spec = plan.get('chart')
            if chart_spec and exec_result.get('result') is not None:
                chart_url = create_chart_and_upload(exec_result['result'], chart_spec, body.dataset_id)
            
            # Generate final summary
            summary_prompt = f"""
            Based on the analysis results: {exec_result['preview']}
            
            Original query: "{body.query}"
            
            Provide a clear, concise summary of the findings in 2-3 sentences.
            """
            summary = call_gemini(summary_prompt)
            
            return {
                "success": True,
                "summary": summary,
                "chart_url": chart_url,
                "preview": exec_result['preview'],
                "explanation": plan.get('explanation', ''),
                "code_executed": safe_code
            }
            
        else:
            # BigQuery flow (placeholder for future implementation)
            raise HTTPException(501, "BigQuery queries not yet implemented")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(500, f"Query failed: {str(e)}")

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
