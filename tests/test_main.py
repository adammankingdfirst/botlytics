import pytest
import pandas as pd
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
import io

# Import the app
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from main import app, sanitize_pandas_code, run_pandas_code_safe, parse_llm_response

client = TestClient(app)

class TestAPI:
    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "running"
    
    def test_health_check(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert "status" in response.json()
    
    @patch('main.storage.Client')
    def test_upload_csv(self, mock_storage):
        # Mock GCS client
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_storage.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        # Create test CSV
        csv_content = "name,age,city\nJohn,25,NYC\nJane,30,LA"
        
        response = client.post(
            "/api/v1/upload",
            files={"file": ("test.csv", csv_content, "text/csv")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "dataset_id" in data
        assert "columns" in data
        assert data["columns"] == ["name", "age", "city"]

class TestCodeSanitization:
    def test_sanitize_safe_code(self):
        safe_code = "result = df.groupby('product')['sales'].sum().reset_index()"
        columns = ['product', 'sales', 'date']
        
        sanitized = sanitize_pandas_code(safe_code, columns)
        assert sanitized == safe_code
    
    def test_sanitize_dangerous_code(self):
        dangerous_codes = [
            "import os; result = df.head()",
            "exec('print(1)'); result = df.head()",
            "eval('1+1'); result = df.head()",
            "open('file.txt'); result = df.head()",
            "__import__('os'); result = df.head()"
        ]
        
        columns = ['product', 'sales']
        
        for code in dangerous_codes:
            with pytest.raises(Exception):
                sanitize_pandas_code(code, columns)
    
    def test_sanitize_missing_result(self):
        code = "df.groupby('product')['sales'].sum()"
        columns = ['product', 'sales']
        
        with pytest.raises(Exception):
            sanitize_pandas_code(code, columns)
    
    def test_sanitize_disallowed_methods(self):
        code = "result = df.to_sql('table', connection)"
        columns = ['product', 'sales']
        
        with pytest.raises(Exception):
            sanitize_pandas_code(code, columns)

class TestCodeExecution:
    def test_safe_execution(self):
        df = pd.DataFrame({
            'product': ['A', 'B', 'A', 'B'],
            'sales': [100, 200, 150, 250]
        })
        
        code = "result = df.groupby('product')['sales'].sum().reset_index()"
        
        result = run_pandas_code_safe(df, code)
        
        assert result['success'] == True
        assert 'result' in result
        assert 'preview' in result
    
    def test_execution_with_error(self):
        df = pd.DataFrame({'a': [1, 2, 3]})
        
        # Code that will fail
        code = "result = df['nonexistent_column'].sum()"
        
        result = run_pandas_code_safe(df, code)
        
        assert result['success'] == False
        assert 'error' in result

class TestLLMResponseParsing:
    def test_parse_valid_json(self):
        response = '''
        {
            "mode": "pandas",
            "code": "result = df.head()",
            "chart": {"type": "bar", "x": "col1", "y": "col2", "title": "Test"},
            "explain": "Shows data preview"
        }
        '''
        
        parsed = parse_llm_response(response)
        
        assert parsed['mode'] == 'pandas'
        assert 'code' in parsed
        assert 'chart' in parsed
    
    def test_parse_json_with_extra_text(self):
        response = '''
        Here's the analysis:
        
        {
            "mode": "pandas",
            "code": "result = df.head()",
            "chart": {"type": "table", "title": "Data"},
            "explain": "Preview"
        }
        
        This should work well.
        '''
        
        parsed = parse_llm_response(response)
        
        assert parsed['mode'] == 'pandas'
        assert 'code' in parsed
    
    def test_parse_invalid_json(self):
        response = "This is not JSON at all, just text about data analysis."
        
        parsed = parse_llm_response(response)
        
        # Should return fallback structure
        assert 'mode' in parsed
        assert 'code' in parsed
        assert 'chart' in parsed

class TestIntegration:
    @patch('main.call_gemini')
    @patch('main.storage.Client')
    def test_query_endpoint_pandas(self, mock_storage, mock_gemini):
        # Mock storage
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_storage.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.exists.return_value = True
        
        # Mock CSV data
        csv_data = "product,sales\nA,100\nB,200"
        mock_blob.download_as_bytes.return_value = csv_data.encode()
        
        # Mock Gemini response
        mock_gemini.return_value = json.dumps({
            "mode": "pandas",
            "code": "result = df.groupby('product')['sales'].sum().reset_index()",
            "chart": {"type": "bar", "x": "product", "y": "sales", "title": "Sales by Product"},
            "explain": "Shows total sales by product"
        })
        
        # Test query
        response = client.post("/api/v1/query", json={
            "dataset_id": "test-123",
            "query": "Show sales by product"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] == True
        assert 'summary' in data
        assert 'table_preview' in data

if __name__ == "__main__":
    pytest.main([__file__])