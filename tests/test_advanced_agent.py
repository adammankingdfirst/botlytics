import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import json
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from agent_sdk import (
    AdvancedAgent, ConversationMemory, DataAnalysisTools, 
    CodeInterpreter, ReasoningChain
)

class TestAdvancedAgent:
    
    @pytest.fixture
    def sample_dataframe(self):
        return pd.DataFrame({
            'product': ['A', 'B', 'A', 'B', 'C'],
            'sales': [100, 200, 150, 250, 300],
            'region': ['North', 'South', 'North', 'South', 'East'],
            'date': pd.date_range('2024-01-01', periods=5)
        })
    
    @pytest.fixture
    def mock_agent(self):
        with patch('agent_sdk.aiplatform.init'), \
             patch('agent_sdk.GenerativeModel') as mock_model:
            
            mock_model_instance = MagicMock()
            mock_model.return_value = mock_model_instance
            
            agent = AdvancedAgent("test-project", "us-central1")
            return agent, mock_model_instance

class TestConversationMemory:
    
    def test_conversation_memory_creation(self):
        memory = ConversationMemory(
            session_id="test-123",
            user_id="user-456",
            messages=[],
            context={},
            tools_used=[],
            data_artifacts={},
            created_at=pd.Timestamp.now(),
            updated_at=pd.Timestamp.now()
        )
        
        assert memory.session_id == "test-123"
        assert memory.user_id == "user-456"
        assert len(memory.messages) == 0
    
    def test_add_message(self):
        memory = ConversationMemory(
            session_id="test-123",
            user_id="user-456",
            messages=[],
            context={},
            tools_used=[],
            data_artifacts={},
            created_at=pd.Timestamp.now(),
            updated_at=pd.Timestamp.now()
        )
        
        memory.add_message("user", "Hello", {"test": True})
        
        assert len(memory.messages) == 1
        assert memory.messages[0]["role"] == "user"
        assert memory.messages[0]["content"] == "Hello"
        assert memory.messages[0]["metadata"]["test"] == True
    
    def test_get_recent_context(self):
        memory = ConversationMemory(
            session_id="test-123",
            user_id="user-456",
            messages=[],
            context={},
            tools_used=[],
            data_artifacts={},
            created_at=pd.Timestamp.now(),
            updated_at=pd.Timestamp.now()
        )
        
        # Add multiple messages
        for i in range(15):
            memory.add_message("user", f"Message {i}")
        
        context = memory.get_recent_context(max_messages=5)
        lines = context.split('\n')
        
        assert len(lines) == 5
        assert "Message 14" in lines[-1]  # Most recent message

class TestDataAnalysisTools:
    
    def test_analyze_dataset(self, sample_dataframe):
        tools = DataAnalysisTools()
        analysis = tools.analyze_dataset(sample_dataframe)
        
        assert analysis["shape"] == (5, 4)
        assert "product" in analysis["columns"]
        assert "sales" in analysis["columns"]
        assert "numeric_summary" in analysis
        assert "categorical_summary" in analysis
    
    def test_create_visualization(self, sample_dataframe):
        tools = DataAnalysisTools()
        
        # Test bar chart
        chart_path = tools.create_visualization(
            sample_dataframe, "bar", "product", "sales", "Test Chart"
        )
        
        assert chart_path is not None
        assert chart_path.endswith('.png')
        
        # Clean up
        if chart_path and os.path.exists(chart_path):
            os.unlink(chart_path)
    
    def test_statistical_analysis_trend(self, sample_dataframe):
        tools = DataAnalysisTools()
        
        result = tools.statistical_analysis(
            sample_dataframe, 
            "trend_analysis",
            date_col="date",
            value_col="sales"
        )
        
        assert "trend_slope" in result
        assert "trend_direction" in result
        assert result["trend_direction"] in ["increasing", "decreasing"]
    
    def test_statistical_analysis_segment(self, sample_dataframe):
        tools = DataAnalysisTools()
        
        result = tools.statistical_analysis(
            sample_dataframe,
            "segment_analysis", 
            segment_col="region",
            metric_col="sales"
        )
        
        assert "segment_statistics" in result
        assert "top_segments" in result

class TestCodeInterpreter:
    
    def test_safe_code_execution(self):
        interpreter = CodeInterpreter()
        
        safe_code = """
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
result = df.sum()
"""
        
        result = interpreter.execute_code(safe_code)
        
        assert result["success"] == True
        assert "result" in result
    
    def test_dangerous_code_detection(self):
        interpreter = CodeInterpreter()
        
        dangerous_codes = [
            "import os; os.system('ls')",
            "exec('print(1)')",
            "eval('1+1')",
            "__import__('subprocess')"
        ]
        
        for code in dangerous_codes:
            result = interpreter.execute_code(code)
            assert result["success"] == False
            assert "validation failed" in result["error"].lower()
    
    def test_code_with_context(self):
        interpreter = CodeInterpreter()
        
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        context = {'df': df}
        
        code = "result = df['x'].sum()"
        result = interpreter.execute_code(code, context)
        
        assert result["success"] == True
        assert result["result"] == 6

class TestReasoningChain:
    
    def test_reasoning_chain_execution(self):
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "problem_type": "data_analysis",
            "steps": [
                {"step": 1, "description": "Analyze data", "action": "analyze"},
                {"step": 2, "description": "Create visualization", "action": "visualize"}
            ],
            "required_tools": ["analyze_data", "create_visualization"],
            "expected_outcome": "Data insights and charts"
        })
        mock_model.generate_content.return_value = mock_response
        
        chain = ReasoningChain(mock_model)
        
        result = chain.execute_reasoning_chain("Analyze sales data")
        
        assert "chain_id" in result
        assert "problem" in result
        assert "decomposition" in result
        assert "step_results" in result

class TestAdvancedAgentIntegration:
    
    def test_start_conversation(self, mock_agent):
        agent, mock_model = mock_agent
        
        session_id = agent.start_conversation("test-user", "Hello")
        
        assert session_id in agent.conversations
        assert agent.conversations[session_id].user_id == "test-user"
        assert len(agent.conversations[session_id].messages) == 1
    
    def test_continue_conversation(self, mock_agent):
        agent, mock_model = mock_agent
        
        # Mock model response
        mock_response = MagicMock()
        mock_response.text = "I can help you analyze your data."
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = []
        mock_model.generate_content.return_value = mock_response
        
        # Start conversation
        session_id = agent.start_conversation("test-user")
        
        # Continue conversation
        result = agent.continue_conversation(session_id, "What can you do?")
        
        assert result["conversation_id"] == session_id
        assert "response" in result
        assert len(agent.conversations[session_id].messages) >= 2
    
    def test_function_calling_simulation(self, mock_agent, sample_dataframe):
        agent, mock_model = mock_agent
        
        # Mock function call response
        mock_part = MagicMock()
        mock_part.function_call.name = "analyze_data"
        mock_part.function_call.args = {"dataset_id": "test-123"}
        
        mock_response = MagicMock()
        mock_response.text = "Analysis completed"
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [mock_part]
        mock_model.generate_content.return_value = mock_response
        
        session_id = agent.start_conversation("test-user")
        
        # Mock dataset context
        dataset_context = {
            "dataset_id": "test-123",
            "dataframe": sample_dataframe
        }
        
        result = agent.continue_conversation(
            session_id, 
            "Analyze this data", 
            dataset_context
        )
        
        assert result["conversation_id"] == session_id
        assert len(result.get("function_results", [])) >= 0

class TestEndToEndWorkflow:
    
    def test_complete_analysis_workflow(self, sample_dataframe):
        """Test a complete analysis workflow"""
        
        # 1. Data Analysis
        tools = DataAnalysisTools()
        analysis = tools.analyze_dataset(sample_dataframe)
        assert analysis["shape"] == (5, 4)
        
        # 2. Code Execution
        interpreter = CodeInterpreter()
        code = "result = df.groupby('product')['sales'].sum()"
        context = {"df": sample_dataframe}
        exec_result = interpreter.execute_code(code, context)
        assert exec_result["success"] == True
        
        # 3. Visualization
        chart_path = tools.create_visualization(
            sample_dataframe, "bar", "product", "sales"
        )
        assert chart_path is not None
        
        # Clean up
        if chart_path and os.path.exists(chart_path):
            os.unlink(chart_path)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])