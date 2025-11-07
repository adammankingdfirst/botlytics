"""
Advanced AI Agent implementation using Google's Agent SDK
Implements: Tool calling, Multi-turn conversations, Data analysis, Code interpreter, Reasoning chains
"""

import logging
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from google.cloud import aiplatform
from google.cloud.aiplatform import gapic
from google.cloud import texttospeech
from google.cloud import speech
from vertexai.generative_models import GenerativeModel, Tool, FunctionDeclaration
import vertexai.preview.generative_models as generative_models
import base64
import io
import wave

logger = logging.getLogger(__name__)

class AccessibilityTools:
    """Accessibility tools for text-to-speech, speech-to-text, and other accessibility features"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        
        # Initialize Google Cloud TTS and STT clients
        try:
            self.tts_client = texttospeech.TextToSpeechClient()
            self.stt_client = speech.SpeechClient()
        except Exception as e:
            logger.warning(f"Failed to initialize accessibility clients: {e}")
            self.tts_client = None
            self.stt_client = None
    
    def text_to_speech(self, text: str, language_code: str = "en-US", 
                      voice_name: str = None, speaking_rate: float = 1.0,
                      pitch: float = 0.0, volume_gain_db: float = 0.0) -> Dict[str, Any]:
        """Convert text to speech with customizable voice options"""
        if not self.tts_client:
            return {"error": "Text-to-speech client not available"}
        
        try:
            # Prepare synthesis input
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Configure voice parameters
            if not voice_name:
                # Default to neural voices for better quality
                voice_name = "en-US-Neural2-F" if language_code.startswith("en") else None
            
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name,
                ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
            )
            
            # Configure audio output
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=speaking_rate,
                pitch=pitch,
                volume_gain_db=volume_gain_db,
                effects_profile_id=["telephony-class-application"]  # Enhanced for clarity
            )
            
            # Perform text-to-speech synthesis
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            # Encode audio content as base64 for web delivery
            audio_base64 = base64.b64encode(response.audio_content).decode('utf-8')
            
            return {
                "success": True,
                "audio_base64": audio_base64,
                "audio_format": "mp3",
                "text": text,
                "language": language_code,
                "voice": voice_name,
                "duration_estimate": len(text) / 150 * 60,  # Rough estimate in seconds
                "accessibility_features": {
                    "speaking_rate": speaking_rate,
                    "pitch": pitch,
                    "volume": volume_gain_db
                }
            }
            
        except Exception as e:
            logger.error(f"Text-to-speech error: {e}")
            return {"error": f"TTS failed: {str(e)}"}
    
    def speech_to_text(self, audio_base64: str, language_code: str = "en-US",
                      enable_automatic_punctuation: bool = True,
                      enable_word_time_offsets: bool = True) -> Dict[str, Any]:
        """Convert speech to text with enhanced accuracy features"""
        if not self.stt_client:
            return {"error": "Speech-to-text client not available"}
        
        try:
            # Decode base64 audio
            audio_content = base64.b64decode(audio_base64)
            
            # Configure recognition
            audio = speech.RecognitionAudio(content=audio_content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
                sample_rate_hertz=48000,
                language_code=language_code,
                enable_automatic_punctuation=enable_automatic_punctuation,
                enable_word_time_offsets=enable_word_time_offsets,
                model="latest_long",  # Best accuracy for longer audio
                use_enhanced=True,    # Enhanced model for better accuracy
                profanity_filter=False,
                speech_contexts=[
                    speech.SpeechContext(
                        phrases=["data analysis", "visualization", "pandas", "chart", "graph", "statistics"]
                    )
                ]
            )
            
            # Perform speech recognition
            response = self.stt_client.recognize(config=config, audio=audio)
            
            # Process results
            transcripts = []
            word_info = []
            
            for result in response.results:
                alternative = result.alternatives[0]
                transcripts.append({
                    "transcript": alternative.transcript,
                    "confidence": alternative.confidence
                })
                
                # Extract word-level timing information
                if enable_word_time_offsets:
                    for word in alternative.words:
                        word_info.append({
                            "word": word.word,
                            "start_time": word.start_time.total_seconds(),
                            "end_time": word.end_time.total_seconds(),
                            "confidence": getattr(word, 'confidence', 0.0)
                        })
            
            # Combine all transcripts
            full_transcript = " ".join([t["transcript"] for t in transcripts])
            avg_confidence = np.mean([t["confidence"] for t in transcripts]) if transcripts else 0.0
            
            return {
                "success": True,
                "transcript": full_transcript,
                "confidence": avg_confidence,
                "language": language_code,
                "transcripts": transcripts,
                "word_info": word_info,
                "accessibility_features": {
                    "automatic_punctuation": enable_automatic_punctuation,
                    "word_timing": enable_word_time_offsets,
                    "enhanced_model": True
                }
            }
            
        except Exception as e:
            logger.error(f"Speech-to-text error: {e}")
            return {"error": f"STT failed: {str(e)}"}
    
    def generate_audio_description(self, chart_data: Dict[str, Any], 
                                 chart_type: str) -> str:
        """Generate detailed audio descriptions for charts and visualizations"""
        try:
            if chart_type == "bar":
                description = self._describe_bar_chart(chart_data)
            elif chart_type == "line":
                description = self._describe_line_chart(chart_data)
            elif chart_type == "scatter":
                description = self._describe_scatter_plot(chart_data)
            elif chart_type == "pie":
                description = self._describe_pie_chart(chart_data)
            else:
                description = f"This is a {chart_type} chart showing data visualization."
            
            # Add accessibility context
            description += " This chart is part of your data analysis results. "
            description += "You can ask me to explain specific parts in more detail or create alternative visualizations."
            
            return description
            
        except Exception as e:
            logger.error(f"Audio description error: {e}")
            return "Chart visualization generated. Please ask me to describe specific aspects you'd like to know about."
    
    def _describe_bar_chart(self, data: Dict[str, Any]) -> str:
        """Generate description for bar charts"""
        try:
            # Extract key information
            title = data.get("title", "Bar chart")
            x_label = data.get("x_label", "categories")
            y_label = data.get("y_label", "values")
            
            description = f"{title}. This bar chart shows {y_label} across different {x_label}. "
            
            # Describe data points if available
            if "data_points" in data:
                points = data["data_points"]
                if len(points) <= 5:
                    for point in points:
                        description += f"{point['x']} has a value of {point['y']}. "
                else:
                    # Summarize for many data points
                    max_point = max(points, key=lambda p: p['y'])
                    min_point = min(points, key=lambda p: p['y'])
                    description += f"The highest value is {max_point['y']} for {max_point['x']}. "
                    description += f"The lowest value is {min_point['y']} for {min_point['x']}. "
                    description += f"There are {len(points)} total categories shown."
            
            return description
            
        except Exception as e:
            return "Bar chart showing categorical data with corresponding values."
    
    def _describe_line_chart(self, data: Dict[str, Any]) -> str:
        """Generate description for line charts"""
        try:
            title = data.get("title", "Line chart")
            x_label = data.get("x_label", "time")
            y_label = data.get("y_label", "values")
            
            description = f"{title}. This line chart shows how {y_label} changes over {x_label}. "
            
            # Analyze trend if data points available
            if "data_points" in data:
                points = data["data_points"]
                if len(points) >= 2:
                    start_val = points[0]['y']
                    end_val = points[-1]['y']
                    
                    if end_val > start_val:
                        trend = "increasing"
                    elif end_val < start_val:
                        trend = "decreasing"
                    else:
                        trend = "stable"
                    
                    description += f"The overall trend is {trend}, starting at {start_val} and ending at {end_val}. "
                    
                    # Find peaks and valleys
                    max_point = max(points, key=lambda p: p['y'])
                    min_point = min(points, key=lambda p: p['y'])
                    description += f"The peak value is {max_point['y']} and the lowest point is {min_point['y']}."
            
            return description
            
        except Exception as e:
            return "Line chart showing data trends over time or sequence."
    
    def _describe_scatter_plot(self, data: Dict[str, Any]) -> str:
        """Generate description for scatter plots"""
        try:
            title = data.get("title", "Scatter plot")
            x_label = data.get("x_label", "x-axis")
            y_label = data.get("y_label", "y-axis")
            
            description = f"{title}. This scatter plot shows the relationship between {x_label} and {y_label}. "
            
            # Analyze correlation if data available
            if "correlation" in data:
                corr = data["correlation"]
                if corr > 0.7:
                    description += "There is a strong positive correlation between the variables. "
                elif corr > 0.3:
                    description += "There is a moderate positive correlation between the variables. "
                elif corr < -0.7:
                    description += "There is a strong negative correlation between the variables. "
                elif corr < -0.3:
                    description += "There is a moderate negative correlation between the variables. "
                else:
                    description += "There is little to no correlation between the variables. "
            
            return description
            
        except Exception as e:
            return "Scatter plot showing the relationship between two variables."
    
    def _describe_pie_chart(self, data: Dict[str, Any]) -> str:
        """Generate description for pie charts"""
        try:
            title = data.get("title", "Pie chart")
            description = f"{title}. This pie chart shows the proportional breakdown of categories. "
            
            if "data_points" in data:
                points = data["data_points"]
                total = sum(point['value'] for point in points)
                
                # Describe largest segments
                sorted_points = sorted(points, key=lambda p: p['value'], reverse=True)
                for i, point in enumerate(sorted_points[:3]):  # Top 3 segments
                    percentage = (point['value'] / total) * 100
                    description += f"{point['label']} represents {percentage:.1f}% of the total. "
            
            return description
            
        except Exception as e:
            return "Pie chart showing proportional data distribution."
    
    def create_accessible_summary(self, analysis_result: Dict[str, Any]) -> Dict[str, str]:
        """Create accessible summaries for different user needs"""
        
        summaries = {}
        
        # Brief summary for screen readers
        summaries["screen_reader"] = self._create_screen_reader_summary(analysis_result)
        
        # Detailed summary for audio description
        summaries["audio_description"] = self._create_audio_description_summary(analysis_result)
        
        # Simple summary for cognitive accessibility
        summaries["simple_language"] = self._create_simple_summary(analysis_result)
        
        # Technical summary for advanced users
        summaries["technical"] = self._create_technical_summary(analysis_result)
        
        return summaries
    
    def _create_screen_reader_summary(self, result: Dict[str, Any]) -> str:
        """Create concise summary optimized for screen readers"""
        summary = "Data analysis complete. "
        
        if "shape" in result:
            rows, cols = result["shape"]
            summary += f"Dataset contains {rows} rows and {cols} columns. "
        
        if "key_findings" in result:
            summary += f"Key finding: {result['key_findings'][0]}. "
        
        summary += "Use navigation commands to explore detailed results."
        return summary
    
    def _create_audio_description_summary(self, result: Dict[str, Any]) -> str:
        """Create detailed audio description"""
        summary = "Here's a comprehensive overview of your data analysis results. "
        
        # Add detailed descriptions of findings
        if "numeric_summary" in result:
            summary += "The numerical analysis shows statistical measures for your data columns. "
        
        if "correlations" in result:
            summary += "Correlation analysis reveals relationships between different variables. "
        
        if "trends" in result:
            summary += "Trend analysis identifies patterns and changes over time. "
        
        summary += "You can ask me to explain any specific aspect in more detail, "
        summary += "create visualizations, or perform additional analysis."
        
        return summary
    
    def _create_simple_summary(self, result: Dict[str, Any]) -> str:
        """Create summary using simple language"""
        summary = "I looked at your data. "
        
        if "shape" in result:
            rows, cols = result["shape"]
            summary += f"Your data has {rows} rows of information and {cols} different types of information. "
        
        summary += "I found some interesting patterns. "
        summary += "You can ask me questions about what I found."
        
        return summary
    
    def _create_technical_summary(self, result: Dict[str, Any]) -> str:
        """Create technical summary for advanced users"""
        summary = "Analysis completed with the following technical details: "
        
        if "shape" in result:
            summary += f"Dataset dimensions: {result['shape']}. "
        
        if "dtypes" in result:
            summary += f"Data types: {len(result['dtypes'])} columns analyzed. "
        
        if "missing_values" in result:
            total_missing = sum(result["missing_values"].values())
            summary += f"Missing values: {total_missing} total across all columns. "
        
        summary += "Full statistical analysis and metadata available in detailed results."
        
        return summary


@dataclass
class ConversationMemory:
    """Memory structure for multi-turn conversations"""

    session_id: str
    user_id: str
    messages: List[Dict[str, Any]]
    context: Dict[str, Any]
    tools_used: List[str]
    data_artifacts: Dict[str, Any]  # Store datasets, charts, analysis results
    created_at: datetime
    updated_at: datetime

    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Add message to conversation history"""
        self.messages.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {},
            }
        )
        self.updated_at = datetime.now()

    def get_recent_context(self, max_messages: int = 10) -> str:
        """Get recent conversation context for the agent"""
        recent_messages = self.messages[-max_messages:]
        context_parts = []

        for msg in recent_messages:
            context_parts.append(f"{msg['role']}: {msg['content']}")

        return "\n".join(context_parts)


class DataAnalysisTools:
    """Built-in data analysis tools for the agent"""

    @staticmethod
    def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive dataset analysis"""
        analysis = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "numeric_summary": {},
            "categorical_summary": {},
            "correlations": {},
            "outliers": {},
        }

        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis["numeric_summary"] = df[numeric_cols].describe().to_dict()

            # Correlation matrix
            if len(numeric_cols) > 1:
                analysis["correlations"] = df[numeric_cols].corr().to_dict()

            # Outlier detection using IQR
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
                analysis["outliers"][col] = len(outliers)

        # Categorical columns analysis
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                analysis["categorical_summary"][col] = {
                    "unique_values": df[col].nunique(),
                    "top_values": df[col].value_counts().head().to_dict(),
                    "missing_count": df[col].isnull().sum(),
                }

        return analysis

    @staticmethod
    def create_visualization(
        df: pd.DataFrame,
        chart_type: str,
        x_col: str = None,
        y_col: str = None,
        title: str = None,
    ) -> str:
        """Create various types of visualizations"""
        try:
            plt.figure(figsize=(12, 8))

            if chart_type == "histogram" and x_col:
                plt.hist(df[x_col].dropna(), bins=30, alpha=0.7, edgecolor="black")
                plt.xlabel(x_col)
                plt.ylabel("Frequency")
                plt.title(title or f"Distribution of {x_col}")

            elif chart_type == "scatter" and x_col and y_col:
                plt.scatter(df[x_col], df[y_col], alpha=0.6)
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(title or f"{y_col} vs {x_col}")

            elif chart_type == "bar" and x_col and y_col:
                data_grouped = (
                    df.groupby(x_col)[y_col].sum().sort_values(ascending=False)
                )
                plt.bar(range(len(data_grouped)), data_grouped.values)
                plt.xticks(range(len(data_grouped)), data_grouped.index, rotation=45)
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(title or f"{y_col} by {x_col}")

            elif chart_type == "line" and x_col and y_col:
                df_sorted = df.sort_values(x_col)
                plt.plot(df_sorted[x_col], df_sorted[y_col], marker="o")
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(title or f"{y_col} over {x_col}")

            elif chart_type == "correlation_heatmap":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    corr_matrix = df[numeric_cols].corr()
                    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
                    plt.title(title or "Correlation Heatmap")

            plt.tight_layout()

            # Save plot
            chart_filename = f"/tmp/chart_{uuid.uuid4().hex}.png"
            plt.savefig(chart_filename, dpi=300, bbox_inches="tight")
            plt.close()

            return chart_filename

        except Exception as e:
            logger.error(f"Visualization error: {e}")
            return None

    @staticmethod
    def statistical_analysis(
        df: pd.DataFrame, analysis_type: str, **kwargs
    ) -> Dict[str, Any]:
        """Perform statistical analysis"""
        results = {}

        if (
            analysis_type == "trend_analysis"
            and "date_col" in kwargs
            and "value_col" in kwargs
        ):
            date_col, value_col = kwargs["date_col"], kwargs["value_col"]
            df[date_col] = pd.to_datetime(df[date_col])
            df_sorted = df.sort_values(date_col)

            # Calculate trend
            x = np.arange(len(df_sorted))
            y = df_sorted[value_col].values
            slope, intercept = np.polyfit(x, y, 1)

            results = {
                "trend_slope": slope,
                "trend_direction": "increasing" if slope > 0 else "decreasing",
                "r_squared": np.corrcoef(x, y)[0, 1] ** 2,
                "start_value": y[0],
                "end_value": y[-1],
                "total_change": y[-1] - y[0],
                "percent_change": ((y[-1] - y[0]) / y[0]) * 100 if y[0] != 0 else 0,
            }

        elif (
            analysis_type == "segment_analysis"
            and "segment_col" in kwargs
            and "metric_col" in kwargs
        ):
            segment_col, metric_col = kwargs["segment_col"], kwargs["metric_col"]

            segment_stats = (
                df.groupby(segment_col)[metric_col]
                .agg(["count", "mean", "median", "std", "min", "max", "sum"])
                .round(2)
            )

            results = {
                "segment_statistics": segment_stats.to_dict(),
                "top_segments": segment_stats.sort_values("sum", ascending=False)
                .head()
                .to_dict(),
                "segment_distribution": df[segment_col].value_counts().to_dict(),
            }

        return results


class CodeInterpreter:
    """Safe code interpreter with advanced execution capabilities"""

    def __init__(self):
        self.allowed_imports = {
            "pandas",
            "numpy",
            "matplotlib.pyplot",
            "seaborn",
            "plotly.express",
            "plotly.graph_objects",
            "scipy.stats",
            "sklearn",
            "datetime",
            "json",
            "math",
        }
        self.execution_history = []

    def execute_code(self, code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute code with enhanced safety and context"""
        import ast
        import sys
        from io import StringIO
        
        # Validate code length
        if len(code) > 10000:
            return {"success": False, "error": "Code too long (max 10000 characters)"}
        
        if not code.strip():
            return {"success": False, "error": "Empty code provided"}

        # Validate code safety
        try:
            tree = ast.parse(code)
            self._validate_ast(tree)
        except SyntaxError as e:
            return {"success": False, "error": f"Syntax error: {e}"}
        except Exception as e:
            return {"success": False, "error": f"Code validation failed: {e}"}

        # Prepare execution context
        safe_globals = {
            "__builtins__": {
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "min": min,
                "max": max,
                "sum": sum,
                "abs": abs,
                "round": round,
                "sorted": sorted,
                "enumerate": enumerate,
                "zip": zip,
                "range": range,
                "print": print,
                "type": type,
                "isinstance": isinstance,
            },
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns,
            "px": px,
            "go": go,
        }

        if context:
            safe_globals.update(context)

        # Capture output
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            # Execute code
            local_vars = {}
            exec(code, safe_globals, local_vars)

            # Get output
            output = captured_output.getvalue()

            # Store execution history
            self.execution_history.append(
                {
                    "code": code,
                    "timestamp": datetime.now().isoformat(),
                    "success": True,
                    "output": output,
                }
            )

            return {
                "success": True,
                "output": output,
                "variables": {
                    k: str(v) for k, v in local_vars.items() if not k.startswith("_")
                },
                "result": local_vars.get("result"),
            }

        except Exception as e:
            error_msg = str(e)
            self.execution_history.append(
                {
                    "code": code,
                    "timestamp": datetime.now().isoformat(),
                    "success": False,
                    "error": error_msg,
                }
            )

            return {"success": False, "error": error_msg}

        finally:
            sys.stdout = old_stdout

    def _validate_ast(self, tree):
        """Validate AST for dangerous operations"""
        # Check for imports (not allowed)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise ValueError("Import statements not allowed in code execution")
        
        # Check for dangerous function calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    dangerous_funcs = [
                        "exec", "eval", "compile", "__import__", 
                        "open", "input", "breakpoint", "exit", "quit"
                    ]
                    if node.func.id in dangerous_funcs:
                        raise ValueError(f"Dangerous function call: {node.func.id}")
                elif isinstance(node.func, ast.Attribute):
                    dangerous_methods = [
                        "system", "popen", "subprocess", "execfile",
                        "read_csv", "to_csv", "to_sql", "read_sql"
                    ]
                    if node.func.attr in dangerous_methods:
                        raise ValueError(f"Dangerous method call: {node.func.attr}")


class ReasoningChain:
    """Advanced reasoning chains for complex problem solving"""

    def __init__(self, agent_model: GenerativeModel):
        self.model = agent_model
        self.chain_history = []

    def execute_reasoning_chain(
        self, problem: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute a multi-step reasoning chain"""

        # Step 1: Problem decomposition
        decomposition_prompt = f"""
        Break down this complex problem into smaller, manageable steps:
        Problem: {problem}
        Context: {json.dumps(context or {}, indent=2)}
        
        Provide a JSON response with:
        {{
            "problem_type": "data_analysis|visualization|statistical_analysis|prediction",
            "steps": [
                {{"step": 1, "description": "...", "action": "analyze|visualize|calculate|interpret"}},
                ...
            ],
            "required_tools": ["tool1", "tool2", ...],
            "expected_outcome": "..."
        }}
        """

        decomposition = self._call_model(decomposition_prompt)

        # Step 2: Execute each step
        results = []
        for step in decomposition.get("steps", []):
            step_result = self._execute_reasoning_step(step, context)
            results.append(step_result)

            # Update context with step results
            if context is None:
                context = {}
            context[f"step_{step['step']}_result"] = step_result

        # Step 3: Synthesis and final reasoning
        synthesis_prompt = f"""
        Synthesize the results from the reasoning chain:
        Original problem: {problem}
        Step results: {json.dumps(results, indent=2)}
        
        Provide a comprehensive analysis including:
        1. Key findings
        2. Insights and patterns
        3. Recommendations
        4. Confidence level
        5. Limitations and assumptions
        """

        final_synthesis = self._call_model(synthesis_prompt)

        chain_result = {
            "problem": problem,
            "decomposition": decomposition,
            "step_results": results,
            "synthesis": final_synthesis,
            "chain_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
        }

        self.chain_history.append(chain_result)
        return chain_result

    def _execute_reasoning_step(
        self, step: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single reasoning step"""
        action = step.get("action", "analyze")
        description = step.get("description", "")

        if action == "analyze":
            # Perform data analysis
            analysis_prompt = f"""
            Perform analysis for: {description}
            Available context: {json.dumps(context, indent=2)}
            
            Provide specific analysis results and insights.
            """
            result = self._call_model(analysis_prompt)

        elif action == "visualize":
            # Create visualization
            result = {"action": "visualization", "description": description}

        elif action == "calculate":
            # Perform calculations
            calc_prompt = f"""
            Perform calculations for: {description}
            Context: {json.dumps(context, indent=2)}
            
            Show step-by-step calculations and final results.
            """
            result = self._call_model(calc_prompt)

        else:
            result = {
                "action": action,
                "description": description,
                "status": "completed",
            }

        return {
            "step": step.get("step"),
            "action": action,
            "description": description,
            "result": result,
            "timestamp": datetime.now().isoformat(),
        }

    def _call_model(self, prompt: str) -> Dict[str, Any]:
        """Call the model with error handling"""
        try:
            response = self.model.generate_content(prompt)

            # Try to parse as JSON first
            try:
                return json.loads(response.text)
            except json.JSONDecodeError:
                # Return as text if not JSON
                return {"text": response.text}

        except Exception as e:
            logger.error(f"Model call failed: {e}")
            return {"error": str(e)}


class AdvancedAgent:
    """Advanced AI Agent with tool calling, memory, and reasoning capabilities"""

    def __init__(self, project_id: str, location: str = "us-central1"):
        self.project_id = project_id
        self.location = location

        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)

        # Initialize components
        self.model = GenerativeModel("gemini-1.5-pro")
        self.conversations: Dict[str, ConversationMemory] = {}
        self.data_tools = DataAnalysisTools()
        self.code_interpreter = CodeInterpreter()
        self.reasoning_chain = ReasoningChain(self.model)
        self.accessibility_tools = AccessibilityTools(project_id)

        # Define available tools
        self.tools = self._setup_tools()

        logger.info("Advanced Agent initialized with all capabilities including accessibility features")

    def _setup_tools(self) -> List[Tool]:
        """Setup function calling tools"""

        # Data analysis tool
        analyze_data_func = FunctionDeclaration(
            name="analyze_data",
            description="Analyze a dataset and provide comprehensive insights",
            parameters={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "ID of the dataset to analyze",
                    },
                    "analysis_type": {
                        "type": "string",
                        "description": "Type of analysis: basic, statistical, trend, segment",
                    },
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific columns to analyze",
                    },
                },
                "required": ["dataset_id"],
            },
        )

        # Visualization tool
        create_chart_func = FunctionDeclaration(
            name="create_visualization",
            description="Create charts and visualizations from data",
            parameters={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "ID of the dataset",
                    },
                    "chart_type": {
                        "type": "string",
                        "description": "Type of chart: bar, line, scatter, histogram, heatmap",
                    },
                    "x_column": {"type": "string", "description": "X-axis column"},
                    "y_column": {"type": "string", "description": "Y-axis column"},
                    "title": {"type": "string", "description": "Chart title"},
                },
                "required": ["dataset_id", "chart_type"],
            },
        )

        # Code execution tool
        execute_code_func = FunctionDeclaration(
            name="execute_code",
            description="Execute Python code for data analysis",
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                    "context_vars": {
                        "type": "object",
                        "description": "Variables to include in execution context",
                    },
                },
                "required": ["code"],
            },
        )

        # Statistical analysis tool
        statistical_analysis_func = FunctionDeclaration(
            name="statistical_analysis",
            description="Perform advanced statistical analysis",
            parameters={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Dataset ID"},
                    "analysis_type": {
                        "type": "string",
                        "description": "Analysis type: trend_analysis, segment_analysis, correlation",
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Analysis-specific parameters",
                    },
                },
                "required": ["dataset_id", "analysis_type"],
            },
        )

        # Text-to-speech tool
        text_to_speech_func = FunctionDeclaration(
            name="text_to_speech",
            description="Convert text to speech with customizable voice options for accessibility",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to convert to speech"},
                    "language_code": {"type": "string", "description": "Language code (e.g., en-US, es-ES)"},
                    "voice_name": {"type": "string", "description": "Specific voice name"},
                    "speaking_rate": {"type": "number", "description": "Speaking rate (0.25 to 4.0)"},
                    "pitch": {"type": "number", "description": "Voice pitch (-20.0 to 20.0)"}
                },
                "required": ["text"]
            }
        )
        
        # Speech-to-text tool
        speech_to_text_func = FunctionDeclaration(
            name="speech_to_text",
            description="Convert speech audio to text for voice input accessibility",
            parameters={
                "type": "object",
                "properties": {
                    "audio_base64": {"type": "string", "description": "Base64 encoded audio data"},
                    "language_code": {"type": "string", "description": "Language code for recognition"},
                    "enable_punctuation": {"type": "boolean", "description": "Enable automatic punctuation"}
                },
                "required": ["audio_base64"]
            }
        )
        
        # Audio description tool
        audio_description_func = FunctionDeclaration(
            name="generate_audio_description",
            description="Generate detailed audio descriptions for charts and visualizations",
            parameters={
                "type": "object",
                "properties": {
                    "chart_data": {"type": "object", "description": "Chart data and metadata"},
                    "chart_type": {"type": "string", "description": "Type of chart (bar, line, scatter, pie)"},
                    "detail_level": {"type": "string", "description": "Level of detail (brief, detailed, technical)"}
                },
                "required": ["chart_data", "chart_type"]
            }
        )

        return [
            Tool(function_declarations=[analyze_data_func]),
            Tool(function_declarations=[create_chart_func]),
            Tool(function_declarations=[execute_code_func]),
            Tool(function_declarations=[statistical_analysis_func]),
            Tool(function_declarations=[text_to_speech_func]),
            Tool(function_declarations=[speech_to_text_func]),
            Tool(function_declarations=[audio_description_func]),
        ]

    def start_conversation(self, user_id: str, initial_message: str = None) -> str:
        """Start a new conversation with memory"""
        session_id = str(uuid.uuid4())

        self.conversations[session_id] = ConversationMemory(
            session_id=session_id,
            user_id=user_id,
            messages=[],
            context={},
            tools_used=[],
            data_artifacts={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        if initial_message:
            self.conversations[session_id].add_message("user", initial_message)

        logger.info(f"Started conversation {session_id} for user {user_id}")
        return session_id

    def continue_conversation(
        self, session_id: str, user_message: str, dataset_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Continue a multi-turn conversation with full context"""

        if session_id not in self.conversations:
            raise ValueError(f"Conversation {session_id} not found")

        conversation = self.conversations[session_id]
        conversation.add_message("user", user_message)

        # Build context-aware prompt
        context_prompt = self._build_context_prompt(
            conversation, user_message, dataset_context
        )

        try:
            # Generate response with tools
            response = self.model.generate_content(
                context_prompt,
                tools=self.tools,
                generation_config=generative_models.GenerationConfig(
                    temperature=0.1, top_p=0.8, max_output_tokens=2048
                ),
            )

            # Process function calls if any
            function_results = []
            if response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "function_call"):
                        func_result = self._execute_function_call(
                            part.function_call, conversation, dataset_context
                        )
                        function_results.append(func_result)
                        conversation.tools_used.append(part.function_call.name)

            # Get final response text
            response_text = (
                response.text
                if response.text
                else "I've executed the requested analysis."
            )

            # Add assistant response to memory
            conversation.add_message(
                "assistant",
                response_text,
                {
                    "function_calls": len(function_results),
                    "tools_used": (
                        conversation.tools_used[-len(function_results) :]
                        if function_results
                        else []
                    ),
                },
            )

            return {
                "response": response_text,
                "function_results": function_results,
                "conversation_id": session_id,
                "tools_used": conversation.tools_used,
                "context_updated": bool(function_results),
            }

        except Exception as e:
            logger.error(f"Conversation error: {e}")
            error_response = (
                f"I encountered an error: {str(e)}. Let me try a different approach."
            )
            conversation.add_message("assistant", error_response)

            return {
                "response": error_response,
                "error": str(e),
                "conversation_id": session_id,
            }

    def _build_context_prompt(
        self,
        conversation: ConversationMemory,
        current_message: str,
        dataset_context: Dict[str, Any] = None,
    ) -> str:
        """Build context-aware prompt for the agent"""

        system_prompt = """You are an advanced data analysis agent with the following capabilities:
        
        1. TOOL CALLING: You can call functions to analyze data, create visualizations, execute code, and perform statistical analysis
        2. MULTI-TURN MEMORY: You remember the entire conversation history and context
        3. DATA ANALYSIS: You have built-in tools for comprehensive data analysis
        4. CODE INTERPRETER: You can execute Python code safely for complex analysis
        5. REASONING CHAINS: You can break down complex problems into steps
        
        Available tools:
        - analyze_data: Comprehensive dataset analysis
        - create_visualization: Create charts and plots
        - execute_code: Run Python code for analysis
        - statistical_analysis: Advanced statistical methods
        
        Always consider the conversation history and use appropriate tools to provide comprehensive answers.
        """

        # Add conversation history
        history_context = conversation.get_recent_context(max_messages=10)

        # Add dataset context if available
        dataset_info = ""
        if dataset_context:
            dataset_info = (
                f"\nCurrent dataset context:\n{json.dumps(dataset_context, indent=2)}"
            )

        # Add data artifacts from previous analysis
        artifacts_info = ""
        if conversation.data_artifacts:
            artifacts_info = f"\nPrevious analysis artifacts:\n{json.dumps(conversation.data_artifacts, indent=2)}"

        full_prompt = f"""{system_prompt}
        
        Conversation history:
        {history_context}
        {dataset_info}
        {artifacts_info}
        
        Current user message: {current_message}
        
        Provide a helpful response and use appropriate tools as needed."""

        return full_prompt

    def _execute_function_call(
        self,
        function_call,
        conversation: ConversationMemory,
        dataset_context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Execute a function call and return results"""

        func_name = function_call.name
        func_args = dict(function_call.args)

        try:
            if func_name == "analyze_data":
                result = self._tool_analyze_data(func_args, dataset_context)

            elif func_name == "create_visualization":
                result = self._tool_create_visualization(func_args, dataset_context)

            elif func_name == "execute_code":
                result = self._tool_execute_code(func_args, dataset_context)

            elif func_name == "statistical_analysis":
                result = self._tool_statistical_analysis(func_args, dataset_context)
                
            elif func_name == "text_to_speech":
                result = self._tool_text_to_speech(func_args)
                
            elif func_name == "speech_to_text":
                result = self._tool_speech_to_text(func_args)
                
            elif func_name == "generate_audio_description":
                result = self._tool_generate_audio_description(func_args)

            else:
                result = {"error": f"Unknown function: {func_name}"}

            # Store result in conversation artifacts
            conversation.data_artifacts[f"{func_name}_{datetime.now().isoformat()}"] = (
                result
            )

            return {
                "function": func_name,
                "arguments": func_args,
                "result": result,
                "success": "error" not in result,
            }

        except Exception as e:
            logger.error(f"Function execution error: {e}")
            return {
                "function": func_name,
                "arguments": func_args,
                "error": str(e),
                "success": False,
            }

    def _tool_analyze_data(
        self, args: Dict[str, Any], dataset_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute data analysis tool"""
        if not dataset_context or "dataframe" not in dataset_context:
            return {"error": "No dataset available for analysis"}

        df = dataset_context["dataframe"]
        analysis_type = args.get("analysis_type", "basic")

        if analysis_type == "basic":
            return self.data_tools.analyze_dataset(df)
        else:
            return {"analysis_type": analysis_type, "status": "completed"}

    def _tool_create_visualization(
        self, args: Dict[str, Any], dataset_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute visualization tool"""
        if not dataset_context or "dataframe" not in dataset_context:
            return {"error": "No dataset available for visualization"}

        df = dataset_context["dataframe"]
        chart_type = args.get("chart_type", "bar")
        x_col = args.get("x_column")
        y_col = args.get("y_column")
        title = args.get("title")

        chart_path = self.data_tools.create_visualization(
            df, chart_type, x_col, y_col, title
        )

        return {
            "chart_type": chart_type,
            "chart_path": chart_path,
            "success": chart_path is not None,
        }

    def _tool_execute_code(
        self, args: Dict[str, Any], dataset_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute code interpreter tool"""
        code = args.get("code", "")
        context_vars = args.get("context_vars", {})

        # Add dataset to context if available
        if dataset_context and "dataframe" in dataset_context:
            context_vars["df"] = dataset_context["dataframe"]

        return self.code_interpreter.execute_code(code, context_vars)

    def _tool_statistical_analysis(
        self, args: Dict[str, Any], dataset_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute statistical analysis tool"""
        if not dataset_context or "dataframe" not in dataset_context:
            return {"error": "No dataset available for statistical analysis"}

        df = dataset_context["dataframe"]
        analysis_type = args.get("analysis_type")
        parameters = args.get("parameters", {})

        return self.data_tools.statistical_analysis(df, analysis_type, **parameters)

    def execute_reasoning_chain(
        self, problem: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute advanced reasoning chain"""
        return self.reasoning_chain.execute_reasoning_chain(problem, context)

    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get conversation summary and insights"""
        if session_id not in self.conversations:
            return {"error": "Conversation not found"}

        conversation = self.conversations[session_id]

        return {
            "session_id": session_id,
            "user_id": conversation.user_id,
            "message_count": len(conversation.messages),
            "tools_used": list(set(conversation.tools_used)),
            "artifacts_count": len(conversation.data_artifacts),
            "duration": (
                conversation.updated_at - conversation.created_at
            ).total_seconds(),
            "last_activity": conversation.updated_at.isoformat(),
        }
    
    # ============================================================================
    # ACCESSIBILITY TOOL IMPLEMENTATIONS
    # ============================================================================
    
    def _tool_text_to_speech(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute text-to-speech tool"""
        text = args.get('text', '')
        language_code = args.get('language_code', 'en-US')
        voice_name = args.get('voice_name')
        speaking_rate = args.get('speaking_rate', 1.0)
        pitch = args.get('pitch', 0.0)
        
        return self.accessibility_tools.text_to_speech(
            text=text,
            language_code=language_code,
            voice_name=voice_name,
            speaking_rate=speaking_rate,
            pitch=pitch
        )
    
    def _tool_speech_to_text(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute speech-to-text tool"""
        audio_base64 = args.get('audio_base64', '')
        language_code = args.get('language_code', 'en-US')
        enable_punctuation = args.get('enable_punctuation', True)
        
        return self.accessibility_tools.speech_to_text(
            audio_base64=audio_base64,
            language_code=language_code,
            enable_automatic_punctuation=enable_punctuation
        )
    
    def _tool_generate_audio_description(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute audio description generation tool"""
        chart_data = args.get('chart_data', {})
        chart_type = args.get('chart_type', 'unknown')
        detail_level = args.get('detail_level', 'detailed')
        
        description = self.accessibility_tools.generate_audio_description(chart_data, chart_type)
        
        # Generate TTS for the description if requested
        tts_result = None
        if detail_level != 'text_only':
            tts_result = self.accessibility_tools.text_to_speech(description)
        
        return {
            "description": description,
            "chart_type": chart_type,
            "detail_level": detail_level,
            "audio_description": tts_result,
            "accessibility_features": {
                "screen_reader_optimized": True,
                "audio_available": tts_result is not None,
                "detailed_description": len(description) > 100
            }
        }
