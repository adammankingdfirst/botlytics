"""
Monitoring and observability utilities for Botlytics
"""

import logging
import time
import functools
from typing import Callable, Any
from google.cloud import monitoring_v3
from google.cloud import logging as cloud_logging
from prometheus_client import Counter, Histogram, generate_latest
import os

# Initialize Google Cloud Logging
if os.environ.get("GCP_PROJECT_ID"):
    try:
        client = cloud_logging.Client()
        client.setup_logging()
    except Exception as e:
        print(f"Failed to setup Cloud Logging: {e}")

# Prometheus metrics
REQUEST_COUNT = Counter('botlytics_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('botlytics_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
LLM_CALLS = Counter('botlytics_llm_calls_total', 'Total LLM calls', ['model', 'status'])
CODE_EXECUTIONS = Counter('botlytics_code_executions_total', 'Code executions', ['mode', 'status'])
CHART_GENERATIONS = Counter('botlytics_chart_generations_total', 'Chart generations', ['type', 'status'])

logger = logging.getLogger(__name__)

def monitor_endpoint(func: Callable) -> Callable:
    """Decorator to monitor API endpoints"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        method = "POST"  # Most of our endpoints are POST
        endpoint = func.__name__
        status = "success"
        
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            status = "error"
            logger.error(f"Error in {endpoint}: {str(e)}")
            raise
        finally:
            duration = time.time() - start_time
            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
            REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
            
            # Log performance metrics
            logger.info(f"Endpoint {endpoint} completed in {duration:.3f}s with status {status}")
    
    return wrapper

def monitor_llm_call(model: str):
    """Decorator to monitor LLM calls"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            status = "success"
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                logger.error(f"LLM call failed for model {model}: {str(e)}")
                raise
            finally:
                duration = time.time() - start_time
                LLM_CALLS.labels(model=model, status=status).inc()
                logger.info(f"LLM call to {model} completed in {duration:.3f}s with status {status}")
        
        return wrapper
    return decorator

def monitor_code_execution(mode: str):
    """Decorator to monitor code execution"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            status = "success"
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                if not result.get('success', True):
                    status = "error"
                return result
            except Exception as e:
                status = "error"
                logger.error(f"Code execution failed for mode {mode}: {str(e)}")
                raise
            finally:
                duration = time.time() - start_time
                CODE_EXECUTIONS.labels(mode=mode, status=status).inc()
                logger.info(f"Code execution ({mode}) completed in {duration:.3f}s with status {status}")
        
        return wrapper
    return decorator

def monitor_chart_generation(chart_type: str):
    """Monitor chart generation"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            status = "success"
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                if result is None:
                    status = "error"
                return result
            except Exception as e:
                status = "error"
                logger.error(f"Chart generation failed for type {chart_type}: {str(e)}")
                raise
            finally:
                duration = time.time() - start_time
                CHART_GENERATIONS.labels(type=chart_type, status=status).inc()
                logger.info(f"Chart generation ({chart_type}) completed in {duration:.3f}s with status {status}")
        
        return wrapper
    return decorator

def log_security_event(event_type: str, details: dict):
    """Log security-related events"""
    logger.warning(f"Security event: {event_type}", extra={
        "event_type": event_type,
        "details": details,
        "severity": "WARNING"
    })

def log_performance_metrics(operation: str, duration: float, metadata: dict = None):
    """Log performance metrics"""
    logger.info(f"Performance: {operation} took {duration:.3f}s", extra={
        "operation": operation,
        "duration": duration,
        "metadata": metadata or {}
    })

class StructuredLogger:
    """Structured logging for better observability"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def log_request(self, method: str, path: str, status_code: int, duration: float, user_id: str = None):
        """Log HTTP request"""
        self.logger.info("HTTP request", extra={
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration": duration,
            "user_id": user_id,
            "event_type": "http_request"
        })
    
    def log_llm_interaction(self, model: str, prompt_length: int, response_length: int, duration: float):
        """Log LLM interaction"""
        self.logger.info("LLM interaction", extra={
            "model": model,
            "prompt_length": prompt_length,
            "response_length": response_length,
            "duration": duration,
            "event_type": "llm_interaction"
        })
    
    def log_data_processing(self, operation: str, rows: int, columns: int, duration: float):
        """Log data processing operation"""
        self.logger.info("Data processing", extra={
            "operation": operation,
            "rows": rows,
            "columns": columns,
            "duration": duration,
            "event_type": "data_processing"
        })
    
    def log_error(self, error: Exception, context: dict = None):
        """Log error with context"""
        self.logger.error(f"Error: {str(error)}", extra={
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
            "event_type": "error"
        })

# Global structured logger instance
structured_logger = StructuredLogger("botlytics")

def get_metrics():
    """Get Prometheus metrics"""
    return generate_latest()

def setup_custom_metrics():
    """Setup custom Cloud Monitoring metrics"""
    if not os.environ.get("GCP_PROJECT_ID"):
        return
    
    try:
        client = monitoring_v3.MetricServiceClient()
        project_name = f"projects/{os.environ.get('GCP_PROJECT_ID')}"
        
        # Define custom metrics
        custom_metrics = [
            {
                "type": "custom.googleapis.com/botlytics/llm_response_time",
                "display_name": "LLM Response Time",
                "description": "Time taken for LLM to respond",
                "metric_kind": monitoring_v3.MetricDescriptor.MetricKind.GAUGE,
                "value_type": monitoring_v3.MetricDescriptor.ValueType.DOUBLE,
            },
            {
                "type": "custom.googleapis.com/botlytics/code_execution_success_rate",
                "display_name": "Code Execution Success Rate",
                "description": "Success rate of code execution",
                "metric_kind": monitoring_v3.MetricDescriptor.MetricKind.GAUGE,
                "value_type": monitoring_v3.MetricDescriptor.ValueType.DOUBLE,
            }
        ]
        
        for metric_config in custom_metrics:
            descriptor = monitoring_v3.MetricDescriptor(
                type=metric_config["type"],
                display_name=metric_config["display_name"],
                description=metric_config["description"],
                metric_kind=metric_config["metric_kind"],
                value_type=metric_config["value_type"],
            )
            
            try:
                client.create_metric_descriptor(
                    name=project_name,
                    metric_descriptor=descriptor
                )
                logger.info(f"Created custom metric: {metric_config['type']}")
            except Exception as e:
                if "already exists" not in str(e):
                    logger.warning(f"Failed to create metric {metric_config['type']}: {e}")
    
    except Exception as e:
        logger.warning(f"Failed to setup custom metrics: {e}")

# Initialize custom metrics on import
setup_custom_metrics()