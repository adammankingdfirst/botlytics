# 🏗️ Botlytics Architecture Documentation

## System Overview

Botlytics is an enterprise-grade AI data analytics platform built on Google Cloud Platform, featuring advanced agent capabilities, multi-turn conversations, and comprehensive accessibility features.

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Streamlit Frontend (Cloud Run)                          │   │
│  │  • Multi-tab interface (Chat, Reasoning, Code, Analysis) │   │
│  │  • Real-time updates and progress indicators             │   │
│  │  • Accessibility features (TTS, STT, high contrast)      │   │
│  │  • Responsive design with session management             │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓ HTTP/REST
┌─────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  FastAPI Backend (Cloud Run)                             │   │
│  │  • RESTful API endpoints                                 │   │
│  │  • Request validation and sanitization                   │   │
│  │  • Security middleware (CORS, headers, rate limiting)    │   │
│  │  • Monitoring and observability (Prometheus, Cloud)      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        BUSINESS LOGIC LAYER                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Advanced Agent (agent_sdk.py)                           │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  Conversation Memory                                │  │   │
│  │  │  • Session management                               │  │   │
│  │  │  • Message history                                  │  │   │
│  │  │  • Context tracking                                 │  │   │
│  │  │  • Data artifacts storage                           │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  Tool Calling & Function Execution                 │  │   │
│  │  │  • Data analysis tools                             │  │   │
│  │  │  • Visualization tools                             │  │   │
│  │  │  • Code interpreter                                │  │   │
│  │  │  • Statistical analysis                            │  │   │
│  │  │  • Accessibility tools (TTS, STT, audio desc)      │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  Reasoning Chains                                  │  │   │
│  │  │  • Problem decomposition                           │  │   │
│  │  │  • Step execution                                  │  │   │
│  │  │  • Result synthesis                                │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        DATA & AI LAYER                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Vertex AI   │  │  Cloud       │  │  Accessibility APIs  │  │
│  │  • Gemini    │  │  Storage     │  │  • Text-to-Speech    │  │
│  │  • Agent SDK │  │  • CSV files │  │  • Speech-to-Text    │  │
│  │  • Function  │  │  • Charts    │  │  • Neural voices     │  │
│  │    Calling   │  │  • Artifacts │  │  • Multi-language    │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  BigQuery    │  │  Cloud       │  │  Cloud Monitoring    │  │
│  │  • Large     │  │  Logging     │  │  • Metrics           │  │
│  │    datasets  │  │  • Structured│  │  • Alerts            │  │
│  │  • SQL       │  │  • Audit     │  │  • Dashboards        │  │
│  │    queries   │  │    trails    │  │  • Traces            │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Frontend (Streamlit)

**Technology**: Python 3.11, Streamlit 1.28.1
**Deployment**: Cloud Run (serverless)
**Features**:
- Multi-tab interface for different workflows
- Real-time chat with AI agent
- File upload with validation
- Accessibility controls (TTS, high contrast, large text)
- Session state management
- Responsive design

**Key Files**:
- `frontend/app.py` - Main application
- `frontend/requirements.txt` - Dependencies
- `frontend.dockerfile` - Container configuration

### 2. Backend (FastAPI)

**Technology**: Python 3.11, FastAPI 0.104.1, Uvicorn
**Deployment**: Cloud Run (serverless)
**Features**:
- RESTful API with OpenAPI documentation
- Request validation with Pydantic
- Security middleware (CORS, headers, sanitization)
- Monitoring with Prometheus metrics
- Structured logging with Cloud Logging
- Health checks and readiness probes

**Key Files**:
- `backend/main.py` - API endpoints and business logic
- `backend/agent_sdk.py` - Advanced agent implementation
- `backend/monitoring.py` - Observability utilities
- `backend/requirements.txt` - Dependencies
- `.dockerfile` - Container configuration

### 3. Advanced Agent

**Core Capabilities**:

#### a) Conversation Memory
```python
class ConversationMemory:
    - session_id: Unique conversation identifier
    - user_id: User tracking
    - messages: Complete message history
    - context: Conversation context and state
    - tools_used: Track tool usage
    - data_artifacts: Store analysis results
```

**Features**:
- Multi-turn conversations with full context
- Session persistence across requests
- Context-aware responses
- Tool usage tracking
- Data artifact storage

#### b) Tool Calling & Function Execution

**Available Tools**:
1. **analyze_data** - Comprehensive dataset analysis
   - Basic statistics
   - Missing value analysis
   - Correlation matrices
   - Outlier detection

2. **create_visualization** - Chart generation
   - Bar, line, scatter, histogram, heatmap
   - Matplotlib and Plotly support
   - Automatic chart upload to GCS

3. **execute_code** - Safe Python execution
   - AST validation
   - Whitelist approach (40+ pandas operations)
   - Resource limits (memory, timeout)
   - Sandboxed environment

4. **statistical_analysis** - Advanced analytics
   - Trend analysis
   - Segment analysis
   - Correlation analysis
   - Time series analysis

5. **text_to_speech** - Accessibility
   - 40+ languages
   - Neural voices
   - Customizable rate, pitch, volume
   - MP3 output

6. **speech_to_text** - Voice input
   - High accuracy recognition
   - Automatic punctuation
   - Word-level timing
   - Multi-language support

7. **generate_audio_description** - Chart accessibility
   - Detailed descriptions
   - Multiple formats (screen reader, simple, technical)
   - Automatic TTS generation

#### c) Code Interpreter

**Security Features**:
- AST validation before execution
- Whitelist of allowed operations
- No imports, exec, eval, or system calls
- Resource limits (100MB memory, 30s timeout)
- Sandboxed execution environment
- Execution history tracking

**Allowed Operations**:
```python
# Pandas operations (40+)
groupby, sum, mean, count, head, tail, sort_values, 
reset_index, merge, concat, drop, fillna, dropna,
to_datetime, describe, value_counts, corr, pivot_table,
rolling, resample, apply, transform, filter, etc.

# NumPy operations
Array operations, mathematical functions, statistics

# Visualization
Matplotlib, Seaborn, Plotly (limited to safe operations)
```

#### d) Reasoning Chains

**Process**:
1. **Problem Decomposition** - Break complex problems into steps
2. **Step Execution** - Execute each step with appropriate tools
3. **Context Building** - Each step builds on previous results
4. **Synthesis** - Combine results into comprehensive insights

**Use Cases**:
- Complex data analysis workflows
- Multi-step problem solving
- Strategic recommendations
- Comprehensive reporting

### 4. Data Storage

#### Cloud Storage (GCS)
- **CSV Files**: User-uploaded datasets
- **Charts**: Generated visualizations
- **Artifacts**: Analysis results and intermediate data
- **Lifecycle**: Automatic cleanup policies

#### BigQuery (Optional)
- **Large Datasets**: > 1M rows
- **SQL Queries**: Complex analytical queries
- **Cost Controls**: Query byte limits
- **Performance**: Optimized for analytics

### 5. AI & ML Services

#### Vertex AI
- **Gemini 1.5 Pro**: Primary LLM
- **Function Calling**: Tool orchestration
- **Agent SDK**: Advanced agent capabilities
- **Context Window**: Large context for conversations

#### Accessibility APIs
- **Text-to-Speech**: Google Cloud TTS with Neural2 voices
- **Speech-to-Text**: Google Cloud STT with enhanced models
- **Multi-language**: 40+ languages supported

### 6. Monitoring & Observability

#### Prometheus Metrics
```python
# Request metrics
botlytics_requests_total{method, endpoint, status}
botlytics_request_duration_seconds{method, endpoint}

# LLM metrics
botlytics_llm_calls_total{model, status}

# Code execution metrics
botlytics_code_executions_total{mode, status}

# Chart generation metrics
botlytics_chart_generations_total{type, status}
```

#### Cloud Monitoring
- Custom metrics for LLM response times
- Code execution success rates
- Resource utilization
- Error rates and patterns

#### Structured Logging
```python
{
  "event_type": "http_request",
  "method": "POST",
  "path": "/api/v1/query",
  "status_code": 200,
  "duration": 1.234,
  "user_id": "user-123"
}
```

## Security Architecture

### 1. Input Validation
- File size limits (50MB)
- Dataset size limits (1M rows, 1000 columns)
- Code length limits (10,000 characters)
- SQL query validation
- CSV parsing with error handling

### 2. Code Execution Security
```python
# Multi-layer security
1. AST validation (syntax and structure)
2. Whitelist approach (only approved operations)
3. Dangerous pattern detection (exec, eval, imports)
4. Resource limits (memory, CPU, timeout)
5. Sandboxed environment (restricted builtins)
6. Execution history (audit trail)
```

### 3. Network Security
- HTTPS enforcement (Cloud Run)
- CORS configuration
- Security headers (XSS, CSRF, clickjacking)
- Service account isolation
- VPC integration (optional)

### 4. Data Security
- Temporary file storage
- Automatic cleanup
- No PII logging
- Encrypted at rest (GCS)
- Encrypted in transit (TLS)

## Deployment Architecture

### Cloud Run Configuration

#### Backend Service
```yaml
Service: botlytics-backend
Memory: 2Gi
CPU: 2
Timeout: 300s (5 minutes)
Max Instances: 20
Min Instances: 1 (always warm)
Concurrency: 80
Execution Environment: gen2
CPU Boost: enabled
```

#### Frontend Service
```yaml
Service: botlytics-frontend
Memory: 1Gi
CPU: 1
Timeout: 300s
Max Instances: 10
Min Instances: 0 (scale to zero)
Concurrency: 50
Execution Environment: gen2
```

### CI/CD Pipeline

**GitHub Actions Workflow**:
1. **Test** - Run pytest suite
2. **Lint** - Code quality checks
3. **Security Scan** - Vulnerability detection
4. **Build** - Docker image creation
5. **Push** - Container Registry upload
6. **Deploy** - Cloud Run deployment
7. **Health Check** - Verify deployment
8. **Integration Tests** - End-to-end validation

## Performance Characteristics

### Latency
- **Cold Start**: < 10 seconds (with cpu-boost)
- **Warm Requests**: < 1 second
- **LLM Calls**: 2-5 seconds (depends on complexity)
- **Code Execution**: < 30 seconds (timeout limit)
- **File Upload**: < 5 seconds (for 50MB)

### Throughput
- **Concurrent Requests**: 80 per instance (backend)
- **Max Instances**: 20 (1600 concurrent requests)
- **Request Rate**: 1000+ requests/minute
- **Data Processing**: 1M rows in < 10 seconds

### Scalability
- **Horizontal**: Auto-scaling from 1-20 instances
- **Vertical**: 2Gi memory, 2 CPU per instance
- **Storage**: Unlimited (GCS)
- **Conversations**: Unlimited (in-memory per instance)

## Data Flow

### 1. File Upload Flow
```
User → Frontend → Backend → Validation → GCS → Dataset ID → User
```

### 2. Query Flow
```
User → Frontend → Backend → Agent → LLM → Tool Selection → 
Tool Execution → Result Processing → Response → User
```

### 3. Conversation Flow
```
User → Message → Backend → Agent → Memory Retrieval → 
Context Building → LLM → Function Calls → Tool Execution → 
Memory Update → Response → User
```

### 4. Code Execution Flow
```
User → Code → Backend → AST Validation → Sanitization → 
Sandboxed Execution → Result Capture → Response → User
```

## Error Handling

### Error Categories
1. **Validation Errors** (400) - Invalid input
2. **Not Found Errors** (404) - Resource not found
3. **Server Errors** (500) - Internal errors
4. **Timeout Errors** (504) - Operation timeout

### Error Response Format
```json
{
  "detail": "Error message",
  "error_type": "ValidationError",
  "timestamp": "2024-01-01T00:00:00Z",
  "request_id": "uuid"
}
```

### Recovery Strategies
- Automatic retry for transient errors
- Graceful degradation for service failures
- Fallback responses for LLM errors
- User-friendly error messages

## Cost Optimization

### Strategies
1. **Auto-scaling** - Scale to zero for frontend
2. **Resource Limits** - Prevent runaway costs
3. **Query Limits** - BigQuery byte limits
4. **File Limits** - Storage size limits
5. **Caching** - Query result caching
6. **Efficient Queries** - Optimized SQL and pandas

### Cost Breakdown (Monthly)
- Cloud Run Backend: $15-25
- Cloud Run Frontend: $5-10
- Cloud Storage: $0.02/GB
- Vertex AI: $0.000125/1K chars
- TTS/STT: Pay per use
- **Total**: $30-50 for moderate usage

## Future Enhancements

### Planned Features
1. **Real-time Collaboration** - Multi-user sessions
2. **Advanced Visualizations** - Interactive dashboards
3. **ML Model Training** - Custom model support
4. **Data Pipelines** - Automated workflows
5. **API Rate Limiting** - Per-user quotas
6. **Authentication** - OAuth2, SSO
7. **Database Integration** - PostgreSQL, MySQL
8. **Streaming Responses** - Real-time updates

### Scalability Improvements
1. **Redis Cache** - Conversation persistence
2. **Cloud SQL** - Structured data storage
3. **Pub/Sub** - Async processing
4. **Cloud Tasks** - Background jobs
5. **CDN** - Static asset delivery

## Maintenance

### Regular Tasks
- **Weekly**: Review logs and metrics
- **Monthly**: Update dependencies
- **Quarterly**: Security audit
- **Annually**: Architecture review

### Monitoring Checklist
- [ ] Error rates < 1%
- [ ] Latency p95 < 2 seconds
- [ ] Uptime > 99.5%
- [ ] Cost within budget
- [ ] No security vulnerabilities

---

**Version**: 1.0.0
**Last Updated**: 2024
**Status**: Production Ready ✅
