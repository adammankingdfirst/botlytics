# 📊 Botlytics

Botlytics lets anyone explore and visualize their data in natural language using Google's Gemini AI.

## 🏗️ Architecture

```
┌───────────────────────────────┐
│         Web Frontend          │
│  (Streamlit UI)              │
└──────────────┬────────────────┘
               │ HTTPS / JSON API
┌──────────────▼────────────────┐
│   FastAPI Backend (Cloud Run) │
│   - Receives user query       │
│   - Calls Gemini via Vertex AI│
│   - Manages auth/storage      │
└──────────────┬────────────────┘
               │
┌──────────────▼────────────────┐
│   Gemini AI (Vertex AI)       │
│   - Generates pandas code     │
│   - Creates visualizations    │
│   - Provides insights         │
└──────────────┬────────────────┘
               │
┌──────────────▼────────────────┐
│  Data Storage & Processing    │
│ - Google Cloud Storage        │
│ - BigQuery (future)           │
│ - Chart Generation            │
└───────────────────────────────┘
```

## 🚀 Quick Start

### 1. GCP Setup

Follow the instructions in `gcp-setup.md` to:
- Create GCP project
- Enable required APIs (Vertex AI, Cloud Run, Storage, BigQuery)
- Create service account with minimal IAM permissions
- Set up storage bucket

### 2. Local Development

```bash
# Clone and setup
git clone <your-repo>
cd botlytics

# Copy environment template
cp .env.example .env
# Edit .env with your GCP settings

# Install backend dependencies
cd backend
pip install -r requirements.txt

# Install frontend dependencies  
cd ../frontend
pip install -r requirements.txt

# Start services (use run-local.sh or manual)
chmod +x run-local.sh
./run-local.sh
```

### 3. Deploy to Cloud Run

```bash
# Make sure you're authenticated with GCP
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Deploy
chmod +x deploy.sh
./deploy.sh
```

## 📁 Project Structure

```
botlytics/
├── backend/
│   ├── main.py              # FastAPI application
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── app.py              # Streamlit UI
│   └── requirements.txt    # Frontend dependencies
├── .dockerfile             # Container configuration
├── deploy.sh              # Cloud Run deployment
├── run-local.sh           # Local development
├── gcp-setup.md           # GCP configuration guide
├── sample_data.csv        # Test data
└── README.md              # This file
```

## 🔧 API Endpoints

- `POST /api/v1/upload` - Upload CSV file
- `POST /api/v1/query` - Query data with natural language
- `GET /api/v1/datasets/{id}` - Get dataset information
- `GET /` - Health check

## 💡 Example Queries

- "What are the total sales by product?"
- "Show me sales trends over time"
- "Which region has the highest sales?"
- "Create a chart showing sales by category"
- "What's the average sales per day?"

## 🔒 Security Features

- Input sanitization for generated code
- Restricted pandas operations
- GCP IAM with minimal permissions
- Secure file handling
- CORS configuration

## 🛠️ Development

### Backend (FastAPI)
```bash
cd backend
uvicorn main:app --reload --port 8080
```

### Frontend (Streamlit)
```bash
cd frontend
streamlit run app.py --server.port 8501
```

### Testing
```bash
# Test with sample data
curl -X POST "http://localhost:8080/api/v1/upload" \
  -F "file=@sample_data.csv"

# Test query
curl -X POST "http://localhost:8080/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"dataset_id":"your-id","query":"Show total sales by product"}'
```

## 📊 Features

- ✅ CSV file upload and processing
- ✅ Natural language to pandas code generation
- ✅ Automatic chart creation
- ✅ Secure code execution
- ✅ Cloud storage integration
- ✅ Streamlit web interface
- ✅ Cloud Run deployment
- 🔄 BigQuery integration (planned)
- 🔄 Advanced visualizations (planned)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

See LICENSE file for details.
