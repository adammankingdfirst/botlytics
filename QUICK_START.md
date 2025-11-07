# 🚀 Botlytics Quick Start Guide

## Prerequisites
- Google Cloud Platform account
- `gcloud` CLI installed and authenticated
- Docker installed (for local testing)
- Python 3.11+ (for local development)

## 5-Minute Setup

### 1. Clone and Configure (2 minutes)
```bash
# Clone repository
git clone <your-repo>
cd botlytics

# Copy environment template
cp .env.example .env

# Edit .env with your GCP settings
# Required: GCP_PROJECT_ID, GCS_BUCKET
```

### 2. GCP Setup (2 minutes)
```bash
# Set your project ID
export PROJECT_ID="botlytics-$(date +%s)"
export BUCKET_NAME="botlytics-data-$PROJECT_ID"

# Create project
gcloud projects create $PROJECT_ID
gcloud config set project $PROJECT_ID

# Enable APIs (takes ~1 minute)
gcloud services enable \
  aiplatform.googleapis.com \
  run.googleapis.com \
  storage.googleapis.com \
  texttospeech.googleapis.com \
  speech.googleapis.com

# Create service account
gcloud iam service-accounts create botlytics-sa \
  --display-name="Botlytics Service Account"

# Grant permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:botlytics-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:botlytics-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

# Create storage bucket
gsutil mb gs://$BUCKET_NAME
```

### 3. Deploy to Cloud Run (1 minute)
```bash
# Update .env with your values
echo "GCP_PROJECT_ID=$PROJECT_ID" > .env
echo "GCS_BUCKET=$BUCKET_NAME" >> .env
echo "GCP_LOCATION=us-central1" >> .env

# Deploy (automated script)
chmod +x deploy-full-stack.sh
./deploy-full-stack.sh
```

**That's it!** Your application is now live on Cloud Run.

## Test Your Deployment

### Quick Health Check
```bash
# Get your backend URL
BACKEND_URL=$(gcloud run services describe botlytics-backend \
  --region=us-central1 --format='value(status.url)')

# Test health endpoint
curl $BACKEND_URL/api/v1/health

# Expected output:
# {"status":"healthy","checks":{...}}
```

### Run Comprehensive Tests
```bash
# Get URLs
BACKEND_URL=$(gcloud run services describe botlytics-backend \
  --region=us-central1 --format='value(status.url)')
FRONTEND_URL=$(gcloud run services describe botlytics-frontend \
  --region=us-central1 --format='value(status.url)')

# Run Cloud Run tests
python test-cloud-run.py \
  --backend-url $BACKEND_URL \
  --frontend-url $FRONTEND_URL \
  --wait 30
```

## Local Development

### Option 1: Docker (Recommended)
```bash
# Build and run backend
docker build -t botlytics -f .dockerfile .
docker run -p 8080:8080 --env-file .env botlytics

# In another terminal, run frontend
docker build -t botlytics-frontend -f frontend.dockerfile .
docker run -p 8501:8501 \
  -e API_BASE_URL=http://localhost:8080 \
  botlytics-frontend
```

### Option 2: Docker Compose
```bash
# Run backend only
docker-compose up backend

# Run full stack
docker-compose --profile full-stack up
```

### Option 3: Manual (Development)
```bash
# Terminal 1: Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8080 --reload

# Terminal 2: Frontend
cd frontend
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

## Common Commands

### View Logs
```bash
# Backend logs
gcloud run services logs read botlytics-backend \
  --region=us-central1 --limit=50

# Frontend logs
gcloud run services logs read botlytics-frontend \
  --region=us-central1 --limit=50
```

### Update Deployment
```bash
# Rebuild and redeploy backend
docker build -t gcr.io/$PROJECT_ID/botlytics-backend:latest -f .dockerfile .
docker push gcr.io/$PROJECT_ID/botlytics-backend:latest
gcloud run deploy botlytics-backend \
  --image gcr.io/$PROJECT_ID/botlytics-backend:latest \
  --region=us-central1

# Rebuild and redeploy frontend
docker build -t gcr.io/$PROJECT_ID/botlytics-frontend:latest -f frontend.dockerfile .
docker push gcr.io/$PROJECT_ID/botlytics-frontend:latest
gcloud run deploy botlytics-frontend \
  --image gcr.io/$PROJECT_ID/botlytics-frontend:latest \
  --region=us-central1
```

### Scale Services
```bash
# Increase backend capacity
gcloud run services update botlytics-backend \
  --region=us-central1 \
  --max-instances=50 \
  --min-instances=2

# Scale frontend to zero when idle
gcloud run services update botlytics-frontend \
  --region=us-central1 \
  --min-instances=0
```

### Monitor Costs
```bash
# View current month costs
gcloud billing accounts list
gcloud billing projects describe $PROJECT_ID

# Set budget alert (recommended)
# Go to: https://console.cloud.google.com/billing/budgets
```

## API Endpoints

### Core Endpoints
```bash
# Health check
GET /api/v1/health

# Upload CSV
POST /api/v1/upload
Content-Type: multipart/form-data
Body: file=@data.csv

# Start conversation
POST /api/v1/conversation/start
Body: {"user_id": "user-123"}

# Continue conversation
POST /api/v1/conversation/continue
Body: {
  "session_id": "uuid",
  "message": "Analyze my data",
  "dataset_id": "dataset-uuid"
}

# Execute code
POST /api/v1/code-interpreter
Body: {
  "code": "result = df.head()",
  "dataset_id": "dataset-uuid"
}

# Reasoning chain
POST /api/v1/reasoning-chain
Body: {
  "problem": "Analyze sales trends",
  "dataset_id": "dataset-uuid"
}
```

### Accessibility Endpoints
```bash
# Text-to-speech
POST /api/v1/accessibility/text-to-speech
Body: {
  "text": "Hello world",
  "language_code": "en-US"
}

# Audio description
POST /api/v1/accessibility/audio-description
Body: {
  "chart_data": {...},
  "chart_type": "bar"
}
```

## Troubleshooting

### Issue: Cold Start Timeout
```bash
# Solution: Enable CPU boost
gcloud run services update botlytics-backend \
  --region=us-central1 \
  --cpu-boost
```

### Issue: Memory Errors
```bash
# Solution: Increase memory
gcloud run services update botlytics-backend \
  --region=us-central1 \
  --memory=4Gi
```

### Issue: Permission Denied
```bash
# Solution: Check service account permissions
gcloud projects get-iam-policy $PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:botlytics-sa@*"
```

### Issue: Frontend Can't Reach Backend
```bash
# Solution: Update backend URL
BACKEND_URL=$(gcloud run services describe botlytics-backend \
  --region=us-central1 --format='value(status.url)')

gcloud run services update botlytics-frontend \
  --region=us-central1 \
  --set-env-vars="API_BASE_URL=$BACKEND_URL"
```

## Example Usage

### 1. Upload Data
```bash
curl -X POST $BACKEND_URL/api/v1/upload \
  -F "file=@sample_data.csv"

# Response:
# {
#   "dataset_id": "abc-123",
#   "columns": ["product", "sales", "date"],
#   "rows": 1000
# }
```

### 2. Start Conversation
```bash
curl -X POST $BACKEND_URL/api/v1/conversation/start \
  -H "Content-Type: application/json" \
  -d '{"user_id":"demo-user"}'

# Response:
# {
#   "session_id": "xyz-789",
#   "message": "Hello! I'm your advanced data analysis agent..."
# }
```

### 3. Analyze Data
```bash
curl -X POST $BACKEND_URL/api/v1/conversation/continue \
  -H "Content-Type: application/json" \
  -d '{
    "session_id":"xyz-789",
    "message":"Show me sales trends",
    "dataset_id":"abc-123"
  }'

# Response:
# {
#   "response": "I've analyzed your sales data...",
#   "tools_used": ["analyze_data", "create_visualization"],
#   "function_results": [...]
# }
```

## Next Steps

1. ✅ **Explore the UI**: Open your frontend URL in a browser
2. ✅ **Upload Sample Data**: Use `sample_data.csv`
3. ✅ **Try Conversations**: Ask questions about your data
4. ✅ **Test Accessibility**: Enable TTS and audio descriptions
5. ✅ **Review Monitoring**: Check Cloud Console for metrics

## Resources

- **Full Documentation**: See `README.md`
- **Architecture**: See `ARCHITECTURE.md`
- **Production Checklist**: See `PRODUCTION_CHECKLIST.md`
- **Refactoring Summary**: See `REFACTORING_SUMMARY.md`
- **API Docs**: Visit `$BACKEND_URL/docs`

## Support

### Check Status
```bash
# Backend health
curl $BACKEND_URL/api/v1/health | jq

# Metrics
curl $BACKEND_URL/metrics
```

### View Logs
```bash
# Recent errors
gcloud run services logs read botlytics-backend \
  --region=us-central1 \
  --filter="severity>=ERROR" \
  --limit=20
```

### Get Help
1. Check logs for errors
2. Review `PRODUCTION_CHECKLIST.md`
3. Run diagnostic tests: `python test-cloud-run.py`
4. Check GitHub Issues
5. Review Cloud Console

---

**Quick Start Complete!** 🎉

Your Botlytics application is now running on Google Cloud Run with:
- ✅ Advanced AI agent with tool calling
- ✅ Multi-turn conversations with memory
- ✅ Safe code execution
- ✅ Comprehensive accessibility features
- ✅ Production-grade monitoring
- ✅ Auto-scaling and high availability

**Access your application**:
- Frontend: `$FRONTEND_URL`
- Backend API: `$BACKEND_URL`
- API Docs: `$BACKEND_URL/docs`

Enjoy your enterprise-grade data analytics platform! 🚀
