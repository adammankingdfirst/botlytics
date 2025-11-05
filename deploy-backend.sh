#!/bin/bash

# Deploy FastAPI Backend to Google Cloud Run

set -e

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"your-project-id"}
SERVICE_NAME="botlytics-backend"
REGION=${GCP_LOCATION:-"us-central1"}
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "🚀 Deploying Botlytics Backend to Google Cloud Run..."

# Validate environment
if [ -z "$GCP_PROJECT_ID" ] || [ -z "$GCS_BUCKET" ]; then
    echo "❌ Error: GCP_PROJECT_ID and GCS_BUCKET must be set"
    exit 1
fi

# Build and push Docker image
echo "📦 Building backend Docker image..."
docker build -t $IMAGE_NAME -f .dockerfile .

echo "📤 Pushing backend image to Container Registry..."
docker push $IMAGE_NAME

# Deploy backend to Cloud Run
echo "🌐 Deploying backend to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --set-env-vars="GCP_PROJECT_ID=$PROJECT_ID,GCS_BUCKET=$GCS_BUCKET,GCP_LOCATION=$REGION" \
  --service-account="botlytics-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --memory=2Gi \
  --cpu=2 \
  --timeout=300 \
  --max-instances=20 \
  --min-instances=1 \
  --concurrency=80 \
  --execution-environment=gen2 \
  --cpu-boost

# Get backend service URL
BACKEND_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')

echo ""
echo "✅ Backend deployment complete!"
echo "🔗 Backend URL: $BACKEND_URL"
echo ""
echo "🧪 Test the backend:"
echo "curl $BACKEND_URL/api/v1/health"
echo ""
echo "📝 Save this URL for frontend deployment:"
echo "export BACKEND_URL=$BACKEND_URL"