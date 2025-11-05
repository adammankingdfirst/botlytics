#!/bin/bash

# Deploy Streamlit Frontend to Google Cloud Run

set -e

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"your-project-id"}
SERVICE_NAME="botlytics-frontend"
REGION=${GCP_LOCATION:-"us-central1"}
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"
BACKEND_URL=${BACKEND_URL:-"https://botlytics-backend-[hash]-uc.a.run.app"}

echo "🎨 Deploying Botlytics Frontend to Google Cloud Run..."

# Validate environment
if [ -z "$GCP_PROJECT_ID" ]; then
    echo "❌ Error: GCP_PROJECT_ID must be set"
    exit 1
fi

if [ -z "$BACKEND_URL" ]; then
    echo "⚠️  Warning: BACKEND_URL not set. Using default placeholder."
    echo "   Make sure to update this after backend deployment."
fi

# Build and push Docker image
echo "📦 Building frontend Docker image..."
docker build -t $IMAGE_NAME -f frontend.dockerfile .

echo "📤 Pushing frontend image to Container Registry..."
docker push $IMAGE_NAME

# Deploy frontend to Cloud Run
echo "🌐 Deploying frontend to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --set-env-vars="API_BASE_URL=$BACKEND_URL" \
  --memory=1Gi \
  --cpu=1 \
  --timeout=300 \
  --max-instances=10 \
  --min-instances=0 \
  --concurrency=50 \
  --execution-environment=gen2

# Get frontend service URL
FRONTEND_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')

echo ""
echo "✅ Frontend deployment complete!"
echo "🎨 Frontend URL: $FRONTEND_URL"
echo "🔗 Backend URL: $BACKEND_URL"
echo ""
echo "🧪 Test the application:"
echo "Open: $FRONTEND_URL"
echo ""
echo "📊 Your Botlytics application is now live!"