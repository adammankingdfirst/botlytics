#!/bin/bash

# Deployment script for Botlytics to Google Cloud Run

set -e

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"your-project-id"}
SERVICE_NAME="botlytics-api"
REGION=${GCP_LOCATION:-"us-central1"}
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "🚀 Deploying Botlytics to Google Cloud Run..."

# Build and push Docker image
echo "📦 Building Docker image..."
docker build -t $IMAGE_NAME -f .dockerfile .

echo "📤 Pushing image to Container Registry..."
docker push $IMAGE_NAME

# Deploy to Cloud Run
echo "🌐 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --set-env-vars="GCP_PROJECT_ID=$PROJECT_ID,GCS_BUCKET=$GCS_BUCKET,GCP_LOCATION=$REGION" \
  --service-account="botlytics-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --memory=2Gi \
  --cpu=1 \
  --timeout=300 \
  --max-instances=10

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')

echo "✅ Deployment complete!"
echo "🌐 Service URL: $SERVICE_URL"
echo ""
echo "🧪 Test the deployment:"
echo "curl $SERVICE_URL/"
echo ""
echo "📊 Update your frontend to use: $SERVICE_URL"