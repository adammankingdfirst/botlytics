#!/bin/bash

# Enhanced deployment script for Botlytics to Google Cloud Run

set -e

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"your-project-id"}
SERVICE_NAME="botlytics-api"
REGION=${GCP_LOCATION:-"us-central1"}
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"
SERVICE_ACCOUNT="botlytics-sa@$PROJECT_ID.iam.gserviceaccount.com"

echo "🚀 Deploying Botlytics to Google Cloud Run..."

# Validate environment
if [ -z "$GCP_PROJECT_ID" ] || [ -z "$GCS_BUCKET" ]; then
    echo "❌ Error: GCP_PROJECT_ID and GCS_BUCKET must be set"
    exit 1
fi

# Build and push Docker image
echo "📦 Building Docker image..."
docker build -t $IMAGE_NAME -f .dockerfile .

echo "📤 Pushing image to Container Registry..."
docker push $IMAGE_NAME

# Deploy to Cloud Run with enhanced security and monitoring
echo "🌐 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --set-env-vars="GCP_PROJECT_ID=$PROJECT_ID,GCS_BUCKET=$GCS_BUCKET,GCP_LOCATION=$REGION" \
  --service-account="$SERVICE_ACCOUNT" \
  --memory=2Gi \
  --cpu=2 \
  --timeout=300 \
  --max-instances=20 \
  --min-instances=1 \
  --concurrency=80 \
  --execution-environment=gen2 \
  --cpu-boost \
  --session-affinity

# Configure traffic allocation (100% to latest)
echo "🔄 Configuring traffic..."
gcloud run services update-traffic $SERVICE_NAME \
  --region=$REGION \
  --to-latest

# Set up monitoring and alerting
echo "📊 Setting up monitoring..."
gcloud logging sinks create botlytics-errors \
  bigquery.googleapis.com/projects/$PROJECT_ID/datasets/botlytics_logs \
  --log-filter='resource.type="cloud_run_revision" AND resource.labels.service_name="'$SERVICE_NAME'" AND severity>=ERROR' \
  --region=$REGION || echo "Logging sink already exists"

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')

echo ""
echo "✅ Deployment complete!"
echo "🌐 Service URL: $SERVICE_URL"
echo "🔍 Health check: $SERVICE_URL/api/v1/health"
echo ""
echo "🧪 Test the deployment:"
echo "curl $SERVICE_URL/api/v1/health"
echo ""
echo "📊 Update your frontend to use: $SERVICE_URL"
echo ""
echo "🔧 Monitoring:"
echo "- Logs: gcloud logging read 'resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"$SERVICE_NAME\"' --limit 50"
echo "- Metrics: https://console.cloud.google.com/run/detail/$REGION/$SERVICE_NAME/metrics"