#!/bin/bash

# Deploy Complete Botlytics Stack to Google Cloud Run

set -e

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"your-project-id"}
REGION=${GCP_LOCATION:-"us-central1"}

echo "🚀 Deploying Complete Botlytics Stack to Google Cloud Run..."
echo "📋 Project: $PROJECT_ID"
echo "🌍 Region: $REGION"
echo ""

# Step 1: Deploy Backend
echo "🔧 Step 1: Deploying Backend..."
./deploy-backend.sh

# Get backend URL for frontend
BACKEND_URL=$(gcloud run services describe botlytics-backend --region=$REGION --format='value(status.url)')
export BACKEND_URL

echo ""
echo "⏳ Waiting 30 seconds for backend to be fully ready..."
sleep 30

# Step 2: Deploy Frontend
echo "🎨 Step 2: Deploying Frontend..."
./deploy-frontend.sh

# Get frontend URL
FRONTEND_URL=$(gcloud run services describe botlytics-frontend --region=$REGION --format='value(status.url)')

echo ""
echo "🎉 DEPLOYMENT COMPLETE!"
echo "=================================="
echo "🔗 Backend API: $BACKEND_URL"
echo "🎨 Frontend App: $FRONTEND_URL"
echo "📊 API Docs: $BACKEND_URL/docs"
echo "🔍 Health Check: $BACKEND_URL/api/v1/health"
echo ""
echo "🧪 Test the complete application:"
echo "1. Open: $FRONTEND_URL"
echo "2. Upload a CSV file"
echo "3. Start a conversation with the AI agent"
echo "4. Try accessibility features in the Accessibility tab"
echo ""
echo "🎯 Your enterprise-grade data analytics platform is now live!"