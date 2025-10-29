#!/bin/bash

# Local development script

set -e

echo "🚀 Starting Botlytics locally..."

# Check if environment variables are set
if [ -z "$GCP_PROJECT_ID" ]; then
    echo "⚠️  Warning: GCP_PROJECT_ID not set"
fi

if [ -z "$GCS_BUCKET" ]; then
    echo "⚠️  Warning: GCS_BUCKET not set"
fi

if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "⚠️  Warning: GOOGLE_APPLICATION_CREDENTIALS not set"
fi

# Start backend
echo "🔧 Starting FastAPI backend..."
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start frontend in another terminal (optional)
echo "🎨 To start the frontend, run in another terminal:"
echo "cd frontend && streamlit run app.py --server.port 8501"

echo ""
echo "✅ Backend running at: http://localhost:8080"
echo "📊 API docs at: http://localhost:8080/docs"
echo "🎨 Frontend (manual): cd frontend && streamlit run app.py"
echo ""
echo "Press Ctrl+C to stop..."

# Wait for interrupt
trap "kill $BACKEND_PID" EXIT
wait $BACKEND_PID