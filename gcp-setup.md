# GCP Setup Instructions

## 1. Create GCP Project
```bash
# Set your project ID
export PROJECT_ID="botlytics-$(date +%s)"
gcloud projects create $PROJECT_ID
gcloud config set project $PROJECT_ID
```

## 2. Enable Required APIs
```bash
gcloud services enable \
  aiplatform.googleapis.com \
  run.googleapis.com \
  storage.googleapis.com \
  bigquery.googleapis.com \
  cloudbuild.googleapis.com
```

## 3. Create Service Account with Minimal IAM
```bash
# Create service account
gcloud iam service-accounts create botlytics-sa \
  --display-name="Botlytics Service Account"

# Grant minimal required permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:botlytics-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:botlytics-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:botlytics-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataEditor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:botlytics-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/bigquery.jobUser"

# Create and download service account key
gcloud iam service-accounts keys create service-account-key.json \
  --iam-account=botlytics-sa@$PROJECT_ID.iam.gserviceaccount.com
```

## 4. Create Storage Bucket
```bash
export BUCKET_NAME="botlytics-data-$PROJECT_ID"
gsutil mb gs://$BUCKET_NAME
```

## 5. Environment Variables
```bash
export GCP_PROJECT_ID=$PROJECT_ID
export GCS_BUCKET=$BUCKET_NAME
export GOOGLE_APPLICATION_CREDENTIALS="./service-account-key.json"
```