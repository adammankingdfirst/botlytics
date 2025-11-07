# 🚀 Production Deployment Checklist for GCP Cloud Run

## Pre-Deployment Checklist

### ✅ GCP Setup
- [ ] GCP Project created and configured
- [ ] All required APIs enabled (Vertex AI, Cloud Run, Storage, BigQuery, TTS, STT)
- [ ] Service account created with minimal permissions
- [ ] Service account key downloaded (for local testing only)
- [ ] GCS bucket created for data storage
- [ ] Environment variables configured

### ✅ Code Quality
- [ ] All tests passing (`pytest tests/ -v`)
- [ ] No linting errors (`flake8 backend/`)
- [ ] Code formatted (`black backend/`)
- [ ] Security scan passed (`safety check`)
- [ ] Diagnostics clean (no TypeScript/Python errors)

### ✅ Docker Configuration
- [ ] Backend Dockerfile builds successfully
- [ ] Frontend Dockerfile builds successfully
- [ ] Docker Compose works locally
- [ ] Health checks configured
- [ ] Non-root user configured
- [ ] Security headers enabled

### ✅ Security Verification
- [ ] Code sanitization working (dangerous patterns blocked)
- [ ] Resource limits configured (memory, CPU, timeout)
- [ ] Input validation on all endpoints
- [ ] SQL injection prevention verified
- [ ] File upload size limits enforced
- [ ] CORS configured appropriately
- [ ] Secrets not in code (use environment variables)

## Deployment Steps

### 1. Local Testing
```bash
# Test locally first
docker-compose up backend

# Run integration tests
python test-integration.py --backend-url http://localhost:8080

# Test with sample data
curl -X POST http://localhost:8080/api/v1/upload \
  -F "file=@sample_data.csv"
```

### 2. Deploy Backend
```bash
# Set environment variables
export GCP_PROJECT_ID="your-project-id"
export GCS_BUCKET="your-bucket-name"
export GCP_LOCATION="us-central1"

# Deploy backend
./deploy-backend.sh

# Verify deployment
BACKEND_URL=$(gcloud run services describe botlytics-backend \
  --region=$GCP_LOCATION --format='value(status.url)')

curl $BACKEND_URL/api/v1/health
```

### 3. Deploy Frontend
```bash
# Deploy frontend with backend URL
export BACKEND_URL="https://botlytics-backend-xyz.run.app"
./deploy-frontend.sh

# Verify deployment
FRONTEND_URL=$(gcloud run services describe botlytics-frontend \
  --region=$GCP_LOCATION --format='value(status.url)')

curl $FRONTEND_URL/_stcore/health
```

### 4. Run Cloud Run Tests
```bash
# Comprehensive Cloud Run testing
python test-cloud-run.py \
  --backend-url $BACKEND_URL \
  --frontend-url $FRONTEND_URL \
  --wait 30
```

## Post-Deployment Verification

### ✅ Health Checks
- [ ] Backend health endpoint returns 200
- [ ] All service checks show "ok" or "not_configured"
- [ ] Frontend health endpoint returns 200
- [ ] Metrics endpoint accessible

### ✅ Functionality Tests
- [ ] File upload works
- [ ] Conversation start/continue works
- [ ] Code interpreter executes safely
- [ ] Reasoning chains complete
- [ ] Accessibility features work (TTS, STT, audio descriptions)
- [ ] Visualizations generate correctly

### ✅ Performance Tests
- [ ] Cold start < 10 seconds (with cpu-boost)
- [ ] Warm requests < 1 second
- [ ] Concurrent requests handled (10+ simultaneous)
- [ ] Memory usage within limits
- [ ] No timeout errors

### ✅ Security Tests
- [ ] Dangerous code blocked
- [ ] Invalid inputs rejected
- [ ] Error messages don't leak sensitive info
- [ ] Rate limiting works (if configured)
- [ ] Authentication works (if configured)

## Monitoring Setup

### ✅ Cloud Monitoring
- [ ] Custom metrics configured
- [ ] Alerting policies created
- [ ] Log-based metrics enabled
- [ ] Error reporting configured
- [ ] Uptime checks configured

### ✅ Prometheus Metrics
- [ ] Metrics endpoint accessible
- [ ] Request counters working
- [ ] Duration histograms working
- [ ] LLM call metrics tracking
- [ ] Code execution metrics tracking

## Production Configuration

### ✅ Cloud Run Settings
```bash
# Recommended production settings
--memory=2Gi                    # Adequate for data processing
--cpu=2                         # Good performance
--timeout=300                   # 5 minutes for complex operations
--max-instances=20              # Scale up to 20 instances
--min-instances=1               # Always warm (no cold starts)
--concurrency=80                # Handle 80 concurrent requests
--execution-environment=gen2    # Better performance
--cpu-boost                     # Faster cold starts
```

### ✅ Environment Variables
```bash
# Required
GCP_PROJECT_ID=your-project-id
GCS_BUCKET=your-bucket-name
GCP_LOCATION=us-central1

# Optional
LOG_LEVEL=INFO
MAX_FILE_SIZE=52428800  # 50MB
MAX_DATASET_ROWS=1000000
```

## Cost Optimization

### ✅ Cost Controls
- [ ] BigQuery query limits configured (100MB per query)
- [ ] File size limits enforced (50MB)
- [ ] Dataset size limits enforced (1M rows)
- [ ] Request timeouts configured
- [ ] Auto-scaling limits set

### ✅ Estimated Monthly Costs
- **Cloud Run Backend**: $15-25/month (with min-instances=1)
- **Cloud Run Frontend**: $5-10/month (with min-instances=0)
- **Cloud Storage**: $0.02/GB/month
- **Vertex AI**: $0.000125 per 1K characters
- **TTS/STT**: Pay per use
- **Total Estimated**: $30-50/month for moderate usage

## Troubleshooting

### Common Issues

**1. Cold Start Timeout**
```bash
# Solution: Enable cpu-boost and increase timeout
gcloud run services update botlytics-backend \
  --cpu-boost \
  --timeout=300
```

**2. Memory Errors**
```bash
# Solution: Increase memory allocation
gcloud run services update botlytics-backend \
  --memory=4Gi
```

**3. Permission Errors**
```bash
# Solution: Verify service account permissions
gcloud projects get-iam-policy $GCP_PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:botlytics-sa@*"
```

**4. Frontend Can't Reach Backend**
```bash
# Solution: Verify BACKEND_URL environment variable
gcloud run services describe botlytics-frontend \
  --region=$GCP_LOCATION \
  --format='value(spec.template.spec.containers[0].env)'
```

## Rollback Procedure

If deployment fails:

```bash
# Rollback backend to previous revision
gcloud run services update-traffic botlytics-backend \
  --to-revisions=PREVIOUS_REVISION=100 \
  --region=$GCP_LOCATION

# Rollback frontend to previous revision
gcloud run services update-traffic botlytics-frontend \
  --to-revisions=PREVIOUS_REVISION=100 \
  --region=$GCP_LOCATION
```

## Maintenance

### Regular Tasks
- [ ] Review logs weekly
- [ ] Check error rates
- [ ] Monitor costs
- [ ] Update dependencies monthly
- [ ] Review security advisories
- [ ] Test disaster recovery

### Updates
```bash
# Update backend
docker build -t gcr.io/$GCP_PROJECT_ID/botlytics-backend:latest -f .dockerfile .
docker push gcr.io/$GCP_PROJECT_ID/botlytics-backend:latest
gcloud run deploy botlytics-backend --image gcr.io/$GCP_PROJECT_ID/botlytics-backend:latest

# Update frontend
docker build -t gcr.io/$GCP_PROJECT_ID/botlytics-frontend:latest -f frontend.dockerfile .
docker push gcr.io/$GCP_PROJECT_ID/botlytics-frontend:latest
gcloud run deploy botlytics-frontend --image gcr.io/$GCP_PROJECT_ID/botlytics-frontend:latest
```

## Success Criteria

✅ **Deployment is successful when:**
- All health checks pass
- All Cloud Run tests pass (>90%)
- Cold start < 10 seconds
- Warm requests < 1 second
- No security vulnerabilities
- Error rate < 1%
- Uptime > 99.5%

## Support

For issues:
1. Check logs: `gcloud run services logs read botlytics-backend --region=$GCP_LOCATION`
2. Review metrics in Cloud Console
3. Run diagnostic tests: `python test-cloud-run.py`
4. Check GitHub Issues
5. Review documentation

---

**Last Updated**: 2024
**Version**: 1.0.0
**Status**: Production Ready ✅
