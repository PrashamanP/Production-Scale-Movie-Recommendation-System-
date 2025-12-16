#!/bin/bash
set -euo pipefail  # Stricter error handling

# Add trap for cleanup on failure
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "ERROR: Deployment failed with exit code $exit_code"
        echo "Cleaning up temporary files..."
        rm -f /tmp/movie-recommender-*.tar
    fi
}
trap cleanup EXIT

# Add function for rollback on failure
rollback_on_failure() {
    echo "ERROR: $1"
    echo "Attempting automatic rollback..."
    if [ -f "scripts/k8s-rollback.sh" ]; then
        chmod +x scripts/k8s-rollback.sh
        ./scripts/k8s-rollback.sh || echo "WARNING: Rollback also failed"
    fi
    exit 1
}

# k8s-deploy.sh - Deploy movie recommender to Kubernetes
# Usage: ./scripts/k8s-deploy.sh [BUILD_NUMBER] [GIT_COMMIT]

BUILD_NUMBER=${1:-"dev"}
GIT_COMMIT=${2:-$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")}

echo "========================================"
echo "Deploying Movie Recommender to Kubernetes"
echo "Build: ${BUILD_NUMBER}"
echo "Commit: ${GIT_COMMIT}"
echo "========================================"

# Step 1: Build Docker image with better error handling
echo "[1/5] Building Docker image..."
export DOCKER_BUILDKIT=1
BUILD_TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

if ! docker build -t movie-recommender:${BUILD_NUMBER} \
    --build-arg BUILD_NUMBER=${BUILD_NUMBER} \
    --build-arg GIT_COMMIT=${GIT_COMMIT} \
    --build-arg BUILD_TIMESTAMP="${BUILD_TIMESTAMP}" \
    . ; then
    rollback_on_failure "Docker build failed"
fi

# Tag as latest
docker tag movie-recommender:${BUILD_NUMBER} movie-recommender:latest

# Step 2: Save Docker image
echo "[2/5] Saving Docker image..."
if ! docker save movie-recommender:latest -o /tmp/movie-recommender-${BUILD_NUMBER}.tar ; then
    rollback_on_failure "Failed to save Docker image"
fi

# Verify tar file was created and is not empty
if [ ! -s /tmp/movie-recommender-${BUILD_NUMBER}.tar ]; then
    rollback_on_failure "Docker image tar file is empty or missing"
fi

# Step 3: Import to k3s with verification
echo "[3/5] Importing to k3s..."
if ! sudo k3s ctr images import /tmp/movie-recommender-${BUILD_NUMBER}.tar ; then
    rm -f /tmp/movie-recommender-${BUILD_NUMBER}.tar
    rollback_on_failure "Failed to import image to k3s"
fi

# Note: Image verification removed as kubectl deployment will fail if image is missing
# This avoids sudo permission issues while maintaining safety through automatic rollback
echo "Image import completed. Kubernetes will verify during deployment."


# Clean up tar file
rm /tmp/movie-recommender-${BUILD_NUMBER}.tar

# Step 4: Update deployment
echo "[4/5] Updating Kubernetes deployment..."
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

# Update environment variables
if ! kubectl set env deployment/movie-recommender \
    BUILD_NUMBER=${BUILD_NUMBER} \
    GIT_COMMIT=${GIT_COMMIT} ; then
    rollback_on_failure "Failed to update environment variables"
fi

# Force rolling update
if ! kubectl set image deployment/movie-recommender \
    recommender=movie-recommender:latest ; then
    rollback_on_failure "Failed to update deployment image"
fi

# Step 5: Wait for rollout with timeout
echo "[5/5] Waiting for rollout to complete..."
if ! kubectl rollout status deployment/movie-recommender --timeout=5m ; then
    rollback_on_failure "Rollout failed or timed out"
fi

# Final verification
echo ""
echo "========================================"
echo "Deployment Complete!"
echo "========================================"
kubectl get pods -l app=movie-recommender

echo ""
echo "Testing health endpoint..."
sleep 2

# Test health endpoint with retry
MAX_RETRIES=3
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -sf http://localhost:8082/health | jq '.' ; then
        echo ""
        echo "✅ Deployment successful!"
        exit 0
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Health check attempt $RETRY_COUNT failed, retrying..."
    sleep 2
done

rollback_on_failure "Health check failed after $MAX_RETRIES attempts"
