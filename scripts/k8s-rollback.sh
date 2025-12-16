#!/bin/bash
set -e  # Exit on any error

# k8s-rollback.sh - Rollback movie recommender deployment
# Usage: ./scripts/k8s-rollback.sh

echo "========================================"
echo "Rolling back Movie Recommender Deployment"
echo "========================================"

export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

# Show current rollout history
echo "Current rollout history:"
kubectl rollout history deployment/movie-recommender
echo ""

# Perform rollback
echo "Initiating rollback..."
kubectl rollout undo deployment/movie-recommender

# Wait for rollback to complete
echo "Waiting for rollback to complete..."
kubectl rollout status deployment/movie-recommender --timeout=5m

# Verify deployment
echo ""
echo "========================================"
echo "Rollback Complete!"
echo "========================================"
kubectl get pods -l app=movie-recommender
echo ""
echo "Current deployment details:"
kubectl describe deployment movie-recommender | grep -A 3 "^Pod Template:"
echo ""
echo "Testing health endpoint..."
sleep 2  # Give pods a moment to be ready
curl -s http://localhost:8082/health | jq '.' || echo "Health check failed"
echo ""
echo "Rollback successful!"
