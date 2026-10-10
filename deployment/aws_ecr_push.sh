#!/usr/bin/env bash
# ==============================================================================
# Mailora ML Classifier - AWS ECR Deployment Helper (Bash)
# ==============================================================================
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-}"

if [ -z "$AWS_ACCOUNT_ID" ]; then
    read -rp "Enter your 12-digit AWS Account ID: " AWS_ACCOUNT_ID
fi

if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo "[ERROR] AWS Account ID is required."
    exit 1
fi

REPO_NAME="mailora-email-classifier"
IMAGE_TAG="v1.1.0"
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}"

echo "===================================================================="
echo " [1/4] Ensuring ECR Repository exists: ${REPO_NAME}"
echo "===================================================================="
aws ecr describe-repositories --repository-names "${REPO_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1 || \
    aws ecr create-repository --repository-name "${REPO_NAME}" --region "${AWS_REGION}"

echo "===================================================================="
echo " [2/4] Authenticating Docker with AWS ECR"
echo "===================================================================="
aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "===================================================================="
echo " [3/4] Building production Docker container"
echo "===================================================================="
docker build -t "${REPO_NAME}:${IMAGE_TAG}" -f Dockerfile .
docker tag "${REPO_NAME}:${IMAGE_TAG}" "${ECR_URI}:${IMAGE_TAG}"
docker tag "${REPO_NAME}:${IMAGE_TAG}" "${ECR_URI}:latest"

echo "===================================================================="
echo " [4/4] Pushing image to AWS ECR: ${ECR_URI}"
echo "===================================================================="
docker push "${ECR_URI}:${IMAGE_TAG}"
docker push "${ECR_URI}:latest"

echo "===================================================================="
echo " SUCCESS: Image deployed to AWS ECR!"
echo " Image URI: ${ECR_URI}:latest"
echo "===================================================================="
