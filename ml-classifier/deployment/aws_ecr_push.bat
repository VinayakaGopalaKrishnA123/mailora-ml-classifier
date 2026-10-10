@echo off
setlocal enabledelayedexpansion

echo ====================================================================
echo        Mailora ML Classifier - AWS ECR Deployment Helper
echo ====================================================================
echo.

if "%AWS_REGION%"=="" set AWS_REGION=us-east-1
if "%AWS_ACCOUNT_ID%"=="" (
    set /p AWS_ACCOUNT_ID="Enter your 12-digit AWS Account ID: "
)

if "%AWS_ACCOUNT_ID%"=="" (
    echo [ERROR] AWS Account ID is required.
    exit /b 1
)

set REPO_NAME=mailora-email-classifier
set IMAGE_TAG=v1.1.0
set ECR_URI=%AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com/%REPO_NAME%

echo.
echo [1/4] Ensuring ECR Repository exists: %REPO_NAME%...
aws ecr describe-repositories --repository-names %REPO_NAME% --region %AWS_REGION% >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Repository does not exist. Creating repository...
    aws ecr create-repository --repository-name %REPO_NAME% --region %AWS_REGION%
)

echo.
echo [2/4] Authenticating Docker with AWS ECR...
aws ecr get-login-password --region %AWS_REGION% | docker login --username AWS --password-stdin %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com
if %ERRORLEVEL% neq 0 (
    echo [ERROR] ECR Docker authentication failed. Please verify 'aws configure' credentials.
    exit /b 1
)

echo.
echo [3/4] Building production Docker container...
docker build -t %REPO_NAME%:%IMAGE_TAG% -f Dockerfile .
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker build failed.
    exit /b 1
)

docker tag %REPO_NAME%:%IMAGE_TAG% %ECR_URI%:%IMAGE_TAG%
docker tag %REPO_NAME%:%IMAGE_TAG% %ECR_URI%:latest

echo.
echo [4/4] Pushing image to AWS ECR: %ECR_URI%...
docker push %ECR_URI%:%IMAGE_TAG%
docker push %ECR_URI%:latest

echo.
echo ====================================================================
echo  SUCCESS: Image pushed to AWS ECR!
echo  Image URI: %ECR_URI%:latest
echo.
echo  Next Step:
echo  Deploy this image in AWS App Runner or AWS ECS Fargate:
echo  1. Open AWS App Runner Console -> Create Service
echo  2. Select 'Container Registry' -> Amazon ECR
echo  3. Choose URI: %ECR_URI%:latest
echo  4. Port: 8000
echo ====================================================================
