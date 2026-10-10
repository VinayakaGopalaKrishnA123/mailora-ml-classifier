# Mailora ML Classifier — AWS Cloud Deployment Guide

This guide explains how to deploy the **Mailora ML Email Classifier (v1.1.0)** microservice to **Amazon Web Services (AWS)** so that your Mailora backend and frontend can access it over a secure public HTTPS endpoint 24/7.

---

## Architecture Overview

```
 ┌─────────────────────────────────────────────────────────┐
 │                   Mailora Production                    │
 │         Node.js Server / Kafka / React Frontend         │
 └────────────────────────────┬────────────────────────────┘
                              │ HTTPS POST /predict
                              ▼
 ┌─────────────────────────────────────────────────────────┐
 │                        AWS Cloud                        │
 │                                                         │
 │    AWS App Runner / ECS / EC2 (FastAPI ML Container)    │
 │    • Port 8000 (HTTPS TLS auto-managed)                │
 │    • Baseline Model v1.1.0 (sub-5ms inference)          │
 │    • Auto-scaling & automatic health checks             │
 └─────────────────────────────────────────────────────────┘
```

---

## Deployment Options at a Glance

| Method | Ease | Cost | Recommended For |
| :--- | :---: | :---: | :--- |
| **Method 1: AWS App Runner (GitHub)** | ⭐⭐⭐⭐⭐ (Easiest, No Docker needed) | ~$5–$15/mo (Pay-as-you-go) | **Recommended for fast production deployment** |
| **Method 2: AWS App Runner (via ECR)**| ⭐⭐⭐⭐ (Docker container) | ~$5–$15/mo | Complete container isolation & CI/CD |
| **Method 3: AWS EC2 (t3.micro / t4g)** | ⭐⭐⭐ (Standard Linux VM) | **Free Tier eligible** | Full server control or minimum fixed cost |
| **Method 4: AWS Lightsail Containers**| ⭐⭐⭐⭐ (Simple GUI) | $7/mo fixed | Fixed monthly budget with zero surprises |

---

## Method 1: AWS App Runner via GitHub (Recommended — Zero CLI Needed)

AWS App Runner is AWS's fully managed serverless container service. It provisions automated SSL/TLS certificates, provides a public HTTPS endpoint, monitors health checks, and automatically scales up and down.

### Step 1: Push This Code to GitHub
1. Create a repository on GitHub (e.g., `mailora-ml-classifier`).
2. In your terminal (`c:\Users\srini\Desktop\Model`):
   ```bash
   git init
   git add .
   git commit -m "feat: Mailora ML Classifier v1.1.0 with false positive fix"
   git remote add origin https://github.com/<your-username>/mailora-ml-classifier.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Create App Runner Service in AWS Console
1. Log into your [AWS Management Console](https://console.aws.amazon.com/).
2. In the top search bar, type **App Runner** and click it.
3. Click **Create service**.
4. In **Source**:
   - Select **Source code repository**.
   - Click **Add new** under GitHub connection, authorize AWS with your GitHub account, and select your repository: `mailora-ml-classifier` (Branch: `main`).
   - Under **Deployment trigger**, choose **Automatic** (so any new git push automatically redeploys).
5. In **Configure build**:
   - **Runtime**: Select **Python 3**.
   - **Build command**:
     ```bash
     pip install -r requirements.txt
     ```
   - **Start command**:
     ```bash
     uvicorn src.api:app --host 0.0.0.0 --port 8000
     ```
   - **Port**: Enter `8000`.
6. In **Configure service**:
   - **Service name**: `mailora-email-classifier`
   - **CPU & Memory**: `1 vCPU`, `2 GB` (plenty for high-throughput baseline).
   - Under **Health check**, leave path as `/health`.
7. Click **Next** -> **Create & deploy**.

Within 3–5 minutes, AWS App Runner will provide you with a permanent HTTPS URL, such as:
`https://abc123xyz.us-east-1.awsapprunner.com`

---

## Method 2: AWS App Runner via Amazon ECR (Using Docker)

If you prefer building and pushing a Docker container:

### Step 1: Install AWS CLI (if not installed)
In PowerShell (as Administrator):
```powershell
winget install Amazon.AWSCLI
```
Then configure your AWS credentials:
```powershell
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Enter default region (e.g., us-east-1)
# Enter output format (json)
```

### Step 2: Push Image to Amazon ECR
Start **Docker Desktop** on your machine, then run the provided batch script:
```cmd
deployment\aws_ecr_push.bat
```
*(Or enter your 12-digit AWS Account ID when prompted).*

The script will automatically:
1. Create the ECR repository `mailora-email-classifier` if it doesn't exist.
2. Authenticate Docker with your private AWS ECR registry.
3. Build the production Docker image using [Dockerfile](file:///c:/Users/srini/Desktop/Model/Dockerfile).
4. Tag and push the image to: `<account-id>.dkr.ecr.<region>.amazonaws.com/mailora-email-classifier:latest`.

### Step 3: Deploy in App Runner
1. In AWS App Runner Console -> **Create service**.
2. Select **Container registry** -> **Amazon ECR**.
3. Browse and select image URI: `mailora-email-classifier:latest`.
4. Deployment settings: **Automatic**.
5. Port: `8000`. Click **Deploy**.

---

## Method 3: Deploy on AWS EC2 (Free-Tier Eligible)

If you want to use an EC2 virtual machine:

### Step 1: Launch an EC2 Instance
1. In AWS Console, go to **EC2** -> **Launch Instance**.
2. **Name**: `mailora-ml-server`
3. **AMI**: Ubuntu 24.04 LTS (x86_64).
4. **Instance Type**: `t3.micro` or `t2.micro` (Free-tier eligible).
5. **Key pair**: Create or select an existing SSH key pair.
6. **Network Settings**:
   - Check **Allow SSH traffic** (Port 22).
   - Check **Allow HTTP traffic** (Port 80).
   - Click **Edit** and add a Custom TCP Rule for Port `8000` (Source: `0.0.0.0/0` or Mailora server IP).
7. Click **Launch Instance**.

### Step 2: Connect and Set Up
SSH into your instance:
```bash
ssh -i your-key.pem ubuntu@<ec2-public-ip>
```

Run on the server:
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python & Git
sudo apt install -y python3-pip python3-venv git

# Clone repository
git clone https://github.com/<your-username>/mailora-ml-classifier.git
cd mailora-ml-classifier

# Create virtual environment and install
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Run as a Persistent Systemd Service
Create the service configuration:
```bash
sudo nano /etc/systemd/system/mailora-classifier.service
```

Paste:
```ini
[Unit]
Description=Mailora ML Email Classifier FastAPI Service
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/mailora-ml-classifier
ExecStart=/home/ubuntu/mailora-ml-classifier/venv/bin/uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=5
Environment=MODEL_TYPE=baseline

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable mailora-classifier
sudo systemctl start mailora-classifier
sudo systemctl status mailora-classifier
```

Your EC2 service is now live at: `http://<ec2-public-ip>:8000`

---

## Connecting Mailora to Your Cloud Deployment

Once deployed, link Mailora's backend to your cloud endpoint:

### In Mailora Server `.env`:
File: `C:\Users\srini\Desktop\Mailora\mail-or-a-frontend-backend\server\.env`
```env
# Replace with your AWS App Runner HTTPS URL or EC2 Public IP
EMAIL_CLASSIFIER_URL=https://abc123xyz.us-east-1.awsapprunner.com
EMAIL_CLASSIFIER_ENABLED=true
EMAIL_CLASSIFIER_TIMEOUT_MS=3000
```

### Verify Connection:
Test from your local terminal:
```bash
# Health Check
curl -X GET https://abc123xyz.us-east-1.awsapprunner.com/health

# Prediction Test (Glasp newsletter test)
curl -X POST https://abc123xyz.us-east-1.awsapprunner.com/predict \
  -H "Content-Type: application/json" \
  -d '{"subject": "Why AI hasn'\''t replaced software engineers, and won'\''t", "body": "Glasp newsletter with 5 articles on AI and engineering."}'
```
Expected response:
```json
{
  "label": "NON_CAREER_RELATED",
  "is_career_related": false,
  "confidence": 0.8552,
  "career_probability": 0.1448,
  "model_version": "1.1.0"
}
```
