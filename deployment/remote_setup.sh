#!/usr/bin/env bash
set -e

echo "=== [1/5] Updating packages and installing dependencies ==="
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv git curl

echo "=== [2/5] Cloning Mailora ML Classifier repository ==="
cd /home/ubuntu
if [ -d "mailora-ml-classifier" ]; then
    cd mailora-ml-classifier
    git pull origin main
else
    git clone https://github.com/VinayakaGopalaKrishnA123/mailora-ml-classifier.git
    cd mailora-ml-classifier
fi

echo "=== [3/5] Setting up Python virtual environment and dependencies ==="
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

echo "=== [4/5] Configuring systemd service ==="
sudo tee /etc/systemd/system/mailora-classifier.service > /dev/null << 'EOF'
[Unit]
Description=Mailora ML Email Classifier Service
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/mailora-ml-classifier
ExecStart=/home/ubuntu/mailora-ml-classifier/venv/bin/uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=3
Environment=MODEL_TYPE=baseline

[Install]
WantedBy=multi-user.target
EOF

echo "=== [5/5] Enabling and starting service ==="
sudo systemctl daemon-reload
sudo systemctl enable mailora-classifier
sudo systemctl restart mailora-classifier
sleep 3
sudo systemctl status mailora-classifier --no-pager
echo ""
echo "=== Testing Local Health Endpoint ==="
curl -s http://127.0.0.1:8000/health
echo ""
echo "=== Deployment on EC2 successfully finished! ==="
