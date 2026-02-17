#!/bin/bash
# Manual deployment script for StudyMate API on Lightsail
# Run this ON the Lightsail instance if you need to deploy manually
# (The GitHub Actions workflow handles this automatically)

set -e

APP_DIR="${APP_DIR:-/home/ec2-user/studymate-api}"
VENV_DIR="$APP_DIR/venv"
SERVICE_NAME="studymate-api"

echo "🚀 Starting manual deployment..."

# Navigate to app directory
cd "$APP_DIR"

# Pull latest code
echo "📥 Pulling latest code from main branch..."
git fetch origin
git reset --hard origin/main

# Activate virtual environment
echo "🐍 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install/update dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Copy service file and reload
echo "⚙️  Updating systemd service..."
sudo cp studymate-api.service /etc/systemd/system/studymate-api.service
sudo systemctl daemon-reload

# Restart services
echo "🔄 Restarting services..."
sudo systemctl restart "$SERVICE_NAME"
sudo systemctl restart nginx

# Show status
echo ""
echo "📊 Service status:"
sudo systemctl status "$SERVICE_NAME" --no-pager || echo "⚠️  Service status unavailable"

echo ""
echo "✅ Manual deployment completed!"
