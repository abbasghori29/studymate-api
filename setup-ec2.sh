#!/bin/bash
# Initial Lightsail setup script (Amazon Linux 2023)
# Run this ONCE on your Lightsail instance for manual setup
# NOTE: The GitHub Actions workflow handles all of this automatically

set -e

echo "🚀 Setting up Lightsail instance for StudyMate API..."

# Update system
echo "📦 Updating system packages..."
sudo dnf update -y

# Install Python 3.11 and required packages
echo "🐍 Installing Python 3.11..."
sudo dnf install -y python3.11 python3.11-pip python3.11-devel git

# Install system dependencies
echo "📦 Installing build dependencies..."
sudo dnf install -y gcc gcc-c++ make cmake nginx openssl

# Setup swap file (safety net for 2GB RAM)
if [ ! -f /swapfile ]; then
    echo "💾 Creating 1GB swap file..."
    sudo fallocate -l 1G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

# Create app directory
APP_DIR="/home/ec2-user/studymate-api"
echo "📁 Creating app directory: $APP_DIR"
mkdir -p "$APP_DIR"
cd "$APP_DIR"

# Clone repository (if not already cloned)
if [ ! -d ".git" ]; then
    echo "📥 Clone your repository here:"
    echo "    git clone https://github.com/YOUR_USER/studymate-api.git ."
fi

# Create virtual environment
echo "🐍 Creating virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "📥 Installing Python dependencies..."
pip install -r requirements.txt

# Copy systemd service file
echo "⚙️  Setting up systemd service..."
sudo cp studymate-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable studymate-api

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file template..."
    cat > .env << 'EOF'
PROJECT_NAME=StudyMate API
VERSION=0.1.0
DEBUG=False
HOST=0.0.0.0
PORT=8005

DATABASE_URL=sqlite:///./app.db

SECRET_KEY=change-this-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

BACKEND_CORS_ORIGINS=["http://localhost:3000","http://localhost:8005"]

OPENAI_API_KEY=your-openai-api-key-here
VECTOR_STORE_PATH=faiss_index_openai

PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_INDEX_NAME=cafs-chatbot-memory
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
EOF
    chmod 600 .env
    echo "⚠️  Please edit .env file with your actual keys!"
fi

# Set permissions
echo "🔒 Setting permissions..."
chmod +x deploy.sh
chown -R ec2-user:ec2-user "$APP_DIR"

echo ""
echo "✅ Lightsail setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file: nano $APP_DIR/.env"
echo "2. Start the service: sudo systemctl start studymate-api"
echo "3. Check status: sudo systemctl status studymate-api"
echo "4. View logs: sudo journalctl -u studymate-api -f"
