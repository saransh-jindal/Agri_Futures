#!/bin/bash

# Exit on error
set -e

# Configuration
PIPELINE_DIR="/opt/ml-pipeline"
USER="mluser"
GROUP="mluser"
LOG_DIR="/var/log/ml-pipeline"
CONFIG_DIR="/etc/ml-pipeline"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root"
    exit 1
fi

# Create user and group if they don't exist
if ! id "$USER" &>/dev/null; then
    echo "Creating user $USER..."
    useradd -r -s /bin/false "$USER"
fi

# Create directories
echo "Creating directories..."
mkdir -p "$PIPELINE_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$CONFIG_DIR"

# Copy files
echo "Copying files..."
cp -r src "$PIPELINE_DIR/"
cp -r config "$PIPELINE_DIR/"
cp ml-scheduler.service /etc/systemd/system/

# Set permissions
echo "Setting permissions..."
chown -R "$USER:$GROUP" "$PIPELINE_DIR"
chown -R "$USER:$GROUP" "$LOG_DIR"
chown -R "$USER:$GROUP" "$CONFIG_DIR"
chmod 755 "$PIPELINE_DIR"
chmod 755 "$LOG_DIR"
chmod 755 "$CONFIG_DIR"

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Reload systemd
echo "Reloading systemd..."
systemctl daemon-reload

# Enable and start service
echo "Enabling and starting service..."
systemctl enable ml-scheduler
systemctl start ml-scheduler

echo "Installation completed successfully!"
echo "Service status:"
systemctl status ml-scheduler 
