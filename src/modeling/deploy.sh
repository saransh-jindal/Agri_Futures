#!/bin/bash

# Exit on error
set -e

# Configuration
PIPELINE_DIR="/opt/ml-pipeline"
BACKUP_DIR="/opt/ml-pipeline/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="backup_${TIMESTAMP}"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root"
    exit 1
fi

# Create backup
echo "Creating backup..."
mkdir -p "$BACKUP_DIR"
tar -czf "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" -C "$PIPELINE_DIR" .

# Stop service
echo "Stopping service..."
systemctl stop ml-scheduler

# Update files
echo "Updating files..."
cp -r src "$PIPELINE_DIR/"
cp -r config "$PIPELINE_DIR/"

# Install dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt

# Start service
echo "Starting service..."
systemctl start ml-scheduler

# Check service status
echo "Checking service status..."
if systemctl is-active --quiet ml-scheduler; then
    echo "Deployment completed successfully!"
    echo "Service is running."
else
    echo "Deployment completed, but service failed to start."
    echo "Rolling back to previous version..."
    ./rollback.sh
    exit 1
fi

# Clean up old backups (keep last 5)
echo "Cleaning up old backups..."
ls -t "${BACKUP_DIR}"/backup_*.tar.gz | tail -n +6 | xargs -r rm

echo "Deployment process completed!" 
