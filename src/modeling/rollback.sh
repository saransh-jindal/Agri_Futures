#!/bin/bash

# Exit on error
set -e

# Configuration
PIPELINE_DIR="/opt/ml-pipeline"
BACKUP_DIR="/opt/ml-pipeline/backups"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root"
    exit 1
fi

# Get latest backup
LATEST_BACKUP=$(ls -t "${BACKUP_DIR}"/backup_*.tar.gz | head -n1)

if [ -z "$LATEST_BACKUP" ]; then
    echo "No backup found!"
    exit 1
fi

echo "Rolling back to: $LATEST_BACKUP"

# Stop service
echo "Stopping service..."
systemctl stop ml-scheduler

# Restore from backup
echo "Restoring from backup..."
rm -rf "${PIPELINE_DIR:?}"/*
tar -xzf "$LATEST_BACKUP" -C "$PIPELINE_DIR"

# Start service
echo "Starting service..."
systemctl start ml-scheduler

# Check service status
echo "Checking service status..."
if systemctl is-active --quiet ml-scheduler; then
    echo "Rollback completed successfully!"
    echo "Service is running."
else
    echo "Rollback completed, but service failed to start."
    echo "Please check the logs for more information."
    exit 1
fi

echo "Rollback process completed!" 
