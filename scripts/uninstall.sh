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

# Stop and disable service
echo "Stopping and disabling service..."
systemctl stop ml-scheduler
systemctl disable ml-scheduler

# Remove systemd service
echo "Removing systemd service..."
rm -f /etc/systemd/system/ml-scheduler.service
systemctl daemon-reload

# Remove directories
echo "Removing directories..."
rm -rf "$PIPELINE_DIR"
rm -rf "$LOG_DIR"
rm -rf "$CONFIG_DIR"

# Remove user and group
echo "Removing user and group..."
userdel "$USER" 2>/dev/null || true
groupdel "$GROUP" 2>/dev/null || true

echo "Uninstallation completed successfully!" 
