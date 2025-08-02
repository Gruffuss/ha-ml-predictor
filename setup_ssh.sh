#!/bin/bash

# Script to setup SSH connection to ha-ml-predictor LXC container
# Run this manually to setup SSH keys

echo "Setting up SSH connection to ha-ml-predictor LXC container..."

# Copy SSH key to container (you'll need to enter password: ha-ml-predictor)
ssh-copy-id -i ~/.ssh/ha-ml-predictor.pub root@192.168.51.112

echo "SSH key setup complete. Testing connection..."

# Test connection
ssh -i ~/.ssh/ha-ml-predictor -o StrictHostKeyChecking=no root@192.168.51.112 "echo 'SSH connection successful!'"