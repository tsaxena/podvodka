#!/bin/bash
set -e

SWEEP_ID=$(wandb sweep sweep.yml 2>&1 | grep "wandb agent" | awk '{print $NF}')
echo "Starting sweep agent for: $SWEEP_ID"
wandb agent "$SWEEP_ID"
