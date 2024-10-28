#!/bin/bash


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE_DATA_DIR="${SCRIPT_DIR}/data/datasets"

cd "${SCRIPT_DIR}"
export DATASET_NAME="motion-forecasting"  # sensor, lidar, motion-forecasting or tbv.
export TARGET_DIR="$BASE_DATA_DIR"  # Target directory on your machine.
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/$DATASET_NAME/*" $TARGET_DIR
