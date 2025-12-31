#!/bin/bash
# Alexandra AI - YouTube Video Creator
# Simple launcher for the video pipeline

cd "$(dirname "$0")"

# Run the pipeline
python3 video_pipeline.py "$@"
