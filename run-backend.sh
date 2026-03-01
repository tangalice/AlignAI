#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python pose_server.py
