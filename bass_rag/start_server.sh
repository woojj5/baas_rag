#!/bin/bash
cd /home/keti_spark1/j309
source .venv/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH
uvicorn app.refrag_server:app --host 0.0.0.0 --port 8012 --reload
