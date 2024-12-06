#!/usr/bin/env bash

echo "Building project package..."
python3 -m pip install -r requirements.txt

echo "Collecting static files..."
python3 manage.py  collectstatic --noinput


