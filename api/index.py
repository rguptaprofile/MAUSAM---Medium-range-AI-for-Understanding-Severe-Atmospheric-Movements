"""
Vercel Serverless Function Entrypoint for MAUSAM.
Exposes FastAPI application, OpenAPI schema, Swagger docs (/docs),
and real-time live satellite/NWP streaming directly on Vercel.
"""
import os
import sys

# Ensure root directory and current working directory are in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in [root_dir, os.getcwd()]:
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.app.main import app

# Export app for Vercel
handler = app
