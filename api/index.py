"""
Vercel Serverless Function Entrypoint for MAUSAM.
Exposes FastAPI application, OpenAPI schema, Swagger docs (/docs),
and real-time live satellite/NWP streaming directly on Vercel.
"""
import os
import sys

# Ensure root directory is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.app.main import app

# Vercel ASGI Handler
handler = app
