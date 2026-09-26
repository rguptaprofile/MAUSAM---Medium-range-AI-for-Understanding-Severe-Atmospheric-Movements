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

try:
    from backend.app.main import app
    handler = app
except Exception as e:
    import traceback
    traceback.print_exc()
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    app = FastAPI(title="MAUSAM Fallback")
    err_msg = traceback.format_exc()
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    def catch_all(path: str):
        return JSONResponse(status_code=500, content={"error": "FastAPI initialization failed", "details": err_msg})
    handler = app
