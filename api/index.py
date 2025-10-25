# api/index.py
import os
from flask import Flask, jsonify, request
from crew_setup import run_crew

app = Flask(__name__)

@app.get("/")
def health():
    return jsonify({
        "status": "ok",
        "message": "Customer Support Analysis Crew API (Flask on Vercel)."
    })

@app.post("/run")
def run():
    # Optional: Accept a custom query for your tool (not strictly used in the mocked tool)
    payload = request.get_json(silent=True) or {}
    data_query = payload.get("data_query", "last quarter support data")

    # Hard guard for required env var
    if not os.getenv("GEMINI_API_KEY"):
        return jsonify({
            "error": "Missing GEMINI_API_KEY environment variable on server."
        }), 500

    try:
        report = run_crew(data_query=data_query)
        return jsonify({
            "ok": True,
            "data_query": data_query,
            "report": report
        })
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500
