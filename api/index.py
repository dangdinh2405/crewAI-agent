import os
from flask import Flask, jsonify, request
from crew_setup import run_crew, run_agent1, run_agent5
from flask_cors import CORS

app = Flask(__name__)  

# CHỈ cho phép frontend của bạn
CORS(app,
     resources={r"/*": {
         "origins": ["https://insurechat-five.vercel.app"],
         "supports_credentials": True
     }})

@app.get("/")
def health():
    return jsonify({
        "status": "ok",
        "message": "Customer Support Analysis Crew API (Flask on Vercel)."
    })

@app.post("/run")
def run():
    payload = request.get_json(silent=True) or {}
    data_query = payload.get("data_query", "")
    
    if not os.getenv("GEMINI_API_KEY"):
        return jsonify({"error": "Missing GEMINI_API_KEY environment variable on server."}), 500

    try:
        report = run_crew(data_query=data_query)
        return jsonify({"ok": True, "data_query": data_query, "report": report})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/agent1")
def agent1():
    """Agent đầu tiên - Data Analyst"""
    payload = request.get_json(silent=True) or {}
    data_query = payload.get("data_query", "")
    feedback = payload.get("feedback", "")
    
    if not os.getenv("GEMINI_API_KEY"):
        return jsonify({"error": "Missing GEMINI_API_KEY environment variable on server."}), 500

    try:
        result = run_agent1(data_query=data_query, feedback=feedback)
        return jsonify({"ok": True, "agent": "data_analyst", "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/agent5")
def agent5():
    """Agent thứ 5 - Report Writer"""
    payload = request.get_json(silent=True) or {}
    data_query = payload.get("data_query", "")
    analysis_result = payload.get("analysis_result", "")
    optimization_result = payload.get("optimization_result", "")
    feedback = payload.get("feedback", "")
    
    if not os.getenv("GEMINI_API_KEY"):
        return jsonify({"error": "Missing GEMINI_API_KEY environment variable on server."}), 500

    try:
        result = run_agent5(
            data_query=data_query,
            analysis_result=analysis_result,
            optimization_result=optimization_result,
            feedback=feedback
        )
        return jsonify({"ok": True, "agent": "report_writer", "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)