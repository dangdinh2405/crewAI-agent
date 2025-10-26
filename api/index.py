import os
from flask import Flask, jsonify, request, send_from_directory, abort
from crew_setup import run_crew, run_agent1, run_agent2, run_agent3, run_agent4, run_agent5
from flask_cors import CORS
import json
import time
from pathlib import Path

app = Flask(__name__)  
OUTPUTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))

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
    payload = request.get_data(as_text=True) or ""
    payload = json.loads(payload)
    data_query = payload.get("data_query", "")
    feedback = payload.get("feedback", "")
    
    if not os.getenv("GEMINI_API_KEY"):
        return jsonify({"error": "Missing GEMINI_API_KEY environment variable on server."}), 500

    try:
        result = run_agent1(data_query=data_query, feedback=feedback)
        return jsonify({"ok": True, "agent": "data_analyst", "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/agent2")
def agent2():
    time.sleep(10)
    """Agent thứ 2 - Process Optimizer"""
    payload = request.get_data(as_text=True) or ""
    payload = json.loads(payload)
    analysis_result = payload.get("analysis_result", "")
    
    if not os.getenv("GEMINI_API_KEY"):
        return jsonify({"error": "Missing GEMINI_API_KEY environment variable on server."}), 500

    try:
        result = run_agent2(
            analysis_result=analysis_result
        )
        return jsonify({"ok": True, "agent": "process_optimizer", "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/agent3")
def agent3():
    """Agent thứ 3 - Additional Analysis"""
    time.sleep(10)
    payload = request.get_data(as_text=True) or ""
    payload = json.loads(payload)
    analysis_result = payload.get("analysis_result", "")
    optimization_result = payload.get("optimization_result", "")
    
    if not os.getenv("GEMINI_API_KEY"):
        return jsonify({"error": "Missing GEMINI_API_KEY environment variable on server."}), 500

    try:
        result = run_agent3(
            analysis_result=analysis_result,
            optimization_result=optimization_result
        )
        return jsonify({"ok": True, "agent": "additional_analyst", "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/agent4")
def agent4():
    """Agent thứ 4 - Quality Assurance"""
    time.sleep(60)
    payload = request.get_data(as_text=True) or ""
    payload = json.loads(payload)
    analysis_result = payload.get("analysis_result", "")
    optimization_result = payload.get("optimization_result", "")
    additional_insights = payload.get("additional_insights", "")
    
    if not os.getenv("GEMINI_API_KEY"):
        return jsonify({"error": "Missing GEMINI_API_KEY environment variable on server."}), 500

    try:
        result = run_agent4(
            analysis_result=analysis_result,
            optimization_result=optimization_result,
            additional_insights=additional_insights
        )
        return jsonify({"ok": True, "agent": "quality_assurance", "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/agent5")
def agent5():
    """Agent thứ 5 - Report Writer"""
    time.sleep(10)
    payload = request.get_data(as_text=True) or ""
    payload = json.loads(payload)
    analysis_result = payload.get("analysis_result", "")
    optimization_result = payload.get("optimization_result", "")
    additional_insights = payload.get("additional_insights", "")
    qa_result = payload.get("qa_result", "")
    feedback = payload.get("feedback", "")
    
    if not os.getenv("GEMINI_API_KEY"):
        return jsonify({"error": "Missing GEMINI_API_KEY environment variable on server."}), 500

    try:
        result = run_agent5(
            analysis_result=analysis_result,
            optimization_result=optimization_result,
            additional_insights=additional_insights,
            qa_result=qa_result,
            feedback=feedback
        )
        return jsonify({"ok": True, "agent": "report_writer", "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.get("/outputs/<path:filename>")
def get_output_file(filename: str):
    try:
        # Resolve and ensure the requested file is inside the outputs directory
        requested_path = Path(OUTPUTS_DIR) / filename
        resolved_requested = requested_path.resolve()
        resolved_outputs = Path(OUTPUTS_DIR).resolve()
        if not str(resolved_requested).startswith(str(resolved_outputs)) or not resolved_requested.is_file():
            return jsonify({"ok": False, "error": "File not found"}), 404

        # send_from_directory handles proper headers; filename is relative to OUTPUTS_DIR
        return send_from_directory(resolved_outputs, filename)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)