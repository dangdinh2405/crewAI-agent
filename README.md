Create .env 
Set 
GEMINI_API_KEY = ...
CREWAI_TRACING_ENABLED=true
SERPER_API_KEY= ...

python -m flask --app api/index.py run --port 3000 --host 0.0.0.0
