call venv\Scripts\activate.bat
start uvicorn api_server:app --host 0.0.0.0 --port 8000
python chat_gui.py