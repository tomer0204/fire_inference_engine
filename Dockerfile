FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

COPY ingest ./ingest
COPY orchestrator ./orchestrator 
COPY configs ./configs
COPY storage ./storage
COPY core ./core 
EXPOSE 5001

CMD ["python", "-m", "orchestrator.server"]
