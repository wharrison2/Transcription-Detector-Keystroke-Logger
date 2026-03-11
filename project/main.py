from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import uuid
from pathlib import Path

app = FastAPI()

# Allow browser -> local server communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

@app.post("/submit")
def submit(data: dict):
    pid = data.get("participant_id", str(uuid.uuid4()))
    with open(DATA_DIR / f"{pid}.json", "w") as f:
        json.dump(data, f, indent=2)
    return {"status": "ok"}
