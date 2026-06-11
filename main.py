from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import tempfile
import os

# -----------------------
# App init
# -----------------------
app = FastAPI()

# -----------------------
# CORS FIX (CRITICAL)
# -----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5000",
        "http://localhost:5000",
        "http://127.0.0.1:5500",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Load Whisper Model
# -----------------------
model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

# -----------------------
# Health check route
# -----------------------
@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "Whisper API is running"
    }

# -----------------------
# Transcription endpoint
# -----------------------
@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    # Run Whisper
    segments, info = model.transcribe(tmp_path)

    text = " ".join(segment.text for segment in segments)

    # Optional cleanup
    try:
        os.remove(tmp_path)
    except:
        pass

    return {
        "text": text,
        "language": info.language
    }