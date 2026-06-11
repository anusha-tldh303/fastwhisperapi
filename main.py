from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import tempfile

app = FastAPI()

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await audio.read())
        path = tmp.name

    segments, info = model.transcribe(path)

    text = " ".join(
        segment.text for segment in segments
    )

    return {
        "text": text,
        "language": info.language
    }