import os
import tempfile
from pathlib import Path
from typing import Optional

import ctranslate2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

app = FastAPI(title="Whisper ASR API", version="1.0.0")

model: Optional[WhisperModel] = None
_runtime_device: Optional[str] = None
_runtime_compute_type: Optional[str] = None


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def resolve_device() -> str:
    requested = os.environ.get("WHISPER_DEVICE", "auto").strip().lower()
    cuda_count = ctranslate2.get_cuda_device_count()

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if cuda_count == 0:
            raise RuntimeError("WHISPER_DEVICE=cuda but no CUDA devices are visible to CTranslate2")
        return "cuda"
    if requested in ("auto", ""):
        return "cuda" if cuda_count > 0 else "cpu"
    raise ValueError(f"Invalid WHISPER_DEVICE={requested!r}; use auto, cuda, or cpu")


def resolve_compute_type(device: str) -> str:
    explicit = os.environ.get("WHISPER_COMPUTE_TYPE", "").strip()
    if explicit:
        return explicit
    return "float16" if device == "cuda" else "int8"


def load_model() -> None:
    global model, _runtime_device, _runtime_compute_type

    model_name = os.environ.get("WHISPER_MODEL", "base").strip()
    print(f"Loading faster-whisper model {model_name!r}...")

    _runtime_device = resolve_device()
    _runtime_compute_type = resolve_compute_type(_runtime_device)
    print(f"Using device={_runtime_device}, compute_type={_runtime_compute_type}")

    model = WhisperModel(
        model_name,
        device=_runtime_device,
        compute_type=_runtime_compute_type,
    )
    print("Model loaded successfully!")


@app.on_event("startup")
async def startup_event() -> None:
    load_model()


@app.get("/")
async def root() -> dict:
    return {
        "message": "Whisper ASR API (faster-whisper / CTranslate2)",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "transcribe": "/transcribe (POST)",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health() -> dict:
    cuda_count = ctranslate2.get_cuda_device_count()
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": _runtime_device,
        "compute_type": _runtime_compute_type,
        "cuda_devices_visible": cuda_count,
        "backend": "faster-whisper",
    }


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)) -> JSONResponse:
    """
    Transcribe audio file using faster-whisper.

    Supported formats: mp3, mp4, mpeg, mpga, m4a, wav, webm
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    allowed_extensions = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}
    file_ext = Path(file.filename or "").suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(sorted(allowed_extensions))}",
        )

    temp_file_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        beam_size = int(os.environ.get("WHISPER_BEAM_SIZE", "5"))
        vad_filter = _env_flag("WHISPER_VAD_FILTER", default=False)

        print(f"Transcribing file: {file.filename}")
        segments_gen, info = model.transcribe(
            temp_file_path,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )

        segments_list = list(segments_gen)
        text = "".join(seg.text for seg in segments_list)

        return JSONResponse(
            content={
                "filename": file.filename,
                "text": text,
                "language": getattr(info, "language", None) or "unknown",
                "segments": [
                    {"start": seg.start, "end": seg.end, "text": seg.text}
                    for seg in segments_list
                ],
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}") from e
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


@app.get("/models")
async def list_models() -> dict:
    current = os.environ.get("WHISPER_MODEL", "base").strip()
    return {
        "available_models": ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        "current_model": current,
        "backend": "faster-whisper",
        "model_info": {
            "tiny": "Fastest, lower accuracy",
            "base": "Good speed / accuracy tradeoff (default)",
            "small": "Better accuracy",
            "medium": "High accuracy",
            "large": "Best accuracy (v1)",
            "large-v2": "Large v2",
            "large-v3": "Large v3",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
