from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import whisper
import torch
import tempfile
import os
from pathlib import Path

app = FastAPI(title="Whisper ASR API", version="1.0.0")

# Global variable to store model
model = None

def load_model():
    """Load Whisper model on startup"""
    global model
    print("Loading Whisper model...")
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model (using 'base' for balance of speed and accuracy)
    # Options: tiny, base, small, medium, large
    model = whisper.load_model("base", device=device)
    print("Model loaded successfully!")

@app.on_event("startup")
async def startup_event():
    """Load model when API starts"""
    load_model()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Whisper ASR API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "transcribe": "/transcribe (POST)",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device,
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe audio file using Whisper
    
    Supported formats: mp3, mp4, mpeg, mpga, m4a, wav, webm
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Check file extension
    allowed_extensions = {'.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Transcribe
        print(f"Transcribing file: {file.filename}")
        result = model.transcribe(temp_file_path)
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        return JSONResponse(content={
            "filename": file.filename,
            "text": result["text"],
            "language": result.get("language", "unknown"),
            "segments": [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"]
                }
                for seg in result.get("segments", [])
            ]
        })
    
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.get("/models")
async def list_models():
    """List available Whisper models"""
    return {
        "available_models": ["tiny", "base", "small", "medium", "large"],
        "current_model": "base",
        "model_info": {
            "tiny": "39M params, fastest",
            "base": "74M params, good balance",
            "small": "244M params, better accuracy",
            "medium": "769M params, high accuracy",
            "large": "1550M params, best accuracy"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)