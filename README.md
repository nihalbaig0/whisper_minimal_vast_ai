# Whisper ASR API

A minimal FastAPI-based Automatic Speech Recognition (ASR) service using OpenAI's Whisper model with GPU acceleration support.

## Features

- 🎤 **Audio Transcription**: Convert speech to text using Whisper models
- 🚀 **GPU Acceleration**: CUDA support for faster processing
- 📁 **Multiple Formats**: Support for MP3, WAV, M4A, WebM, and more
- 🔍 **RESTful API**: Clean FastAPI endpoints with auto-documentation
- 🐳 **Docker Ready**: Containerized deployment with NVIDIA CUDA
- 📊 **Health Monitoring**: Built-in health checks and model status
- 🧪 **Test Suite**: Comprehensive API testing script

## Quick Start

### Local Development

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the API**:
   ```bash
   python main.py
   ```

3. **Access the API**:
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

### Docker Deployment

1. **Build the image**:
   ```bash
   docker build -t whisper-asr .
   ```

2. **Run with GPU support**:
   ```bash
   docker run --gpus all -p 8000:8000 whisper-asr
   ```

3. **Run without GPU**:
   ```bash
   docker run -p 8000:8000 whisper-asr
   ```

## API Endpoints

### `GET /`
Root endpoint with API information

### `GET /health`
Health check endpoint showing:
- Service status
- Model loading status
- CUDA availability
- Device being used

### `GET /models`
List available Whisper models and their specifications

### `POST /transcribe`
Transcribe audio file to text

**Request**: Multipart form data with audio file
**Supported formats**: `.mp3`, `.mp4`, `.mpeg`, `.mpga`, `.m4a`, `.wav`, `.webm`

**Response**:
```json
{
  "filename": "audio.wav",
  "text": "Transcribed text here...",
  "language": "en",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "First segment"
    }
  ]
}
```

## Testing

Use the provided test script to verify API functionality:

```bash
# Basic health tests
python test_api.py http://localhost:8000

# Test with audio file
python test_api.py http://localhost:8000 male.wav
```

## Model Information

The API uses the Whisper "base" model by default, providing a good balance of speed and accuracy. Available models:

| Model | Parameters | Speed | Accuracy |
|-------|------------|-------|----------|
| tiny  | 39M        | Fastest | Lower    |
| base  | 74M        | Fast   | Good     |
| small | 244M       | Medium | Better   |
| medium| 769M       | Slow   | High     |
| large | 1550M      | Slowest| Best     |

## Requirements

- Python 3.8+
- PyTorch with CUDA support (optional but recommended)
- FFmpeg for audio processing
- NVIDIA GPU (optional, for acceleration)

