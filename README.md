# Whisper ASR API (faster-whisper)

Minimal [FastAPI](https://fastapi.tiangolo.com/) service for speech-to-text using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper on [CTranslate2](https://github.com/OpenNMT/CTranslate2)). No PyTorch dependency for inference.

## Prerequisites

- **Python** 3.10 or newer (3.10 matches the Docker image)
- **FFmpeg** on your `PATH` (used to decode uploaded audio)
- **GPU (optional):** NVIDIA driver + CUDA libraries compatible with the `ctranslate2` wheel you install; for Docker, the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) so `docker run --gpus all` works

## Run locally

1. From the repository root (the directory that contains `main.py` and `requirements.txt`), create a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the API:

   ```bash
   python main.py
   ```

   Or explicitly:

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

4. Open **http://localhost:8000/docs** for interactive API documentation.

The first request may be slow while the model is downloaded and cached (default model: `base`).

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `base` | Model size or path (e.g. `tiny`, `small`, `large-v3`). |
| `WHISPER_DEVICE` | `auto` | `auto` (CUDA if visible, else CPU), `cuda`, or `cpu`. |
| `WHISPER_COMPUTE_TYPE` | *(auto)* | `float16` on GPU, `int8` on CPU if unset. Override e.g. `int8_float16` on GPU. |
| `WHISPER_BEAM_SIZE` | `5` | Decoding beam size (lower is faster, less accurate). |
| `WHISPER_VAD_FILTER` | `false` | Set to `true` to enable VAD filtering (often faster on long files with silence). |

Example (CPU-only machine):

```bash
WHISPER_DEVICE=cpu WHISPER_MODEL=base python main.py
```

## Run with Docker

Build (CUDA 12.3 + cuDNN 9 runtime base, aligned with upstream faster-whisper images):

```bash
docker build -t whisper-asr .
```

Run **with GPU**:

```bash
docker run --rm --gpus all -p 8000:8000 whisper-asr
```

Run **CPU-only** (no GPU passthrough):

```bash
docker run --rm -e WHISPER_DEVICE=cpu -p 8000:8000 whisper-asr
```

Optional: pass model and compute settings:

```bash
docker run --rm --gpus all -e WHISPER_MODEL=small -e WHISPER_COMPUTE_TYPE=float16 -p 8000:8000 whisper-asr
```

Health check: **http://localhost:8000/health** (shows device, compute type, and `cuda_devices_visible`).

## Traefik + HTTPS

Same layout as [traefik_template/docker-compose.traefik.yml](traefik_template/docker-compose.traefik.yml) and [traefik_template/docker-compose.yml](traefik_template/docker-compose.yml): **Traefik v2.3**, Let’s Encrypt **DNS-01** via **Cloudflare** (no inbound `:443` required for ACME), HTTP→HTTPS redirect, dashboard + API behind **HTTP Basic auth** (`USERNAME` / `HASHED_PASSWORD` in `.env`). The app service is **`whisper`**, internal port **8000**, GPU via `deploy.resources.reservations.devices`. See [Traefik Readme.md](Traefik%20Readme.md) and [Traefik ACME DNS / Cloudflare](https://doc.traefik.io/traefik/https/acme/#cloudflare).

### Cloudflare API token (DNS-01)

1. In [Cloudflare Dashboard](https://dash.cloudflare.com) → **My Profile** → **API Tokens** → **Create Token**.
2. Use **Edit zone DNS** or a custom token with **Zone → DNS → Edit** and **Zone → Zone → Read**, scoped to your **zone**.
3. Copy the token into `.env` as **`CF_DNS_API_TOKEN`** (shown once).

Your domain’s DNS must be **on Cloudflare** for this API to create `_acme-challenge` TXT records. Keep **A**/**AAAA** records for `WHISPER_HOST` and `TRAEFIK_HOST` pointing at your server (or tunnel) as usual.

### Deploy

1. Create the shared network: `docker network create traefik-public`
2. Copy [.env.example](.env.example) to `.env` and set **`WHISPER_HOST`**, **`TRAEFIK_HOST`**, **`CF_DNS_API_TOKEN`**, **`EMAIL`**, **`USERNAME`**, **`HASHED_PASSWORD`** (APR1 hash):

   ```bash
   export USERNAME=admin
   export PASSWORD=changethis
   export HASHED_PASSWORD=$(openssl passwd -apr1 $PASSWORD)
   ```

3. Start Traefik, then the app (from repo root):

   ```bash
   docker compose -f docker-compose.traefik.yml up -d
   docker compose -f docker-compose.yml up -d
   ```

   Optional extra replicas (same labels → Traefik load balancing):  
   `docker compose -f docker-compose.yml up -d --scale whisper=3`

**Note:** DNS-01 fixes **certificate issuance** without public **:443** on your hostname (e.g. [vast.ai](https://docs.vast.ai/documentation/instances/connect/networking) port mapping). Clients must still reach your service (correct **A** record, `https://IP:port`, or a tunnel/proxy).

### GPU Docker troubleshooting

**Error: `failed to fulfil mount request: open /usr/bin/nvidia-cuda-mps-control: no such file or directory`**

The NVIDIA Container Toolkit bind-mounts several host utilities (including CUDA **MPS** tools) into the container. If that binary is not on the host, GPU containers fail before your app starts.

1. **Check whether the file exists:**

   ```bash
   ls -l /usr/bin/nvidia-cuda-mps-control
   ```

2. **If it is missing, install compute utilities for your driver** (package name tracks the driver series, e.g. **575** from `nvidia-smi`):

   ```bash
   sudo apt update
   apt-cache search nvidia-compute-utils | head -20
   sudo apt install nvidia-compute-utils-575
   ```

   Replace `575` with the branch that matches your installed NVIDIA driver, then:

   ```bash
   sudo systemctl restart docker
   ```

3. **If you use Docker from Snap** (especially **edge**), a known CDI default can cause this even when drivers look fine. Try forcing the legacy NVIDIA hook:

   ```bash
   docker run --rm --runtime=nvidia --gpus '"all,driver=nvidia.runtime-hook"' -p 8000:8000 whisper-asr
   ```

4. **Sanity check** (any failure here is the host/toolkit, not this repo):

   ```bash
   docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
   ```

## API overview

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service info |
| `GET` | `/health` | Health and runtime device info |
| `GET` | `/models` | Available model names and current `WHISPER_MODEL` |
| `POST` | `/transcribe` | Multipart upload: form field `file` = audio file |

Supported extensions: `.mp3`, `.mp4`, `.mpeg`, `.mpga`, `.m4a`, `.wav`, `.webm`.

Example response from `/transcribe`:

```json
{
  "filename": "audio.wav",
  "text": "Full transcript text",
  "language": "en",
  "segments": [
    {"start": 0.0, "end": 2.5, "text": " Hello"}
  ]
}
```

## Tests

With the server running:

```bash
pip install requests
python test_api.py http://localhost:8000
python test_api.py http://localhost:8000 path/to/sample.wav
```

## Notes

- **Driver vs image:** The Docker base is CUDA **12.3** runtime. Your host NVIDIA driver must be new enough to run that container (recent drivers reporting CUDA 12.x are typically fine).
- **VRAM:** Larger models need more GPU memory; use `WHISPER_MODEL=tiny` or `base` if you hit OOM errors.
