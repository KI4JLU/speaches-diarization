# speaches-diarization

Custom **arm64 (GB10/sbsa)** [Speaches](https://github.com/speaches-ai/speaches)
image, **diarization-only**, for the HAWKI transcription stack on an NVIDIA
DGX Spark (GB10, aarch64, 128 GB unified memory).

Internal reference: Kanban card KI-310 and
`transcription-stack-requirements.md` §3 in the HAWKI repo.

## Why this exists

The official arm64 Speaches image (`ghcr.io/speaches-ai/speaches:latest-cuda`)
ships **without torch and without pyannote** (verified 2026-07-14 and again
2026-07-15 against the current pull): its arm64 variant is built from an old,
pre-diarization tree. The diarization endpoints HAWKI depends on simply don't
exist there.

pyannote itself runs fine on the GB10 — torch ships CUDA sbsa wheels, and vLLM
proves the platform daily on the same box. So this repo builds Speaches from
source at a **pinned upstream commit** with the full dependency lockfile
(torch 2.8.0 + pyannote-audio 4.0.4) and adds two small patches.

### Pinned commit

`993994f7984bf3fe9655b267448328cf66fccb42` (upstream master, 2026-04-17).

The production Speaches worker reports version `0.8.3`, but that string is
hardcoded in upstream `main.py` — the worker actually runs a master build.
The endpoint schemas of this commit match the worker's live `openapi.json`
field-for-field (verified 2026-07-15), so HAWKI's `CustomSpeachesProvider`
works against both without changes.

## Patches

| Patch | What | Why |
|---|---|---|
| `0001-serialize-diarization-jobs.patch` | Global semaphore around the diarization run, size from `DIARIZATION_MAX_CONCURRENCY` (default `1`) | A whole-file pyannote run occupies the GPU continuously for minutes and competes with realtime STT decode on unified memory. Interim requirement while the realtime model shares the box: one job at a time. Excess requests queue (client timeouts already budget for this). |
| `0002-speaker-count-constraints.patch` | Adds optional `num_speakers` / `min_speakers` / `max_speakers` form fields, passed through to the pyannote pipeline | HAWKI already sends `num_speakers=1` or `min_speakers=2`; upstream (and the current production worker!) silently ignore these fields — the UI speaker-count choice had no server-side effect. pyannote supports the kwargs natively. |

Both endpoints HAWKI uses are preserved unchanged otherwise:

- `POST /v1/audio/diarization` — incl. `known_speaker_names[]` /
  `known_speaker_references[]` (Speaches' own embedding matching) and
  `response_format=json|rttm`
- `POST /v1/audio/speech/timestamps` — silero VAD

## Configuration

Environment variables (image defaults in parentheses):

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | **Required at first start** — `pyannote/speaker-diarization-community-1` is a gated HF repo. Cached afterwards. |
| `API_KEY` | If set, all API requests require it as Bearer token (`/health` stays public). HAWKI setting: `diarization_api_key`. |
| `DIARIZATION_MAX_CONCURRENCY` (`1`) | Concurrent diarization runs. Keep at 1 while the GPU is shared with a latency-sensitive workload. |
| `PRELOAD_MODELS` (`["pyannote/speaker-diarization-community-1"]`) | Downloaded at startup. The pipeline is loaded onto the GPU on first request and stays resident (upstream hardcodes TTL −1). |
| `ENABLE_UI` (`false`) | Gradio UI off. |

Note: whisper/TTS code is present in the image (it's a full Speaches build)
but no such models are configured or preloaded — the CTranslate2/ARM-CUDA gap
stays irrelevant because those endpoints are never called.

## Build

CI builds and pushes `ghcr.io/ki4jlu/speaches-diarization:latest` on every
push to `main` (native arm64 runner).

Manually on the target host (build args only needed behind a proxy):

```sh
git clone https://github.com/KI4JLU/speaches-diarization.git
cd speaches-diarization
sudo docker build \
  --build-arg http_proxy=$HTTP_PROXY_URL \
  --build-arg https_proxy=$HTTP_PROXY_URL \
  -t speaches-diarization:local .
```

## Deploy

See `compose.example.yaml` — merge the service block into the STT host's
compose stack (port **8003**). `HF_TOKEN` and `DIARIZATION_API_KEY` go into
the host's `.env`.

HAWKI switch-over is settings-only: point `diarization_base_url` /
`diarization_api_key` at the new host (via its TLS front, path routed to
`:8003`). No HAWKI code changes.

## Smoke test

```sh
curl -s http://localhost:8003/health
curl -s -X POST http://localhost:8003/v1/audio/speech/timestamps \
  -H "Authorization: Bearer $API_KEY" \
  -F file=@sample.wav
curl -s -X POST http://localhost:8003/v1/audio/diarization \
  -H "Authorization: Bearer $API_KEY" \
  -F model=pyannote/speaker-diarization-community-1 \
  -F min_speakers=2 \
  -F file=@sample.wav
```

## License

Speaches is MIT-licensed; the patches in this repo are derived from and
distributed under the same MIT license.
