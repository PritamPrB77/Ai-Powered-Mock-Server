# AI Dynamic Mock Server

Production-ready FastAPI backend that:

- accepts any OpenAPI YAML/JSON spec
- dynamically registers routes from that spec
- validates requests with `openapi-core`
- generates responses using selectable providers: local Ollama or OpenRouter (with optional Seq2Seq)
- validates generated responses structurally (OpenAPI) and semantically (sentence-transformers)
- supports multi-response generation (`n <= 10`)
- maintains in-memory context across requests
- exposes a minimal UI at `/` for spec upload and endpoint viewing

## 1 Setup

```bash
pip install -r requirements.txt
```

Create `.env` from `.env.example` and set:

```env
GENERATION_PROVIDER=openrouter

OPENROUTER_ENABLED=true
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=mistralai/mistral-7b-instruct
OPENROUTER_HISTORY_TURNS=5

OLLAMA_ENABLED=true
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=qwen3:5
OLLAMA_NUM_PREDICT=256

SEQ2SEQ_ENABLED=false
SEQ2SEQ_MODEL_NAME=google/flan-t5-base
```

Notes:
- Switch provider with `GENERATION_PROVIDER=ollama` or `GENERATION_PROVIDER=openrouter`.
- OpenRouter is the default provider and runs with runtime context memory.
- Ollama default URL is `http://localhost:11434/api/generate`.

## 2 Run

```bash
uvicorn app.main:app --reload
```

## 3 Use

1. Open `http://127.0.0.1:8000/`.
2. Upload or paste OpenAPI YAML/JSON spec.
3. Check generated endpoints from `/registered-endpoints`.
4. Test dynamic endpoints in Postman.

## 4 Multi-response

Use either:

- query param: `?n=3`
- request body field: `"n": 3`

The server caps `n` to `10`.
