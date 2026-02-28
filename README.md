# AI Dynamic Mock Server

Production-ready FastAPI backend that:

- accepts any OpenAPI YAML/JSON spec
- dynamically registers routes from that spec
- validates requests with `openapi-core`
- generates responses using a real Seq2Seq model (`flan-t5`) with OpenRouter fallback
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
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=mistralai/mistral-7b-instruct
OPENROUTER_ENABLED=true
OPENROUTER_FALLBACK_ENABLED=true
SEQ2SEQ_ENABLED=true
SEQ2SEQ_MODEL_NAME=google/flan-t5-base
SEMANTIC_VALIDATION_ENABLED=true
```

Notes:
- Seq2Seq runs first.
- OpenRouter is used as fallback when Seq2Seq output is invalid/unavailable.
- For fastest local mode, set `OPENROUTER_ENABLED=false` and `SEMANTIC_VALIDATION_ENABLED=false`.

## 2 Run Backend

Main command:

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Equivalent short command:

```bash
uvicorn app.main:app --reload
```

If you want port `8010`:

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8010
```

## 3 User Guide

1. Open `http://127.0.0.1:8000/`.
2. Upload or paste OpenAPI YAML/JSON spec.
3. Check generated endpoints from `/registered-endpoints`.
4. Send requests to generated endpoints from Postman/curl.
5. If needed, add `?n=3` to return multiple generated responses.

## 4 Multi-response

Use either:

- query param: `?n=3`
- request body field: `"n": 3`

The server caps `n` to `10`.

## 5 About `uvicorn*.log` Files (Important or Not?)

Files like:
- `uvicorn.out.log`
- `uvicorn.err.log`
- `uvicorn_8010.out.log`
- `uvicorn_8010.err.log`

are runtime log files created when Uvicorn output is redirected to files.

- Important for debugging startup/runtime errors.
- Not required for the backend to run.
- Safe to delete when the server is stopped (if you do not need old logs).
