"""
main.py — FastAPI application: REST + SSE endpoints + static file serving.

Endpoints:
  POST /search        → accepts sketch + filters, returns search_id
  GET  /search-stream/{search_id}  → SSE stream of pipeline stage events
  GET  /              → serves static/index.html
"""

import os
import json
import uuid
import asyncio
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline import run_pipeline

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Shazam for Stocks", version="1.0.0")

# Static files (index.html, style.css)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------------------------------
# Data loading (done once at startup)
# ---------------------------------------------------------------------------

WINDOWS_PATH = os.path.join("data", "windows.npy")
META_PATH = os.path.join("data", "meta.json")

_all_windows: list[dict] = []
_meta_map: dict[str, dict] = {}


def load_data() -> None:
    """Load windows.npy and meta.json into memory at startup."""
    global _all_windows, _meta_map

    if not os.path.exists(WINDOWS_PATH):
        print(f"[WARN] {WINDOWS_PATH} not found — run prepare_data.py first")
        return
    if not os.path.exists(META_PATH):
        print(f"[WARN] {META_PATH} not found — run prepare_data.py first")
        return

    raw = np.load(WINDOWS_PATH, allow_pickle=True)
    _all_windows = list(raw)
    print(f"[INFO] Loaded {len(_all_windows)} windows from {WINDOWS_PATH}")

    with open(META_PATH) as f:
        meta_list = json.load(f)
    _meta_map = {m["symbol"]: m for m in meta_list}
    print(f"[INFO] Loaded {len(_meta_map)} symbols from {META_PATH}")


load_data()

# ---------------------------------------------------------------------------
# In-memory search result store (search_id -> pipeline args)
# ---------------------------------------------------------------------------

# Maps search_id -> (sketch, filters) so the SSE endpoint can retrieve them
_pending: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class Filters(BaseModel):
    """Filters sent from the frontend canvas UI."""

    large: bool = True
    mid: bool = True
    small: bool = False
    window_days: int = 60
    smoothing: float = 2.0


class SearchRequest(BaseModel):
    """Payload from POST /search."""

    sketch: list[float]       # 50 z-normalized floats
    filters: Filters


class SearchResponse(BaseModel):
    """Response from POST /search."""

    search_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    """Serve the main index.html."""
    with open(os.path.join("static", "index.html"), encoding="utf-8") as f:
        return HTMLResponse(
            content=f.read(),
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            }
        )


@app.post("/search", response_model=SearchResponse)
async def post_search(body: SearchRequest) -> SearchResponse:
    """
    Accept sketch + filters, store them keyed by a new search_id.

    Returns the search_id so the client can open the SSE stream.
    """
    if len(body.sketch) != 50:
        raise HTTPException(status_code=422, detail="sketch must have exactly 50 values")

    search_id = str(uuid.uuid4())
    _pending[search_id] = {
        "sketch": np.array(body.sketch, dtype=np.float64),
        "filters": body.filters.model_dump(),
    }
    return SearchResponse(search_id=search_id)


@app.get("/search-stream/{search_id}")
async def search_stream(search_id: str) -> StreamingResponse:
    """
    SSE endpoint: run the pipeline and stream one event per stage.

    Returns text/event-stream with JSON payloads matching SSE Event Schema.
    """
    if search_id not in _pending:
        raise HTTPException(status_code=404, detail="search_id not found")

    payload = _pending.pop(search_id)
    sketch: np.ndarray = payload["sketch"]
    filters: dict = payload["filters"]

    async def event_generator():
        """Async generator that yields SSE-formatted strings."""
        try:
            async for event in run_pipeline(
                sketch=sketch,
                filters=filters,
                all_windows=_all_windows,
                meta_map=_meta_map,
                search_id=search_id,
            ):
                data = json.dumps(event)
                yield f"data: {data}\n\n"
                await asyncio.sleep(0)   # yield control to event loop
        except Exception as e:
            error_event = json.dumps({"stage": -1, "status": "error", "message": str(e)})
            yield f"data: {error_event}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",       # disable nginx buffering
        },
    )


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=2000, reload=True)
