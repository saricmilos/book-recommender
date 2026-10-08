# app/deploy.py
import os
import pickle
import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.sparse import load_npz

from src.item_cf import recommend_similar_items_new
from app.book_info import get_book_info  # <--- import new module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MODEL_DIR") or os.path.join(BASE_DIR, "..", "models")
ITEM_SIM_PATH = os.path.join(MODEL_DIR, "topk_item_sim.npz")
ITEM_ENCODER_PATH = os.path.join(MODEL_DIR, "item_encoder.pkl")

item_sim_matrix = None
item_encoder = None
models_loaded = False
book_titles_list = []

class BookRequest(BaseModel):
    book_title: str
    top_k: int = 10

# --- FastAPI lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global item_sim_matrix, item_encoder, models_loaded, book_titles_list
    logger.info(f"Starting startup sequence. Looking for models in: {MODEL_DIR}")

    try:
        if os.path.exists(ITEM_SIM_PATH):
            item_sim_matrix = load_npz(ITEM_SIM_PATH)
            logger.info(f"Loaded item_sim_matrix from {ITEM_SIM_PATH}")
    except Exception as e:
        logger.error(f"Failed to load item_sim_matrix: {e}")

    try:
        if os.path.exists(ITEM_ENCODER_PATH):
            with open(ITEM_ENCODER_PATH, "rb") as f:
                item_encoder = pickle.load(f)
            logger.info(f"Loaded item_encoder from {ITEM_ENCODER_PATH}")
            if hasattr(item_encoder, "classes_"):
                book_titles_list = sorted(item_encoder.classes_.tolist())
    except Exception as e:
        logger.error(f"Failed to load item_encoder: {e}")

    models_loaded = item_sim_matrix is not None and item_encoder is not None
    if not models_loaded:
        logger.warning("⚠️ MODELS NOT LOADED.")

    yield
    logger.info("Shutting down API...")

# --- FastAPI setup ---
app = FastAPI(title="BookVerse Recommendation API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://cassiopeiai.com",
        "https://saricmilos.com",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "online", "models_loaded": models_loaded, "total_books": len(book_titles_list)}

@app.get("/search_books")
def search_books(query: str, limit: int = 20):
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Server initializing models.")
    query = query.strip()
    if not query:
        return {"query": query, "results": [], "total_matches": 0}
    query_lower = query.lower()
    matches = [title for title in book_titles_list if query_lower in title.lower()]
    return {"query": query, "results": matches[:limit], "total_matches": len(matches)}

@app.api_route("/recommend_books", methods=["GET", "POST"])
async def recommend_books(book_title: str = Query(None), top_k: int = Query(10), request_body: BookRequest = Body(None)):
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    target_title = request_body.book_title if request_body else book_title
    target_k = request_body.top_k if request_body else top_k
    if not target_title:
        raise HTTPException(status_code=400, detail="book_title is required.")

    try:
        recommendations = recommend_similar_items_new(
            item_title=target_title,
            item_encoder=item_encoder,
            item_sim_matrix=item_sim_matrix,
            k=target_k,
        )
        tasks = [get_book_info(t) for t in [target_title] + list(recommendations)]
        results = await asyncio.gather(*tasks)
        return {"book_title": results[0], "recommendations": results[1:]}
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Book '{target_title}' not in database.")
    except Exception as e:
        logger.exception("Recommendation failure")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("deploy:app", host="0.0.0.0", port=port)