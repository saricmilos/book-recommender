# app/deploy.py
import os
import pickle
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from item_cf import recommend_similar_items
from fastapi import Body, Query

logger = logging.getLogger("uvicorn.error")

MODEL_DIR = os.environ.get("MODEL_DIR") or os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_DIR = os.path.abspath(MODEL_DIR)
ITEM_SIM_PATH = os.path.join(MODEL_DIR, "item_sim_matrix.pkl")
ITEM_ENCODER_PATH = os.path.join(MODEL_DIR, "item_encoder.pkl")

# In-memory placeholders
item_sim_matrix = None
item_encoder = None
models_loaded = False
book_titles_list = []

class BookRequest(BaseModel):
    book_title: str
    top_k: int = 10

# Lifespan handler to replace deprecated on_event("startup")
@asynccontextmanager
async def lifespan(app: FastAPI):
    global item_sim_matrix, item_encoder, models_loaded, book_titles_list
    # Startup logic
    try:
        with open(ITEM_SIM_PATH, "rb") as f:
            item_sim_matrix = pickle.load(f)
        logger.info(f"Loaded item_sim_matrix from {ITEM_SIM_PATH}")
    except Exception as e:
        logger.error(f"Failed to load item_sim_matrix: {e}")
        item_sim_matrix = None

    try:
        with open(ITEM_ENCODER_PATH, "rb") as f:
            item_encoder = pickle.load(f)
        logger.info(f"Loaded item_encoder from {ITEM_ENCODER_PATH}")
        
        # Extract book titles from encoder
        if hasattr(item_encoder, 'classes_'):
            book_titles_list = sorted(item_encoder.classes_.tolist())
            logger.info(f"Extracted {len(book_titles_list)} book titles from encoder")
        else:
            logger.warning("item_encoder does not have 'classes_' attribute")
            
    except Exception as e:
        logger.error(f"Failed to load item_encoder: {e}")
        item_encoder = None

    models_loaded = item_sim_matrix is not None and item_encoder is not None
    if not models_loaded:
        logger.warning("Models not fully loaded; endpoints will return 503 until resolved.")

    yield  # control passes to FastAPI app while running

    # Shutdown logic (optional)
    logger.info("Shutting down API...")

# Initialize FastAPI with lifespan
app = FastAPI(title="Item-based CF API", lifespan=lifespan)

# Allow CORS for frontend (replace "*" with your domain in production)
app.add_middleware(
    CORSMiddleware,
        allow_origins=[
        "https://cassiopeiai.com",
        "https://saricmilos.com",
        "http://localhost:3000",   # keep for local dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "API is up. Welcome to book recommender!"}

@app.get("/search_books/")
def search_books(query: str, limit: int = 20):
    """
    Search for books by title prefix/substring.
    Returns matching book titles for autocomplete.
    """
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded on server.")
    
    if not book_titles_list:
        raise HTTPException(status_code=500, detail="Book titles not available.")
    
    # Validate query
    query = query.strip()
    if len(query) < 1:
        raise HTTPException(status_code=400, detail="Query must be at least 1 character.")
    
    # Search for matches (case-insensitive)
    query_lower = query.lower()
    matches = [
        title for title in book_titles_list 
        if query_lower in title.lower()
    ]
    
    # Limit results
    matches = matches[:limit]
    
    logger.info(f"Search query '{query}' returned {len(matches)} results")
    
    return {
        "query": query,
        "results": matches,
        "total_matches": len(matches)
    }


@app.api_route("/recommend_books/", methods=["GET", "POST"])
def recommend_books(
    book_title: str = Query(None),
    top_k: int = Query(10),
    request_body: BookRequest = Body(None)
):
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded on server.")

    if request_body:
        book_title = request_body.book_title
        top_k = request_body.top_k

    if not book_title:
        raise HTTPException(status_code=400, detail="book_title is required.")

    try:
        recommendations = recommend_similar_items(
            item_title=book_title,
            item_encoder=item_encoder,
            item_sim_matrix=item_sim_matrix,
            k=top_k,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Book '{book_title}' not found.")
    except Exception as e:
        logger.exception("Unexpected error during recommendation")
        raise HTTPException(status_code=500, detail="Internal server error")

    return {"book_title": book_title, "recommendations": list(recommendations)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "deploy:app",
        host="0.0.0.0",
        port=8000
    )