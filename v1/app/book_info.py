# app/book_info.py
import re
import logging
import asyncio
import httpx

logger = logging.getLogger("uvicorn.error")
openlibrary_cache = {}  # cache work IDs by cleaned title

# --- Helper functions ---
def clean_title(title: str) -> str:
    """Remove subtitles, numbering, punctuation for search."""
    title = re.sub(r'\(.*?\)', '', title)         # remove parentheses
    title = re.sub(r'#\d+', '', title)            # remove #numbers
    title = title.split(':')[0]                    # remove subtitles
    title = re.sub(r'[^a-zA-Z0-9 ]', '', title)  # remove punctuation
    return title.strip()

async def get_book_info(title: str) -> dict:
    """Fetch book info from Open Library, always returning full metadata."""
    cleaned_title = clean_title(title)
    if cleaned_title in openlibrary_cache:
        work_key = openlibrary_cache[cleaned_title]
        return await fetch_work_info(title, work_key)

    search_url = "https://openlibrary.org/search.json"
    params = {"title": cleaned_title}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(search_url, params=params)
            resp.raise_for_status()
            data = resp.json()
            docs = data.get("docs", [])
            if not docs:
                return {"title": title, "author": "Unknown Author"}

            # Take first doc to ignore edition details
            book_doc = docs[0]
            work_key = book_doc.get("key")
            if work_key:
                openlibrary_cache[cleaned_title] = work_key
                return await fetch_work_info(title, work_key, book_doc)
            else:
                return {"title": title, "author": "Unknown Author"}

    except Exception as e:
        logger.error(f"Open Library error for '{title}': {e}")
        return {"title": title, "author": "Unknown Author"}

async def fetch_work_info(title: str, work_key: str, book_doc: dict = None) -> dict:
    """Fetch detailed metadata for a work key from Open Library."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            work_url = f"https://openlibrary.org{work_key}.json"
            work_resp = await client.get(work_url)
            work_resp.raise_for_status()
            work_info = work_resp.json()

            description = work_info.get("description")
            if isinstance(description, dict):
                description = description.get("value")
            elif not isinstance(description, str):
                description = None

            cover_id = book_doc.get("cover_edition_key") if book_doc else None
            if not cover_id and work_info.get("covers"):
                cover_id = work_info["covers"][0]

            return {
                "title": book_doc.get("title") if book_doc else title,
                "author": ", ".join(book_doc.get("author_name", [])) if book_doc and book_doc.get("author_name") else "Unknown Author",
                "first_publish_year": book_doc.get("first_publish_year") if book_doc else None,
                "subjects": ", ".join(book_doc.get("subject", [])) if book_doc and book_doc.get("subject") else None,
                "number_of_pages": work_info.get("number_of_pages"),
                "publishers": ", ".join(book_doc.get("publisher", [])) if book_doc and book_doc.get("publisher") else None,
                "languages": [lang.get("key").split("/")[-1] for lang in work_info.get("languages", [])] if work_info.get("languages") else None,
                "description": description,
                "openlibrary_id": work_key,
                "cover_urls": {
                    "small": f"https://covers.openlibrary.org/b/olid/{cover_id}-S.jpg" if cover_id else None,
                    "medium": f"https://covers.openlibrary.org/b/olid/{cover_id}-M.jpg" if cover_id else None,
                    "large": f"https://covers.openlibrary.org/b/olid/{cover_id}-L.jpg" if cover_id else None,
                },
                "openlibrary_url": f"https://openlibrary.org{work_key}"
            }
    except Exception as e:
        logger.error(f"Open Library work fetch error for '{title}': {e}")
        return {"title": title, "author": "Unknown Author"}