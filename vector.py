from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
import uuid
import logging
from typing import List, Optional

# EPUB parsing dependencies
from ebooklib import epub
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _extract_paragraphs(epub_path: str) -> List[str]:
    """Extract paragraph-like blocks from an EPUB file."""
    book = epub.read_epub(epub_path)
    parts: List[str] = []
    for item in book.get_items_of_type(epub.ITEM_DOCUMENT):
        html = item.get_content()
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n")
        # split into paragraphs by blank lines and keep non-empty paragraphs
        for p in [p.strip() for p in text.split("\n\n")]:
            if p:
                parts.append(" ".join(p.split()))
    logger.info("Extracted %d paragraphs from EPUB %s", len(parts), epub_path)
    return parts


def _chunk_paragraphs(paragraphs: List[str], max_chars: int = 1200) -> List[str]:
    """Chunk consecutive paragraphs into larger pieces (<= max_chars)."""
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for p in paragraphs:
        if cur_len + len(p) + 1 <= max_chars:
            cur.append(p)
            cur_len += len(p) + 1
        else:
            if cur:
                chunks.append(" ".join(cur))
            cur = [p]
            cur_len = len(p) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def index_csv(csv_path: str, chroma: Chroma, id_prefix: str = "csv_") -> None:
    """Read CSV and add documents to the provided Chroma instance.

    Requires a CSV with at least a text column (commonly 'Text') — tries a few fallbacks.
    Adds metadata 'source' and optional 'chapter' / 'verse' when available.
    """
    csv_path = os.path.expanduser(csv_path)
    logger.info("Indexing CSV: %s", csv_path)
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error("Failed to read CSV %s: %s", csv_path, e)
        return

    # Determine text and book columns heuristically
    text_col = None
    for candidate in ("Text", "text", "Content", "content", "body", "Body"):
        if candidate in df.columns:
            text_col = candidate
            break
    book_col = None
    for candidate in ("Book", "book", "Source", "source", "Title", "title"):
        if candidate in df.columns:
            book_col = candidate
            break

    if text_col is None:
        logger.error("CSV at %s doesn't contain a recognized text column. Columns: %s", csv_path, list(df.columns))
        return

    documents = []
    ids = []
    for i, row in df.iterrows():
        book_part = ""
        if book_col and not pd.isna(row.get(book_col)):
            book_part = str(row.get(book_col)) + " "
        text_part = "" if pd.isna(row.get(text_col)) else str(row.get(text_col))
        page_content = (book_part + text_part).strip()
        if not page_content:
            continue
        metadata = {"source": os.path.basename(csv_path)}
        if "Chapter" in df.columns and not pd.isna(row.get("Chapter")):
            metadata["chapter"] = row.get("Chapter")
        if "chapter" in df.columns and not pd.isna(row.get("chapter")):
            metadata["chapter"] = row.get("chapter")
        if "Verse" in df.columns and not pd.isna(row.get("Verse")):
            metadata["verse"] = row.get("Verse")
        if "verse" in df.columns and not pd.isna(row.get("verse")):
            metadata["verse"] = row.get("verse")

        doc_id = f"{id_prefix}{i}"
        documents.append(Document(page_content=page_content, metadata=metadata, id=doc_id))
        ids.append(doc_id)

    if documents:
        logger.info("Adding %d CSV documents to Chroma", len(documents))
        chroma.add_documents(documents=documents, ids=ids)
    else:
        logger.info("No CSV documents found to add.")


def index_epub(epub_path: str, chroma: Chroma, id_prefix: str = "epub_") -> None:
    """Extract paragraphs from EPUB, chunk them, and add to Chroma."""
    epub_path = os.path.expanduser(epub_path)
    logger.info("Indexing EPUB: %s", epub_path)
    try:
        paragraphs = _extract_paragraphs(epub_path)
    except Exception as e:
        logger.error("Failed to extract EPUB %s: %s", epub_path, e)
        return

    chunks = _chunk_paragraphs(paragraphs)
    docs = [Document(page_content=c, metadata={"source": os.path.basename(epub_path)}) for c in chunks]
    ids = [f"{id_prefix}{uuid.uuid4()}" for _ in docs]
    if docs:
        logger.info("Adding %d EPUB documents to Chroma", len(docs))
        chroma.add_documents(documents=docs, ids=ids)
    else:
        logger.info("No chunks produced from EPUB; nothing added.")


def _clear_dir(path: str) -> None:
    """Remove files and empty directories under path."""
    if not os.path.exists(path):
        return
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            try:
                os.remove(os.path.join(root, name))
            except Exception:
                pass
        for name in dirs:
            try:
                os.rmdir(os.path.join(root, name))
            except Exception:
                pass


def get_retriever(
    csv_path: str = os.path.expanduser(
        "~/Documents/REPO/Prive/KISHA/DEVELOP/bible_databases/formats/csv/KJV.csv"
    ),
    epub_path: str = os.path.expanduser(
        "~/Documents/REPO/Prive/KISHA/DEVELOP/bible_databases/formats/The-MacArthur-Bible-Commentary.epub"
    ),
    db_location: str = "./chrome_langchain_db",
    model_name: str = "mxbai-embed-large",
    k: int = 5,
    reindex: bool = False,
):
    """
    Build and return a Chroma retriever that indexes ONLY the provided CSV and EPUB.

    Behavior:
      - Uses a dedicated persist directory: '<db_location>/epub_csv_only' so other DB files are not consulted.
      - If that directory is empty, indexes both CSV and EPUB (CSV first, then EPUB).
      - Pass reindex=True to clear only that subdirectory and rebuild from the two sources.
    """
    db_location = os.path.expanduser(db_location)
    os.makedirs(db_location, exist_ok=True)
    persist_dir = os.path.join(db_location, "epub_csv_only")
    os.makedirs(persist_dir, exist_ok=True)

    if reindex:
        logger.info("Reindex requested — clearing persist dir %s", persist_dir)
        _clear_dir(persist_dir)

    embeddings = OllamaEmbeddings(model=model_name)
    chroma = Chroma(collection_name="bible_epub_csv", persist_directory=persist_dir, embedding_function=embeddings)

    # Only consider the dedicated persist_dir when deciding whether to index.
    need_index = not any(os.scandir(persist_dir))  # True if empty
    if need_index:
        logger.info("No persisted DB found in %s. Indexing CSV + EPUB only...", persist_dir)
        # Index CSV (only if path exists and is a file)
        if csv_path and os.path.isfile(os.path.expanduser(csv_path)):
            try:
                index_csv(csv_path, chroma)
            except Exception as e:
                logger.exception("Error while indexing CSV %s: %s", csv_path, e)
        else:
            logger.warning("CSV path not provided or not a file; skipping CSV indexing: %s", csv_path)

        # Index EPUB (only if path exists and is a file)
        if epub_path and os.path.isfile(os.path.expanduser(epub_path)):
            try:
                index_epub(epub_path, chroma)
            except Exception as e:
                logger.exception("Error while indexing EPUB %s: %s", epub_path, e)
        else:
            logger.warning("EPUB path not provided or not a file; skipping EPUB indexing: %s", epub_path)

        # attempt to persist (some wrappers may not need it)
        try:
            chroma.persist()
        except Exception:
            # Some Chroma wrappers persist automatically — ignore if not supported.
            pass
    else:
        logger.info("Found existing persisted DB in %s — skipping indexing (use reindex=True to force)", persist_dir)

    retriever = chroma.as_retriever(search_kwargs={"k": k})
    return retriever
