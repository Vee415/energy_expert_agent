"""Document ingestion pipeline for ENTSO-E energy reports.

Loads PDFs, chunks them (with table-aware splitting), creates embeddings,
and uploads to Qdrant vector DB.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

from pydantic import Field
from pydantic_settings import BaseSettings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DocumentConfig(BaseSettings):
    """Configuration for document ingestion pipeline."""

    # Paths
    documents_dir: str = Field(default="data/documents", description="Directory containing PDF files")

    # Chunking settings
    chunk_size: int = Field(default=512, description="Token size for each chunk")
    chunk_overlap: int = Field(default=100, description="Overlap between chunks")

    # Embedding settings
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Sentence-transformers model name")
    embed_batch_size: int = Field(default=32, description="Batch size for embedding generation")

    # Qdrant settings
    qdrant_host: str = Field(default="localhost", description="Qdrant server host")
    qdrant_port: int = Field(default=6333, description="Qdrant server port")
    collection_name: str = Field(default="entsoe_reports", description="Qdrant collection name")
    qdrant_batch_size: int = Field(default=100, description="Batch size for Qdrant uploads")

    # Vector dimensions (depends on embedding model)
    vector_size: int = Field(default=384, description="Embedding vector dimension (384 for all-MiniLM-L6-v2)")

    class Config:
        env_file = ".env"
        env_prefix = ""


def load_documents(config: DocumentConfig) -> List[Dict[str, Any]]:
    """Load all PDF documents from the documents directory.

    Args:
        config: DocumentConfig with documents_dir path

    Returns:
        List of document dictionaries with page_content and metadata
    """
    docs_dir = Path(config.documents_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    pdf_files = list(docs_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {docs_dir}")

    all_documents = []

    for pdf_path in pdf_files:
        logger.info(f"Loading: {pdf_path.name}")
        try:
            loader = PyMuPDFLoader(str(pdf_path))
            documents = loader.load()

            # Add source filename to metadata
            for doc in documents:
                doc.metadata["source"] = pdf_path.name
                doc.metadata["file_path"] = str(pdf_path)

            all_documents.extend(documents)
            logger.info(f"  Loaded {len(documents)} pages from {pdf_path.name}")

        except Exception as e:
            logger.warning(f"  Failed to load {pdf_path.name}: {e}")
            continue

    logger.info(f"Total pages loaded: {len(all_documents)}")
    return all_documents


def _is_tabular_line(line: str) -> bool:
    """Check if a line looks like a table row.

    A line is considered tabular if it contains tab characters or has
    multiple column-like separations (3+ consecutive spaces between words).
    Uses pure string operations for speed on PDF-extracted text.
    """
    stripped = line.strip()
    if not stripped:
        return False
    # Skip very long lines (prose, not tables)
    if len(stripped) > 300:
        return False
    # Lines with tab characters are tabular
    if "\t" in stripped:
        return True
    # Lines with pipe separators are tabular
    if "|" in stripped and stripped.count("|") >= 2:
        return True
    # Check for 3+ consecutive spaces (column alignment)
    # Simple string split — no regex needed
    parts = stripped.split("   ")  # 3 spaces
    return len(parts) >= 3


def detect_tables(page_text: str) -> List[Tuple[int, int]]:
    """Detect table regions in page text.

    Scans line-by-line to find contiguous runs of tabular lines.
    A table region requires at least 2 consecutive tabular lines (header + data).
    Pages with more than 200 lines skip table detection (likely malformed extraction).

    Args:
        page_text: Raw text content of a PDF page

    Returns:
        List of (start_line, end_line) tuples marking table regions
        (inclusive start, exclusive end, like Python slicing)
    """
    lines = page_text.split("\n")
    if not lines or len(lines) > 200:
        return []

    # Classify each line as tabular or not
    tabular_flags = [_is_tabular_line(line) for line in lines]

    # Find contiguous runs of tabular lines
    table_regions = []
    i = 0
    while i < len(lines):
        if tabular_flags[i]:
            # Start of a potential table region
            start = i
            tabular_count = 1
            consecutive_non_tabular = 0
            j = i + 1
            while j < len(lines):
                if tabular_flags[j]:
                    consecutive_non_tabular = 0
                    tabular_count += 1
                else:
                    consecutive_non_tabular += 1
                    if consecutive_non_tabular > 1:
                        break
                j += 1

            # Trim trailing non-tabular lines
            end = j - consecutive_non_tabular if consecutive_non_tabular > 0 else j

            # Only count as a table if at least 2 lines are tabular
            if tabular_count >= 2:
                table_regions.append((start, end))
            # Always advance at least 1 line to avoid infinite loop
            i = max(end, start + 1)
        else:
            i += 1

    return table_regions


def split_with_table_awareness(
    page_text: str,
    page_metadata: Dict[str, Any],
    text_splitter: RecursiveCharacterTextSplitter,
    chunk_id_start: int
) -> Tuple[List[Dict[str, Any]], int]:
    """Split a single page's text into chunks, keeping tables intact.

    Detects table regions in the page text and keeps them as whole chunks
    (never split across boundaries). Narrative text between tables is split
    using RecursiveCharacterTextSplitter as before.

    Args:
        page_text: Raw text content of a PDF page
        page_metadata: Metadata dict to attach to each chunk
        text_splitter: Pre-created RecursiveCharacterTextSplitter instance
        chunk_id_start: Starting chunk_id for this page

    Returns:
        Tuple of (list of chunk dicts, next chunk_id)
    """
    lines = page_text.split("\n")
    table_regions = detect_tables(page_text)

    chunks = []
    chunk_id = chunk_id_start

    # Build ordered list of content regions
    # Each region is either "narrative" or "table"
    regions = []
    prev_end = 0
    for table_start, table_end in table_regions:
        # Narrative text before this table
        if table_start > prev_end:
            narrative_text = "\n".join(lines[prev_end:table_start]).strip()
            if narrative_text:
                regions.append(("narrative", narrative_text))
        # Table text
        table_text = "\n".join(lines[table_start:table_end]).strip()
        if table_text:
            regions.append(("table", table_text))
        prev_end = table_end

    # Narrative text after last table
    if prev_end < len(lines):
        remaining_text = "\n".join(lines[prev_end:]).strip()
        if remaining_text:
            regions.append(("narrative", remaining_text))

    # If no regions detected (no tables), treat whole page as narrative
    if not regions:
        stripped_text = page_text.strip()
        if stripped_text:
            regions.append(("narrative", stripped_text))

    # Process each region
    for content_type, text in regions:
        if content_type == "table":
            # Tables are kept as single chunks, even if oversized
            chunks.append({
                "text": text,
                "metadata": {
                    **page_metadata,
                    "chunk_id": chunk_id,
                    "chunk_index": 0,
                    "total_chunks_in_region": 1,
                    "content_type": "table",
                }
            })
            chunk_id += 1
        else:
            # Narrative text is split normally
            sub_chunks = text_splitter.split_text(text)
            for i, chunk_text in enumerate(sub_chunks):
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        **page_metadata,
                        "chunk_id": chunk_id,
                        "chunk_index": i,
                        "total_chunks_in_region": len(sub_chunks),
                        "content_type": "narrative",
                    }
                })
                chunk_id += 1

    return chunks, chunk_id


def chunk_documents(
    documents: List[Dict[str, Any]],
    config: DocumentConfig
) -> List[Dict[str, Any]]:
    """Split documents into chunks, keeping tables intact.

    Uses table-aware splitting: tables are detected and kept as whole chunks,
    while narrative text is split using RecursiveCharacterTextSplitter.

    Args:
        documents: List of loaded documents
        config: DocumentConfig with chunk_size and chunk_overlap

    Returns:
        List of chunk dictionaries with text and metadata
    """
    chunks = []
    chunk_id = 0
    table_count = 0

    # Create splitter once, reuse for all pages
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    for idx, doc in enumerate(documents):
        source = doc.metadata.get("source", "?")
        page = doc.metadata.get("page", "?")
        page_len = len(doc.page_content)
        line_count = doc.page_content.count("\n") + 1
        if idx < 5 or idx % 200 == 0:
            logger.info(f"  Chunking page {idx+1}/{len(documents)}: {source} p{page} ({page_len} chars, {line_count} lines)")
        try:
            doc_chunks, chunk_id = split_with_table_awareness(
                page_text=doc.page_content,
                page_metadata=doc.metadata,
                text_splitter=text_splitter,
                chunk_id_start=chunk_id,
            )
        except Exception as e:
            logger.warning(f"  Table detection failed on {source} p{page}: {e}. Using narrative fallback.")
            doc_chunks = []
            for i, chunk_text in enumerate(text_splitter.split_text(doc.page_content)):
                doc_chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        **doc.metadata,
                        "chunk_id": chunk_id,
                        "chunk_index": i,
                        "total_chunks_in_region": 0,
                        "content_type": "narrative",
                    }
                })
                chunk_id += 1
        table_count += sum(1 for c in doc_chunks if c["metadata"].get("content_type") == "table")
        chunks.extend(doc_chunks)

    logger.info(f"Created {len(chunks)} chunks from {len(documents)} pages ({table_count} table chunks)")
    return chunks


def create_embeddings(
    chunks: List[Dict[str, Any]],
    config: DocumentConfig
) -> List[Dict[str, Any]]:
    """Create embeddings for chunks using sentence-transformers.

    Args:
        chunks: List of chunk dictionaries with text
        config: DocumentConfig with embedding_model and embed_batch_size

    Returns:
        List of chunks with embeddings added
    """
    logger.info(f"Loading embedding model: {config.embedding_model}")
    model = SentenceTransformer(config.embedding_model, device="cpu")

    texts = [chunk["text"] for chunk in chunks]
    total_chunks = len(texts)

    logger.info(f"Creating embeddings for {total_chunks} chunks (batch_size={config.embed_batch_size})")

    all_embeddings = []
    for i in range(0, total_chunks, config.embed_batch_size):
        batch = texts[i:i + config.embed_batch_size]
        embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.extend(embeddings)

        if (i // config.embed_batch_size + 1) % 10 == 0 or i == 0:
            logger.info(f"  Processed {min(i + config.embed_batch_size, total_chunks)}/{total_chunks} chunks")

    # Add embeddings to chunks
    for chunk, embedding in zip(chunks, all_embeddings):
        chunk["embedding"] = embedding.tolist()

    logger.info(f"Created {len(all_embeddings)} embeddings (dimension: {len(all_embeddings[0])})")
    return chunks


def ingest_to_qdrant(
    chunks: List[Dict[str, Any]],
    config: DocumentConfig
) -> Dict[str, Any]:
    """Upload chunks with embeddings to Qdrant vector database.

    Args:
        chunks: List of chunks with embeddings
        config: DocumentConfig with Qdrant connection details

    Returns:
        Dictionary with ingestion summary
    """
    logger.info(f"Connecting to Qdrant at {config.qdrant_host}:{config.qdrant_port}")
    client = QdrantClient(host=config.qdrant_host, port=config.qdrant_port)

    # Check if collection exists, create if not
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if config.collection_name not in collection_names:
        logger.info(f"Creating collection: {config.collection_name}")
        client.create_collection(
            collection_name=config.collection_name,
            vectors_config=VectorParams(
                size=config.vector_size,
                distance=Distance.COSINE
            )
        )
    else:
        logger.info(f"Using existing collection: {config.collection_name}")

    # Prepare points for upload
    points = []
    for i, chunk in enumerate(chunks):
        point = PointStruct(
            id=i,
            vector=chunk["embedding"],
            payload={
                "text": chunk["text"],
                **chunk["metadata"]
            }
        )
        points.append(point)

    # Upload in batches
    total_points = len(points)
    logger.info(f"Uploading {total_points} points to Qdrant (batch_size={config.qdrant_batch_size})")

    for i in range(0, total_points, config.qdrant_batch_size):
        batch = points[i:i + config.qdrant_batch_size]
        client.upsert(
            collection_name=config.collection_name,
            points=batch
        )

        if (i // config.qdrant_batch_size + 1) % 5 == 0 or i == 0:
            logger.info(f"  Uploaded {min(i + config.qdrant_batch_size, total_points)}/{total_points} points")

    # Get collection info
    collection_info = client.get_collection(config.collection_name)

    return {
        "collection": config.collection_name,
        "vectors_count": collection_info.points_count,
        "points_uploaded": total_points
    }


def run_ingestion() -> Dict[str, Any]:
    """Main entry point for document ingestion pipeline.

    Returns:
        Dictionary with ingestion summary
    """
    start_time = datetime.now()
    logger.info("=" * 50)
    logger.info("Starting Document Ingestion Pipeline")
    logger.info("=" * 50)

    # Load configuration
    config = DocumentConfig()
    logger.info(f"Documents dir: {config.documents_dir}")
    logger.info(f"Embedding model: {config.embedding_model}")
    logger.info(f"Collection: {config.collection_name}")

    # Step 1: Load documents
    logger.info("\n[1/4] Loading documents...")
    documents = load_documents(config)

    if not documents:
        logger.error("No documents loaded. Exiting.")
        return {"error": "No documents loaded"}

    # Step 2: Chunk documents
    logger.info("\n[2/4] Chunking documents...")
    chunks = chunk_documents(documents, config)

    # Step 3: Create embeddings
    logger.info("\n[3/4] Creating embeddings...")
    chunks = create_embeddings(chunks, config)

    # Step 4: Ingest to Qdrant
    logger.info("\n[4/4] Ingesting to Qdrant...")
    result = ingest_to_qdrant(chunks, config)

    # Summary
    duration = (datetime.now() - start_time).total_seconds()
    logger.info("\n" + "=" * 50)
    logger.info("Ingestion Complete!")
    logger.info("=" * 50)
    logger.info(f"Documents processed: {len(documents)}")
    logger.info(f"Chunks created: {len(chunks)}")
    logger.info(f"Vectors in collection: {result['vectors_count']}")
    logger.info(f"Duration: {duration:.1f}s")

    return {
        "documents": len(documents),
        "chunks": len(chunks),
        **result,
        "duration_seconds": duration
    }


if __name__ == "__main__":
    run_ingestion()
