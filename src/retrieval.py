"""Retrieval module for querying ENTSO-E energy reports.

Handles vector search, reranking, and context assembly.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchRequest

from src.ingestion import DocumentConfig

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result with text and metadata."""
    text: str
    score: float
    source: str
    chunk_id: int
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Retriever:
    """Retrieves and reranks relevant chunks from Qdrant."""

    def __init__(
        self,
        config: Optional[DocumentConfig] = None,
        embedding_model: Optional[str] = None,
        reranker_model: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 10,
        rerank_top_k: int = 5
    ):
        """Initialize the retriever with embedding and reranker models.

        Args:
            config: DocumentConfig with Qdrant connection details
            embedding_model: Model for query embedding (default: all-MiniLM-L6-v2)
            reranker_model: Cross-encoder model for reranking
            top_k: Number of initial results from vector search
            rerank_top_k: Number of results after reranking
        """
        self.config = config or DocumentConfig()
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k

        # Embedding model (same as ingestion)
        self.embed_model_name = embedding_model or self.config.embedding_model
        logger.info(f"Loading embedding model: {self.embed_model_name}")
        self.embedding_model = SentenceTransformer(self.embed_model_name, device="cpu")

        # Reranker model (cross-encoder)
        self.reranker_model_name = reranker_model
        logger.info(f"Loading reranker model: {reranker_model}")
        self.rereranker = CrossEncoder(reranker_model, device="cpu")

        # Qdrant client
        logger.info(f"Connecting to Qdrant at {self.config.qdrant_host}:{self.config.qdrant_port}")
        self.qdrant = QdrantClient(
            host=self.config.qdrant_host,
            port=self.config.qdrant_port
        )

        logger.info("Retriever initialized successfully")

    def embed_query(self, query: str) -> List[float]:
        """Create embedding for a query string.

        Args:
            query: The search query

        Returns:
            Query embedding vector
        """
        return self.embedding_model.encode(query).tolist()

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_source: Optional[str] = None
    ) -> List[SearchResult]:
        """Search for relevant chunks using vector similarity.

        Args:
            query: Search query
            top_k: Number of results to return (default: self.top_k)
            filter_source: Optional filter by source filename

        Returns:
            List of SearchResult objects
        """
        top_k = top_k or self.top_k

        # Create query embedding
        query_vector = self.embed_query(query)

        # Build filter if source specified
        search_filter = None
        if filter_source:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=filter_source)
                    )
                ]
            )

        # Search Qdrant
        logger.debug(f"Searching for: '{query[:50]}...' (top_k={top_k})")
        results = self.qdrant.query_points(
            collection_name=self.config.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=search_filter,
            with_payload=True,
            with_vectors=False
        ).points

        # Convert to SearchResult objects
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                text=result.payload.get("text", ""),
                score=result.score,
                source=result.payload.get("source", "unknown"),
                chunk_id=result.payload.get("chunk_id", -1),
                page_number=result.payload.get("page"),
                metadata={k: v for k, v in result.payload.items() if k != "text"}
            ))

        logger.info(f"Retrieved {len(search_results)} chunks from vector search")
        return search_results

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """Rerank results using cross-encoder for better relevance.

        Args:
            query: Original search query
            results: Initial search results
            top_k: Number of results to return (default: self.rerank_top_k)

        Returns:
            Reranked list of SearchResult objects
        """
        top_k = top_k or self.rerank_top_k

        if not results:
            return []

        # Prepare query-document pairs for cross-encoder
        pairs = [(query, result.text) for result in results]

        # Score pairs
        logger.debug(f"Reranking {len(pairs)} results")
        scores = self.rereranker.predict(pairs)

        # Sort by reranker score (descending)
        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Take top_k
        reranked = []
        for result, score in scored_results[:top_k]:
            result.score = float(score)  # Update score with reranker score
            reranked.append(result)

        logger.info(f"Reranked to top {len(reranked)} results")
        return reranked

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
        filter_source: Optional[str] = None,
        use_hybrid: bool = True
    ) -> List[SearchResult]:
        """Full retrieval pipeline: search + rerank (+ optional hybrid boost).

        Args:
            query: Search query
            top_k: Initial number of results from vector search
            rerank_top_k: Final number of results after reranking
            filter_source: Optional filter by source filename
            use_hybrid: If True, boost results with keyword matches

        Returns:
            Reranked list of SearchResult objects
        """
        # Use defaults if not specified
        top_k = top_k or self.top_k
        rerank_top_k = rerank_top_k or self.rerank_top_k

        # Step 1: Vector search (fetch 3x candidates for better reranking)
        search_k = top_k * 3
        results = self.search(query, top_k=search_k, filter_source=filter_source)

        if not results:
            logger.warning(f"No results found for query: '{query[:50]}...'")
            return []

        # Step 2: Hybrid boost (keyword matching)
        if use_hybrid:
            results = self._hybrid_boost(query, results)

        # Step 3: Rerank
        reranked = self.rerank(query, results, top_k=rerank_top_k)

        return reranked

    def _hybrid_boost(
        self,
        query: str,
        results: List[SearchResult],
        boost: float = 2.0
    ) -> List[SearchResult]:
        """Boost results that contain query keywords.

        This improves retrieval for specific terms like years, regions, metrics.
        """
        import re

        # Extract keywords (numbers, capitalized words, specific terms)
        query_lower = query.lower()

        # Score each result
        for result in results:
            text_lower = result.text.lower()
            source_lower = result.source.lower()
            keyword_matches = 0

            # Check for year mentions — and boost source filename match
            years = re.findall(r'\b(20\d{2})\b', query)
            for year in years:
                if year in text_lower:
                    keyword_matches += 2  # Strong boost for year matches
                if year in source_lower:
                    keyword_matches += 3  # Strongest boost: year in filename

            # Check for region/organization keywords
            regions = ['nordic', 'european', 'germany', 'france', 'italy', 'spain', 'benelux']
            for region in regions:
                if region in query_lower and region in text_lower:
                    keyword_matches += 1.5

            # Check for metric keywords
            metrics = ['mfr', 'afrr', 'fcr', 'volume', 'price', 'gwh', 'euro', 'mw']
            for metric in metrics:
                if metric in query_lower and metric in text_lower:
                    keyword_matches += 1

            # Apply boost
            if keyword_matches > 0:
                result.score += boost * keyword_matches

        # Re-sort by boosted score
        results.sort(key=lambda x: x.score, reverse=True)

        return results

    def get_context_string(
        self,
        results: List[SearchResult],
        include_metadata: bool = True
    ) -> str:
        """Format search results into a context string for the LLM.

        Args:
            results: List of SearchResult objects
            include_metadata: Whether to include source info

        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant context found."

        context_parts = []
        for i, result in enumerate(results, 1):
            # Mark table chunks so the LLM knows to read them as tabular data
            content_type = result.metadata.get("content_type", "narrative")
            type_label = " [TABLE]" if content_type == "table" else ""

            if include_metadata:
                context_parts.append(
                    f"[{i}] Source: {result.source}, Chunk: {result.chunk_id}, Score: {result.score:.3f}{type_label}\n{result.text}"
                )
            else:
                context_parts.append(f"[{i}]{type_label} {result.text}")

        return "\n\n---\n\n".join(context_parts)

    def query_collection_info(self) -> Dict[str, Any]:
        """Get information about the Qdrant collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            collection = self.qdrant.get_collection(self.config.collection_name)
            return {
                "collection_name": self.config.collection_name,
                "vectors_count": collection.points_count,
                "status": collection.status,
                "vector_size": collection.config.params.vectors.size if collection.config.params.vectors else None,
                "distance": str(collection.config.params.vectors.distance) if collection.config.params.vectors else None
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}


def test_retrieval():
    """Quick test of the retrieval pipeline."""
    logging.basicConfig(level=logging.INFO)

    print("Initializing retriever...")
    retriever = Retriever(top_k=10, rerank_top_k=5)

    # Check collection
    info = retriever.query_collection_info()
    print(f"\nCollection info: {info}")

    # Test queries
    test_queries = [
        "What is the balancing report about?",
        "energy market trends in 2024",
        "regional coordination mechanisms"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)

        results = retriever.retrieve(query)

        print(f"\nTop {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result.score:.3f} | Source: {result.source} | Chunk: {result.chunk_id}")
            print(f"   Text: {result.text[:200]}...")

        # Show full context
        context = retriever.get_context_string(results)
        print(f"\nContext length: {len(context)} chars")


if __name__ == "__main__":
    test_retrieval()
