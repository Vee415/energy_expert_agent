"""Intelligent Document Agent with retrieval and answering.

Simple agent that retrieves context and uses LLM to answer questions.
Supports both local (Ollama) and cloud (Claude API) LLMs.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from langchain_ollama import OllamaLLM

from src.retrieval import Retriever

# Optional import for Claude API
try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Response from the agent with metadata."""
    answer: str
    sources: List[str]
    confidence: float
    retrieved_context: str


class DocumentAgent:
    """Agent for querying ENTSO-E energy reports.

    Supports multiple LLM providers:
    - ollama: Local models (default for development)
    - anthropic: Claude API (recommended for production)

    Set LLM_PROVIDER environment variable to switch.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        top_k: int = 5,
        llm_provider: Optional[str] = None
    ):
        """Initialize the document agent.

        Args:
            model_name: Model to use (provider-specific)
            temperature: Sampling temperature
            top_k: Number of chunks to retrieve
            llm_provider: 'ollama' or 'anthropic' (or set LLM_PROVIDER env var)
        """
        self.temperature = temperature
        self.top_k = top_k

        # Determine provider
        self.llm_provider = llm_provider or os.getenv("LLM_PROVIDER", "ollama")

        # Set default models per provider
        if model_name:
            self.model_name = model_name
        elif self.llm_provider == "anthropic":
            self.model_name = "claude-3-haiku-20240307"
        else:
            self.model_name = "gemma3:4b"

        # Initialize retriever
        logger.info("Initializing retriever...")
        self.retriever = Retriever(top_k=top_k, rerank_top_k=top_k)

        # Initialize LLM based on provider
        self._init_llm()

        logger.info("DocumentAgent initialized successfully")

    def _init_llm(self):
        """Initialize the LLM based on provider."""
        if self.llm_provider == "anthropic":
            self._init_anthropic()
        else:
            self._init_ollama()

    def _init_anthropic(self):
        """Initialize Claude API."""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Anthropic support requires: pip install langchain-anthropic"
            )

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable required for Claude API"
            )

        logger.info(f"Initializing Claude API: {self.model_name}")
        self.llm = ChatAnthropic(
            model=self.model_name,
            anthropic_api_key=api_key,
            temperature=self.temperature
        )
        logger.info("Claude API initialized successfully")

    def _init_ollama(self):
        """Initialize local Ollama."""
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        logger.info(f"Initializing Ollama: {self.model_name} at {base_url}")
        try:
            self.llm = OllamaLLM(
                model=self.model_name,
                temperature=self.temperature,
                base_url=base_url
            )
            # Test connection
            _ = self.llm.invoke("Hello")
            logger.info(f"Ollama initialized successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise RuntimeError(
                f"Ollama not accessible at {base_url}.\n"
                "Make sure it's running: ollama serve\n"
                f"Or set OLLAMA_BASE_URL env var.\n"
                "To use Claude API instead: export LLM_PROVIDER=anthropic"
            )

    def _calculate_confidence(self, results) -> float:
        """Calculate confidence based on retrieval scores."""
        if not results:
            return 0.0

        # Average the scores and normalize
        avg_score = sum(r.score for r in results) / len(results)
        # Cross-encoder scores are roughly -10 to +10
        # Normalize to 0-1 range
        confidence = (avg_score + 10) / 20
        return max(0.0, min(1.0, confidence))

    def ask(self, question: str) -> AgentResponse:
        """Ask the agent a question.

        Args:
            question: The user's question

        Returns:
            AgentResponse with answer and metadata
        """
        logger.info(f"Processing question: {question}")

        # Step 1: Retrieve relevant context
        retrieved_results = self.retriever.retrieve(question)
        context = self.retriever.get_context_string(retrieved_results)

        # Get unique sources
        sources = list(set([r.source for r in retrieved_results])) if retrieved_results else []

        # Calculate confidence
        confidence = self._calculate_confidence(retrieved_results)

        # Step 2: Generate answer using LLM
        if not retrieved_results or confidence < 0.3:
            answer = (
                "I don't have sufficient information to answer this question. "
                "The retrieved documents don't contain relevant information about your query."
            )
        else:
            prompt = self._build_prompt(question, context)
            try:
                answer = self.llm.invoke(prompt)
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                answer = "I encountered an error while generating the answer."

        return AgentResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            retrieved_context=context
        )

    def _build_prompt(self, question: str, context: str) -> str:
        """Build prompt for the LLM."""
        return f"""You are an expert assistant analyzing ENTSO-E energy market reports.
Use ONLY the provided context to answer the question. If the context doesn't contain
sufficient information, say "I don't have enough information to answer this question."

Context from ENTSO-E reports:
{'=' * 60}
{context}
{'=' * 60}

Question: {question}

Provide a clear, accurate answer based on the context above. Include specific details
from the reports when possible. If the context contains conflicting information,
acknowledge it.

When context chunks are marked [TABLE], read them as structured data — extract specific
values, compare across rows/columns, and reference exact numbers when answering factual
questions about volumes, prices, or metrics.

Answer:"""

    def chat(self, question: str) -> str:
        """Simple chat interface that returns just the answer."""
        response = self.ask(question)
        return response.answer


def run_interactive():
    """Run interactive agent session."""
    import sys

    print("=" * 60)
    print("ENTSO-E Document Agent")
    print("=" * 60)
    print("\nInitializing... (make sure Ollama is running)")
    print("Start Ollama: ollama serve")
    print()

    try:
        agent = DocumentAgent()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        sys.exit(1)

    print("\nAgent ready! Ask questions about ENTSO-E reports.")
    print("Examples:")
    print("  - What are mFRR and aFRR?")
    print("  - What are the main findings in the balancing report?")
    print("  - How does regional coordination work?")
    print("\nType 'quit' to exit.\n")

    while True:
        try:
            question = input("\nQuestion: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            if not question:
                continue

            print("\nThinking...")
            response = agent.ask(question)

            print(f"\n{'=' * 60}")
            print("ANSWER:")
            print(f"{'=' * 60}")
            print(response.answer)

            print(f"\n{'=' * 60}")
            print(f"Confidence: {response.confidence:.1%}")
            print(f"Sources: {', '.join(response.sources) if response.sources else 'None'}")
            print(f"{'=' * 60}")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    run_interactive()
