"""Uncertainty quantification module for the Document Agent.

Provides 3-signal confidence scoring:
1. Retrieval confidence - relevance of retrieved chunks
2. Answer consistency - semantic alignment between answer and context
3. Self-evaluation - LLM's own assessment of answer quality
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)


@dataclass
class UncertaintySignals:
    """Container for all uncertainty signals."""
    retrieval_confidence: float  # Signal 1: Retrieved chunks relevance
    answer_consistency: float      # Signal 2: Answer-context alignment
    self_evaluation: float         # Signal 3: LLM self-assessment

    # Combined score
    overall_confidence: float

    # Thresholds for decision making
    is_confident: bool           # True if overall > 0.6
    should_answer: bool          # True if overall > 0.3

    # Explanations
    retrieval_explanation: str
    consistency_explanation: str
    self_eval_explanation: str


class UncertaintyQuantifier:
    """Quantifies uncertainty in RAG responses using multiple signals."""

    def __init__(
        self,
        consistency_model: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: Optional[str] = None,
        retrieval_threshold: float = 0.3,
        confidence_threshold: float = 0.6
    ):
        """Initialize uncertainty quantifier.

        Args:
            consistency_model: Model for semantic similarity
            llm_model: Model for self-evaluation (uses same as agent if None)
            retrieval_threshold: Minimum score to consider retrieval valid
            confidence_threshold: Minimum for "high confidence" classification
        """
        self.consistency_model_name = consistency_model
        self.retrieval_threshold = retrieval_threshold
        self.confidence_threshold = confidence_threshold

        # Load consistency model (same as retrieval embedding model)
        logger.info(f"Loading consistency model: {consistency_model}")
        self.consistency_model = SentenceTransformer(consistency_model, device="cpu")

        # LLM for self-evaluation (lazy loaded)
        self.llm_model_name = llm_model or "gemma3:4b"
        self._llm = None

        logger.info("UncertaintyQuantifier initialized")

    @property
    def llm(self):
        """Lazy load LLM for self-evaluation."""
        if self._llm is None:
            import os
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self._llm = OllamaLLM(
                model=self.llm_model_name,
                temperature=0.0,
                base_url=base_url
            )
        return self._llm

    def calculate_retrieval_confidence(
        self,
        retrieval_scores: List[float],
        num_chunks: int
    ) -> Tuple[float, str]:
        """Signal 1: Calculate confidence based on retrieval scores.

        Args:
            retrieval_scores: Reranker scores for retrieved chunks
            num_chunks: Total number of chunks retrieved

        Returns:
            Tuple of (confidence_score, explanation)
        """
        if not retrieval_scores or num_chunks == 0:
            return 0.0, "No chunks retrieved"

        # Cross-encoder scores typically range from -10 to +10
        # Normalize to 0-1 range
        normalized_scores = [(s + 10) / 20 for s in retrieval_scores]
        avg_score = np.mean(normalized_scores)

        # Check if top result is relevant enough
        top_score = normalized_scores[0] if normalized_scores else 0

        # Weighted combination: 70% average, 30% top score
        confidence = 0.7 * avg_score + 0.3 * top_score
        confidence = max(0.0, min(1.0, confidence))

        # Generate explanation
        if confidence >= 0.7:
            explanation = f"High relevance retrieved (top score: {top_score:.2f}, avg: {avg_score:.2f})"
        elif confidence >= 0.4:
            explanation = f"Moderate relevance retrieved (top score: {top_score:.2f}, avg: {avg_score:.2f})"
        else:
            explanation = f"Low relevance retrieved (top score: {top_score:.2f}, avg: {avg_score:.2f})"

        return confidence, explanation

    def calculate_answer_consistency(
        self,
        answer: str,
        context_chunks: List[str]
    ) -> Tuple[float, str]:
        """Signal 2: Calculate semantic consistency between answer and context.

        Args:
            answer: Generated answer text
            context_chunks: List of retrieved context chunks

        Returns:
            Tuple of (consistency_score, explanation)
        """
        if not answer or not context_chunks:
            return 0.0, "No answer or context provided"

        # Encode answer
        answer_embedding = self.consistency_model.encode(answer, convert_to_tensor=True)

        # Encode context chunks
        chunk_embeddings = self.consistency_model.encode(
            context_chunks,
            convert_to_tensor=True
        )

        # Calculate cosine similarities
        from sentence_transformers.util import cos_sim
        similarities = cos_sim(answer_embedding, chunk_embeddings)[0]

        # Get max and average similarity
        max_sim = float(similarities.max())
        avg_sim = float(similarities.mean())

        # Weighted combination: 60% max, 40% average
        consistency = 0.6 * max_sim + 0.4 * avg_sim

        # Generate explanation
        if consistency >= 0.7:
            explanation = f"Answer highly consistent with context (max sim: {max_sim:.2f})"
        elif consistency >= 0.5:
            explanation = f"Answer moderately consistent with context (max sim: {max_sim:.2f})"
        elif consistency >= 0.3:
            explanation = f"Answer loosely related to context (max sim: {max_sim:.2f})"
        else:
            explanation = f"Answer poorly aligned with context (max sim: {max_sim:.2f}) - possible hallucination"

        return consistency, explanation

    def self_evaluation(
        self,
        question: str,
        answer: str,
        context: str
    ) -> Tuple[float, str]:
        """Signal 3: LLM self-evaluation of answer quality.

        Args:
            question: Original question
            answer: Generated answer
            context: Retrieved context

        Returns:
            Tuple of (confidence_score, explanation)
        """
        prompt = f"""You are evaluating the quality of an AI-generated answer. Rate the following:

Context from documents:
{'='*40}
{context[:1500]}
{'='*40}

Question: {question}

Generated Answer: {answer}

Evaluate on these criteria:
1. Does the answer directly address the question? (0-33 points)
2. Is the answer supported by the context? (0-33 points)
3. Is the answer complete and accurate? (0-34 points)

Total score will be 0-100. Provide ONLY a numeric score (0-100) followed by a brief reason.

Format: SCORE: <number> | REASON: <explanation>

Evaluation:"""

        try:
            response = self.llm.invoke(prompt)

            # Parse score from response
            score_match = re.search(r'(\d+)', response)
            if score_match:
                score = int(score_match.group(1))
                score = max(0, min(100, score))  # Clamp to 0-100
                normalized_score = score / 100.0
            else:
                normalized_score = 0.5  # Default if parsing fails

            # Extract reason
            reason_match = re.search(r'REASON:\s*(.+)', response, re.IGNORECASE)
            if reason_match:
                explanation = reason_match.group(1).strip()
            else:
                explanation = response[:100] if response else "Self-evaluation completed"

            return normalized_score, explanation

        except Exception as e:
            logger.warning(f"Self-evaluation failed: {e}")
            return 0.5, "Self-evaluation unavailable"

    def quantify_uncertainty(
        self,
        question: str,
        answer: str,
        context_chunks: List[str],
        retrieval_scores: List[float],
        num_chunks: int
    ) -> UncertaintySignals:
        """Calculate all uncertainty signals and combine them.

        Args:
            question: Original question
            answer: Generated answer
            context_chunks: Retrieved context chunks
            retrieval_scores: Scores from retriever
            num_chunks: Number of chunks retrieved

        Returns:
            UncertaintySignals with all confidence metrics
        """
        # Signal 1: Retrieval confidence
        retrieval_conf, retrieval_exp = self.calculate_retrieval_confidence(
            retrieval_scores, num_chunks
        )

        # Signal 2: Answer consistency
        consistency, consistency_exp = self.calculate_answer_consistency(
            answer, context_chunks
        )

        # Signal 3: Self-evaluation
        context_text = "\n\n".join(context_chunks)
        self_eval, self_eval_exp = self.self_evaluation(
            question, answer, context_text
        )

        # Combine signals with weights
        # Retrieval is most important (found good context)
        # Consistency catches hallucinations
        # Self-eval provides sanity check
        weights = {
            'retrieval': 0.4,
            'consistency': 0.35,
            'self_eval': 0.25
        }

        overall = (
            weights['retrieval'] * retrieval_conf +
            weights['consistency'] * consistency +
            weights['self_eval'] * self_eval
        )

        # Decision thresholds
        is_confident = overall >= self.confidence_threshold
        should_answer = overall >= self.retrieval_threshold

        return UncertaintySignals(
            retrieval_confidence=retrieval_conf,
            answer_consistency=consistency,
            self_evaluation=self_eval,
            overall_confidence=overall,
            is_confident=is_confident,
            should_answer=should_answer,
            retrieval_explanation=retrieval_exp,
            consistency_explanation=consistency_exp,
            self_eval_explanation=self_eval_exp
        )

    def get_confidence_report(self, signals: UncertaintySignals) -> str:
        """Generate a human-readable confidence report."""
        report = f"""
Confidence Report
{'='*50}
Overall Confidence: {signals.overall_confidence:.1%}
Decision: {'ANSWER' if signals.should_answer else 'DECLINE'}
({'HIGH' if signals.is_confident else 'LOW'} confidence)

Signal Breakdown:
  1. Retrieval Quality:  {signals.retrieval_confidence:.1%}
     -> {signals.retrieval_explanation}

  2. Answer Consistency: {signals.answer_consistency:.1%}
     -> {signals.consistency_explanation}

  3. Self-Evaluation:    {signals.self_evaluation:.1%}
     -> {signals.self_eval_explanation}

{'='*50}
"""
        return report


class UncertaintyAwareAgentResponse:
    """Enhanced response with uncertainty information."""

    def __init__(
        self,
        answer: str,
        sources: List[str],
        uncertainty_signals: UncertaintySignals,
        retrieved_context: str
    ):
        self.answer = answer
        self.sources = sources
        self.signals = uncertainty_signals
        self.context = retrieved_context

    @property
    def confident_answer(self) -> str:
        """Get answer only if confident, otherwise explanation."""
        if self.signals.should_answer:
            if self.signals.is_confident:
                return self.answer
            else:
                return f"{self.answer}\n\n[Note: Low confidence ({self.signals.overall_confidence:.0%}) - verify this information]"
        else:
            return (
                "I don't have sufficient information to answer this question confidently "
                f"(confidence: {self.signals.overall_confidence:.0%}). "
                "The retrieved documents don't contain relevant information."
            )


def test_uncertainty():
    """Test uncertainty quantification with examples."""
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Initializing Uncertainty Quantifier...")
    uq = UncertaintyQuantifier()

    # Test cases
    test_cases = [
        {
            "name": "Good retrieval - Balancing report",
            "question": "What are mFRR and aFRR?",
            "answer": "mFRR (manual Frequency Restoration Reserve) and aFRR (automatic Frequency Restoration Reserve) are balancing services used in European electricity markets.",
            "chunks": [
                "The FRR process comprises the activation of aFRR also known as secondary frequency control reserve",
                "mFRR balancing capacity for upward regulation is procured in a national market",
                "aFRR is a process under development where procured volumes are constantly increasing"
            ],
            "scores": [6.7, 6.5, 5.8]  # High scores
        },
        {
            "name": "Poor retrieval - Tesla CEO (out of domain)",
            "question": "Who is the CEO of Tesla?",
            "answer": "Elon Musk is the CEO of Tesla.",
            "chunks": [
                "Operator Company Ltd 23February 2016 EirGrid plc",
                "Landsnet hf 23February 2016 Terna Rete Elettrica Nazionale",
                "Litgrid AB 23February 2016 AS Augstsprieguma"
            ],
            "scores": [-10.0, -10.4, -10.5]  # Very negative scores
        }
    ]

    for test in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {test['name']}")
        print(f"{'='*60}")

        signals = uq.quantify_uncertainty(
            question=test["question"],
            answer=test["answer"],
            context_chunks=test["chunks"],
            retrieval_scores=test["scores"],
            num_chunks=len(test["chunks"])
        )

        print(uq.get_confidence_report(signals))


if __name__ == "__main__":
    test_uncertainty()
