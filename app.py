"""Streamlit frontend for the Intelligent Document Agent.

Interactive web interface for querying ENTSO-E energy reports
with uncertainty visualization.
"""

import sys
import logging
from pathlib import Path

import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agent import DocumentAgent
from src.uncertainty import UncertaintyQuantifier

# Configure logging
logging.basicConfig(level=logging.WARNING)

# Page configuration
st.set_page_config(
    page_title="ENTSO-E Document Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_agent():
    """Initialize or get cached agent."""
    if "agent" not in st.session_state:
        with st.spinner("Initializing agent... (this may take a moment)"):
            st.session_state.agent = DocumentAgent()
    return st.session_state.agent


def initialize_uncertainty():
    """Initialize or get cached uncertainty quantifier."""
    if "uncertainty" not in st.session_state:
        with st.spinner("Loading uncertainty models..."):
            st.session_state.uncertainty = UncertaintyQuantifier()
    return st.session_state.uncertainty


def format_confidence_bar(confidence: float) -> str:
    """Create a colored confidence bar."""
    if confidence >= 0.7:
        color = "🟢"
    elif confidence >= 0.4:
        color = "🟡"
    else:
        color = "🔴"

    filled = int(confidence * 10)
    bar = color * filled + "⬜" * (10 - filled)
    return bar


def main():
    """Main Streamlit application."""

    # Header
    st.title("⚡ ENTSO-E Document Agent")
    st.markdown("""
    *Intelligent Q&A over European energy market reports with uncertainty awareness*

    Ask questions about balancing reports, market reports, regional coordination,
    and implementation monitoring documents.
    """)

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        top_k = st.slider(
            "Number of chunks to retrieve",
            min_value=3,
            max_value=10,
            value=5,
            help="How many document chunks to retrieve for context"
        )

        show_context = st.checkbox(
            "Show retrieved context",
            value=False,
            help="Display the raw text chunks used to answer"
        )

        st.divider()

        st.header("About")
        st.markdown("""
        This agent uses:
        - **RAG** (Retrieval-Augmented Generation)
        - **Vector search** with Qdrant
        - **Reranking** with cross-encoder
        - **3-signal uncertainty quantification**

        Documents: 8 ENTSO-E PDFs (~78 MB)
        """)

        st.divider()

        st.header("Example Questions")
        st.markdown("""
        - What are mFRR and aFRR?
        - What are the main findings in the balancing report?
        - How does regional coordination work?
        - What are electricity market trends in 2024?
        - Who is the CEO of Tesla? (test out-of-domain)
        """)

    # Initialize components
    try:
        agent = initialize_agent()
        uncertainty = initialize_uncertainty()
    except Exception as e:
        st.error(f"Failed to initialize: {e}")
        st.info("Make sure Ollama is running: `ollama serve`")
        st.stop()

    # Query input
    st.divider()
    question = st.text_input(
        "Ask a question:",
        placeholder="What are the main findings in the balancing report?",
        key="question_input"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        submit = st.button("Ask", type="primary", use_container_width=True)
    with col2:
        st.caption("Press Enter or click Ask to submit")

    # Process query
    if submit and question:
        with st.spinner("Retrieving relevant documents..."):
            # Get response from agent
            response = agent.ask(question)

            # Get detailed uncertainty signals
            retrieved_chunks = response.retrieved_context.split("\n\n---\n\n")
            chunk_texts = [c.split("\n")[-1] if "\n" in c else c for c in retrieved_chunks if c.strip()]

            # Get scores from retrieved results (stored in agent's retriever)
            from src.retrieval import Retriever
            temp_retriever = Retriever(top_k=top_k)
            temp_results = temp_retriever.retrieve(question)
            retrieval_scores = [r.score for r in temp_results]

        with st.spinner("Analyzing uncertainty..."):
            # Calculate uncertainty signals
            signals = uncertainty.quantify_uncertainty(
                question=question,
                answer=response.answer,
                context_chunks=chunk_texts[:top_k],
                retrieval_scores=retrieval_scores[:top_k],
                num_chunks=len(temp_results)
            )

        # Display results
        st.divider()

        # Answer section
        st.header("Answer")

        if signals.should_answer:
            if signals.is_confident:
                st.success(response.answer)
            else:
                st.warning(response.answer + "\n\n⚠️ **Low confidence** - Please verify this information")
        else:
            st.error("I don't have sufficient information to answer this question confidently.")
            st.info("The retrieved documents don't contain relevant information about your query.")

        # Confidence metrics
        st.divider()
        st.header("Confidence Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Overall Confidence",
                f"{signals.overall_confidence:.0%}",
                help="Combined confidence score"
            )
            st.markdown(format_confidence_bar(signals.overall_confidence))

        with col2:
            st.metric(
                "Retrieval Quality",
                f"{signals.retrieval_confidence:.0%}",
                help="Relevance of retrieved chunks"
            )
            st.progress(signals.retrieval_confidence, text="")

        with col3:
            st.metric(
                "Answer Consistency",
                f"{signals.answer_consistency:.0%}",
                help="Alignment between answer and context"
            )
            st.progress(signals.answer_consistency, text="")

        # Detailed breakdown
        with st.expander("View detailed uncertainty breakdown"):
            st.subheader("Signal Explanations")

            st.markdown(f"""
            **1. Retrieval Quality ({signals.retrieval_confidence:.1%})**
            > {signals.retrieval_explanation}

            **2. Answer Consistency ({signals.answer_consistency:.1%})**
            > {signals.consistency_explanation}

            **3. Self-Evaluation ({signals.self_evaluation:.1%})**
            > {signals.self_eval_explanation}
            """)

            st.divider()

            st.markdown(f"""
            **Decision Logic:**
            - Should Answer: {'✅ Yes' if signals.should_answer else '❌ No'}
            - High Confidence: {'✅ Yes' if signals.is_confident else '❌ No'}
            - Threshold: {signals.overall_confidence:.1%} {'≥' if signals.is_confident else '<'} 60%
            """)

        # Sources
        st.divider()
        st.header("Sources")

        if response.sources:
            for source in response.sources:
                st.markdown(f"📄 {source}")
        else:
            st.markdown("*No sources retrieved*")

        # Retrieved context
        if show_context and response.retrieved_context:
            st.divider()
            st.header("Retrieved Context")
            st.markdown("*Chunks used to generate the answer:*")
            st.text_area("Context", response.retrieved_context, height=300)

    # Footer
    st.divider()
    st.caption("Built with LangChain, Qdrant, sentence-transformers, and Streamlit")


if __name__ == "__main__":
    main()
