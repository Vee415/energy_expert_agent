"""Test the Document Agent with a sample question."""

import logging
import sys

logging.basicConfig(level=logging.INFO)

def main():
    print("=" * 60)
    print("Document Agent Test")
    print("=" * 60)
    print("\nMake sure Ollama is running!")
    print("If not, start it with: ollama serve")
    print()

    try:
        from src.agent import DocumentAgent

        print("Initializing agent...")
        agent = DocumentAgent(model_name="gemma3:4b")

        # Test questions
        test_questions = [
            "What is the main purpose of the ENTSO-E balancing reports?",
            "What are mFRR and aFRR?",
        ]

        for question in test_questions:
            print(f"\n{'=' * 60}")
            print(f"Question: {question}")
            print(f"{'=' * 60}\n")

            response = agent.ask(question)

            print("ANSWER:")
            print("-" * 60)
            print(response.answer)
            print("-" * 60)
            print(f"\nConfidence: {response.confidence:.2%}")
            print(f"Sources: {', '.join(response.sources) if response.sources else 'None'}")
            print(f"\nRetrieved context (first 500 chars):")
            print(response.retrieved_context[:500])

    except ImportError as e:
        print(f"Import error: {e}")
        print("\nYou may need to install additional packages:")
        print("  pip install langchain-ollama")
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Make sure you have the model: ollama pull gemma3:4b")
        print(f"3. Check if Ollama is accessible at http://localhost:11434")


if __name__ == "__main__":
    main()
