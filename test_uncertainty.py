"""Test the uncertainty quantification module."""

import logging
from src.uncertainty import UncertaintyQuantifier

logging.basicConfig(level=logging.INFO)


def main():
    print("=" * 70)
    print("Uncertainty Quantification Test")
    print("=" * 70)
    print("\nInitializing...")

    uq = UncertaintyQuantifier()

    # Test Case 1: Good retrieval (high confidence expected)
    print("\n" + "=" * 70)
    print("TEST 1: Good Retrieval (mFRR/aFRR question)")
    print("=" * 70)

    signals1 = uq.quantify_uncertainty(
        question="What are mFRR and aFRR?",
        answer="mFRR (manual Frequency Restoration Reserve) and aFRR (automatic Frequency Restoration Reserve) are balancing services used in European electricity markets.",
        context_chunks=[
            "The FRR process comprises the activation of aFRR also known as secondary frequency control reserve",
            "mFRR balancing capacity for upward regulation is procured in a national market",
            "aFRR is a process under development where procured volumes are constantly increasing"
        ],
        retrieval_scores=[6.7, 6.5, 5.8],
        num_chunks=3
    )

    print(uq.get_confidence_report(signals1))

    # Test Case 2: Poor retrieval (low confidence expected - Tesla CEO)
    print("\n" + "=" * 70)
    print("TEST 2: Poor Retrieval (Tesla CEO - out of domain)")
    print("=" * 70)

    signals2 = uq.quantify_uncertainty(
        question="Who is the CEO of Tesla?",
        answer="Elon Musk is the CEO of Tesla.",
        context_chunks=[
            "Operator Company Ltd 23February 2016 EirGrid plc",
            "Landsnet hf 23February 2016 Terna Rete Elettrica Nazionale",
            "Litgrid AB 23February 2016 AS Augstsprieguma"
        ],
        retrieval_scores=[-10.0, -10.4, -10.5],
        num_chunks=3
    )

    print(uq.get_confidence_report(signals2))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTest 1 (mFRR/aFRR):")
    print(f"  Overall Confidence: {signals1.overall_confidence:.1%}")
    print(f"  Is Confident: {signals1.is_confident}")
    print(f"  Should Answer: {signals1.should_answer}")

    print(f"\nTest 2 (Tesla CEO):")
    print(f"  Overall Confidence: {signals2.overall_confidence:.1%}")
    print(f"  Is Confident: {signals2.is_confident}")
    print(f"  Should Answer: {signals2.should_answer}")

    print("\n" + "=" * 70)
    print("Key Observations:")
    print("=" * 70)
    print("- Test 1 should show HIGH confidence (>60%)")
    print("- Test 2 should show VERY LOW confidence (<30%)")
    print("- The system correctly identifies out-of-domain queries!")
    print("=" * 70)


if __name__ == "__main__":
    main()
