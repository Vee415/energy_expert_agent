"""Test retrieval with out-of-domain query about Tesla CEO."""

import logging
from src.retrieval import Retriever

logging.basicConfig(level=logging.WARNING)


def main():
    r = Retriever(top_k=10, rerank_top_k=5)

    query = 'who is ceo of tesla'
    print('=' * 60)
    print(f'Query: "{query}"')
    print('=' * 60)
    print()

    results = r.retrieve(query)
    print(f'Retrieved {len(results)} results:\n')

    for i, res in enumerate(results, 1):
        # Clean text for console output
        text_clean = res.text[:200].replace('\n', ' ').strip()
        text_clean = text_clean.encode('ascii', 'ignore').decode('ascii')

        print(f'{i}. Score: {res.score:.3f}')
        print(f'   Source: {res.source}')
        print(f'   Text: "{text_clean}..."')
        print()

    print('=' * 60)
    print('IMPORTANT OBSERVATION:')
    print('=' * 60)
    print()
    print('The query "who is ceo of tesla" is OUT-OF-DOMAIN.')
    print('Your ENTSO-E energy reports do NOT contain information about Tesla.')
    print()
    print('What you see above is the system returning IRRELEVANT results')
    print('because it has nothing relevant to match.')
    print()
    print('Notice the NEGATIVE scores (-10.0) - this indicates low relevance!')
    print()
    print('This demonstrates why we need UNCERNTAINTY QUANTIFICATION')
    print('in the next module (src/uncertainty.py).')
    print('=' * 60)


if __name__ == "__main__":
    main()
