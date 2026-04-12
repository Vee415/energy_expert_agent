"""Test script for retrieval module with multiple queries."""

import logging
from src.retrieval import Retriever

logging.basicConfig(level=logging.WARNING)


def test_multiple_queries():
    """Test retrieval with multiple diverse queries."""

    # Initialize retriever
    r = Retriever(top_k=10, rerank_top_k=5)

    # Test queries covering different aspects
    test_queries = [
        'What are the main findings in the balancing report?',
        'energy market trends in 2024',
        'regional coordination mechanisms',
        'implementation monitoring report 2021',
        'market report 2023 electricity prices',
        'balancing energy activation',
        'What is mFRR and aFRR?',
        'European electricity market coupling',
        'transmission system operators cooperation',
        'renewable energy integration challenges'
    ]

    print('=' * 70)
    print('RETRIEVAL TEST - Multiple Queries')
    print('=' * 70)

    for i, query in enumerate(test_queries, 1):
        print(f'\nQuery {i}/{len(test_queries)}: {query}')
        print('-' * 70)

        results = r.retrieve(query)

        print(f'Retrieved {len(results)} results:\n')
        for j, res in enumerate(results, 1):
            # Clean text for console output
            text_clean = res.text[:150].replace('\n', ' ').strip()
            text_clean = text_clean.encode('ascii', 'ignore').decode('ascii')

            print(f'  {j}. Score: {res.score:.3f} | {res.source} | Chunk {res.chunk_id}')
            print(f'     "{text_clean}..."')

        print()

    print('=' * 70)
    print('All queries completed!')
    print('=' * 70)


if __name__ == "__main__":
    test_multiple_queries()
