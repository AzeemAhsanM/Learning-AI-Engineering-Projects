
import os

from dotenv import load_dotenv
from google import genai
import chromadb

# 1. Load the API key and create the Gemini client.
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY was not found in .env")

gemini = genai.Client(api_key=api_key)

EMBEDDING_MODEL = "gemini-embedding-2"

# 2. Connect to the existing persistent ChromaDB database.
chroma = chromadb.PersistentClient(path="./chroma_db")

collection = chroma.get_collection(
    name="company_knowledge_gemini",
    embedding_function=None,
)

# 3. Define questions and their expected relevant chunks.
test_cases = [
    {
        "question": "What software does Acme develop?",
        "expected": "Acme Technologies develops financial management software",
    },
    {
        "question": "When was Acme founded?",
        "expected": "Acme Technologies was founded in 2020.",
    },
    {
        "question": "How many employees does Acme have?",
        "expected": "The company has 25 employees.",
    },
    {
        "question": "Who is the CEO of Acme Technologies?",
        "expected": None,
    },
]

K = 3
recall_scores = []
reciprocal_ranks = []

# 4. Evaluate each test question.
for test in test_cases:
    question = test["question"]
    expected = test["expected"]

    # Generate the question embedding.
    response = gemini.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question,
    )

    query_vector = response.embeddings[0].values

    # Retrieve the top K chunks.
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(K, collection.count()),
        include=["documents", "distances"],
    )

    documents = results["documents"][0]
    distances = results["distances"][0]

    print("\nQuestion:", question)

    if expected is None:
        print("Expected result: No answer-bearing chunk")
        print("Nearest distance:", round(distances[0], 4))
        print("Retrieved chunks still need relevance review.")
        continue

    # Find the rank of the expected chunk.
    relevant_ranks = [
        rank
        for rank, document in enumerate(documents, start=1)
        if expected.lower() in document.lower()
    ]

    # Calculate Recall@K for this one expected chunk.
    recall = 1.0 if relevant_ranks else 0.0
    recall_scores.append(recall)

    # Calculate reciprocal rank.
    reciprocal_rank = (
        1.0 / relevant_ranks[0]
        if relevant_ranks
        else 0.0
    )
    reciprocal_ranks.append(reciprocal_rank)

    print("Expected chunk found:", bool(relevant_ranks))
    print("Recall@K:", recall)
    print("Reciprocal rank:", round(reciprocal_rank, 3))

    print("Retrieved results:")
    for rank, (document, distance) in enumerate(
        zip(documents, distances), start=1
    ):
        print(f"  {rank}. distance={distance:.4f} | {document}")

# 5. Calculate aggregate metrics for answerable questions.
if recall_scores:
    mean_recall = sum(recall_scores) / len(recall_scores)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    print("\n===== Evaluation summary =====")
    print(f"Recall@{K}: {mean_recall:.3f}")
    print(f"MRR: {mrr:.3f}")
    print("Answerable questions:", len(recall_scores))
    print("Unanswerable questions:", sum(
        test["expected"] is None for test in test_cases
    ))
