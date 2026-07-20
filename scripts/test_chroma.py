from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings


docs = [
    Document(page_content="RAG combines information retrieval with language generation."),
    Document(page_content="PubMed contains biomedical literature and abstracts."),
    Document(page_content="Chroma is a vector database for storing document embeddings."),
    Document(page_content="Medical question answering systems can use retrieved papers to improve factual accuracy."),
]

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory="./archive/experiments/indexes/chroma_test_db"
)

results = db.similarity_search("Which database stores document embeddings?", k=2)

print("Search results:")
for i, r in enumerate(results, 1):
    print(f"\n[{i}] {r.page_content}")
