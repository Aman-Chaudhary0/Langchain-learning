# %%
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# ==================================================
# 1. Create Documents
# ==================================================
# %%
doc1 = Document(
    page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
    metadata={"team": "Royal Challengers Bangalore"}
)

doc2 = Document(
    page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
    metadata={"team": "Mumbai Indians"}
)

doc3 = Document(
    page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
    metadata={"team": "Chennai Super Kings"}
)

doc4 = Document(
    page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
    metadata={"team": "Mumbai Indians"}
)

doc5 = Document(
    page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
    metadata={"team": "Chennai Super Kings"}
)

docs = [doc1, doc2, doc3, doc4, doc5]


# ==================================================
# 2. Create Local Hugging Face Embeddings
# ==================================================
# %%
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ==================================================
# 3. Create Chroma Vector Store
# ==================================================
# %%
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="my_chroma_db",
    collection_name="sample"
)

print("Chroma vector store created.")


# ==================================================
# 4. Add Documents
# ==================================================
# %%
ids = vector_store.add_documents(docs)

print("\nDocuments added successfully.")

for id in ids:
    print(id)


# ==================================================
# 5. View Documents
# ==================================================
# %%
data = vector_store.get(
    include=["documents", "metadatas"]
)

print("\nStored Documents:")

for i in range(len(data["documents"])):
    print("\nID:", data["ids"][i])
    print("Document:", data["documents"][i])
    print("Metadata:", data["metadatas"][i])


# ==================================================
# 6. Similarity Search
# ==================================================
# %%
results = vector_store.similarity_search(
    query="Who among these are a bowler?",
    k=2
)

print("\nSimilarity Search Results:")

for doc in results:
    print("\nDocument:", doc.page_content)
    print("Metadata:", doc.metadata)


# ==================================================
# 7. Similarity Search With Score
# ==================================================
# %%
results = vector_store.similarity_search_with_score(
    query="Who among these are a bowler?",
    k=2
)

print("\nSimilarity Search With Score:")

for doc, score in results:
    print("\nDocument:", doc.page_content)
    print("Metadata:", doc.metadata)
    print("Score:", score)


# ==================================================
# 8. Metadata Filtering
# ==================================================
# %%
results = vector_store.similarity_search(
    query="Who is a player?",
    k=10,
    filter={"team": "Chennai Super Kings"}
)

print("\nChennai Super Kings Players:")

for doc in results:
    print("\n", doc.page_content)


# ==================================================
# 9. Update Document
# ==================================================
# %%
updated_doc = Document(
    page_content="Virat Kohli, the former captain of Royal Challengers Bangalore, is renowned for his aggressive leadership and consistent batting performances. His passion, fitness, and ability to chase targets have made him one of the most dependable players in T20 cricket.",
    metadata={"team": "Royal Challengers Bangalore"}
)

# Get an existing document ID
data = vector_store.get()

document_id = data["ids"][0]

vector_store.update_document(
    document_id=document_id,
    document=updated_doc
)

print("\nDocument updated successfully.")


# ==================================================
# 10. Verify Update
# ==================================================
# %%
data = vector_store.get(
    include=["documents", "metadatas"]
)

print("\nDocuments After Update:")

for i in range(len(data["documents"])):
    print("\nID:", data["ids"][i])
    print("Document:", data["documents"][i])


# ==================================================
# 11. Delete Document
# ==================================================
# %%
vector_store.delete(
    ids=[document_id]
)

print("\nDocument deleted successfully.")


# ==================================================
# 12. Verify Deletion
# ==================================================
# %%
data = vector_store.get(
    include=["documents", "metadatas"]
)

print("\nRemaining Documents:")

for i in range(len(data["documents"])):
    print("\nID:", data["ids"][i])
    print("Document:", data["documents"][i])
# %%
