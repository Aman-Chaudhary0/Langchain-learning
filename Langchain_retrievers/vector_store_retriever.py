#%%
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


#%%
# Step 1: Your source documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]


#%%
# Step 2: Initialize local Hugging Face embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


#%%
# Step 3: Create Chroma vector store in memory
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="my_collection"
)


#%%
# Step 4: Convert vectorstore into a retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)


#%%
# Step 5: Search using retriever
query = "What is Chroma used for?"

results = retriever.invoke(query)


#%%
# Step 6: Display results
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)


#%%
# Step 7: Direct similarity search
results = vectorstore.similarity_search(
    query,
    k=2
)


#%%
# Step 8: Display similarity search results
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
# %%
