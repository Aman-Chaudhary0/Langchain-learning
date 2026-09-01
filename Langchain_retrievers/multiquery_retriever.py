#%%
from langchain_community.vectorstores import FAISS
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace
)
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


#%%
# Relevant health & wellness documents
all_docs = [
    Document(
        page_content="Regular walking boosts heart health and can reduce symptoms of depression.",
        metadata={"source": "H1"}
    ),
    Document(
        page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.",
        metadata={"source": "H2"}
    ),
    Document(
        page_content="Deep sleep is crucial for cellular repair and emotional regulation.",
        metadata={"source": "H3"}
    ),
    Document(
        page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.",
        metadata={"source": "H4"}
    ),
    Document(
        page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.",
        metadata={"source": "H5"}
    ),
    Document(
        page_content="The solar energy system in modern homes helps balance electricity demand.",
        metadata={"source": "I1"}
    ),
    Document(
        page_content="Python balances readability with power, making it a popular system design language.",
        metadata={"source": "I2"}
    ),
    Document(
        page_content="Photosynthesis enables plants to produce energy by converting sunlight.",
        metadata={"source": "I3"}
    ),
    Document(
        page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.",
        metadata={"source": "I4"}
    ),
    Document(
        page_content="Black holes bend spacetime and store immense gravitational energy.",
        metadata={"source": "I5"}
    ),
]


#%%
# Step 1: Hugging Face embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


#%%
# Step 2: Create FAISS vector store
vectorstore = FAISS.from_documents(
    documents=all_docs,
    embedding=embedding_model
)


#%%
# Step 3: Normal similarity retriever
similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)


#%%
# Step 4: Hugging Face API LLM
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.2
)


#%%
# Step 5: Convert to Chat Model
chat_model = ChatHuggingFace(
    llm=llm
)


#%%
# Step 6: MultiQueryRetriever
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(
        search_kwargs={"k": 5}
    ),
    llm=chat_model
)


#%%
# Query
query = "How to improve energy levels and maintain balance?"


#%%
# Similarity search
similarity_results = similarity_retriever.invoke(query)


#%%
# Multi-query search
multiquery_results = multiquery_retriever.invoke(query)


#%%
# Similarity Results
print("\n" + "=" * 100)
print("SIMILARITY SEARCH RESULTS")
print("=" * 100)

for i, doc in enumerate(similarity_results):
    print(f"\n--- Result {i+1} ---")
    print("Source:", doc.metadata["source"])
    print(doc.page_content)


#%%
# Multi Query Results
print("\n" + "=" * 100)
print("MULTI QUERY RESULTS")
print("=" * 100)

for i, doc in enumerate(multiquery_results):
    print(f"\n--- Result {i+1} ---")
    print("Source:", doc.metadata["source"])
    print(doc.page_content)
# %%
