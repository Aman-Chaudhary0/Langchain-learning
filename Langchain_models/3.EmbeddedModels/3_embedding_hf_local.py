from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "This is a test sentence for generating embeddings."
vector_embeddings = embeddings.embed_query(text)

print(str(vector_embeddings))