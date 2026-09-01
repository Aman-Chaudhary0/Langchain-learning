#%%
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace
)

from dotenv import load_dotenv


#%%
load_dotenv()


# ==========================================================
# 1. GET YOUTUBE TRANSCRIPT
# ==========================================================

video_id = "Gfr50f6ZBvo"

try:
    api = YouTubeTranscriptApi()

    transcript_list = api.fetch(
        video_id,
        languages=["en"]
    )

    transcript = " ".join(
        chunk.text for chunk in transcript_list
    )

    print("Transcript loaded successfully.")
    print("Transcript length:", len(transcript))

except TranscriptsDisabled:
    print("No captions available for this video.")
    transcript = ""


#%%
# ==========================================================
# 2. SPLIT TRANSCRIPT INTO CHUNKS
# ==========================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.create_documents([transcript])

print("Number of chunks:", len(chunks))


#%%
# ==========================================================
# 3. CREATE EMBEDDINGS
# ==========================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


#%%
# ==========================================================
# 4. CREATE FAISS VECTOR STORE
# ==========================================================

vector_store = FAISS.from_documents(
    chunks,
    embedding_model
)


#%%
# ==========================================================
# 5. CREATE RETRIEVER
# ==========================================================

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 4
    }
)


#%%
# ==========================================================
# 6. CREATE LLM
# ==========================================================

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",
    task="text-generation",
)

model = ChatHuggingFace(
    llm=llm
)


#%%
# ==========================================================
# 7. CREATE PROMPT
# ==========================================================

prompt = PromptTemplate(
    template="""
You are a helpful assistant.

Answer the question ONLY using the provided YouTube transcript context.

If the answer cannot be found in the context, say:
"I don't know based on the provided video transcript."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=[
        "context",
        "question"
    ]
)


#%%
# ==========================================================
# 8. FORMAT RETRIEVED DOCUMENTS
# ==========================================================

def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )


#%%
# ==========================================================
# 9. CREATE RAG CHAIN
# ==========================================================

parallel_chain = RunnableParallel(
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
)


#%%
rag_chain = (
    parallel_chain
    | prompt
    | model
    | StrOutputParser()
)


#%%
# ==========================================================
# 10. ASK QUESTION
# ==========================================================

question = """
Is the topic of nuclear fusion discussed in this video?
If yes, explain what was discussed.
"""

answer = rag_chain.invoke(question)

print(answer)
# %%
