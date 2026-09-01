#%%
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace
)
from dotenv import load_dotenv


#%%
video_id = "Gfr50f6ZBvo"  # Only the ID, not the full URL

try:
    api = YouTubeTranscriptApi()

    # Fetch transcript
    transcript_list = api.fetch(
        video_id,
        languages=["en"]
    )

    # Convert transcript to plain text
    transcript = " ".join(chunk.text for chunk in transcript_list)

    print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")

#%%
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

#%%
len(chunks)

#%%
chunks[100]


#%%
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = FAISS.from_documents(chunks, embedding_model)

#%%
vector_store.index_to_docstore_id

#%%
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
print(retriever)

#%%
retriever.invoke("what is deepmind?")


#%%
## Augmentation with LLM
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)


#%%
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

#%%
question = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs    = retriever.invoke(question)
print(retrieved_docs)


#%%
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
print(context_text)

#%%
final_prompt = prompt.invoke({"context": context_text, "question": question})
print(final_prompt)



#%%
## Generate answer using LLM
answer = model.invoke(final_prompt)
print(answer.content)
# %%
