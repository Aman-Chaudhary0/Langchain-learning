from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Hello, how are you?")
print(result.content)


