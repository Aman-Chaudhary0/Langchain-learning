from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100,
        return_full_text=False
    )
)

model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content="You are a helpful AI assistant"),
    HumanMessage(content="What is the capital of France?"),

]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))

print(messages)