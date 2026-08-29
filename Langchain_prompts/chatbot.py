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

chat_history = [
    SystemMessage(content="You are a helpful AI assistant")
]

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    chat_history.append(
        HumanMessage(content=user_input)
    )

    result = model.invoke(chat_history)

    chat_history.append(
        AIMessage(content=result.content)
    )

    print("AI:", result.content)

print(chat_history)