from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)


@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return a * b


# Bind tool
llm_tool = model.bind_tools([multiply_numbers])

# User message
messages = [
    HumanMessage(content="Multiply 5 and 6")
]

# 1. LLM decides to call the tool
response = llm_tool.invoke(messages)

# print("AI response:", response)
# print("Tool calls:", response.tool_calls)

# Add AI response to messages
messages.append(response)

# 2. Get tool call
tool_call = response.tool_calls[0]

# 3. Execute tool
result = multiply_numbers.invoke(tool_call["args"])

# print("Tool result:", result)

# 4. Add tool result as ToolMessage
messages.append(
    ToolMessage(
        content=str(result),
        tool_call_id=tool_call["id"]
    )
)

# 5. Send conversation back to LLM
final_response = llm_tool.invoke(messages)

print("Final Response:", final_response.content)