from langchain_core.tools import tool

#custom tools
@tool
def add(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return a * b

class MathToolKit:
    def get_tools(self):
        return [add, multiply]

toolkit = MathToolKit()
tools = toolkit.get_tools()

for tool in tools:
    print(tool.name, "=>", tool.description)