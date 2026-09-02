from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool


class MultiplyInput(BaseModel):
    a: int = Field(
        description="The first number to multiply"
    )

    b: int = Field(
        description="The second number to multiply"
    )


def multiply_numbers(a: int, b: int) -> int:
    return a * b


multiply_tool = StructuredTool.from_function(
    func=multiply_numbers,
    name="multiply_numbers",
    description="Multiplies two numbers.",
    args_schema=MultiplyInput
)


print(multiply_tool.invoke({"a": 3, "b": 4}))
print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args)