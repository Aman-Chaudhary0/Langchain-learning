from langchain_core.tools import tool

# #Step1 - create a function

# def multiply_numbers(a,b):
#     """Multiplies two numbers."""
#     return a*b


# #step-2 - add type hints
# def multiply_numbers(a: int, b: int) -> int:
#     """Multiplies two numbers."""
#     return a*b


#step-3 - add tool decorator
@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return a*b


result = multiply_numbers.invoke({"a":3, "b":4})
print(result)

print(multiply_numbers.name)
print(multiply_numbers.description)
print(multiply_numbers.args)