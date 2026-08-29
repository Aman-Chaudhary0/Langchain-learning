from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv


load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id='Qwen/Qwen2.5-0.5B-Instruct',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)

model = ChatHuggingFace(llm=llm)

# detailed way
template2 = PromptTemplate(
    template='Greet this person in 5 languages. The name of the person is {name}',
    input_variables=['name']
)

# fill the values of the placeholders
prompt = template2.invoke({'name':'nitish'})

result = model.invoke(prompt)

print(result.content)
