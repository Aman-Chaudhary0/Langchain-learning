from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",
    task="text-generation",
)

prompt = PromptTemplate(
    template = 'Generate 5 interesting facts about {topic}.',
    input_variables = ['topic'],
)

output_parser = StrOutputParser()

model = ChatHuggingFace(llm=llm)

chain = prompt | model | output_parser

result =  chain.invoke({'topic': 'space exploration'})

print(result)

chain.get_graph().print_ascii()