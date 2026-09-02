from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model="exaone3.5:7.8b")
template = """
한국어로 대답해줘.

Here is a question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    question = input(": ")
    if question == "q":
        break
    
    result = chain.invoke({"question": question})
    print(result)