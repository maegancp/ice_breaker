import os
from dotenv import load_dotenv

# from langchain.llms import AzureOpenAI
from langchain.chat_models import ChatOpenAI
#  chat model - wrapper around a LLM

from langchain.prompts import PromptTemplate
# a wrapper class around a prompt, allows for different inputs using a combination of a template 

from langchain.chains import LLMChain
# combines multiple components together to create one single coherent application
# can build a more complex chain by combining multiple chains together 

if __name__ == '__main__':
    print("Hello LangChain")
    try:
        load_dotenv()

    except: 
        print("Failed to connect API ")

    comment = 'Not sure what to think about asians'

    summary_template = """
    Give the comment {comment} made, I want you to assume the role of law enforcement and answer the following: 
    1. Starting with "yes", "maybe" or "no", is this hate speech? 
    2. Why?
    """

    summary_prompt_template = PromptTemplate(input_variables = ["comment"], template = summary_template)

    llm = ChatOpenAI(
        temperature = 0.2, 
        model_name = "gpt-4", 
        model_kwargs = {"engine": "test"}
        )

    chain = LLMChain(llm = llm, prompt = summary_prompt_template)

    print(chain.run(comment = comment))