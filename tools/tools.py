from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain import PromptTemplate, LLMChain

from transformers import pipeline

import sys
sys.path.insert(0, "../")
from ice_breaker import comment

from dotenv import find_dotenv, load_dotenv

"""
For models hosted on Hugging Face Hub 

Documentation: 
https://python.langchain.com/docs/integrations/providers/huggingface
"""

huggingface_token = "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"


def get_huggingface_output(comment: str): 
    """
    Retrieves the output from the hugging face hate speech model on whether the comment contains hate speech
    """

    # search for linkedin profile page
    # search.run(text) using serpapi

    try:
        load_dotenv(find_dotenv())

    except: 
        print("Failed to connect API ")

    repo_id = "facebook/roberta-hate-speech-dynabench-r4-target"

    huggingface_model = pipeline("sentiment-analysis", model = repo_id)
    output = huggingface_model(comment)
    
    return output

comment = 'Not sure what to think about asians'
get_huggingface_output(comment)