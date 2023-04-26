# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 18:01:28 2023

@author: marca
"""

from openai_pinecone_tools import generate_response
import configparser
import openai



def get_api_keys(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

    openai_api_key = config.get("API_KEYS", "OpenAI_API_KEY")
    pinecone_api_key = config.get("API_KEYS", "Pinecone_API_KEY")
    pinecone_env = config.get("API_KEYS", "Pinecone_ENV")
    index = config.get("API_KEYS", "Pinecone_Index")
    namespace = config.get("API_KEYS", "Namespace")

    return openai_api_key, pinecone_api_key, pinecone_env, index, namespace

openai_api_key, pinecone_api_key, pinecone_env, index, namespace = get_api_keys('config.ini')

openai.api_key = openai_api_key


CHAT_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"
PINECONE_INDEX = index
PINECONE_NAMESPACE = namespace
PINECONE_API_KEY = pinecone_api_key
PINECONE_ENV = pinecone_env
AGENT_LIST = """
Agent List:

DocAgent:  For basic questions over regular documentation, essays, syllabi, reading materials, and web page text/summaries.
TableAgent: For questions regarding provided comma seperated value text data.
MathAgent: For math-based or quantity-based questions.  It has access to Wolfram Alpha and all it's datasets.
LiteratureAgent: For questions regarding English Literature.  Best used for more subjective and/or nuanced questions around writing and literature, that require more in-depth answers.
"""

def headmaster_agent(query, context, agent_list=AGENT_LIST):
    # Generate ChatGPT messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "You are my Question Answering Delegator.  Your job is to take a query and relevant context for the query.  Then, from a list of question-anwering agents, you choose the one you feel is best suited for answering the question.  You must pick one agent and only one.  You must answer with just the agent's name."},
        ]
        
    for c in context:
        messages.append({"role": "user", "content": f"Context: {c}"})
        
    messages.extend([
            {"role": "user", "content": f"Question:\n{query}"},
            {"role": "user", "content": agent_list}
            ])
        

    
    # Use ChatGPT to generate a Wolfram Alpha natural language query
    selection = generate_response(messages, temperature=0.1, max_tokens=40)
    
    agent = None
    
    if "DocAgent" in selection:
        agent = "DocAgent"
    elif "TableAgent" in selection:
        agent = "TableAgent"
    elif "MathAgent" in selection:
        agent = "MathAgent"
    elif "LiteratureAgent" in selection:
        agent = "LiteratureAgent"
    else:
        print("Headmaster was unsure who to delegate to")
        
    return agent