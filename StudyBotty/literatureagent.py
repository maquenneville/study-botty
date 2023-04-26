# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 21:58:24 2023

@author: marca
"""


from openai_pinecone_tools import generate_response
import openai
import configparser

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


def literature_agent(query, context):
    # Generate ChatGPT messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "You are my English Literature Teacher's Assistant.  Provided a query and relevant context, your job is to provide an answer to the query.  The answers should be nuanced and well-articulated, using the context and your own extensive knowledge of English Literature.  Assume the one asking the question has a grad-school level understanding of English Literature."},
        ]
        
    for c in context:
        messages.append({"role": "user", "content": f"Context: {c}"})
        
    messages.append({"role": "user", "content": f"Query:\n{query}"})
        

    
    # Use ChatGPT to generate a Wolfram Alpha natural language query
    answer = generate_response(messages, temperature=0.4)
    
    return answer