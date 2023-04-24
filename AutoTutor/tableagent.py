# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 00:49:04 2023

@author: marca
"""

from ingester import read_xlsx_file, read_csv_file
from openai_pinecone_tools import generate_response
import pandas as pd
from nltk.tokenize import sent_tokenize
import tiktoken
import configparser
import openai
from openai.error import RateLimitError, InvalidRequestError, APIError
import pinecone
from pinecone import PineconeProtocolError


encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")



def count_tokens(text):
    tokens = len(encoding.encode(text))
    return tokens

def get_api_keys(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

    openai_api_key = config.get("API_KEYS", "OpenAI_API_KEY")
    pinecone_api_key = config.get("API_KEYS", "Pinecone_API_KEY")
    pinecone_env = config.get("API_KEYS", "Pinecone_ENV")
    index = config.get("API_KEYS", "Pinecone_Index")
    namespace = config.get("API_KEYS", "Namespace")
    google_namespace = config.get("API_KEYS", "Google_Namespace")

    return openai_api_key, pinecone_api_key, pinecone_env, index, namespace, google_namespace

openai_api_key, pinecone_api_key, pinecone_env, index, namespace, google_namespace = get_api_keys('config.ini')

openai.api_key = openai_api_key


CHAT_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"
PINECONE_INDEX = index
PINECONE_NAMESPACE = namespace
PINECONE_API_KEY = pinecone_api_key
PINECONE_ENV = pinecone_env

def table_decision_agent(query, context):
    
    messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Your job is to take csv text, and decide if ChatGPT can answer the query using the context.  You must decide, and must answer using either 'yes' or 'no'."},
            {"role": "user", "content": f"Query: {query}\n\nContext: {context}"},
        ]
    
    response = generate_response(
        messages, temperature=0.0, n=1, max_tokens=10, frequency_penalty=0
    )
    can_answer = None
    
    if "yes" in response.lower():
        can_answer = True
    elif "no" in response.lower():
        can_answer = False
        
    else:
        print("Got confused while deciding how to answer your question, I'm sorry!")
        return
    
    return can_answer


def table_agent(query, context):
    
    messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": """You are my Tabular Data Analyzer.  Using provided comma seperated value data, answer the question as truthfully as possible. If the answer is not contained within the text below, attempt to use the context and your knowledge to give an answer.  If the context cannot help you find an answer, say "I don't know." """},
            {"role": "user", "content": f"Comma seperated value text:\n{context}"},
            {"role": "user", "content": f"Q: {query}\nA: "}
        ]
    
    response = generate_response(
        messages, temperature=0.0, n=1, max_tokens=500, frequency_penalty=0
    )

    
    return response