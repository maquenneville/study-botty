# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 20:18:30 2023

@author: marca
"""


import os
import tiktoken
import configparser
import openai
from openai_pinecone_tools import generate_response




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


    return openai_api_key, pinecone_api_key, pinecone_env, index, namespace

openai_api_key, pinecone_api_key, pinecone_env, index, namespace = get_api_keys('config.ini')

openai.api_key = openai_api_key


CHAT_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"
PINECONE_INDEX = index
PINECONE_NAMESPACE = namespace
PINECONE_API_KEY = pinecone_api_key
PINECONE_ENV = pinecone_env



def answer_decision_agent(query, context, answer):
    
    messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Your job is to take a query and context, and an attempted answer to the query.  You then decide if the answer to the query is a satisfactory answer.  You must decide, and must answer using either 'yes' or 'no'."},
            {"role": "user", "content": f"Query: {query}"},
            {"role": "user", "content": f"Context: {context}"},
            {"role": "user", "content": f"Answer: {answer}"}
            
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
    


def construct_prompt(
    question: str,
    context: str,
    separator: str = "\n*"
    
):

    header = (
        """Answer the question as truthfully as possible using the provided context. If the answer is not contained within the text below, attempt to use the context and your knowledge to give an answer.  If the context cannot help you find an answer, say "I don't know."\n\nContext:\n"""
    )

    context = header + context

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": context},
        {"role": "user", "content": f"Q: {question}\nA:"},
    ]

    return messages



def doc_agent(
    query: str,
    context: str,
    show_prompt=False
):
    messages = construct_prompt(query, context)

    if show_prompt:
        print(messages)

    response = generate_response(messages, temperature=0.5, n=1, max_tokens=1000, frequency_penalty=0)
    return response.strip(" \n")


#print(fetch_context_from_pinecone("What are the columns in the Raw Mussel data 2017 (1) table?"))
