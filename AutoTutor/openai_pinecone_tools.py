# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 18:31:41 2023

@author: marca
"""



import tiktoken
import configparser
import openai
from openai.error import RateLimitError, InvalidRequestError, APIError
import pinecone
from pinecone import PineconeProtocolError
import time
from tqdm import tqdm
import pandas as pd




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



def get_embedding(text: str, model: str=EMBEDDING_MODEL):
    while True:
        try:
            result = openai.Embedding.create(
              model=model,
              input=text
            )
            break
        except (APIError, RateLimitError):
            print("OpenAI had an issue, trying again in a few seconds...")
            time.sleep(10)
    return result["data"][0]["embedding"]



def create_embeddings_dataframe(context_chunks):


    # Calculate embeddings for each chunk with a progress bar
    embeddings = []
    for chunk in tqdm(context_chunks, desc="Calculating embeddings"):
        embedding = get_embedding(chunk)
        embeddings.append(embedding)

    # Create the DataFrame with index and chunk columns
    df = pd.DataFrame({"index": range(len(context_chunks)), "chunk": context_chunks})

    # Add the embeddings to the DataFrame in separate columns with the naming convention "embedding{num}"
    embeddings_df = pd.DataFrame(embeddings, columns=[f"embedding{i}" for i in range(1536)])

    # Concatenate the main DataFrame with the embeddings DataFrame
    result_df = pd.concat([df, embeddings_df], axis=1)

    return result_df


def store_embeddings_in_pinecone(namespace=PINECONE_NAMESPACE, index=PINECONE_INDEX, pinecone_api_key=PINECONE_API_KEY, pinecone_env=PINECONE_ENV, dataframe=None):
    # Initialize Pinecone
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

    # Instantiate Pinecone's Index
    pinecone_index = pinecone.Index(index_name=index)

    if dataframe is not None and not dataframe.empty:
        batch_size = 80
        vectors_to_upsert = []
        batch_count = 0

        # Calculate the total number of batches
        total_batches = -(-len(dataframe) // batch_size)

        # Create a tqdm progress bar object
        progress_bar = tqdm(total=total_batches, desc="Loading info into Pinecone")

        for index, row in dataframe.iterrows():
            context_chunk = row["chunk"]
            
            vector = [float(row[f"embedding{i}"]) for i in range(1536)]
            
            pine_index = f"hw_{index}"
            metadata = {"context": context_chunk}
            vectors_to_upsert.append((pine_index, vector, metadata))

            # Upsert when the batch is full or it's the last row
            if len(vectors_to_upsert) == batch_size or index == len(dataframe) - 1:
                while True:
                     
                    try:
                        upsert_response = pinecone_index.upsert(
                            vectors=vectors_to_upsert,
                            namespace=namespace
                        )

                        batch_count += 1
                        vectors_to_upsert = []

                        # Update the progress bar
                        progress_bar.update(1)
                        break

                    except pinecone.core.client.exceptions.ApiException:
                        print("Pinecone is a little overwhelmed, trying again in a few seconds...")
                        time.sleep(10)

        # Close the progress bar after completing all upserts
        progress_bar.close()

    else:
        print("No dataframe to retrieve embeddings")
        
        
        

def fetch_context_from_pinecone(query, top_n=2, index=PINECONE_INDEX, namespace=PINECONE_NAMESPACE, pinecone_api_key=PINECONE_API_KEY, pinecone_env=PINECONE_ENV):
    # Initialize Pinecone
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

    # Generate the query embedding
    query_embedding = get_embedding(query)

    # Query Pinecone for the most similar embeddings
    pinecone_index = pinecone.Index(index_name=index)
    
    
    while True:
        try:
            query_response = pinecone_index.query(
                namespace=namespace,
                top_k=top_n,
                include_values=False,
                include_metadata=True,
                vector=query_embedding
                
            )
            break
        
        except PineconeProtocolError:
            print("Pinecone needs a moment....")
            time.sleep(3)
            continue
    
    # Retrieve metadata for the relevant embeddings
    context_chunks = [match['metadata']['context'] for match in query_response['matches']]
    
    context = "\n".join(context_chunks)

    return context




def generate_response(
    messages, model_engine="gpt-3.5-turbo", temperature=0.5, n=1, max_tokens=4000, frequency_penalty=0
):

    

    # Calculate the number of tokens in the messages
    tokens_used = sum([count_tokens(msg["content"]) for msg in messages])
    tokens_available = 4096 - tokens_used

    # Adjust max_tokens to not exceed the available tokens
    max_tokens = min(max_tokens, (tokens_available - 100))

    # Reduce max_tokens further if the total tokens exceed the model limit
    if tokens_used + max_tokens > 4096:
        max_tokens = 4096 - tokens_used - 10

    if max_tokens < 1:
        max_tokens = 1

    # Generate a response
    max_retries = 10
    retries = 0
    while True:
        if retries < max_retries:
            try:
                completion = openai.ChatCompletion.create(
                    model=model_engine,
                    messages=messages,
                    n=n,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                )
                break
            except (RateLimitError, KeyboardInterrupt):
                time.sleep(60)
                retries += 1
                print("Server overloaded, retrying in a minute")
                continue
        else:
            print("Failed to generate prompt after max retries")
            return
    response = completion.choices[0].message.content
    return response