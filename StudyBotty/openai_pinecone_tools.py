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
import pandas as pd
from tqdm.auto import tqdm
import asyncio
import tqdm.asyncio as async_tqdm
from pydub import AudioSegment
import os
import math
import tempfile
from elevenlabs import set_api_key
import sys


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
    google_api_key = config.get("API_KEYS", "Google_API_KEY")
    wolfram_api_key = config.get("API_KEYS", "Wolfram_API_KEY")
    google_id = config.get("API_KEYS", "Google_Search_ID")
    eleven_labs_api_key = config.get("API_KEYS", "Eleven_Labs_API_KEY")

    return (
        openai_api_key,
        pinecone_api_key,
        pinecone_env,
        index,
        namespace,
        google_namespace,
        google_api_key,
        wolfram_api_key,
        google_id,
        eleven_labs_api_key
    )


(
    openai_api_key,
    pinecone_api_key,
    pinecone_env,
    index,
    namespace,
    google_namespace,
    google_api_key,
    wolfram_api_key,
    google_id,
    eleven_labs_api_key
) = get_api_keys("config.ini")

openai.api_key = openai_api_key



SMART_CHAT_MODEL = "gpt-4"
FAST_CHAT_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"
PINECONE_INDEX = index
PINECONE_NAMESPACE = namespace
PINECONE_API_KEY = pinecone_api_key
PINECONE_ENV = pinecone_env
GOOGLE_NAMESPACE = google_namespace
GOOGLE_API_KEY = google_api_key
GOOGLE_ID = google_id
WOLFRAM_API_KEY = wolfram_api_key
ELEVENLABS_API_KEY = eleven_labs_api_key


def get_embedding(text: str, model: str = EMBEDDING_MODEL):
    while True:
        try:
            result = openai.Embedding.create(model=model, input=text)
            break
        except (APIError, RateLimitError):
            print("OpenAI had an issue, trying again in a few seconds...")
            time.sleep(10)
    return result["data"][0]["embedding"]



def create_embeddings_dataframe(context_chunks):
    # Calculate embeddings for each chunk with a progress bar
    embeddings = []
    progress_bar = tqdm(total=len(context_chunks), desc="Calculating embeddings", position=0)
    
    for chunk in context_chunks:
        embedding = get_embedding(chunk)
        embeddings.append(embedding)
        progress_bar.update(1)  # Increment the progress bar after each embedding calculation
        sys.stdout.flush()

    progress_bar.close()  # Close the progress bar when the loop is finished

    # Create the DataFrame with index and chunk columns
    df = pd.DataFrame({"index": range(len(context_chunks)), "chunk": context_chunks})

    # Add the embeddings to the DataFrame in separate columns with the naming convention "embedding{num}"
    embeddings_df = pd.DataFrame(
        embeddings, columns=[f"embedding{i}" for i in range(1536)]
    )

    # Concatenate the main DataFrame with the embeddings DataFrame
    result_df = pd.concat([df, embeddings_df], axis=1)

    return result_df





def store_embeddings_in_pinecone(
    namespace=PINECONE_NAMESPACE,
    index=PINECONE_INDEX,
    pinecone_api_key=PINECONE_API_KEY,
    pinecone_env=PINECONE_ENV,
    dataframe=None,
):
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
        progress_bar = tqdm(total=total_batches, desc="Loading info into Pinecone", position=0)

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
                            vectors=vectors_to_upsert, namespace=namespace
                        )

                        batch_count += 1
                        vectors_to_upsert = []

                        # Update the progress bar
                        progress_bar.update(1)
                        sys.stdout.flush()
                        break

                    except pinecone.core.client.exceptions.ApiException:
                        print(
                            "Pinecone is a little overwhelmed, trying again in a few seconds..."
                        )
                        time.sleep(10)

        # Close the progress bar after completing all upserts
        progress_bar.close()

    else:
        print("No dataframe to retrieve embeddings")


def fetch_context_from_pinecone(
    query,
    top_n=3,
    index=PINECONE_INDEX,
    namespace=PINECONE_NAMESPACE,
    pinecone_api_key=PINECONE_API_KEY,
    pinecone_env=PINECONE_ENV,
):
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
                vector=query_embedding,
            )
            break

        except PineconeProtocolError:
            print("Pinecone needs a moment....")
            time.sleep(3)
            continue

    # Retrieve metadata for the relevant embeddings
    context_chunks = [
        match["metadata"]["context"] for match in query_response["matches"]
    ]

    return context_chunks


def generate_response(
    messages,
    model=FAST_CHAT_MODEL,
    temperature=0.5,
    n=1,
    max_tokens=4000,
    frequency_penalty=0,
):
    token_ceiling = 4096
    if model == "gpt-4":
        max_tokens = 8000
        token_ceiling = 8000
    # Calculate the number of tokens in the messages
    tokens_used = sum([count_tokens(msg["content"]) for msg in messages])
    tokens_available = token_ceiling - tokens_used

    # Adjust max_tokens to not exceed the available tokens
    max_tokens = min(max_tokens, (tokens_available - 100))

    # Reduce max_tokens further if the total tokens exceed the model limit
    if tokens_used + max_tokens > token_ceiling:
        max_tokens = token_ceiling - tokens_used - 10

    if max_tokens < 1:
        max_tokens = 1

    # Generate a response
    max_retries = 10
    retries = 0
    while True:
        if retries < max_retries:
            try:
                completion = openai.ChatCompletion.create(
                    model=model,
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


def transcribe_using_whisper(audio_file):
    

    # Convert input audio to WAV format
    audio = AudioSegment.from_file(audio_file)
    audio = audio.set_channels(1)  # Ensure mono audio

    size = os.path.getsize(audio_file)
    max_chunk_size = 25000000

    full_trans = []


    if size > max_chunk_size:
        audio_length = len(audio)
        num_chunks = math.ceil(size / max_chunk_size)

        # Calculate the length of each chunk in milliseconds
        chunk_length_ms = math.ceil(audio_length / num_chunks)

        # Split the audio into chunks
        chunks = [audio[i*chunk_length_ms:(i+1)*chunk_length_ms] for i in range(num_chunks)]

        for chunk in chunks:
            # Create a temporary WAV file
            fd, tmp_file_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                # Export the chunk to the temporary file
                chunk.export(tmp_file_path, format="wav")

                with open(tmp_file_path, "rb") as tmp_file:
                    transcript = openai.Audio.transcribe("whisper-1", tmp_file)
                    full_trans.append(transcript.text)

            finally:
                # Remove the temporary WAV file
                os.remove(tmp_file_path)

        full_trans = "".join(full_trans)

    else:
        with open(audio_file, "rb") as audio_file:
            full_trans = openai.Audio.transcribe("whisper-1", audio_file)
            full_trans = full_trans.text

    
    return full_trans
