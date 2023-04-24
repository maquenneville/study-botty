# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 23:21:38 2023

@author: marca
"""


import docx
import openpyxl
import pandas as pd
import chardet
import io
from pdfminer.high_level import extract_text
import os
import tiktoken
import os
from openai_pinecone_tools import generate_response


encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


def read_pdf(file_path):
    try:
        text = extract_text(file_path)
    except Exception as e:
        print(f"Error reading PDF file {file_path}: {str(e)}")
        text = ""
    return text

def read_docx(file_path):
    try:
        doc = docx.Document(file_path)
        text = " ".join([p.text for p in doc.paragraphs])
    except Exception as e:
        print(f"Error reading DOCX file {file_path}: {str(e)}")
        text = ""
    return text



def read_xlsx_file(file_path):
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error reading XLSX file {file_path}: {str(e)}")
        df = None
        
    file_name = os.path.basename(file_path)
    return file_name, df

def read_csv_file(file_path):
    try:
        with open(file_path, "rb") as f:
            encoding = chardet.detect(f.read())["encoding"]
        df = pd.read_csv(file_path, encoding=encoding)
    except Exception as e:
        print(f"Error reading CSV file {file_path}: {str(e)}")
        df = None
    
    file_name = os.path.basename(file_path)
    return file_name, df


def read_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading TXT file {file_path}: {str(e)}")
        text = ""
    return text



def csv_id_agent(context):
    
    if len(context) > 2000:
    
        context = context[:2000]
    
    messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "You are my CSV Indentification Assistant. Your job is to take a block of text provided below, and decide if the text represents all or part of a comma separated value text.  You must decide, and must answer using either 'yes' or 'no'."},
            {"role": "user", "content": f"Text to be identified: {context}"}
        ]
    
    response = generate_response(
        messages, temperature=0.0, n=1, max_tokens=10, frequency_penalty=0
    )
    is_csv = None
    
    if "yes" in response.lower():
        is_csv = True
    elif "no" in response.lower():
        is_csv = False
        
    else:
        print("I can't tell is this is a CSV, I'm sorry!")
        return
    
    return is_csv




def process_table_file(file_path):
    
    def count_tokens(text):
        
        tokens = len(encoding.encode(text))
        return tokens

    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.csv':
        file_name, df = read_csv_file(file_path)
    elif file_ext == '.xlsx':
        file_name, df = read_xlsx_file(file_path)
    else:
        raise ValueError("Unsupported file extension. Please provide a CSV or XLSX file.")
    
    if df is None:
        return None
    
    plain_csv_text = df.to_csv(index=False)
    tokens = count_tokens(plain_csv_text)

    if tokens > 3500:
        token_list = encoding.encode(plain_csv_text)
    
        accumulated_tokens = 0
        split_index = 0
        last_newline_index = 0
    
        newline_token = encoding.encode("\n")[0]
    
        for idx, token in enumerate(token_list):
            accumulated_tokens += 1
    
            if token == newline_token:
                last_newline_index = idx
    
            if accumulated_tokens > 3500:
                split_index = last_newline_index
                break
    
        truncated_text = encoding.decode(token_list[:split_index])


    else:
        truncated_text = plain_csv_text

    output_text = f"{file_name}\n{truncated_text}"
    return output_text


def ingester(file_path):
    extension = file_path.split(".")[-1].lower()
    if extension == "pdf":
        return read_pdf(file_path)
    elif extension in ["doc", "docx"]:
        return read_docx(file_path)
    elif extension in ["csv", "xlsx"]:
        return process_table_file(file_path)
    elif extension == "txt":
        return read_txt(file_path)
    else:
        print(f"Unsupported file type: {extension}")
        return ""
    


def chunk_text(text, chunk_size=1000):
    words = text.split()
    chunks = []
    current_chunk = ""

    for word in words:
        
        # Check if adding the word to the current chunk would exceed the chunk size
        if len(current_chunk) + len(word) + 1 > chunk_size:
            # If so, add the current chunk to the chunks list and start a new chunk with the current word
            chunks.append(current_chunk.strip())
            current_chunk = word
        else:
            # Otherwise, add the word to the current chunk
            current_chunk += f" {word}"

    # Add the last chunk to the chunks list
    if current_chunk:
        chunks.append(current_chunk.strip())
    

    return chunks

def ingest_folder(folder_path, progress=True):
    context_chunks = []

    # List all files in the folder
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    total_files = len(file_paths)

    for i, file_path in enumerate(file_paths):
        if progress:
            print(f"Processing file {i + 1}/{total_files}: {file_path}")

        text = ingester(file_path)

        if csv_id_agent(text):
            context_chunks.append(text)
            
        else:
            chunks = chunk_text(text)
            context_chunks.extend(chunks)

    return context_chunks

#print(ingest_folder(r"C:\Users\marca\Desktop\Coding\ChatGPT\embeddings\DocBot\test_input"))