# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 13:39:19 2023

@author: marca
"""

from openai_pinecone_tools import *
from docagent import *
from ingester import *
from searchagent import *
from mathagent import *
from tableagent import *
from headmasteragent import *
from science_agent import *
from tqdm.auto import tqdm
import threading
import time
import sys


        
class Spinner:
    def __init__(self, message="Thinking..."):
        self._message = message
        self._running = False
        self._spinner_thread = None

    def start(self):
        self._running = True
        self._spinner_thread = threading.Thread(target=self._spin)
        self._spinner_thread.start()

    def stop(self):
        self._running = False
        self._spinner_thread.join()

    def _spin(self):
        spinner_chars = "|/-\\"
        index = 0

        while self._running:
            sys.stdout.write(f"\r{self._message} {spinner_chars[index % len(spinner_chars)]}")
            sys.stdout.flush()
            time.sleep(0.1)
            index += 1

        # Clear the spinner line
        sys.stdout.write("\r" + " " * (len(self._message) + 2))
        sys.stdout.flush()



def main():
    print("\n\nWelcome to AutoTutor!")

    add_docs = input("\n\nWould you like to add a folder of documents? (y/n): ").lower()

    if add_docs == "y":
        folder_path = input("Enter the folder path: ")
        doc_chunks = ingest_folder(folder_path)
        doc_df = create_embeddings_dataframe(doc_chunks)
        store_embeddings_in_pinecone(dataframe=doc_df)

    print("\n\nOk, I'm ready for your questions!\n")

    while True:
        query = input("\nEnter your question or type 'exit' to quit: ")

        if query.lower() == "exit":
            break

        spinner = Spinner()

        # Start spinner with the default description
        spinner.start()

        # Retrieve the context from Pinecone
        context = fetch_context_from_pinecone(query)
        
        faculty = headmaster_agent(query, context)
        
        if faculty == "DocAgent":
            answer = doc_agent(query, context)
        elif faculty == "TableAgent":
            answer = table_agent(query, context)
        elif faculty == "MathAgent":
            answer = math_agent(query, context)
        elif faculty == "LiteratureAgent":
            answer = literature_agent(query, context)
        elif faculty == "ScienceAgent":
            answer = science_agent(query, context)
            
        # Check if ChatGPT answered the query with the given context
        did_answer = answer_decision_agent(query, context, answer)

        if not did_answer:

            # If ChatGPT cannot answer with the given context, use the google_search_agent
            context = google_search_agent(query)
            
            faculty = headmaster_agent(query, context)
            
            if faculty == "DocAgent":
                answer = doc_agent(query, context)
            elif faculty == "TableAgent":
                answer = table_agent(query, context)
            elif faculty == "MathAgent":
                answer = math_agent(query, context)
            elif faculty == "LiteratureAgent":
                answer = literature_agent(query, context)

        # Stop the spinner before printing the answer
        spinner.stop()
        print(f"\nAnswer: {answer}")






if __name__ == "__main__":
    main()