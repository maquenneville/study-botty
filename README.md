# AutoTutor

AutoTutor is an AI-powered question-answering system that helps users find answers to their questions using a combination of pre-loaded documents, tables, and online resources.

# Features
- Ingests documents from various file formats (PDF, DOCX, CSV, XLSX, and TXT).
- Processes and stores document embeddings for quick context retrieval.
- Determines the appropriate agent (DocAgent, TableAgent, or MathAgent, more to come) to answer a given question.
- If an answer isn't found within the pre-loaded documents, AutoTutor searches Google for additional context.
- Utilizes OpenAI's ChatGPT for natural language understanding and generation.

# Setup
- Clone the repository to your chosen directory
- Install the required Python packages:

  pip install -r requirements.txt
  
- Fill in the required fields in the config.ini file, including your Pinecone API key, Pinecone Environment, Pinecone Index name, OpenAI API key, Wolfram Alpha API key, Google Custom Search Engine API key and Google Custom Search Engine ID.

# Usage
- To start AutoTutor, run the main script:

python auto_tutor.py

- AutoTutor will prompt you to add a folder of documents. If you choose to add documents, enter the folder path when prompted. AutoTutor will ingest the documents, process them, and store their embeddings.

Once the setup is complete, AutoTutor will be ready to answer your questions. Enter your question at the prompt, and AutoTutor will use the appropriate agent to find the best answer. If an answer cannot be found within the pre-loaded documents, AutoTutor will search Google for additional context before attempting to answer your question again.

To exit AutoTutor, type "exit" at the question prompt.

# Notes

- This program, while stable, is still in alpha.  It's possible it will get lost at certain points, but for most docs it is a powerful QA bot.
