# DocuBot: PDF Conversational AI Chatbot GPT 3.5
This is a Python GUI application that demonstrates how to build a custom PDF chatbot using LangChain and GPT 3.5 

## Features
- Extracts text content from uploaded PDF documents.
- Splits the extracted text into manageable chunks for efficient processing.
- Converts text chunks into vector embeddings for semantic similarity search.
- Provides a conversational interface where users can query the content of the uploaded PDFs.
- Maintains conversation history to provide context-aware responses.


## Important Files and Folders
1. **`src/.env`**: File with environment variables.
2. **`src/requirements.txt`**: File with dependencies to be installed for the project.
3. **`src/`**: Directory containing the source code for the streamlit app.
4. **`docs/`**: Documentation including meeting minutes and README in PDF format.


## How it works
1. The application gui is built using streamlit
2. The application reads text from PDF files, splits it into chunks
3. Uses OpenAI Embedding API to generate embedding vectors used to find the most relevant content to a user's question
4. Build a conversational retrieval chain using Langchain
5. Use OpenAI GPT API to generate respond based on content in PDF


## Requirements
1. Install the following Python packages:
    ```
    pip install -r requirements.txt
    ```

2. Create a `.env` file in the root directory of the project and add the following environment variables:
    ```
    OPENAI_API_KEY= # Your OpenAI API key
    ```
### Note
> Make sure you are in the `/src` directory

## Code Structure
The code is structured as follows:

- `app.py`: The main application file that defines the Streamlit gui app and the user interface.
    * get_pdf_text function: reads text from PDF files
    * get_text_chunks function: splits text into chunks
    * get_vectorstore function: creates a FAISS vectorstore from text chunks and their embeddings
    * get_conversation_chain function: creates a retrieval chain from vectorstore
    * handle_userinput function: generates response from OpenAI GPT API
- `htmlTemplates.py`: A module that defines HTML templates for the user interface.


## How to run
```py
streamlit run app.py
```
> __Note__: Make sure you are in the `/src` directory
