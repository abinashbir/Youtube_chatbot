# YouTube Chatbot

A Retrieval-Augmented Generation (RAG) based chatbot that allows users to ask questions about the content of a YouTube video.

The project extracts the video's transcript, splits it into smaller chunks, converts the chunks into embeddings, stores them in a FAISS vector database, retrieves relevant context, and uses Google Gemini to generate answers.

## Features

- Extract transcripts from YouTube videos
- Split transcripts into smaller chunks
- Generate embeddings using `all-MiniLM-L6-v2`
- Store and search embeddings using FAISS
- Retrieve relevant transcript sections for a question
- Generate answers using Google Gemini
- Built using LangChain
- Answers are restricted to the provided transcript context

## Architecture

```text
YouTube Video
      |
      v
YouTube Transcript
      |
      v
Text Splitting
      |
      v
Document Chunks
      |
      v
Sentence Transformer
all-MiniLM-L6-v2
      |
      v
FAISS Vector Store
      |
      v
Retriever
      |
      v
Relevant Context
      |
      v
Prompt Template
      |
      v
Gemini 3.6 Flash
      |
      v
Generated Answer