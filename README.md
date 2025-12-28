# Class 6 English RAG Chatbot

A Retrieval-Augmented Generation (RAG) based chatbot designed to help Class 6 English students in India learn stories, poems, grammar, and vocabulary strictly from their syllabus.

This project is built for AI + coding practice, not for marks or exams.

🚀 Project Highlights

📚 Answers only from Class 6 English syllabus

🔍 Uses Vector Database (Chroma) for semantic search

🧠 Implements manual RAG (no black-box chains)

🧑‍🏫 Follows a strict teacher-style prompt (RTCFR)

❌ Refuses out-of-syllabus questions

🖥️ Clean Streamlit UI

🧠 How It Works (High Level)

Syllabus PDFs (inside a ZIP file) are extracted

Text is split into chunks

Chunks are converted into embeddings

Embeddings are stored in a Chroma Vector DB

User asks a question

Relevant syllabus content is retrieved

LLM answers only using retrieved content

This ensures zero hallucination.

🛠️ Tech Stack

Python

Streamlit (Frontend)

LangChain (modular packages)

ChromaDB (Vector Database)

OpenAI Embeddings + LLM

dotenv for API key management
