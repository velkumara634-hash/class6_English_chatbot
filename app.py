import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate

# -------------------------
# Streamlit setup
# -------------------------
st.set_page_config(page_title="Class 6 English RAG Bot")
st.markdown(
    """
    <style>
    .chat-user {
        background-color: #e8f0fe;
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 8px;
    }
    .chat-bot {
        background-color: #f6f8fa;
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 16px;
        border-left: 4px solid #4f8bf9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📘 Class 6 English Chatbot")
st.caption("Learn English stories, poems, and grammar — strictly from your syllabus")

with st.sidebar:
    st.header("ℹ️ How to use")
    st.write(
        """
        • Ask questions from **Class 6 English syllabus**  
        • Stories, poems, grammar, meanings  
        • Simple explanations only  
        """
    )

    st.divider()

    show_context = st.checkbox("🔍 Show retrieved text (debug)")

# -------------------------
# Load .env
# -------------------------
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY not found in .env")
    st.stop()

# -------------------------
# Load Vector DB
# -------------------------
embeddings = OpenAIEmbeddings()

db = Chroma(
    persist_directory="vectordb",
    embedding_function=embeddings
)

# ✅ THIS WAS MISSING
retriever = db.as_retriever(search_kwargs={"k": 3})

# -------------------------
# Prompt
# -------------------------
PROMPT = PromptTemplate(
    template="""
You are an AI English Tutor for Class 6 students.

Rules:
- Use ONLY the given context
- If the answer is not found in context, say exactly:
  "This topic is not available in your Class 6 English syllabus."

Answer format:
Explanation:
(2–4 short sentences)

Example:
(one simple example)

Try This:
(one question)

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)

llm = ChatOpenAI(temperature=0)

# -------------------------
# UI
# -------------------------
query = st.text_input(
    "Ask your Class 6 English question:",
    placeholder="e.g. Tell me the story of How the Dog Found Himself a New Master"
)


if query:
    st.markdown(f"<div class='chat-user'><b>You:</b><br>{query}</div>", unsafe_allow_html=True)

    with st.spinner("Thinking from textbook..."):
        docs = retriever.invoke(query)

    if not docs:
        st.markdown(
            "<div class='chat-bot'><b>Bot:</b><br>"
            "This topic is not available in your Class 6 English syllabus."
            "</div>",
            unsafe_allow_html=True
        )
    else:
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = PROMPT.format(
            context=context,
            question=query
        )

        response = llm.invoke(prompt)

        st.markdown(
            f"<div class='chat-bot'><b>Bot:</b><br>{response.content}</div>",
            unsafe_allow_html=True
        )

        # Optional debug view
        if show_context:
            with st.expander("📄 Retrieved Text"):
                st.write(context)


        response = llm.invoke(prompt)
        st.write(response.content)
