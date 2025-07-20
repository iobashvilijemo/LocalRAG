import os
import tempfile
import time
import streamlit as st
from embedchain import App
import base64
from streamlit_chat import message




def embedchain_bot(db_path):

    return App.from_config(
        config={
            "llm": {"provider": "ollama", "config": {
                "model": "llama3.2:latest", 
                "max_tokens": 250,
                "temperature": 0.1, 
                "stream": True, 
                "base_url": "http://localhost:11434"}},
            
            "vectordb": {"provider": "chroma", "config": {"dir": db_path}},
            
            "embedder": {"provider": "ollama", "config": {
                "model": "mxbai-embed-large:latest", 
                "base_url": "http://localhost:11434"}},
        }
    )

def display_pdf(file):
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="400"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


st.title("Chat with PDF using Embedchain")
st.caption("Upload a PDF file to start analyzing it with LLM.")


db_path = os.path.join(os.getcwd(), "chroma_db")

os.makedirs(db_path, exist_ok=True)
st.write(f"Using temporary directory for database: {db_path}")


if 'app' not in st.session_state:
    st.session_state.app = embedchain_bot(db_path)
if 'messages' not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("Upload PDF")
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

    if pdf_file:
        st.subheader("PDF Preview")
        display_pdf(pdf_file)


if st.button("Process PDF") and pdf_file:
    with st.spinner("Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_file.getvalue())
            temp_file.flush()
            st.session_state.app.db.reset()
            st.session_state.app.add(temp_file.name)
            
    st.success("PDF processed successfully!")
    time.sleep(1)
    os.remove(temp_file.name)


for i, msg in enumerate(st.session_state.messages):
    message(msg["content"], is_user=msg.get("is_user", False), key=str(i))


if prompt := st.chat_input("Ask a question about the PDF"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    message(prompt, is_user=True)


    with st.spinner("Generating response..."):
        response = st.session_state.app.chat(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        message(response)

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()