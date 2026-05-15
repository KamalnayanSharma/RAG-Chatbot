import streamlit as st

from pdf_processing import extract_text_from_pdf
from utils import chunk_text
from vector_store import create_vector_store, search_query
from llm import get_answer

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("PDF Question Answering System")

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    all_text = ""

    for pdf in uploaded_files:

        text = extract_text_from_pdf(pdf)

        all_text += text

    chunks = chunk_text(all_text)

    index, embeddings = create_vector_store(chunks)

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    question = st.chat_input("Ask a question")

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        relevant_chunks = search_query(
            question,
            chunks,
            index
        )

        context = " ".join(
            [chunk["text"] for chunk in relevant_chunks]
        )

        # st.subheader("Retrieved Chunks")
        # for chunk in relevant_chunks:
        #     st.write(f"Chunk ID: {chunk['chunk_id']}")
        #     st.write(chunk["text"])
        #     st.write("------")
        with st.spinner("Thinking..."):

            answer = get_answer(question, context)

        with st.chat_message("assistant"):

            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )