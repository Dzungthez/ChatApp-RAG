import streamlit as st
import os
from multiprocessing import freeze_support
from src import main
from pipeline.text_loader import Loader
import asyncio


def save_uploaded_file(uploaded_file):
    with open(os.path.join("data", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())


async def async_invoke(workflow, question):
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, workflow.invoke, question)
    return response


async def running(workflow):
    st.title("AI Question Answering System")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        save_uploaded_file(uploaded_file)
        st.success(f"Uploaded file: {uploaded_file.name}")

        docs = Loader().load(os.path.join("data", uploaded_file.name))
        st.session_state.documents = docs

    # Initialize session state for messages if not already done
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.header("Chat with AI")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask something about the document"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = await async_invoke(workflow, prompt)
        answer = response['generation']

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)


if __name__ == "__main__":
    freeze_support()
    workflow = main.call_workflow()
    asyncio.run(running(workflow))