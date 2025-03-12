import streamlit as st
from rag_chatbot import RagChatbot
import os

def main():
    st.title("RAG Chatbot")
    bot = RagChatbot()

    file = st.file_uploader(
        label="Upload your pdf file",
        type="pdf"
    )

    if file is not None:
        bot.createVectorDb(file)
        st.success("Document process successfully")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Ask a question")
    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
    
        res = bot.chat(user_query)
        with st.chat_message("assistant"):
            st.markdown(res)

        st.session_state.messages.append({"role": "assistant", "content": res})

if __name__ == "__main__":
    main()