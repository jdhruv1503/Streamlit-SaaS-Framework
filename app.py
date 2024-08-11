import streamlit as st
from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage
import os
from dotenv import load_dotenv
import pickle

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY", "")
if not os.path.exists("_data/chatSessions"):
    os.makedirs("_data/chatSessions")

prompt = st.chat_input("Say something")

if "chatSessions" not in st.session_state:
    st.session_state.chatSessions = []
    if os.path.exists("_data/chatSessions.pickle"):
        with open("_data/chatSessions.pickle", "rb") as f:
            st.session_state.chatSessions = pickle.load(f)
    else:
        st.session_state.chatSessions = [{"name": "Unnamed Chat", "id": 0},{"name": "Unnamed Chat 2", "id": 1}]
        with open("_data/chatSessions.pickle", "wb") as f:
            pickle.dump(st.session_state.chatSessions, f)
    
    st.session_state.currentChatNumber = len(st.session_state.chatSessions)-1
    st.session_state.currentChatId = len(st.session_state.chatSessions)-1

    st.session_state.chatSessionData = []

    # Display chat histories
    for (i, chat) in enumerate(st.session_state.chatSessions):
        if os.path.exists(f"_data/chatSessions/{i}.pickle"):
            with open(f"_data/chatSessions/{i}.pickle", "rb") as f:
                st.session_state.chatSessionData.append(pickle.load(f))
        else:
            st.session_state.chatSessionData.append({"messages": []})


def get_llm_stream(messages):
    llm = Groq(model="llama-3.1-70b-versatile", openai_api_key=groq_api_key)
    stream = llm.stream_chat(messages = [ChatMessage(role=ms["role"], content=ms["content"]) for ms in messages])

    for word in stream:
        yield word.delta

def get_chat_page(i):
    for message in st.session_state.chatSessionData[st.session_state.currentChatId]["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt:
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.chatSessionData[i]["messages"].append({"role": "user", "content": prompt})

        stream = get_llm_stream(st.session_state.chatSessionData[i]["messages"])

        with st.chat_message("assistant"):  
            response = st.write_stream(stream)

        # Add assistant response to chat history
        st.session_state.chatSessionData[i]["messages"].append({"role": "assistant", "content": response})

        # Write new chat history to file
        with open(f"_data/chatSessions/{i}.pickle", "wb") as f:
            pickle.dump(st.session_state.chatSessionData[i], f)

with st.sidebar:
    st.title("🦜🔗 Streamlit LLM SaaS Quickstart App")

    with st.expander("Chat Sessions", True):
        pages = []
        st.navigation([st.Page(get_chat_page(chat["id"]), title=chat['name']) for chat in st.session_state.chatSessions])