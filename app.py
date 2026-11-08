"""Web interface for the Bedrock chatbot. Run with: streamlit run app.py"""
import streamlit as st
from botocore.exceptions import BotoCoreError, ClientError

from bot import (
    MAX_TURNS,
    MODEL_ID,
    Reply,
    load_history,
    save_history,
    trim_history,
)

st.set_page_config(page_title="Bedrock Chatbot")
st.title("Bedrock Chatbot")

# --- Session state: runs only on the FIRST load, survives reruns ---
if "messages" not in st.session_state:
    st.session_state.messages = trim_history(load_history())
    st.session_state.total_cost = 0.0
    st.session_state.total_tokens = 0

# --- Draw the conversation so far ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"][0]["text"])

# --- Handle a new question ---
question = st.chat_input("Ask me anything")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({"role": "user", "content": [{"text": question}]})
    reply = Reply(st.session_state.messages)

    with st.chat_message("assistant"):
        try:
            st.write_stream(reply.stream())
        except ClientError as error:
            code = error.response["Error"]["Code"]
            detail = error.response["Error"]["Message"]
            st.error(f"AWS error: {code} — {detail}")
            st.session_state.messages.pop()
            st.stop()
        except BotoCoreError as error:
            st.error(f"Connection problem: {error}")
            st.session_state.messages.pop()
            st.stop()

        if reply.stop_reason == "max_tokens":
            st.warning("Reply was cut off — hit the max token limit.")

    st.session_state.messages.append({"role": "assistant", "content": [{"text": reply.text}]})
    st.session_state.messages = trim_history(st.session_state.messages)
    save_history(st.session_state.messages)

    st.session_state.total_cost += reply.cost()
    st.session_state.total_tokens += reply.usage["totalTokens"]

# --- Sidebar: drawn LAST so it shows the latest numbers ---
with st.sidebar:
    st.header("Session")
    st.caption(f"Model: {MODEL_ID}")
    st.metric("Session cost", f"${st.session_state.total_cost:.6f}")
    st.metric("Tokens used", f"{st.session_state.total_tokens:,}")
    st.caption(f"Memory: {len(st.session_state.messages) // 2}/{MAX_TURNS} turns")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        save_history([])
        st.rerun()
