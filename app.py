"""Web interface for the Bedrock chatbot. Run with: streamlit run app.py"""
import streamlit as st
from botocore.exceptions import BotoCoreError, ClientError

from bot import (
    MAX_TURNS,
    MODEL_ID,
    Turn,
    count_turns,
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
    texts = [block["text"] for block in message["content"] if "text" in block]
    if texts:
        with st.chat_message(message["role"]):
            st.markdown("\n\n".join(texts))

# --- Handle a new question ---
question = st.chat_input("Ask me anything")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    start = len(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": [{"text": question}]})
    turn = Turn(st.session_state.messages)

    with st.chat_message("assistant"):
        try:
            st.write_stream(turn.stream())
        except ClientError as error:
            code = error.response["Error"]["Code"]
            detail = error.response["Error"]["Message"]
            st.error(f"AWS error: {code} — {detail}")
            del st.session_state.messages[start:]
            st.stop()
        except BotoCoreError as error:
            st.error(f"Connection problem: {error}")
            del st.session_state.messages[start:]
            st.stop()

        if turn.tools_used:
            st.caption(f"Tools used: {', '.join(turn.tools_used)}")
        if turn.stop_reason == "max_tokens":
            st.warning("Reply was cut off — hit the max token limit.")

    st.session_state.messages = trim_history(st.session_state.messages)
    save_history(st.session_state.messages)

    st.session_state.total_cost += turn.cost()
    st.session_state.total_tokens += turn.input_tokens + turn.output_tokens

# --- Sidebar: drawn LAST so it shows the latest numbers ---
with st.sidebar:
    st.header("Session")
    st.caption(f"Model: {MODEL_ID}")
    st.metric("Session cost", f"${st.session_state.total_cost:.6f}")
    st.metric("Tokens used", f"{st.session_state.total_tokens:,}")
    st.caption(f"Memory: {count_turns(st.session_state.messages)}/{MAX_TURNS} turns")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        save_history([])
        st.rerun()
