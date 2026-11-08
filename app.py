"""Web interface for the Learn2Lead assistant. Run with: streamlit run app.py"""
import streamlit as st
from botocore.exceptions import BotoCoreError, ClientError

from bot import MAX_TURNS, MODEL_ID, Turn, count_turns, trim_history

STARTER_QUESTIONS = [
    "What is Learn2Lead?",
    "How much does tutoring cost?",
    "What subjects do you offer?",
    "How do I join my class?",
]

st.set_page_config(page_title="Learn2Lead Assistant")
st.title("Learn2Lead Assistant")
st.caption("Ask about our tutoring, pricing and subjects, or how to use the site.")

# --- Session state: runs only on the FIRST load, survives reruns ---
# Each visitor gets their own conversation, kept only for this browser session.
# (The terminal app saves history to a file; a shared file would leak chats between visitors.)
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.total_cost = 0.0
    st.session_state.total_tokens = 0

# --- Welcome message and starter questions for new visitors ---
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("Hi! I can tell you about Learn2Lead's tutoring, or help you use the site. "
                    "What would you like to know?")
    columns = st.columns(2)
    for i, starter in enumerate(STARTER_QUESTIONS):
        if columns[i % 2].button(starter, use_container_width=True):
            st.session_state.pending_question = starter
            st.rerun()

# --- Draw the conversation so far ---
for message in st.session_state.messages:
    texts = [block["text"] for block in message["content"] if "text" in block]
    if texts:
        with st.chat_message(message["role"]):
            st.markdown("\n\n".join(texts))

# --- Handle a new question (typed, or a starter button) ---
question = st.chat_input("Ask about Learn2Lead") or st.session_state.pop("pending_question", None)

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
            st.error("Sorry, something went wrong. Please try again in a moment.")
            st.caption(f"AWS error: {error.response['Error']['Code']}")
            del st.session_state.messages[start:]
            st.stop()
        except BotoCoreError:
            st.error("Sorry, I couldn't connect. Please try again in a moment.")
            del st.session_state.messages[start:]
            st.stop()

        if turn.sources:
            st.caption("Sources: " + " · ".join(turn.sources))
        if turn.stop_reason == "max_tokens":
            st.warning("Reply was cut off — hit the max token limit.")

    st.session_state.messages = trim_history(st.session_state.messages)
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
        st.rerun()
