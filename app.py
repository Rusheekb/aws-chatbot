"""Web interface for the Learn2Lead assistant. Run with: streamlit run app.py"""
import hmac
import os

import streamlit as st
from botocore.exceptions import BotoCoreError, ClientError

import rag
from bot import MAX_TURNS, MODEL_ID, Turn, count_turns, trim_history

STARTER_QUESTIONS = [
    "What is Learn2Lead?",
    "How much does tutoring cost?",
    "What subjects do you offer?",
    "How do I join my class?",
]

# Limits for a public demo, so nobody can run up the AWS bill.
APP_PASSWORD = os.getenv("APP_PASSWORD", "")          # empty = no password
MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS_PER_SESSION", "20"))
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "500"))


def password_ok():
    """Ask for APP_PASSWORD until it's entered. Always True when no password is set."""
    if not APP_PASSWORD or st.session_state.get("authenticated"):
        return True
    entered = st.text_input("Demo password", type="password")
    if entered:
        # compare_digest takes the same time whether the first or last character is wrong,
        # so the password can't be guessed by timing the response.
        if hmac.compare_digest(entered.encode(), APP_PASSWORD.encode()):
            st.session_state.authenticated = True
            st.rerun()
        st.error("That password isn't right.")
    return False


@st.cache_resource(show_spinner="Getting ready…")
def warm_up():
    """Build the search index once per server, before the first visitor asks anything."""
    rag.ensure_index()
    return True


st.set_page_config(page_title="Learn2Lead Assistant")
st.title("Learn2Lead Assistant")
st.caption("Ask about our tutoring, pricing and subjects, or how to use the site.")

if not password_ok():
    st.stop()
warm_up()

# --- Session state: runs only on the FIRST load, survives reruns ---
# Each visitor gets their own conversation, kept only for this browser session.
# (The terminal app saves history to a file; a shared file would leak chats between visitors.)
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.total_cost = 0.0
    st.session_state.total_tokens = 0
    st.session_state.questions_asked = 0

questions_left = MAX_QUESTIONS - st.session_state.questions_asked

# --- Welcome message, plus starter questions until the conversation begins ---
with st.chat_message("assistant"):
    st.markdown("Hi! I can tell you about Learn2Lead's tutoring, or help you use the site. "
                "What would you like to know?")
if not st.session_state.messages:
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
if questions_left <= 0:
    st.info(f"You've reached this demo's limit of {MAX_QUESTIONS} questions. Thanks for trying it! "
            "For anything else, email learn2leadtutoring@gmail.com.")
question = st.chat_input(
    "Ask about Learn2Lead", disabled=questions_left <= 0, max_chars=MAX_QUESTION_CHARS
) or st.session_state.pop("pending_question", None)

if question and questions_left > 0:
    if len(question) > MAX_QUESTION_CHARS:      # also checked here, not only in the browser
        st.warning(f"Please keep questions under {MAX_QUESTION_CHARS} characters.")
        st.stop()
    st.session_state.questions_asked += 1

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
    st.caption(f"Questions left: {max(MAX_QUESTIONS - st.session_state.questions_asked, 0)}/{MAX_QUESTIONS}")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
