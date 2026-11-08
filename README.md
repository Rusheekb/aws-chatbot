# AWS Bedrock Chatbot

A command-line chatbot built with Python and Amazon Bedrock (Nova 2 Lite).

## Features
- Streaming responses
- Conversation memory with a sliding window
- Per-turn and session cost tracking
- Graceful error handling
- Configurable via `.env` and `system_prompt.txt`
- Conversation history saved between sessions
- Web UI built with Streamlit

## Setup
1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your AWS credentials
4. Make sure the model is enabled in the Bedrock console for your region

## Usage
**Terminal:**
```bash
python3 chatbot.py
```
Commands: `clear` resets memory (and the saved history), `quit` exits and shows a cost summary.

**Web UI:**
```bash
streamlit run app.py
```
Opens at http://localhost:8501. Both interfaces share the same conversation history.

## Configuration
| Setting | Default | Description |
|---|---|---|
| `MODEL_ID` | `global.amazon.nova-2-lite-v1:0` | Bedrock model to use |
| `MAX_TOKENS` | `300` | Max length of each reply |
| `TEMPERATURE` | `0.7` | 0 = focused, 1 = creative |
| `MAX_TURNS` | `3` | Exchanges remembered |
| `INPUT_PRICE_PER_M` / `OUTPUT_PRICE_PER_M` | `0.30` / `2.50` | Prices for cost tracking |

Edit `system_prompt.txt` to change the bot's personality.
