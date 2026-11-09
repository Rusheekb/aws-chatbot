# Deploying the demo

The web app can run for free on [Streamlit Community Cloud](https://streamlit.io/cloud). The only costs are Bedrock calls, about $0.0012 per question. These steps keep a public demo from running up your AWS bill.

## 1. Create a separate AWS user for the demo

Never deploy your own keys. If the demo's key leaks, it should only be able to do what the demo does.

1. In the IAM console, create a policy from [`iam-policy.json`](iam-policy.json). Replace `ACCOUNT_ID` with your 12-digit AWS account ID first.
   It allows exactly three things: streaming and non-streaming chat with Nova 2 Lite, and Titan embeddings. Nothing else.
2. Create a user (for example `learn2lead-demo`) with **only** that policy attached, and create an access key for it.
3. Check the key does what the app needs and nothing more:
   ```bash
   AWS_ACCESS_KEY_ID=<demo key> AWS_SECRET_ACCESS_KEY=<demo secret> AWS_DEFAULT_REGION=us-east-2 \
       python3 deploy/check_permissions.py
   ```
   All four lines should say `OK`. (With your own, broader key, the last check reports `FAIL`, because your key can use models the demo doesn't need.)

If a model call fails with `AccessDeniedException` after deploying, the error message names the exact resource ARN that was denied; add that ARN to the policy.

## 2. Deploy on Streamlit Community Cloud

1. Push the repository to GitHub. It contains no secrets: `.env` and the generated `doc_index.json` are in `.gitignore`.
2. At [share.streamlit.io](https://share.streamlit.io), choose **Create app**, pick the repository, branch `main` and file `app.py`. Under **Advanced settings**, choose Python 3.13.
3. Paste the secrets (TOML format):
   ```toml
   AWS_ACCESS_KEY_ID = "<demo key>"
   AWS_SECRET_ACCESS_KEY = "<demo secret>"
   AWS_DEFAULT_REGION = "us-east-2"
   APP_PASSWORD = "<something you share with reviewers>"
   MAX_QUESTIONS_PER_SESSION = "20"
   ```
   Streamlit makes top-level secrets available as environment variables, which is where boto3 and the app read them.
4. Deploy. On first start the app builds its search index (about 70 embedding calls, a few seconds, well under $0.001).

## 3. Protections in the app

| Setting | Default | What it prevents |
|---|---|---|
| `APP_PASSWORD` | none | Strangers using the demo. Leave it empty to make the demo open. |
| `MAX_QUESTIONS_PER_SESSION` | 20 | One visitor asking hundreds of questions |
| `MAX_QUESTION_CHARS` | 500 | Very long inputs (input tokens cost money) |
| `MAX_TOKENS` | 300 | Very long answers (output tokens cost most) |

The question limit is per browser session, so someone determined can reload the page to reset it. The password, the AWS budget alert and the restricted demo user are the real protections. For heavier traffic, add a shared limit (for example a daily counter in a database).

## 4. If something goes wrong

- **The key leaks:** deactivate it in IAM immediately and create a new one. The restricted policy limits the damage to Nova 2 Lite and Titan calls.
- **Costs climb:** your AWS budget alert emails you. Delete the app on share.streamlit.io, or deactivate the key, to stop spending at once.
- **The demo is no longer needed:** delete the app, then delete the IAM user.
