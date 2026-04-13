# Deploy to Streamlit Community Cloud — Free, No Docker, No Azure

## Cost: $0

| Resource | Cost |
|---|---|
| GitHub (hosts your code) | Free |
| Streamlit Community Cloud (runs the app) | Free |
| Google Gemini API (AI agent) | Free tier |

---

## One-time setup steps

### Step 1 — Push code to GitHub

1. Go to https://github.com/new and create a **private** repository
   (e.g. `sports-leaderboard`)
2. In your `Leaderboard` folder, open a terminal and run:

```bash
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/<your-username>/sports-leaderboard.git
git push -u origin main
```

> `config.yaml` is in `.gitignore` so your API key is never uploaded.

---

### Step 2 — Connect to Streamlit Community Cloud

1. Go to **https://share.streamlit.io** and sign in with your GitHub account
2. Click **"Create app"**
3. Fill in:
   - **Repository**: `<your-username>/sports-leaderboard`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
4. Click **"Advanced settings"** before deploying (next step)

---

### Step 3 — Add your secrets

In the **Advanced settings → Secrets** box, paste exactly this
(replace the api_key with your real key):

```toml
[gemini]
api_key           = "AIzaSyAG94lfTPMdN1Z45pgOXymjUEAdo9KbIi8"
model_name        = "gemini-2.0-flash"
temperature       = 0.7
top_p             = 0.95
max_output_tokens = 4096

[app]
data_dir       = "/tmp"
default_rounds = 12
```

Click **"Deploy!"**

---

### Step 4 — Share the link

Streamlit gives you a URL like:
```
https://<your-username>-sports-leaderboard-streamlit-app-xxxx.streamlit.app
```

Share this link with all players. They open it in any phone browser — no app install needed.

---

## Updating the app after code changes

```bash
git add .
git commit -m "update"
git push
```

Streamlit Cloud detects the push and redeploys automatically within ~30 seconds.

---

## Updating secrets (e.g. new API key)

Streamlit Cloud → your app → **Settings → Secrets** → edit → **Save**.
The app restarts automatically.

---

## Notes on data persistence

Scores and schedule are stored in `/tmp` on the Streamlit Cloud server.
`/tmp` survives for the duration of the tournament session (hours) but
resets if the app is restarted or redeployed.

For a 2–3 hour home tournament this is perfectly fine — the app will
not restart on its own during an active session.
