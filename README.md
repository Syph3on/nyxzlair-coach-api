# Coach API — backend for the calorie tracker chat

This wraps your `agent.py` LangGraph logic in a small web API (`main.py`)
so your website can call it. It's a **separate service** from your React
site — different repo, different host.

## Run it locally first

```bash
python3 -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your real keys (same two keys
from your original `_env` file):
```
GEMINI_API_KEY=...
SERPAPI_API_KEY=...
```

Run it:
```bash
uvicorn main:app --reload --port 8000
```

Test it in your browser at `http://localhost:8000/docs` — you'll see an
interactive page where you can try the `/chat` endpoint directly.

## Deploy it (Render — free tier, works well for this)

1. Push this `agent-server` folder to its own GitHub repo (separate from
   your website repo).
2. Go to [render.com](https://render.com) → sign up → **New** →
   **Web Service** → connect that repo.
3. Settings:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Under **Environment**, add `GEMINI_API_KEY` and `SERPAPI_API_KEY` —
   same values as your `.env`. Never put these in your frontend.
5. Deploy. You'll get a URL like `https://your-app.onrender.com`.
6. Once deployed, come back to `main.py` and tighten the CORS line —
   replace `allow_origins=["*"]` with your actual Vercel URL, so random
   sites can't call your API and burn your API quota.

Note: Render's free tier spins the service down when idle, so the first
request after a while will be slow (~30s cold start) — normal, not a bug.

## Connect it to your frontend

In your React project's `.env`, add:
```
VITE_AGENT_API_URL=https://your-app.onrender.com
```
(use `http://localhost:8000` while testing locally)

## One deviation from your original agent.py

I changed the model from `gemini-3.1-flash-lite` to `gemini-2.5-flash`,
and I pass the API key explicitly rather than relying on the library's
default env var name (it expects `GOOGLE_API_KEY`, but your `_env` file
uses `GEMINI_API_KEY`). If `gemini-3.1-flash-lite` is a real model you
have access to, just change the `model=` line in `main.py` back.

I also removed the CLI-only pieces (file attachments, the interactive
`input()` loop, the heartbeat console logger) since those don't apply to
a web request/response — the sarcasm and search-routing logic is intact.
