# Running the full system in Google Colab (with real models) + ngrok

This mirrors how the midterm notebook was run, extended with a FastAPI
backend and a static HTML/CSS/JS frontend served from the same process,
tunnelled to the internet with ngrok so it can be demoed in the
face-cam video.

## 1. Open a Colab notebook with a GPU runtime
Runtime → Change runtime type → GPU (T4 or better).

## 2. Get the project files into Colab
Either `git clone` your GitHub repo, or zip/upload this folder and
unzip it in `/content/`.

```python
!git clone https://github.com/<your-username>/<your-repo>.git
%cd <your-repo>/backend
```

## 3. Put the data files in place
```
backend/data/Family_Law.pdf
backend/data/Labor_Law.pdf
backend/data/Indexsheet.csv
```
(Same files used in the midterm project — copy or `!wget` them in.)

## 4. Install dependencies
```python
!pip install -q -r requirements.txt
```

## 5. Get a free ngrok authtoken
Sign up at https://dashboard.ngrok.com (free tier is enough) and copy
your authtoken.

## 6. Launch the backend + tunnel
```python
import subprocess, time
from pyngrok import ngrok

ngrok.set_auth_token("YOUR_NGROK_TOKEN")

# LEGAL_RAG_MOCK is NOT set here -> real models load (GPU required)
proc = subprocess.Popen(
    ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
    cwd="."
)
time.sleep 15  # give the models time to load — check logs if it's still loading

public_url = ngrok.connect(8000)
print("Open your app at:", public_url)
```

Open the printed `https://....ngrok-free.app` URL — it serves the
chat GUI directly (FastAPI mounts the frontend at `/`), and the GUI
talks to the same origin's `/api/*` routes, so no extra CORS/config
is needed.

## 7. Sanity check before recording the demo video
```
GET  <ngrok-url>/api/health   -> {"status":"ok","mock_mode":false,"rag_ready":true}
```

## Developing without a GPU (frontend / agents / FastAPI only)
```bash
cd backend
pip install fastapi uvicorn pydantic requests pyngrok
LEGAL_RAG_MOCK=1 uvicorn main:app --reload --port 8000
```
This skips loading Qwen/bge-m3/bge-reranker entirely and returns
clearly-labelled `[MOCK MODE]` answers, so you can build/test the
booking agent, database, and GUI on a laptop, then switch to the real
models only in Colab for the final run.
