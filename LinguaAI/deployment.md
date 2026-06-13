# LinguaAI – Deployment Guide

## Streamlit Cloud (Recommended)

1. Push project to a public GitHub repo
2. Go to https://share.streamlit.io → New App
3. Select repo, branch `main`, main file `app.py`
4. Click **Deploy** — Streamlit Cloud handles the rest

No secrets or API keys needed — LinguaAI uses only free public APIs.

---

## Hugging Face Spaces

1. Create Space at https://huggingface.co/new-space
2. SDK: **Streamlit**, Hardware: CPU Basic (free)
3. Upload: `app.py`, `requirements.txt`, `utils/`, `.streamlit/`
4. Space auto-builds and deploys

---

## Render (Free Tier)

1. New Web Service → connect GitHub repo
2. Build command: `pip install -r requirements.txt`
3. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Plan: Free

---

## Railway

1. New project → Deploy from GitHub
2. Set start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
3. Railway auto-detects Python and installs requirements

---

## Local Docker

```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t linguaai .
docker run -p 8501:8501 linguaai
```
