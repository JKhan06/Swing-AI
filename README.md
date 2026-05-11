# SwingAI

AI-powered golf swing analysis platform that uses computer vision and LLM-based feedback to analyze swing mechanics in real time.

## Features

- Pose estimation using MediaPipe
- AI-generated swing feedback with Google Gemini
- FastAPI backend for video processing
- React frontend for interactive analysis
- Upload and analyze golf swing videos

## Tech Stack

Frontend: React.js, TypeScript, Tailwind CSS
Backend: FastAPI, Python
AI/ML: MediaPipe, Google Gemini
Deployment/Tools: Git, GitHub, REST APIs

## Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run the dev server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000` with docs at `http://127.0.0.1:8000/docs`.

## Frontend Setup

```bash
cd frontend
npm install
```

### Run the dev server

```bash
npm run dev
```

The app will be available at `http://localhost:3000`. Expects the backend running on `http://localhost:8000`.
