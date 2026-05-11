# SwingAI

Backend–frontend project scaffold.

## Backend

- **Path**: `backend/`
- **Framework**: FastAPI

### Setup

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

## Frontend

- **Path**: `frontend/` (to be added later)

