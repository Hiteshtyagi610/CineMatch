# Backend (FastAPI) image. Run the Streamlit frontend separately —
# `streamlit run frontend/app.py` — pointing API_BASE at this container.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

COPY common/ common/
COPY backend/ backend/

# Model artifacts are gitignored (regenerate, don't bake stale ones in).
# Build with: docker build --build-arg TRAIN=1 . -t cinematch-api
# or simply run scripts/train_model.py locally first and COPY models/ here.
COPY models/ models/

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
