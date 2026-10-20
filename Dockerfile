FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY --chown=user:user requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user:user main.py ./main.py
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
