FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .
COPY eval ./eval
COPY data ./data
# Eval runner only; training needs a GPU host with the `train` extra.
ENTRYPOINT ["python", "eval/run.py"]
