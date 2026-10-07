FROM python:3.11-slim AS core

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace
COPY . /workspace
RUN python -m pip install --no-cache-dir -e ".[dev]"

CMD ["python", "-m", "pytest"]

FROM core AS graph
RUN python -m pip install --no-cache-dir torch \
      --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir torch-geometric
CMD ["python", "-m", "pytest", "tests/test_graph.py", "tests/test_graph_models.py", "tests/test_graph_vertical_slice.py"]

FROM core AS final
