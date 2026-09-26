# The explorer with live search, on CPU. Build: docker build -t groundwork .
# Run, with the corpus, model weights and vectors in ./data and the site data in
# ./results/site/data (written by `groundwork site`):
#   docker run --rm -p 8000:8000 -v "$PWD/data:/data" -v "$PWD/results/site/data:/app/site/data" \
#     groundwork --live configs/grid/fixed-bm25.toml configs/grid/fixed-qwen3-rerank.toml
# Give the container about 12 GB. It reads the chunk vectors the evaluation computed and encodes
# only the question. On a CPU the reranker takes about 20 s a question (measured, 6 cores); on an
# Apple GPU, `uv run groundwork serve --site src/explorer/dist --live ...` takes about 0.6 s.

FROM node:24-slim AS explorer
WORKDIR /explorer
COPY src/explorer/package.json src/explorer/package-lock.json ./
RUN npm ci
COPY src/explorer/ ./
RUN npm run build

FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /uvx /bin/
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
COPY pyproject.toml uv.lock README.md ./
COPY src/groundwork src/groundwork
RUN uv sync --locked --no-dev
COPY configs configs
COPY --from=explorer /explorer/dist /app/site
EXPOSE 8000
ENTRYPOINT ["uv", "run", "--no-sync", "groundwork", "--data-dir", "/data", "serve", \
            "--host", "0.0.0.0", "--site", "/app/site", "--device", "cpu"]
