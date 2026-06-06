# Repository Guidelines

## Project Structure & Module Organization
The FastAPI inference surface lives in `main.py`, which loads the BAAI `bge-m3` encoder, materializes FAISS indexes on GPU, and exposes `/search`. Training and evaluation data (`train.csv`, `test.csv`, `ec_story_test.csv`, `tag_answer.csv`) sit at the repo root so scripts can stream them via pandas. Use `generate_train_test.py` to refresh splits from `ec_merged.csv`. Regression runs live in `evaluation_script.py`, which writes colored logs to `bge-m3/*.log`. Dependencies stay in `requirements.txt`.

## Build, Test, and Development Commands
```bash
python -m venv .venv && source .venv/bin/activate  # isolate deps
pip install -r requirements.txt                   # FastAPI, FlagEmbedding, FAISS, loguru
uvicorn main:app --reload --host 0.0.0.0 --port 8000  # serve the retriever
python evaluation_script.py                       # hit the live endpoint and log accuracy
python generate_train_test.py                     # rebuild train/test from ec_merged.csv
```
Run the server with CUDA visible; the FAISS GPU index is created at import time, so restart the process whenever you change data or the embedding model.

## Coding Style & Naming Conventions
Stay close to PEP 8: four-space indentation, `snake_case` for functions/variables, `UpperCamelCase` for Pydantic models, and short, descriptive module names (`evaluation_script.py` pattern). Prefer type hints on public functions, document endpoints with docstrings, and keep constants (`RED`, `GREEN`, etc.) near the top of scripts. Follow the established column names (`question`, `tag`, `answer`) and seed reproducibility helpers (e.g., `np.random.RandomState(42)`). Use vectorized pandas operations; avoid per-row loops unless profiling shows a bottleneck.

## Testing Guidelines
`evaluation_script.py` doubles as the primary regression test: start the FastAPI server locally, then run the script to POST each question to `/search` and verify tag/answer matches. Treat ≥0.6 cosine scores and ≥95% tag accuracy as the go/no-go gate before merging. Extend coverage by adding scenario CSVs (e.g., `ec_story_test.csv`) and capturing their logs under `bge-m3/`. When introducing new data columns or model tweaks, include a minimal smoke test in the PR description to show the service responds end-to-end.

## Commit & Pull Request Guidelines
History shows compact, descriptive summaries (`bge-m3 codes`, `Initial commit`). Keep that style: one-line, imperative subjects ≤50 chars, optionally followed by details in the body. Reference issues (`Fixes #12`) when relevant and describe data/model changes explicitly (“refresh train/test split with 2024-05 data”). Pull requests should explain why accuracy shifts, attach the latest evaluation log snippet, list any new CSVs or config knobs, and include rollback notes if the embedding checkpoint changes.
