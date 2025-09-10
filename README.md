# 📊 CSV Engine PoC

This is a proof-of-concept **mini tabular engine** built in Python.

It was never intended to be "production-grade" — it was written for fun, as a way to explore the **design intricacies** of streaming, filtering, transformation, and memory-aware data processing over tabular formats.

## 🧠 Motivation

Tabular engines (like Pandas, Polars, DuckDB) feel magical — but what happens under the hood?

This project tries to **recreate that feeling from scratch**, learning how real systems might:

- Stream data from CSVs
- Lazily process rows
- Filter and transform data
- Build composable predicates
- Operate under memory pressure (adaptive caching)
- Normalize and search across fields with ranking

## 🛠️ Features

- ✅ Streaming CSV ingestion (via Python `csv.reader`)
- ✅ Lazy filtering with predicate logic
- ✅ Basic transformation engine (arithmetic, string ops)
- ✅ Adaptive caching mechanism — **hot RAM vs. cold disk**
- ✅ Diacritics-insensitive search (`"café"` == `"cafe"`)
- ✅ DSL-style query parser (`title:"foo bar" tag:csv`)
- ✅ Output: pretty table + JSONL export
- ✅ FastAPI backend with `/search` and `/healthz`
- 🧪 Optional profiling + unit tests (WIP)

## ⚠️ Limitations

This engine is **not performant** compared to mature tools:

> ⚠️ We do **not** recommend building this in Python. Use Pandas, Polars, or a C++/Rust-based backend for any real workload.

This was purely about **reverse-engineering the design principles** of tabular engines, not competing with them.

## 🧪 Testing Adaptive Cache (MIXED Mode)

The system includes a **memory-aware engine** that switches modes:

- `IMMUT_VIEW`: only views data (safe to cache aggressively)
- `MUTATE_TRANSFORM`: modifies rows (higher memory cost)
- `MIXED`: hybrid mode — filters + transforms

The **adaptive cache** stores hot rows in memory and spills cold data to disk once memory exceeds ~25MB.

To test the full pipeline:
- Run a **transformation operation (e.g., column arithmetic)** and combine it with a **filter/search query**
- Ensure your dataset is large enough to exceed the memory cap (`sys.getsizeof` can help estimate)
- Observe the fallback to disk (you’ll see the ⚠️ log about switching to disk mode)

This ensures that **MIXED mode logic + adaptive cache** actually kicks in.

## 🚀 Running

```bash
# Run CLI
python app.py --file Data/snippets.csv --q "cafe" --fields title,body --diacritics-off

# Run API
uvicorn routes:app --reload
curl "http://127.0.0.1:8000/search?q=cafe&file=Data/snippets.csv&fields=title&fields=body&diacritics_off=true"
