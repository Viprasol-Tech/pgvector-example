<p align="center"><img src="docs/assets/logo.png" width="120" alt="Viprasol Tech"></p>

<h1 align="center">pgvector Example</h1>

<p align="center">A practical, offline-friendly Postgres <code>pgvector</code> example — schema, embeddings, and top-k similarity search.</p>

<p align="center"><em>Built and maintained by <a href="https://viprasol.com">Viprasol Tech</a> — Fintech Experts. Full-Stack Builders.</em></p>

<p align="center">
  <a href="https://github.com/Viprasol-Tech/pgvector-example/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/Viprasol-Tech/pgvector-example/ci.yml?branch=main&style=flat-square" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue?style=flat-square" alt="Python">
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/badge/lint-ruff-261230?style=flat-square" alt="Ruff"></a>
  <a href="https://t.me/viprasol_help"><img src="https://img.shields.io/badge/Telegram-chat-26A5E4?style=flat-square&logo=telegram" alt="Telegram"></a>
  <a href="https://github.com/Viprasol-Tech/pgvector-example/stargazers"><img src="https://img.shields.io/github/stars/Viprasol-Tech/pgvector-example?style=flat-square" alt="Stars"></a>
</p>

---

A clean, copy-pasteable reference for vector similarity search on Postgres with [`pgvector`](https://github.com/pgvector/pgvector). It ships a real psycopg backend **and** a pure-Python cosine fallback, so the entire test suite and demo run fully offline — no database required.

## ✨ Features

- 🐘 **Real pgvector backend** — psycopg 3 + `vector` column, IVFFlat index, cosine/L2/inner-product.
- 🔌 **Offline by default** — in-memory cosine store so tests and the demo run with zero infrastructure.
- 🧩 **Pluggable embedders** — a clean `Embedder` ABC with deterministic `FakeEmbedder`.
- 🤖 **Optional real models** — `sentence-transformers` and OpenAI backends, imported lazily.
- 🗄️ **Schema as code** — inspectable DDL helpers for the extension, table, and index.
- 🔎 **Top-k similarity** — `upsert` + cosine `query` with scores and metadata.
- 🧪 **35 offline tests** — fast, deterministic, no network or DB.
- 🛡️ **Typed & strict** — `py.typed`, `mypy --strict`, `ruff` clean.
- 📦 **Single-source packaging** — hatchling, one `pyproject.toml`.
- 🐳 **Docker-ready** — `docker-compose` spins up `pgvector/postgres` for real use.
- ⚙️ **CI matrix** — Ubuntu/macOS/Windows × Python 3.11–3.13.
- 🖥️ **CLI demo** — `pgvector-example "your query"`.

## 🚀 Quickstart

```bash
git clone https://github.com/Viprasol-Tech/pgvector-example.git
cd pgvector-example
pip install -e ".[dev]"

# Run the fully-offline demo:
python examples/demo.py
# or the CLI:
pgvector-example "how does vector search work in postgres"
```

Everything above runs **without a database**.

## 📖 Usage

```python
from pgvector_example import FakeEmbedder, InMemoryVectorStore

store = InMemoryVectorStore(FakeEmbedder(dim=64))
store.upsert("d1", "pgvector adds vector similarity search to Postgres.")
store.upsert("d2", "Embeddings map text into a dense numeric vector space.")

for hit in store.query("vector search in postgres", top_k=2):
    print(f"{hit.score:.3f}  {hit.id}: {hit.content}")
```

### Using the real Postgres backend

```python
from pgvector_example import FakeEmbedder, PostgresVectorStore

store = PostgresVectorStore(FakeEmbedder(dim=64), dsn="postgresql://...")
store.init_schema()            # enables extension + creates table & index
store.upsert("d1", "hello pgvector")
print(store.query("hello", top_k=5))
```

Install the driver extra and start Postgres with Docker:

```bash
pip install -e ".[postgres]"
docker compose up -d
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
```

See the [DDL helpers](src/pgvector_example/schema.py) to inspect the exact SQL.

## 🗺️ Roadmap

- [ ] HNSW index option alongside IVFFlat
- [ ] Async psycopg backend
- [ ] Metadata filtering in queries
- [ ] Batch upsert via `COPY`
- [ ] Benchmark harness

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) and our [Code of Conduct](CODE_OF_CONDUCT.md). Run `ruff`, `mypy`, and `pytest` before opening a PR.

## 📬 Contact — Viprasol Tech Private Limited

- 🌐 Website: [viprasol.com](https://viprasol.com)
- ✉️ Email: [support@viprasol.com](mailto:support@viprasol.com)
- 💬 Telegram: [t.me/viprasol_help](https://t.me/viprasol_help) · 📱 WhatsApp: +91 96336 52112
- 🐙 GitHub: [@Viprasol-Tech](https://github.com/Viprasol-Tech) · 💼 [LinkedIn](https://www.linkedin.com/in/viprasol/) · 𝕏 [@viprasol](https://twitter.com/viprasol)

> *Viprasol Tech — fintech software, algorithmic trading systems, MT4/MT5 bots, AI voice agents, and B2B SaaS. Need a custom build? [Get in touch](mailto:support@viprasol.com).*

## 📄 License

Released under the [MIT License](LICENSE) © 2025 Viprasol Tech Private Limited.
