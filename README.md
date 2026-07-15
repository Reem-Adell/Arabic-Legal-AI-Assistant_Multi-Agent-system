# Arabic Legal AI Assistant — Hybrid RAG with Legal-Index Routing

A Retrieval-Augmented Generation (RAG) system that answers questions in Arabic about
Egyptian personal-status (family) law and the 2025 labor law, grounded strictly in the
source legislation.

Built as the mid-term project for an AI Agent course. Runs end-to-end on a free
Google Colab GPU runtime with no API keys required.

## What it does

- Answers Arabic legal questions with citations back to the specific law and article
- Routes each query to the correct statute *before* retrieving text, using a
  lightweight metadata index rather than guessing from the full corpus
- Combines lexical (BM25) and semantic (FAISS) retrieval, then re-ranks with a
  cross-encoder for precision
- Supports natural follow-up questions via a custom conversational memory layer
  that contextualizes each new query against the conversation so far
- Refuses to answer when the retrieved documents don't actually support a claim,
  instead of hallucinating a legal opinion

## Architecture

```
User question (Arabic)
        │
        ▼
 Legal Index routing ── picks the relevant law(s) from Indexsheet.csv
        │
        ▼
 Hybrid retrieval ── BM25 (lexical) + FAISS (semantic, BAAI/bge-m3)
        │
        ▼
 Cross-encoder re-ranking ── BAAI/bge-reranker-v2-m3
        │
        ▼
 Grounded generation ── Qwen2.5-7B-Instruct, with conversational memory
        │
        ▼
 Answer + cited article(s), or refusal if unsupported
```

## Models & components

| Component        | Model / Library                     |
|-------------------|--------------------------------------|
| Embeddings        | `BAAI/bge-m3`                       |
| Re-ranker         | `BAAI/bge-reranker-v2-m3`           |
| LLM               | `Qwen2.5-7B-Instruct` (4-bit)       |
| Lexical retrieval | `rank_bm25`                         |
| Vector store      | FAISS (CPU)                         |
| PDF parsing       | PyMuPDF (`fitz`)                    |
| Orchestration     | LangChain (core, community, HF)     |

## Data

| File | Description |
|---|---|
| `Family_Law.pdf` | Bundle of five Egyptian personal-status statutes (each restarts article numbering from 1) |
| `Labor_Law.pdf` | 2025 Egyptian Labor Law |
| `Indexsheet.csv` | Legal index: maps topics/keywords to law name, part, and article range — used for query routing before retrieval |

The notebook also supports swapping the CSV index for a live Google Sheet via a
`USE_GOOGLE_SHEET` flag, for teams that want to update the index without touching code.

## Notable engineering challenges solved

Egyptian legal PDFs turned out to have several quirks that broke naive parsing —
documented here so they're not accidentally re-broken:

1. **Bundled statutes with restarting article numbers** — `Family_Law.pdf` contains
   five distinct laws, each starting again from Article 1. Internal segment markers
   in the text are used to detect boundaries and tag each chunk with the correct
   official law name.
2. **Reversed Eastern Arabic-Indic numerals** — `Labor_Law.pdf` encodes some article
   numbers in Eastern Arabic-Indic digits that come out of PyMuPDF extraction
   reversed (e.g. Article 14 extracts as "٤١"). A dedicated parser detects and
   corrects this.
3. **Spelled-out ordinal article numbers** — part of the family law bundle uses
   ordinal words instead of digits, handled by a regex + word-to-integer lookup.
4. A post-extraction sanity check flags duplicate article numbers per law, catching
   any parsing regressions before they reach the index.

## Running it

1. Open `legal-ai-assistant-hybrid-rag.ipynb` in Google Colab
2. Set the runtime to **GPU** (Runtime → Change runtime type → T4/A100)
3. Upload `Family_Law.pdf`, `Labor_Law.pdf`, and `Indexsheet.csv` alongside the
   notebook when prompted
4. Run all cells top to bottom — the first cell installs all dependencies, no API
   keys needed

### Requirements (for local/reference use)

```
pip install -r requirements.txt
```

Note: the notebook is written for a Colab GPU runtime; running the 7B LLM locally
requires a GPU with enough VRAM (4-bit quantized, ~6GB+ recommended).

## Repository structure

```
.
├── legal-ai-assistant-hybrid-rag.ipynb   # Main deliverable — full pipeline
├── Family_Law.pdf                        # Source legislation (5 bundled statutes)
├── Labor_Law.pdf                         # Source legislation (2025 Labor Law)
├── Indexsheet.csv                        # Legal index used for query routing
├── requirements.txt
└── README.md
```

## Status

- Core pipeline (parsing, indexing, hybrid retrieval, re-ranking, generation,
  memory) is complete and tested end-to-end
- Evaluation against the course's required test questions is in progress
- Google Sheets integration is implemented behind a flag but not yet the default path

## License

MIT — see `LICENSE`.
