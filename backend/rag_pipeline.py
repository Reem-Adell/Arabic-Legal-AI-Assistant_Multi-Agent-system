"""
rag_pipeline.py

This is the midterm "Arabic Legal AI Assistant" hybrid RAG pipeline
(BM25 + FAISS + BAAI/bge-m3 embeddings + BAAI/bge-reranker-v2-m3 +
Qwen2.5-7B-Instruct + Legal-Index routing), repackaged as an
importable class instead of notebook cells, so main.py (FastAPI)
can call it directly.

Run modes
---------
1. REAL mode (default): loads the actual models. Needs a GPU
   (Colab T4/L4 recommended) and the same PDFs / Indexsheet.csv
   used in the midterm project. Put them under backend/data/ or
   set the *_PATH environment variables below.

2. MOCK mode: set the environment variable LEGAL_RAG_MOCK=1.
   No models are loaded; answer_query() returns a clearly-labelled
   placeholder answer plus a routed law name, so the FastAPI app,
   the multi-agent orchestrator, and the frontend can all be
   developed/tested on a laptop with no GPU. This is an EXPLICIT
   switch (never a silent fallback) — the app prints which mode it
   is running in on startup.
"""

import os
import re
import glob
from dataclasses import dataclass
from typing import List, Dict, Optional

MOCK_MODE = os.environ.get("LEGAL_RAG_MOCK", "0") == "1"

DATA_DIR = os.environ.get("LEGAL_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
FAMILY_LAW_PATH = os.environ.get("FAMILY_LAW_PATH", os.path.join(DATA_DIR, "Family_Law.pdf"))
LABOR_LAW_PATH = os.environ.get("LABOR_LAW_PATH", os.path.join(DATA_DIR, "Labor_Law.pdf"))
INDEX_CSV_PATH = os.environ.get("INDEX_CSV_PATH", os.path.join(DATA_DIR, "Indexsheet.csv"))

LAW_FILES = {
    "قانون الأحوال الشخصية": FAMILY_LAW_PATH,
    "قانون العمل": LABOR_LAW_PATH,
}

# Maps a routed law name -> the lawyer specialization the office uses
# for booking. Kept close to the routing layer since it depends on it.
LAW_TO_SPECIALIZATION = {
    "قانون الأحوال الشخصية": "أحوال شخصية",
    "قانون العمل": "قانون العمل",
}

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

TOP_K_FINAL = 4
TOP_K_HYBRID = 10
ROUTING_CONFIDENCE_THRESHOLD = 0.35

REFUSAL_MESSAGE = (
    "لم أتمكن من العثور على نص قانوني في المستندات المتاحة يدعم الإجابة على هذا السؤال."
)

SYSTEM_INSTRUCTIONS = """أنت مساعد قانوني متخصص في القانون المصري.
اكتب إجابتك كاملة باللغة العربية الفصحى فقط. لا تستخدم أي لغة أخرى إطلاقاً.
أجب فقط بالاعتماد على النصوص القانونية المسترجعة أدناه. لا تخترع أي معلومة قانونية أبداً.
إذا لم تكن النصوص المسترجعة كافية للإجابة، أجب حرفياً بالعبارة التالية فقط:
"{refusal}"

عند الإجابة، التزم بالتنسيق التالي:
القانون: <اسم القانون>
المادة: <رقم المادة إن وجد>
الإجابة: <شرح موجز بالعربية بالاعتماد على النص فقط>
المصدر: <اسم القانون / المادة>
""".format(refusal=REFUSAL_MESSAGE)


@dataclass
class RagAnswer:
    answer_text: str
    routed_law: Optional[str]
    routed_subtopic: Optional[str]
    specialization: Optional[str]
    sources: List[str]


class LegalRAGPipeline:
    """Loads once at FastAPI startup and is reused across requests."""

    def __init__(self):
        self.ready = False
        self.embedding_model = None
        self.vector_stores = {}
        self.bm25_retrievers = {}
        self.reranker = None
        self.tokenizer = None
        self.llm_model = None
        self.legal_index_df = None
        self.legal_index_embeddings = None

    # ------------------------------------------------------------------
    def load(self):
        if MOCK_MODE:
            print("[rag_pipeline] LEGAL_RAG_MOCK=1 -> running in MOCK MODE "
                  "(no models loaded, canned answers only).")
            self.ready = True
            return

        print("[rag_pipeline] Loading real models — this needs a GPU and can take a few minutes...")
        import pandas as pd
        import numpy as np
        import fitz
        import torch
        from langchain_core.documents import Document
        from langchain_community.retrievers import BM25Retriever
        from langchain.retrievers import EnsembleRetriever
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from sentence_transformers import CrossEncoder
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        self._np = np
        self._Document = Document
        self._BM25Retriever = BM25Retriever
        self._EnsembleRetriever = EnsembleRetriever
        self._FAISS = FAISS
        self._torch = torch

        if not os.path.exists(INDEX_CSV_PATH):
            raise FileNotFoundError(
                f"Legal Index CSV not found at '{INDEX_CSV_PATH}'. "
                f"Upload Indexsheet.csv into backend/data/ (explicit failure, "
                f"no hardcoded fallback — same principle as the midterm project)."
            )
        self.legal_index_df = self._load_legal_index_from_csv(INDEX_CSV_PATH, pd)

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        law_documents = self._load_and_chunk_laws(fitz, Document)
        for law_name, docs in law_documents.items():
            if not docs:
                continue
            self.vector_stores[law_name] = FAISS.from_documents(docs, self.embedding_model)
            bm25 = BM25Retriever.from_documents(docs)
            bm25.k = TOP_K_HYBRID
            self.bm25_retrievers[law_name] = bm25
            print(f"[rag_pipeline] Built retrievers for {law_name} ({len(docs)} chunks).")

        legal_index_texts = (
            self.legal_index_df["subtopic"].fillna("") + " — " +
            self.legal_index_df["description"].fillna("") + " — " +
            self.legal_index_df["keywords"].fillna("")
        ).tolist()
        self.legal_index_embeddings = np.array(self.embedding_model.embed_documents(legal_index_texts))

        # Free the embedding model's GPU memory now that all indexes are
        # built. It's still used at query time (one short text per request),
        # so CPU inference here is fast enough and avoids the embedding
        # model, reranker, and 7B LLM all competing for GPU memory at once
        # — that combination is what caused the OOM crash during generate().
        if torch.cuda.is_available():
            self.embedding_model.client = self.embedding_model.client.to("cpu")
            torch.cuda.empty_cache()

        self.reranker = CrossEncoder(RERANKER_MODEL_NAME, max_length=512,
                                      device="cpu" if torch.cuda.is_available() else None)

        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        if torch.cuda.is_available():
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=torch.float16)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME, quantization_config=bnb, device_map="auto")
        else:
            self.llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, dtype=torch.float32)

        self.ready = True
        print("[rag_pipeline] Ready.")

    # ------------------------------------------------------------------
    # NOTE: chunking / index-loading logic below is carried over from the
    # midterm notebook (article-splitting, reversed-digit repair, spelled
    # out ordinal handling). Trimmed of markdown commentary; behavior
    # unchanged.
    def _load_legal_index_from_csv(self, path, pd):
        COLUMN_MAP = {
            "Main_Topic": "main_topic", "Subtopic": "subtopic", "Law_Name": "law_name",
            "Part": "chapter", "Keywords": "keywords", "Description": "description",
        }
        df = pd.read_csv(path, encoding="utf-8-sig").dropna(how="all")
        df.columns = [str(c).strip() for c in df.columns]
        df = df.rename(columns=COLUMN_MAP)

        def normalize_law_name(name):
            return "قانون العمل" if "عمل" in str(name) else "قانون الأحوال الشخصية"

        df["law_name_original"] = df["law_name"]
        df["law_name"] = df["law_name"].apply(normalize_law_name)
        if "article_range" not in df.columns:
            df["article_range"] = ""
        return df

    def _load_and_chunk_laws(self, fitz, Document):
        # Delegates to the same regex-based article splitter developed and
        # validated in the midterm notebook. Kept in a separate helper
        # module so this file stays focused on orchestration; see
        # midterm_parsing.py for the exact tested implementation.
        from midterm_parsing import build_law_documents
        return build_law_documents(LAW_FILES, Document)

    # ------------------------------------------------------------------
    def route_query(self, query: str, top_n: int = 2) -> List[dict]:
        if MOCK_MODE:
            return []
        np = self._np
        q_emb = np.array(self.embedding_model.embed_query(query)).reshape(1, -1)

        def cosine_sim(a, b):
            a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
            b_n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
            return a_n @ b_n.T

        sims = cosine_sim(q_emb, self.legal_index_embeddings)[0]
        order = np.argsort(-sims)[:top_n]
        matches = []
        for idx in order:
            if sims[idx] >= ROUTING_CONFIDENCE_THRESHOLD:
                row = self.legal_index_df.iloc[idx].to_dict()
                row["similarity"] = float(sims[idx])
                matches.append(row)
        return matches

    def _build_ensemble_retriever(self, law_name: str):
        if law_name not in self.vector_stores:
            return None
        dense = self.vector_stores[law_name].as_retriever(search_kwargs={"k": TOP_K_HYBRID})
        sparse = self.bm25_retrievers[law_name]
        return self._EnsembleRetriever(retrievers=[sparse, dense], weights=[0.5, 0.5])

    def hybrid_retrieve(self, query: str, routed: List[dict]):
        target_laws = list({r["law_name"] for r in routed}) if routed else list(self.vector_stores.keys())
        candidates, seen = [], set()
        for law_name in target_laws:
            ensemble = self._build_ensemble_retriever(law_name)
            if ensemble is None:
                continue
            for doc in ensemble.invoke(query):
                key = (doc.metadata.get("law_name"), doc.metadata.get("article"), doc.page_content[:50])
                if key not in seen:
                    seen.add(key)
                    candidates.append(doc)
        if not candidates:
            return []
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.reranker.predict(pairs)
        ranked = [d for _, d in sorted(zip(scores, candidates), key=lambda x: -x[0])]
        return ranked[:TOP_K_FINAL]

    def generate(self, prompt: str, max_new_tokens: int = 500) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=6000).to(self.llm_model.device)
        with self._torch.no_grad():
            output = self.llm_model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None, pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # ------------------------------------------------------------------
    def answer_query(self, query: str, history_text: str = "") -> RagAnswer:
        if MOCK_MODE:
            return self._mock_answer(query)

        routed = self.route_query(query)
        specialization, routed_law, routed_subtopic = None, None, None
        if routed:
            routed_law = routed[0]["law_name"]
            routed_subtopic = routed[0]["subtopic"]
            specialization = LAW_TO_SPECIALIZATION.get(routed_law)

        docs = self.hybrid_retrieve(query, routed)
        if not docs:
            return RagAnswer(REFUSAL_MESSAGE, routed_law, routed_subtopic, specialization, [])

        context = "\n\n".join(
            f"[{d.metadata.get('law_name', '')} - مادة {d.metadata.get('article', '')}] {d.page_content}"
            for d in docs
        )
        prompt = f"{SYSTEM_INSTRUCTIONS}\n\nسجل المحادثة:\n{history_text}\n\nالنصوص المسترجعة:\n{context}\n\nسؤال المستخدم: {query}"
        answer_text = self.generate(prompt)
        sources = [f"{d.metadata.get('law_name', '')} - مادة {d.metadata.get('article', '')}" for d in docs]
        return RagAnswer(answer_text, routed_law, routed_subtopic, specialization, sources)

    # ------------------------------------------------------------------
    def _mock_answer(self, query: str) -> RagAnswer:
        """Deterministic canned response used only when LEGAL_RAG_MOCK=1,
        purely so the FastAPI + agents + frontend can be built/tested
        without a GPU. Clearly labelled as mock, never silently used."""
        if any(k in query for k in ["عمل", "أجر", "إجازة", "فصل", "عامل"]):
            routed_law, subtopic = "قانون العمل", "إنهاء عقد العمل"
        else:
            routed_law, subtopic = "قانون الأحوال الشخصية", "الحضانة"
        specialization = LAW_TO_SPECIALIZATION.get(routed_law)
        answer_text = (
            f"[MOCK MODE — لا يوجد نموذج فعلي محمّل]\n"
            f"القانون: {routed_law}\nالمادة: (تجريبي)\n"
            f"الإجابة: هذه إجابة تجريبية لاختبار الواجهة والمسارات فقط.\n"
            f"المصدر: {routed_law} / {subtopic}"
        )
        return RagAnswer(answer_text, routed_law, subtopic, specialization, [f"{routed_law} (mock)"])


# Singleton used by main.py
pipeline = LegalRAGPipeline()
