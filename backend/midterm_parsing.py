"""
midterm_parsing.py

Article-level PDF splitting logic, carried over UNCHANGED from the
midterm "Arabic Legal AI Assistant" notebook, where it was tested
against the real Family_Law.pdf / Labor_Law.pdf and validated to
correctly handle:
  1. bundled statutes whose article numbering restarts partway through
     the PDF (the family-law compilation),
  2. reversed Eastern Arabic-Indic numerals in the labor-law PDF export,
  3. spelled-out ordinal article references ("مكرر", "مكرر ثانيا", ...).

Kept as its own module so rag_pipeline.py stays focused on
orchestration (routing, retrieval, generation) rather than parsing.
"""

import os
import re
import unicodedata
from typing import Dict, List

DIGIT_TRANS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

ARTICLE_PATTERN = re.compile(
    r"(?:^|\n)[\s\)\(]*م\s*ا\s*د\s*[ةه][\s\)\(:]*([0-9٠-٩۰-۹]+)"
    r"[\s\)\(]*(مكرر[اً]*(?:\s*(?:ثانيا|ثالثا|رابعا))?)?",
    re.MULTILINE)

REVERSE_DIGITS = {
    "قانون العمل": True,
    "قانون الأحوال الشخصية": False,
}

PART_NAMES = {
    "قانون الأحوال الشخصية": {
        1: "قانون رقم 25 لسنة 1920 (النفقة)",
        2: "قانون رقم 25 لسنة 1929 (أحكام الأحوال الشخصية)",
        3: "قانون رقم 1 لسنة 2000 (إجراءات التقاضي)",
        4: "قرار وزير العدل 1086 لسنة 2000",
        5: "قرار وزير العدل 1087 لسنة 2000 (الرؤية)",
        6: "قرار وزير العدل 1088 لسنة 2000 (الجرد)",
        7: "قرار وزير العدل 1089 لسنة 2000 (الأخصائيون)",
        8: "قرار وزير العدل 1090 لسنة 2000 (السجل)",
        9: "قانون رقم 10 لسنة 2004 (محاكم الأسرة)",
    }
}


def fix_reversed_digits(text: str) -> str:
    return re.sub(r"[٠-٩]+", lambda m: m.group(0)[::-1], text)


def extract_pdf_text(fitz, path: str, reverse_digits: bool = False) -> str:
    doc = fitz.open(path)
    pages = [page.get_text() for page in doc]
    doc.close()
    text = "\n".join(pages)
    text = unicodedata.normalize("NFKC", text)
    if reverse_digits:
        text = fix_reversed_digits(text)
    return text


def _candidates(tok: str) -> set:
    opts = [""]
    for ch in tok:
        subs = ["1", "9", "0"] if ch == "1" else [ch]
        opts = [p + s for p in opts for s in subs]
    opts += [o[::-1] for o in opts]
    return {int(o) for o in opts if o and not o.startswith("0")}


def repair_article_numbers(matches) -> list:
    out, prev, part = [], 0, 1
    for m in matches:
        tok = m.group(1).translate(DIGIT_TRANS)
        suffix = (m.group(2) or "").strip()
        cands = _candidates(tok)
        if suffix and prev in cands:
            num = prev
        elif prev + 1 in cands:
            num = prev + 1
        elif prev + 2 in cands:
            num = prev + 2
        elif 1 in cands and prev >= 2:
            num, part = 1, part + 1
        elif prev in cands:
            num = prev
        else:
            num = min(cands)
        prev = num
        out.append((str(num) + ((" " + suffix) if suffix else ""), part))
    return out


def split_into_articles(Document, law_name: str, full_text: str) -> List:
    matches = list(ARTICLE_PATTERN.finditer(full_text))
    docs = []
    if not matches:
        return [Document(page_content=full_text,
                          metadata={"law_name": law_name, "article": None, "source": law_name})]
    if matches[0].start() > 0:
        preamble = full_text[: matches[0].start()].strip()
        if len(preamble) > 30:
            docs.append(Document(page_content=preamble,
                                  metadata={"law_name": law_name, "article": "مقدمة", "source": law_name}))
    labels = repair_article_numbers(matches)
    part_names = PART_NAMES.get(law_name, {})
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        chunk_text = full_text[start:end].strip()
        if len(chunk_text) < 5:
            continue
        article_no, part = labels[i]
        sub_law = part_names.get(part, "")
        header = f"[المادة {article_no}"
        header += f" — {sub_law}]" if sub_law else "]"
        docs.append(Document(
            page_content=header + "\n" + chunk_text,
            metadata={"law_name": law_name, "article": article_no, "sub_law": sub_law, "source": law_name},
        ))
    return docs


def build_law_documents(law_files: Dict[str, str], Document) -> Dict[str, List]:
    import fitz  # imported here so this module is importable without PyMuPDF in mock mode
    law_docs = {}
    for law_name, path in law_files.items():
        if not os.path.exists(path):
            print(f"[midterm_parsing] Missing file for '{law_name}': {path}")
            continue
        text = extract_pdf_text(fitz, path, reverse_digits=REVERSE_DIGITS.get(law_name, False))
        articles = split_into_articles(Document, law_name, text)
        law_docs[law_name] = articles
        print(f"[midterm_parsing] {law_name}: {len(articles)} article-level chunks from {path}")
    return law_docs
