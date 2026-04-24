from __future__ import annotations

import base64
import io
import re
import logging
from statistics import mean
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import fitz

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF Analyzer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

@app.get("/")
def read_root():
    return {"message": "PDF Analyzer API is running"}

class AnalyzeRequest(BaseModel):
    pdf_base64: str = Field(..., min_length=100)
    analysis_type: str = Field(default="zmist")
    page_number: int = Field(default=2, ge=1)

def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()

def is_leader_fragment(text: str) -> bool:
    clean = normalize_text(text)
    return not clean or re.fullmatch(r"[.\d\s\u2026_]+", clean) is not None

def font_is_bold(font_name: str) -> bool:
    return False

def extract_page_rows_fitz(doc: fitz.Document, page_number: int) -> tuple[list[dict[str, Any]], float]:
    if page_number > len(doc):
        raise HTTPException(status_code=400, detail=f"PDF has only {len(doc)} page(s).")

    page = doc[page_number - 1]
    page_width = page.rect.width
    blocks = page.get_text("dict")["blocks"]
    
    all_spans = []
    for b in blocks:
        if "lines" not in b: continue
        for l in b["lines"]:
            for s in l["spans"]:
                txt = s["text"].strip()
                if not txt: continue
                is_bold_flag = bool(s["flags"] & 16)
                is_bold_name = any(k in s["font"].lower() for k in ["bold", "black", "heavy"])
                all_spans.append({
                    "text": txt, "x": s["bbox"][0], "y": s["bbox"][1], "font_size": s["size"], "is_bold": is_bold_flag or is_bold_name
                })

    if not all_spans: return [], page_width

    buckets = []
    for span in sorted(all_spans, key=lambda x: (x["y"], x["x"])):
        row = next((item for item in buckets if abs(item["y"] - span["y"]) < 5), None)
        if row is None:
            row = {"y": span["y"], "spans": []}
            buckets.append(row)
        row["spans"].append(span)

    raw_rows = []
    for bucket in buckets:
        line_spans = sorted(bucket["spans"], key=lambda x: x["x"])
        content_spans = [s for s in line_spans if not is_leader_fragment(s["text"])]
        if not content_spans: continue
        full_text = " ".join(s["text"] for s in content_spans)
        text_spans = [s for s in content_spans if re.search(r"[А-Яа-яІіЄєЇїҐґA-Za-z]", s["text"])]
        is_row_bold = any(s["is_bold"] for s in text_spans) if text_spans else any(s["is_bold"] for s in content_spans)
        raw_rows.append({
            "text": full_text, "clean": normalize_text(full_text), "x": min(s["x"] for s in content_spans), "y": bucket["y"],
            "font_size": mean(s["font_size"] for s in content_spans), "is_bold": is_row_bold
        })
    return raw_rows, page_width

def analyze_zmist(rows: list[dict[str, Any]], page_width: float) -> dict[str, Any]:
    findings = []
    if not rows: return {"summary": "Текст не знайдено", "findings": ["Порожня сторінка"], "is_success": False}
    if not any("ЗМІСТ" in r["clean"].upper() for r in rows): findings.append('Не знайдено заголовок "ЗМІСТ".')
    major_pattern = r"^(ВСТУП|РОЗДІЛ\s+\d+|ВИСНОВКИ|ДОДАТКИ|СПИСОК ВИКОРИСТАНИХ ДЖЕРЕЛ)"
    for row in rows:
        c = row["clean"].upper()
        if re.match(major_pattern, c):
            text_part = re.split(r"[\.\u2026_]{3,}", row["text"])[0].strip()
            if not row["is_bold"]: findings.append(f'Розділ "{text_part[:30]}" має бути ЖИРНИМ.')
        elif re.match(r"^\d+\.\d+", c):
            text_part = re.split(r"[\.\u2026_]{3,}", row["text"])[0].strip()
            if row["is_bold"]: findings.append(f'Підпункт "{text_part[:30]}" не повинен бути жирним.')
    return {"summary": "Перевірку змісту завершено.", "findings": findings, "is_success": len(findings) == 0}

def analyze_page_numbers(doc: fitz.Document) -> dict[str, Any]:
    findings = []
    for page_num in [1, 2, 3]:
        if page_num > len(doc): continue
        page = doc[page_num - 1]
        rect = page.rect
        search_rect = fitz.Rect(rect.width - 60, 0, rect.width, 60)
        words = page.get_text("words")
        digits = "".join(re.sub(r"[^\d]", "", w[4]) for w in words if fitz.Rect(w[:4]).intersects(search_rect))
        if page_num == 3 and digits != "3": findings.append(f"На 3-й сторінці знайдено '{digits}' замість '3'.")
    return {"summary": "Нумерація перевірена.", "findings": findings, "is_success": len(findings) == 0}

def analyze_general_text(doc: fitz.Document) -> dict[str, Any]:
    findings = []
    highlights = []
    CM = 28.346
    MARGIN_TOLERANCE = 0.5 * CM
    TARGET_LEFT, TARGET_RIGHT, TARGET_TOP, TARGET_BOTTOM = 2.5*CM, 1.0*CM, 2.0*CM, 2.0*CM
    
    def get_page_first_line_info(page) -> dict | None:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" not in b: continue
            for l in b["lines"]:
                text = "".join(s["text"] for s in l["spans"]).strip()
                if not text or (text.isdigit() and l["bbox"][1] < 60): continue
                return {"text": text, "is_bold": bool(l["spans"][0]["flags"] & 16), "starts_with_digit": text[0].isdigit()}
        return None

    pages_with_errors = set()
    for page_num in range(3, len(doc) + 1):
        page = doc[page_num-1]
        width, height = page.rect.width, page.rect.height
        blocks = page.get_text("dict")["blocks"]
        all_elements = []
        for b in blocks:
            if "lines" not in b: continue
            for l in b["lines"]:
                text = "".join(s["text"] for s in l["spans"]).strip()
                if text and not (text.isdigit() and l["bbox"][1] < 60): all_elements.append(l)
        if not all_elements: continue
        
        actual_top = min(l["bbox"][1] for l in all_elements)
        actual_bottom = height - max(l["bbox"][3] for l in all_elements)
        actual_left = min(l["bbox"][0] for l in all_elements)
        actual_right = width - max(l["bbox"][2] for l in all_elements)
        
        p_findings = []
        if abs(actual_left - TARGET_LEFT) > MARGIN_TOLERANCE: p_findings.append("Помилка лівого поля")
        if actual_bottom > 4.5 * CM:
            is_valid = False
            if page_num < len(doc):
                next_info = get_page_first_line_info(doc[page_num])
                if next_info and next_info["is_bold"] and not next_info["starts_with_digit"]: is_valid = True
            if not is_valid: p_findings.append("Порожнє місце знизу")
        
        if p_findings:
            pages_with_errors.add(page_num)
            for f in p_findings: findings.append(f"Стор. {page_num}: {f}")
            
    return {"summary": "Текст перевірено.", "findings": findings, "is_success": len(findings) == 0, "pages_with_errors": sorted(list(pages_with_errors)), "highlights": highlights}

def analyze_chapters(doc: fitz.Document) -> dict[str, Any]:
    findings = []
    pages_with_errors = set()
    for page_num in range(3, len(doc) + 1):
        page = doc[page_num-1]
        blocks = page.get_text("dict")["blocks"]
        lines = []
        for b in blocks:
            if "lines" not in b: continue
            for l in b["lines"]:
                text = "".join(s["text"] for s in l["spans"]).strip()
                if text and not (text.isdigit() and l["bbox"][1] < 60): lines.append(l)
        if not lines: continue
        for idx, line in enumerate(lines):
            text = "".join(s["text"] for s in line["spans"]).strip()
            if re.match(r"^РОЗДІЛ\s+\d+", text.upper()):
                p_f = []
                if idx > 0 and lines[idx-1]["bbox"][1] > 100: p_f.append("РОЗДІЛ не з нової сторінки")
                line_center = (line["bbox"][0] + line["bbox"][2]) / 2
                if abs(line_center - page.rect.width/2) > 45: p_f.append("РОЗДІЛ не по центру")
                if text != text.upper() or not bool(line["spans"][0]["flags"] & 16): p_f.append("РОЗДІЛ має бути ВЕЛИКИМИ ЖИРНИМИ")
                if re.search(r"РОЗДІЛ\s+\d+\.", text): p_f.append("Крапка після номера розділу")
                if idx + 1 < len(lines):
                    next_l = lines[idx+1]
                    next_t = "".join(s["text"] for s in next_l["spans"]).strip()
                    if abs((next_l["bbox"][0]+next_l["bbox"][2])/2 - page.rect.width/2) > 45: p_f.append("Назва не по центру")
                    if next_t != next_t.upper() or not bool(next_l["spans"][0]["flags"] & 16): p_f.append("Назва має бути ВЕЛИКИМИ ЖИРНИМИ")
                    if next_t.endswith("."): p_f.append("Крапка в кінці назви")
                    if idx + 2 < len(lines) and (lines[idx+2]["bbox"][1] - next_l["bbox"][3]) < 20: p_f.append("Відсутній порожній рядок після назви")
                else: p_f.append("Не знайдено назву розділу")
                if p_f:
                    pages_with_errors.add(page_num)
                    for f in p_f: findings.append(f"Стор. {page_num}: {f}")
    return {"summary": "Розділи перевірено.", "findings": findings, "is_success": len(findings) == 0, "pages_with_errors": sorted(list(pages_with_errors))}

def analyze_subchapters(doc: fitz.Document) -> dict[str, Any]:
    findings = []
    pages_with_errors = set()
    CM = 28.346
    for page_num in range(3, len(doc) + 1):
        page = doc[page_num-1]
        blocks = page.get_text("dict")["blocks"]
        lines = []
        for b in blocks:
            if "lines" not in b: continue
            for l in b["lines"]:
                text = "".join(s["text"] for s in l["spans"]).strip()
                if text: lines.append(l)
        for idx, line in enumerate(lines):
            text = "".join(s["text"] for s in line["spans"]).strip()
            if re.match(r"^\d+\.\d+\s+", text):
                p_f = []
                if not bool(line["spans"][0]["flags"] & 16): p_f.append("Підрозділ не жирний")
                indent = line["bbox"][0] - 2.5*CM
                if abs(indent - 1.5*CM) > 0.5*CM: p_f.append("Відступ не 1.5 см")
                # Порожня строка зверху (тільки якщо не початок сторінки)
                if idx > 0 and (line["bbox"][1] - lines[idx-1]["bbox"][3]) < 20: p_f.append("Відсутній рядок зверху")
                if idx + 1 < len(lines) and (lines[idx+1]["bbox"][1] - line["bbox"][3]) < 20: p_f.append("Відсутній рядок знизу")
                if p_f:
                    pages_with_errors.add(page_num)
                    for f in p_f: findings.append(f"Стор. {page_num}: {f}")
    return {"summary": "Підрозділи перевірено.", "findings": findings, "is_success": len(findings) == 0, "pages_with_errors": sorted(list(pages_with_errors))}

@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    try:
        pdf_bytes = base64.b64decode(request.pdf_base64)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if request.analysis_type == "page_numbers": res = analyze_page_numbers(doc)
        elif request.analysis_type == "general_text": res = analyze_general_text(doc)
        elif request.analysis_type == "chapters": res = analyze_chapters(doc)
        elif request.analysis_type == "subchapters": res = analyze_subchapters(doc)
        elif request.analysis_type == "zmist":
            rows, pw = extract_page_rows_fitz(doc, request.page_number)
            res = analyze_zmist(rows, pw)
        else: res = {"summary": "Unknown type", "findings": [], "is_success": False}
        res["analysis_type"] = request.analysis_type
        return res
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
    finally: doc.close()
