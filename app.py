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
    # Ігноруємо фрагменти, що складаються лише з точок, цифр та пробілів (лідери змісту)
    return not clean or re.fullmatch(r"[.\d\s\u2026_]+", clean) is not None

def font_is_bold(font_name: str) -> bool:
    if not font_name:
        return False
    # Ігноруємо загальні назви сімейств, шукаємо саме вказівку на жирність після дефісу або в кінці
    name_low = font_name.lower()
    if any(k in name_low for k in ["bold", "black", "heavy", "demi"]):
        # Але переконуємося, що це не просто назва шрифту "Bold" (буває і таке)
        return True
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
                
                # Визначаємо жирність конкретного спана
                bold_flag = bool(s["flags"] & 4)
                bold_name = font_is_bold(s["font"])
                
                all_spans.append({
                    "text": txt,
                    "x": s["bbox"][0],
                    "y": s["bbox"][1],
                    "font_size": s["size"],
                    "is_bold": bold_flag or bold_name
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
        
        # Рядок вважається жирним, якщо ТЕКСТОВІ спани жирні
        # (ігноруємо жирність крапок або номерів сторінок в кінці)
        text_spans = [s for s in content_spans if re.search(r"[А-Яа-яІіЄєЇїҐґA-Za-z]", s["text"])]
        is_row_bold = any(s["is_bold"] for s in text_spans) if text_spans else any(s["is_bold"] for s in content_spans)

        raw_rows.append({
            "text": full_text,
            "clean": normalize_text(full_text),
            "x": min(s["x"] for s in content_spans),
            "y": bucket["y"],
            "font_size": mean(s["font_size"] for s in content_spans),
            "is_bold": is_row_bold
        })

    # Об'єднуємо розірвані рядки змісту (якщо пункт займає 2 рядки)
    merged_rows = []
    for row in raw_rows:
        # Якщо рядок не починається з ВСТУП/РОЗДІЛ/цифр і є дуже близьким до попереднього
        starts_new = re.match(r"^(ВСТУП|РОЗДІЛ|ВИСНОВКИ|ДОДАТКИ|СПИСОК|\d+\.\d+)", row["clean"], re.I)
        if not starts_new and merged_rows:
            prev = merged_rows[-1]
            if abs(row["y"] - prev["y"]) < 25: # Допуск для міжрядкового інтервалу
                prev["text"] += " " + row["text"]
                prev["clean"] = normalize_text(prev["text"])
                prev["is_bold"] = prev["is_bold"] or row["is_bold"]
                continue
        merged_rows.append(row)

    return merged_rows, page_width

def analyze_zmist(rows: list[dict[str, Any]], page_width: float) -> dict[str, Any]:
    findings: list[str] = []
    if not rows: return {"summary": "Текст не знайдено", "findings": ["Порожня сторінка"], "is_success": False}

    title_found = any("ЗМІСТ" in r["clean"].upper() for r in rows)
    if not title_found: findings.append('Не знайдено заголовок "ЗМІСТ".')

    major_pattern = r"^(ВСТУП|РОЗДІЛ\s+\d+|ВИСНОВКИ|ДОДАТКИ|СПИСОК ВИКОРИСТАНИХ ДЖЕРЕЛ)"
    
    major_rows = []
    sub_rows = []
    
    for row in rows:
        c = row["clean"].upper()
        if re.match(major_pattern, c):
            major_rows.append(row)
        elif re.match(r"^\d+\.\d+", c):
            sub_rows.append(row)

    if not major_rows: findings.append("Не знайдено основні розділи (ВСТУП, РОЗДІЛ...).")

    for row in major_rows:
        # Отримуємо частину тексту ДО крапок або цифр в кінці
        text_part = re.split(r"[\.\u2026_]{3,}", row["text"])[0].strip()
        # Тільки літери
        letters = re.sub(r"[^А-ЯІЄЇҐA-Z]", "", text_part)
        if letters and letters != letters.upper():
            findings.append(f'Пункт "{text_part[:30]}..." має бути ВЕЛИКИМИ ЛІТЕРАМИ.')
        
        if not row["is_bold"]:
            findings.append(f'Розділ "{text_part[:30]}..." має бути ЖИРНИМ.')

    for row in sub_rows:
        if row["is_bold"]:
            findings.append(f'Підпункт "{row["clean"][:30]}..." не повинен бути жирним.')

    return {
        "summary": "Перевірку змісту завершено.",
        "findings": findings,
        "is_success": len(findings) == 0,
        "metrics": {"major": len(major_rows), "sub": len(sub_rows)}
    }

def analyze_page_numbers(doc: fitz.Document) -> dict[str, Any]:
    findings = []
    for page_num in [1, 2, 3]:
        if page_num > len(doc): continue
        page = doc[page_num - 1]
        rect = page.rect
        search_rect = fitz.Rect(rect.width - 60, 0, rect.width, 60)
        words = page.get_text("words")
        digits = "".join(re.sub(r"[^\d]", "", w[4]) for w in words if fitz.Rect(w[:4]).intersects(search_rect))
        if page_num == 1 and digits: findings.append(f"На титульній сторінці знайдено цифри ({digits}).")
        elif page_num == 2 and digits: findings.append(f"На сторінці змісту знайдено цифри ({digits}).")
        elif page_num == 3:
            if not digits: findings.append("На 3-й сторінці не знайдено номер.")
            elif digits != "3": findings.append(f"На 3-й сторінці знайдено '{digits}' замість '3'.")
    return {"summary": "Нумерація перевірена.", "findings": findings, "is_success": len(findings) == 0}

def analyze_general_text(doc: fitz.Document) -> dict[str, Any]:
    findings = []
    highlights = []
    CM = 28.346
    MARGIN_TOLERANCE = 0.5 * CM
    TARGET_LEFT = 2.5 * CM
    TARGET_RIGHT = 1.0 * CM
    TARGET_TOP = 2.0 * CM
    TARGET_BOTTOM = 2.0 * CM
    TARGET_INDENT = 1.5 * CM
    
    def is_major_heading(text: str) -> bool:
        clean = re.sub(r"\s+", " ", text.strip())
        if bool(re.match(r"^(ВСТУП|РОЗДІЛ\s+\d+|ВИСНОВКИ|ДОДАТКИ|СПИСОК ВИКОРИСТАНИХ ДЖЕРЕЛ)", clean.upper())):
            return True
        letters_only = re.sub(r"[^А-ЯІЄЇҐA-Z]", "", clean)
        if len(letters_only) >= 3 and letters_only == letters_only.upper() and clean == clean.upper():
            return True
        return False

    def get_page_major_heading(page) -> str | None:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" not in b: continue
            for l in b["lines"]:
                text = "".join(s["text"] for s in l["spans"]).strip()
                if is_major_heading(text): return text
        return None

    stop_page = len(doc) + 1
    for i in range(2, len(doc)):
        heading = get_page_major_heading(doc[i])
        if heading and "ДОДАТКИ" in heading:
            stop_page = i + 1
            break

    pages_with_errors = set()
    for page_num in range(3, stop_page):
        page = doc[page_num - 1]
        width, height = page.rect.width, page.rect.height
        tables = page.find_tables()
        table_bboxes = [fitz.Rect(t.bbox) for t in tables.tables]
        img_info = page.get_image_info()
        img_bboxes = [fitz.Rect(img["bbox"]) for img in img_info]

        blocks = page.get_text("dict")["blocks"]
        text_lines = []
        for b in blocks:
            if "lines" not in b: continue
            block_rect = fitz.Rect(b["bbox"])
            if any(block_rect.intersects(t_bbox) for t_bbox in table_bboxes): continue
            if any(block_rect.intersects(i_bbox) for i_bbox in img_bboxes): continue
            for l in b["lines"]:
                text = "".join(s["text"] for s in l["spans"]).strip()
                if not text: continue

                # Ігноруємо номери сторінок (тільки цифри у верхній або нижній частині)
                if text.isdigit() and (l["bbox"][1] < 60 or l["bbox"][3] > height - 60):
                    continue

                # Ігноруємо Таблиця ..., Рис ...
                if re.match(r"^(Рис|Табл|Рисунок|Таблиця|Джерело)\.?\s*", text, re.I): continue
                # Ігноруємо формули (рядки з математичними символами та малою кількістю літер)
                letters_count = len(re.sub(r"[^А-Яа-яІіЄєЇїҐґA-Za-z]", "", text))
                if letters_count < 10 and any(c in text for c in "=+-*/∑∫√^"): continue
                
                l["text_content"] = text
                line_center = (l["bbox"][0] + l["bbox"][2]) / 2
                page_center = width / 2
                l["is_centered"] = abs(line_center - page_center) < 40 and (l["bbox"][2] - l["bbox"][0]) < width * 0.7
                first_span = l["spans"][0]
                l["is_bold"] = bool(first_span["flags"] & 4) or font_is_bold(first_span["font"])
                l["starts_with_digit"] = text[0].isdigit()

                text_lines.append(l)

        if not text_lines: continue

        # Ліве поле рахуємо по звичайному тексту (не заголовки, не формули, не центровані)
        left_margin_lines = [l for l in text_lines if not (l.get("is_centered") or l.get("is_bold") or l.get("starts_with_digit")) and (l["bbox"][2] - l["bbox"][0]) > width * 0.4]
        
        ref_left = left_margin_lines if left_margin_lines else text_lines
        actual_left = min(l["bbox"][0] for l in ref_left)
        actual_right = width - max(l["bbox"][2] for l in ref_left)
        actual_top = min(l["bbox"][1] for l in text_lines)
        elements_y = [l["bbox"][3] for l in text_lines]
        elements_y.extend([t.y1 for t in table_bboxes])
        elements_y.extend([img.y1 for img in img_bboxes])
        actual_bottom_y = max(elements_y) if elements_y else height
        actual_bottom = height - actual_bottom_y

        page_findings = []
        if abs(actual_left - TARGET_LEFT) > MARGIN_TOLERANCE:
            page_findings.append(f"Ліве поле {actual_left/CM:.1f} см замість 2.5 см")
            highlights.append({"page": page_num, "x": 0, "y": 0, "w": actual_left, "h": height})
        if actual_right < TARGET_RIGHT - MARGIN_TOLERANCE:
            page_findings.append(f"Праве поле {actual_right/CM:.1f} см замість 1.0 см")
            highlights.append({"page": page_num, "x": width - actual_right, "y": 0, "w": actual_right, "h": height})
        if abs(actual_top - TARGET_TOP) > MARGIN_TOLERANCE:
            page_findings.append(f"Верхнє поле {actual_top/CM:.1f} см замість 2.0 см")
            highlights.append({"page": page_num, "x": 0, "y": 0, "w": width, "h": actual_top})
        
        # Порожнє місце знизу
        if actual_bottom > 4.5 * CM:
            is_section_end = False
            if page_num < len(doc):
                next_page_heading = get_page_major_heading(doc[page_num])
                if next_page_heading: is_section_end = True
            if not is_section_end:
                page_findings.append(f"Порожнє місце знизу ({actual_bottom/CM:.1f} см).")
                highlights.append({"page": page_num, "x": 0, "y": actual_bottom_y, "w": width, "h": actual_bottom})

        # Відступи абзаців
        indents = []
        for i, l in enumerate(text_lines):
            # Тільки для довгих рядків основного тексту
            if not (l.get("is_centered") or l.get("is_bold") or l.get("starts_with_digit")) and (l["bbox"][2] - l["bbox"][0]) > width * 0.5:
                indent = l["bbox"][0] - actual_left
                if 10 < indent < 80: # Нормальний діапазон для абзацу
                    indents.append(indent)
                elif indent > 80: # Аномально великий відступ
                    page_findings.append(f"Аномальний відступ рядка ({indent/CM:.1f} см)")

        if indents:
            avg_indent = sum(indents) / len(indents)
            if abs(avg_indent - TARGET_INDENT) > MARGIN_TOLERANCE:
                page_findings.append(f"Відступ першого рядка {avg_indent/CM:.1f} см замість 1.5 см")

        if page_findings:
            pages_with_errors.add(page_num)
            for f in page_findings: findings.append(f"Сторінка {page_num}: {f}")

    return {"summary": "Текст перевірено.", "findings": findings, "is_success": len(findings) == 0, "pages_with_errors": sorted(list(pages_with_errors)), "highlights": highlights}

@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    try:
        pdf_bytes = base64.b64decode(request.pdf_base64)
    except Exception as exc: raise HTTPException(status_code=400, detail="Invalid base64 PDF payload.")
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc: raise HTTPException(status_code=400, detail="Unable to parse PDF.")
    try:
        if request.analysis_type == "page_numbers": result = analyze_page_numbers(doc)
        elif request.analysis_type == "general_text": result = analyze_general_text(doc)
        elif request.analysis_type == "zmist":
            rows, page_width = extract_page_rows_fitz(doc, request.page_number)
            result = analyze_zmist(rows, page_width)
        else: result = {"summary": "Невідомий тип", "findings": [], "is_success": False}
        result["analysis_type"] = request.analysis_type
        result["page_number"] = request.page_number
        return result
    finally: doc.close()
