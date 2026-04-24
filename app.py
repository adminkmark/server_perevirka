from __future__ import annotations

import base64
import io
import re
from statistics import mean
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import fitz


import logging

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
    return not clean or re.fullmatch(r"[.\d\s\u2026]+", clean) is not None

def font_is_bold(font_name: str) -> bool:
    if not font_name:
        return False
    return any(keyword in font_name.lower() for keyword in ["bold", "black", "demi", "heavy"])

def extract_page_rows_fitz(doc: fitz.Document, page_number: int) -> tuple[list[dict[str, Any]], float]:
    if page_number > len(doc):
        raise HTTPException(status_code=400, detail=f"PDF has only {len(doc)} page(s).")

    page = doc[page_number - 1]
    page_width = page.rect.width
    
    # Використовуємо dict для детального аналізу
    blocks = page.get_text("dict")["blocks"]
    rows: list[dict[str, Any]] = []

    all_spans = []
    for b in blocks:
        if "lines" not in b: continue
        for l in b["lines"]:
            for s in l["spans"]:
                txt = s["text"].strip()
                if not txt: continue
                all_spans.append({
                    "text": txt,
                    "x": s["bbox"][0],
                    "y": s["bbox"][1],
                    "h": s["bbox"][3] - s["bbox"][1],
                    "font_size": s["size"],
                    "font_name": s["font"],
                    "is_bold": bool(s["flags"] & 4) or font_is_bold(s["font"])
                })

    if not all_spans:
        return [], page_width

    # Групуємо в рядки за координатою Y
    buckets: list[dict[str, Any]] = []
    for span in sorted(all_spans, key=lambda x: (x["y"], x["x"])):
        # Допуск 3 пункти для одного рядка
        row = next((item for item in buckets if abs(item["y"] - span["y"]) < 4), None)
        if row is None:
            row = {"y": span["y"], "spans": []}
            buckets.append(row)
        row["spans"].append(span)

    for bucket in buckets:
        line_spans = sorted(bucket["spans"], key=lambda x: x["x"])
        # Відфільтровуємо лідери (точки в змісті)
        content_spans = [s for s in line_spans if not is_leader_fragment(s["text"])]
        
        if not content_spans:
            continue
            
        full_text = " ".join(s["text"] for s in content_spans)
        clean_text = normalize_text(full_text)
        
        if not clean_text:
            continue

        rows.append({
            "text": full_text,
            "clean": clean_text,
            "x": min(s["x"] for s in content_spans),
            "y": bucket["y"],
            "font_size": mean(s["font_size"] for s in content_spans),
            "is_bold": any(s["is_bold"] for s in content_spans),
            "spans": content_spans
        })

    return rows, page_width


def analyze_zmist(rows: list[dict[str, Any]], page_width: float) -> dict[str, Any]:
    findings: list[str] = []

    if not rows:
        return {"summary": "Перевірка змісту не виконана.", "findings": ["Не вдалося зчитати текст сторінки."]}

    # Пошук заголовка "ЗМІСТ"
    title_row = next((row for row in rows if "ЗМІСТ" in row["clean"].upper()), None)
    if not title_row:
        findings.append('Не знайдено заголовок "ЗМІСТ".')

    # Основні пункти: ВСТУП, РОЗДІЛ...
    major_pattern = r"^(ВСТУП|РОЗДІЛ\s+\d+|ВИСНОВКИ|ДОДАТКИ|СПИСОК ВИКОРИСТАНИХ ДЖЕРЕЛ)"
    major_rows = [row for row in rows if re.match(major_pattern, row["clean"], re.IGNORECASE)]
    
    # Підпункти: 1.1, 1.2...
    sub_rows = [row for row in rows if re.match(r"^\d+\.\d+", row["clean"])]

    if not major_rows:
        findings.append("Не знайдено основні пункти змісту (ВСТУП, РОЗДІЛ...).")

    for row in major_rows:
        # Перевірка на ВЕЛИКІ ЛІТЕРИ для основних пунктів
        label = re.split(r"[.\u2026]", row["clean"])[0].strip()
        letters_only = re.sub(r"[0-9«»\"'.\-–—,:; ]", "", label)
        if letters_only and letters_only != letters_only.upper():
            findings.append(f'Пункт "{label}" має бути великими літерами.')

    # Перевірка на жирність основних пунктів
    weak_bold_rows = [row["clean"] for row in major_rows if not row["is_bold"]]
    if weak_bold_rows:
        labels = [f'"{item}"' for item in weak_bold_rows[:3]]
        suffix = " та інші" if len(weak_bold_rows) > 3 else ""
        findings.append(f"Основні пункти змісту мають бути жирними: {', '.join(labels)}{suffix}.")

    # Підпункти не мають бути жирними
    bold_sub_rows = [row["clean"] for row in sub_rows if row["is_bold"]]
    if bold_sub_rows:
        labels = [f'"{item}"' for item in bold_sub_rows[:3]]
        suffix = " та інші" if len(bold_sub_rows) > 3 else ""
        findings.append(f"Підпункти не мають бути жирними: {', '.join(labels)}{suffix}.")

    # Розмір шрифту (беремо середнє по перших пунктах)
    relevant_rows = [row for row in rows if not any(x in row["clean"].upper() for x in ["ЗМІСТ", "СТОР"])]
    if relevant_rows:
        avg_font = mean(row["font_size"] for row in relevant_rows[:10])
        if avg_font < 12.5 or avg_font > 15.5: # Цільовий 14pt
            findings.append(f"Розмір шрифту в змісті виглядає як {avg_font:.1f} pt замість 14 pt.")

    summary = (
        f"Виявлено {len(findings)} відхилення(нь) у змісті."
        if findings
        else "Суттєвих відхилень у змісті не виявлено."
    )

    return {
        "summary": summary,
        "findings": findings,
        "is_success": len(findings) == 0,
        "metrics": {
            "page_width": page_width,
            "major_rows": len(major_rows),
            "sub_rows": len(sub_rows),
        }
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
                if re.match(r"^(Рис|Табл|Рисунок|Таблиця)\.?\s+\d+", text, re.I): continue
                if re.fullmatch(r"\d+", text) and (l["bbox"][1] < 100 or l["bbox"][3] > height - 100): continue
                l["text_content"] = text
                line_center = (l["bbox"][0] + l["bbox"][2]) / 2
                page_center = width / 2
                l["is_centered"] = abs(line_center - page_center) < 40 and (l["bbox"][2] - l["bbox"][0]) < width * 0.7
                first_span = l["spans"][0]
                first_char = first_span["text"].strip()[0] if first_span["text"].strip() else ""
                l["is_bold_or_digit"] = bool(first_span["flags"] & 4) or font_is_bold(first_span["font"]) or first_char.isdigit()
                text_lines.append(l)

        if not text_lines: continue

        left_margin_lines = [l for l in text_lines if not (l.get("is_centered") or l.get("is_bold_or_digit")) and (l["bbox"][2] - l["bbox"][0]) > width * 0.3]
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
        if actual_bottom > 4.5 * CM:
            is_section_end = False
            if page_num < len(doc):
                next_page_heading = get_page_major_heading(doc[page_num])
                if next_page_heading: is_section_end = True
            current_heading = get_page_major_heading(page)
            if current_heading and "ДОДАТКИ" in current_heading.upper(): is_section_end = True
            if not is_section_end:
                page_findings.append(f"Порожнє місце знизу ({actual_bottom/CM:.1f} см). Не повинно бути.")
                highlights.append({"page": page_num, "x": 0, "y": actual_bottom_y, "w": width, "h": actual_bottom})
        elif actual_bottom < TARGET_BOTTOM - MARGIN_TOLERANCE:
            page_findings.append(f"Нижнє поле {actual_bottom/CM:.1f} см замість 2.0 см")
            highlights.append({"page": page_num, "x": 0, "y": actual_bottom_y, "w": width, "h": actual_bottom})

        indents = []
        spacings = []
        for i, l in enumerate(text_lines):
            if not (l.get("is_centered") or l.get("is_bold_or_digit")):
                indent = l["bbox"][0] - actual_left
                if indent > 10: indents.append(indent)
            if i > 0:
                prev_l = text_lines[i-1]
                if l["bbox"][1] > prev_l["bbox"][3]:
                    spacing = l["bbox"][1] - prev_l["bbox"][1]
                    if 15 < spacing < 40: spacings.append(spacing)

        if indents:
            avg_indent = sum(indents) / len(indents)
            if abs(avg_indent - TARGET_INDENT) > MARGIN_TOLERANCE:
                page_findings.append(f"Відступ першого рядка {avg_indent/CM:.1f} см замість 1.5 см")
        if spacings:
            avg_spacing = sum(spacings) / len(spacings)
            if avg_spacing < 20.0:
                page_findings.append(f"Міжрядковий інтервал замалий ({avg_spacing:.1f} pt). Має бути 1.5 (~21 pt).")
        if page_findings:
            pages_with_errors.add(page_num)
            for f in page_findings: findings.append(f"Сторінка {page_num}: {f}")

    is_success = len(findings) == 0
    summary = "Загальний текст відповідає вимогам." if is_success else f"Виявлено помилки на {len(pages_with_errors)} сторінках."
    return {"summary": summary, "findings": findings, "is_success": is_success, "pages_with_errors": sorted(list(pages_with_errors)), "highlights": highlights}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    try:
        pdf_bytes = base64.b64decode(request.pdf_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 PDF payload.") from exc
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Unable to parse PDF.") from exc
    try:
        if request.analysis_type == "page_numbers": result = analyze_page_numbers(doc)
        elif request.analysis_type == "general_text": result = analyze_general_text(doc)
        elif request.analysis_type == "zmist":
            rows, page_width = extract_page_rows_fitz(doc, request.page_number)
            result = analyze_zmist(rows, page_width)
        else:
            result = {"summary": "Невідомий тип аналізу.", "findings": [f'analysis_type "{request.analysis_type}" is not supported.'], "is_success": False}
        result["analysis_type"] = request.analysis_type
        result["page_number"] = request.page_number
        return result
    finally:
        doc.close()
