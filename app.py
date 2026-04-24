from __future__ import annotations

import base64
import io
import re
from collections import defaultdict
from statistics import mean
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pypdf import PdfReader
import fitz


app = FastAPI(title="PDF Analyzer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    pdf_base64: str = Field(..., min_length=100)
    analysis_type: str = Field(default="zmist")
    page_number: int = Field(default=2, ge=1)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def is_leader_fragment(text: str) -> bool:
    clean = normalize_text(text)
    return not clean or re.fullmatch(r"[.\d\s]+", clean) is not None


def font_is_bold(font_dict: dict[str, Any] | None) -> bool:
    if not font_dict:
        return False

    base_font = str(font_dict.get("/BaseFont", ""))
    if re.search(r"bold|black|demi", base_font, re.IGNORECASE):
        return True

    descriptor = font_dict.get("/FontDescriptor")
    if descriptor and isinstance(descriptor, dict):
        try:
            weight = int(descriptor.get("/FontWeight", 0))
            if weight >= 600:
                return True
        except Exception:
            pass

    return False


def extract_page_rows(reader: PdfReader, page_number: int) -> tuple[list[dict[str, Any]], float]:
    if page_number > len(reader.pages):
        raise HTTPException(status_code=400, detail=f"PDF has only {len(reader.pages)} page(s).")

    page = reader.pages[page_number - 1]
    page_width = float(page.mediabox.right) - float(page.mediabox.left)
    spans: list[dict[str, Any]] = []

    def visitor(text: str, cm: list[float], tm: list[float], font_dict: dict[str, Any] | None, font_size: float) -> None:
        clean = normalize_text(text)
        if not clean:
            return

        x = float(tm[4])
        y = float(tm[5])
        spans.append(
            {
                "text": clean,
                "x": x,
                "y": y,
                "font_size": float(font_size),
                "font_name": str((font_dict or {}).get("/BaseFont", "")),
                "is_bold": font_is_bold(font_dict),
            }
        )

    page.extract_text(visitor_text=visitor)

    if not spans:
        return [], page_width

    buckets: list[dict[str, Any]] = []
    for span in sorted(spans, key=lambda item: (-item["y"], item["x"])):
        row = next((item for item in buckets if abs(item["y"] - span["y"]) < 3), None)
        if row is None:
            row = {"y": span["y"], "spans": []}
            buckets.append(row)
        row["spans"].append(span)

    rows: list[dict[str, Any]] = []
    for bucket in buckets:
        line_spans = sorted(bucket["spans"], key=lambda item: item["x"])
        text = normalize_text(" ".join(span["text"] for span in line_spans))
        if is_leader_fragment(text):
            continue

        content_spans = [span for span in line_spans if not is_leader_fragment(span["text"])]
        if not content_spans:
            continue

        rows.append(
            {
                "text": text,
                "clean": text,
                "x": min(span["x"] for span in content_spans),
                "y": bucket["y"],
                "font_size": mean(span["font_size"] for span in content_spans),
                "font_names": sorted({span["font_name"] for span in content_spans if span["font_name"]}),
                "is_bold": any(span["is_bold"] for span in content_spans),
                "spans": content_spans,
            }
        )

    return rows, page_width


def analyze_zmist(rows: list[dict[str, Any]], page_width: float) -> dict[str, Any]:
    findings: list[str] = []

    if not rows:
        return {"summary": "Перевірка змісту не виконана.", "findings": ["Не вдалося зчитати текст сторінки."]}

    title_row = next((row for row in rows if row["clean"].upper() == "ЗМІСТ"), None)
    if not title_row:
        findings.append('Не знайдено заголовок "ЗМІСТ".')

    major_rows = [
        row
        for row in rows
        if re.match(r"^(ВСТУП|РОЗДІЛ\s+\d+|ВИСНОВКИ|ДОДАТКИ|СПИСОК ВИКОРИСТАНИХ ДЖЕРЕЛ)", row["clean"], re.IGNORECASE)
    ]
    sub_rows = [row for row in rows if re.match(r"^\d+\.\d+", row["clean"])]

    if not major_rows:
        findings.append("Не знайдено основні пункти змісту.")

    if not sub_rows:
        findings.append("Не знайдено підпункти у форматі n.n Назва.")

    for row in major_rows:
        label = re.split(r"[.\u2026]", row["clean"])[0].strip()
        letters_only = re.sub(r"[0-9«»\"'.\-–—,:; ]", "", label)
        if letters_only and letters_only != letters_only.upper():
            findings.append(f'Пункт "{label}" має бути великими літерами.')

    weak_bold_rows = [re.split(r"[.\u2026]", row["clean"])[0].strip() for row in major_rows if not row["is_bold"]]
    if weak_bold_rows:
        labels = [f'"{item}"' for item in weak_bold_rows[:5]]
        suffix = " та інші" if len(weak_bold_rows) > 5 else ""
        findings.append(f"Основні пункти змісту не визначені як жирні: {', '.join(labels)}{suffix}.")

    bold_sub_rows = [row["clean"] for row in sub_rows if row["is_bold"]]
    if bold_sub_rows:
        labels = [f'"{item}"' for item in bold_sub_rows[:5]]
        suffix = " та інші" if len(bold_sub_rows) > 5 else ""
        findings.append(f"Підпункти визначені як жирні, хоча мають бути звичайними: {', '.join(labels)}{suffix}.")

    font_rows = [row for row in rows if row["clean"] != "ЗМІСТ"]
    if font_rows:
        avg_font = mean(row["font_size"] for row in font_rows[:12])
        if avg_font < 11.5 or avg_font > 16.8:
            findings.append(f"Розмір основного шрифту виглядає як {avg_font:.1f} pt замість 14 pt.")

    summary = (
        f"Виявлено {len(findings)} відхилення(нь) у змісті."
        if findings
        else "Суттєвих відхилень у змісті не виявлено."
    )

    return {
        "summary": summary,
        "findings": findings,
        "rows": rows,
        "metrics": {
            "page_width": page_width,
            "major_rows": len(major_rows),
            "sub_rows": len(sub_rows),
        },
    }


def analyze_page_numbers(pdf_bytes: bytes) -> dict[str, Any]:
    findings = []
    
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        return {"summary": "Помилка читання PDF", "findings": [str(e)], "is_success": False}
        
    for page_num in [1, 2, 3]:
        if page_num > len(doc):
            if page_num == 3:
                findings.append("Неможливо перевірити 3-тю сторінку (у документі менше 3 сторінок).")
            continue
            
        page = doc[page_num - 1]
        rect = page.rect
        # Кут 2х2 см. 1 см = ~28 точок. 2 см = 60 точок.
        # В PyMuPDF координати від (0,0) у лівому ВЕРХНЬОМУ куті.
        search_rect = fitz.Rect(rect.width - 60, 0, rect.width, 60)
        
        words = page.get_text("words")
        
        digits = ""
        for w in words:
            word_rect = fitz.Rect(w[:4])
            if word_rect.intersects(search_rect):
                clean_word = re.sub(r"[^\d]", "", w[4])
                if clean_word:
                    digits += clean_word
                    
        if page_num == 1:
            if digits:
                findings.append(f"На титульній сторінці у правому верхньому куті знайдено цифри ({digits}). Їх там не повинно бути.")
        elif page_num == 2:
            if digits:
                findings.append(f"На сторінці змісту у правому верхньому куті знайдено цифри ({digits}). Їх там не повинно бути.")
        elif page_num == 3:
            if not digits:
                findings.append("На 3-й сторінці не знайдено цифру у правому верхньому куті (зона 2х2 см).")
            elif digits != "3":
                findings.append(f'Номер на 3-й сторінці має бути "3", але знайдено "{digits}".')
                
    is_success = len(findings) == 0
    summary = "Нумерація сторінок відповідає вимогам." if is_success else "Виявлено помилки нумерації сторінок."
    
    return {
        "summary": summary,
        "findings": findings,
        "is_success": is_success
    }

def analyze_general_text(pdf_bytes: bytes) -> dict[str, Any]:
    findings = []
    highlights = []
    
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        return {"summary": "Помилка читання PDF", "findings": [str(e)], "is_success": False, "highlights": [], "pages_with_errors": []}
        
    CM = 28.346
    MARGIN_TOLERANCE = 0.5 * CM
    TARGET_LEFT = 2.5 * CM
    TARGET_RIGHT = 1.0 * CM
    TARGET_TOP = 2.0 * CM
    TARGET_BOTTOM = 2.0 * CM
    TARGET_INDENT = 1.5 * CM
    
    def is_major_heading(text: str) -> bool:
        clean = re.sub(r"\s+", " ", text.strip()).upper()
        return bool(re.match(r"^(ВСТУП|РОЗДІЛ\s+\d+|ВИСНОВКИ|ДОДАТКИ|СПИСОК ВИКОРИСТАНИХ ДЖЕРЕЛ)", clean))

    def get_page_major_heading(page) -> str | None:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" not in b: continue
            for l in b["lines"]:
                text = "".join(s["text"] for s in l["spans"]).strip()
                if is_major_heading(text):
                    return text
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
        
        # Знаходимо таблиці та рисунки, щоб ігнорувати текст у них
        tables = page.find_tables()
        table_bboxes = [fitz.Rect(t.bbox) for t in tables.tables]
        
        img_info = page.get_image_info()
        img_bboxes = [fitz.Rect(img["bbox"]) for img in img_info]

        blocks = page.get_text("dict")["blocks"]
        text_lines = []
        for b in blocks:
            if "lines" not in b: continue
            
            block_rect = fitz.Rect(b["bbox"])
            # Пропускаємо, якщо текст у таблиці або біля рисунка
            if any(block_rect.intersects(t_bbox) for t_bbox in table_bboxes):
                continue
            if any(block_rect.intersects(i_bbox) for i_bbox in img_bboxes):
                continue

            for l in b["lines"]:
                text = "".join(s["text"] for s in l["spans"]).strip()
                if not text: continue
                
                # Пропускаємо підписи до рисунків/таблиць
                if re.match(r"^(Рис|Табл|Рисунок|Таблиця)\.?\s+\d+", text, re.I):
                    continue

                if re.fullmatch(r"\d+", text):
                    if l["bbox"][1] < 100 or l["bbox"][3] > height - 100:
                        continue
                
                # Ігноруємо центрований текст (формули, розрахунки, заголовки) для аналізу відступів
                line_center = (l["bbox"][0] + l["bbox"][2]) / 2
                page_center = width / 2
                if abs(line_center - page_center) < 40: # Допуск центрування
                    if (l["bbox"][2] - l["bbox"][0]) < width * 0.7: # Якщо рядок не на всю ширину
                        continue

                text_lines.append(l)

        if not text_lines:
            continue

        page_findings = []
        
        # Для лівого поля ігноруємо рядки з жирним шрифтом або цифрами на початку
        left_margin_lines = []
        for l in text_lines:
            first_span = l["spans"][0]
            first_text = first_span["text"].strip()
            first_char = first_text[0] if first_text else ""
            is_bold = bool(first_span["flags"] & 4)
            is_digit = first_char.isdigit()
            if not (is_bold or is_digit):
                left_margin_lines.append(l)
        
        # Якщо є звичайні рядки, рахуємо ліве поле по них. Інакше по всіх.
        ref_left = left_margin_lines if left_margin_lines else text_lines
        actual_left = min(l["bbox"][0] for l in ref_left)
        
        # Праве поле рахуємо по тих самих рядках, що й ліве
        actual_right = width - max(l["bbox"][2] for l in ref_left)
        
        # Верхнє поле рахуємо по ВСІХ рядках (включаючи жирні та цифри)
        actual_top = min(l["bbox"][1] for l in text_lines)
        
        actual_bottom_y = max(l["bbox"][3] for l in text_lines)
        actual_bottom = height - actual_bottom_y

        if abs(actual_left - TARGET_LEFT) > MARGIN_TOLERANCE:
            page_findings.append(f"Ліве поле {actual_left/CM:.1f} см замість 2.5 см")
            highlights.append({"page": page_num, "x": 0, "y": 0, "w": actual_left, "h": height})
            
        if actual_right < TARGET_RIGHT - MARGIN_TOLERANCE:
            page_findings.append(f"Праве поле {actual_right/CM:.1f} см замість 1.0 см")
            highlights.append({"page": page_num, "x": width - actual_right, "y": 0, "w": actual_right, "h": height})

        if abs(actual_top - TARGET_TOP) > MARGIN_TOLERANCE:
            page_findings.append(f"Верхнє поле {actual_top/CM:.1f} см замість 2.0 см")
            highlights.append({"page": page_num, "x": 0, "y": 0, "w": width, "h": actual_top})

        if actual_bottom > 4.0 * CM:
            is_section_end = False
            if page_num < len(doc):
                next_page_heading = get_page_major_heading(doc[page_num])
                if next_page_heading:
                    is_section_end = True
            if not is_section_end:
                page_findings.append(f"Порожнє місце знизу ({actual_bottom/CM:.1f} см). Не повинно бути.")
                highlights.append({"page": page_num, "x": 0, "y": actual_bottom_y, "w": width, "h": actual_bottom})
        elif actual_bottom < TARGET_BOTTOM - MARGIN_TOLERANCE:
            page_findings.append(f"Нижнє поле {actual_bottom/CM:.1f} см замість 2.0 см")
            highlights.append({"page": page_num, "x": 0, "y": actual_bottom_y, "w": width, "h": actual_bottom})

        indents = []
        spacings = []
        for i, l in enumerate(text_lines):
            # Відступ перевіряємо тільки для тексту, що НЕ починається з цифри і НЕ є жирним
            first_span = l["spans"][0]
            first_text = first_span["text"].strip()
            first_char = first_text[0] if first_text else ""
            is_bold = bool(first_span["flags"] & 4)
            is_digit = first_char.isdigit()
            
            indent = l["bbox"][0] - actual_left
            if indent > 10 and not is_bold and not is_digit:
                indents.append(indent)
                
            if i > 0:
                prev_l = text_lines[i-1]
                if l["bbox"][1] > prev_l["bbox"][3]:
                    spacing = l["bbox"][1] - prev_l["bbox"][1]
                    if 15 < spacing < 40:
                        spacings.append(spacing)

        if indents:
            avg_indent = sum(indents) / len(indents)
            if abs(avg_indent - TARGET_INDENT) > MARGIN_TOLERANCE:
                page_findings.append(f"Відступ першого рядка {avg_indent/CM:.1f} см замість 1.5 см")

        if spacings:
            avg_spacing = sum(spacings) / len(spacings)
            # Помилка тільки якщо інтервал менше 20 pt
            if avg_spacing < 20.0:
                page_findings.append(f"Міжрядковий інтервал замалий ({avg_spacing:.1f} pt). Має бути 1.5 (~21 pt).")

        if page_findings:
            pages_with_errors.add(page_num)
            for f in page_findings:
                findings.append(f"Сторінка {page_num}: {f}")

    is_success = len(findings) == 0
    summary = "Загальний текст відповідає вимогам." if is_success else f"Виявлено помилки на {len(pages_with_errors)} сторінках."
    
    return {
        "summary": summary,
        "findings": findings,
        "is_success": is_success,
        "pages_with_errors": sorted(list(pages_with_errors)),
        "highlights": highlights
    }

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
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Unable to parse PDF.") from exc

    if request.analysis_type == "page_numbers":
        result = analyze_page_numbers(pdf_bytes)
    elif request.analysis_type == "general_text":
        result = analyze_general_text(pdf_bytes)
    else:
        rows, page_width = extract_page_rows(reader, request.page_number)
    
        if request.analysis_type == "zmist":
            result = analyze_zmist(rows, page_width)
        else:
            result = {
                "summary": "Невідомий тип аналізу.",
                "findings": [f'analysis_type "{request.analysis_type}" is not supported.'],
                "rows": rows,
                "metrics": {"page_width": page_width},
            }

    result["analysis_type"] = request.analysis_type
    result["page_number"] = request.page_number
    return result
