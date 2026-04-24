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


def extract_page_number_from_rows(rows: list[dict[str, Any]], search_region: dict[str, float] | None = None) -> dict[str, Any] | None:
    def is_pure_number(text: str) -> str | None:
        alpha = re.sub(r"[^A-Za-zА-Яа-яІіЄєЇїҐґ0-9]", "", text)
        if re.fullmatch(r"\d{1,4}", alpha):
            return alpha
        return None

    def in_region(x: float, y: float) -> bool:
        if not search_region:
            return True
        return (search_region["xmin"] <= x <= search_region["xmax"] and 
                search_region["ymin"] <= y <= search_region["ymax"])

    if search_region:
        for row in rows:
            if in_region(row["x"], row["y"]):
                num_text = is_pure_number(row["clean"])
                if num_text:
                    return {"text": num_text, "x": row["x"], "y": row["y"]}
            
            if "spans" in row:
                for span in row["spans"]:
                    if in_region(span["x"], span["y"]):
                        num_text = is_pure_number(span["text"])
                        if num_text:
                            return {"text": num_text, "x": span["x"], "y": span["y"]}
    else:
        for row in rows[:5]:
            num_text = is_pure_number(row["clean"])
            if num_text: return {"text": num_text, "x": row["x"], "y": row["y"]}
        for row in rows[-5:]:
            num_text = is_pure_number(row["clean"])
            if num_text: return {"text": num_text, "x": row["x"], "y": row["y"]}
            
    return None


def analyze_page_numbers(reader: PdfReader) -> dict[str, Any]:
    findings = []
    
    if len(reader.pages) >= 1:
        rows1, _ = extract_page_rows(reader, 1)
        num1 = extract_page_number_from_rows(rows1)
        if num1:
            findings.append(f"На титульній сторінці знайдено номер ({num1['text']}). Нумерації там не повинно бути.")
            
    if len(reader.pages) >= 2:
        rows2, _ = extract_page_rows(reader, 2)
        num2 = extract_page_number_from_rows(rows2)
        if num2:
            findings.append(f"На сторінці змісту знайдено номер ({num2['text']}). Нумерації там не повинно бути.")
            
    if len(reader.pages) >= 3:
        rows3, _ = extract_page_rows(reader, 3)
        h3 = float(reader.pages[2].mediabox.top)
        w3 = float(reader.pages[2].mediabox.right) - float(reader.pages[2].mediabox.left)
        
        search_region = {
            "xmin": w3 - 100,
            "xmax": w3 + 100,
            "ymin": h3 - 100,
            "ymax": h3 + 100
        }
        num3 = extract_page_number_from_rows(rows3, search_region)
        if not num3:
            findings.append("На 3-й сторінці не знайдено номер сторінки у правому верхньому куті (зона 3х3 см).")
        else:
            if num3["text"] != "3":
                findings.append(f'Номер на 3-й сторінці має бути "3", але знайдено "{num3["text"]}".')
    else:
        findings.append("Неможливо перевірити 3-тю сторінку (у документі менше 3 сторінок).")
        
    is_success = len(findings) == 0
    summary = "Нумерація сторінок відповідає вимогам." if is_success else "Виявлено помилки нумерації сторінок."
    
    return {
        "summary": summary,
        "findings": findings,
        "is_success": is_success
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
        result = analyze_page_numbers(reader)
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
