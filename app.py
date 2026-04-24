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

class AnalyzeRequest(BaseModel):
    pdf_base64: str = Field(..., min_length=100)
    analysis_type: str = Field(default="zmist")
    page_number: int = Field(default=2, ge=1)

def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()

def is_leader_fragment(text: str) -> bool:
    clean = normalize_text(text)
    return not clean or re.fullmatch(r"[.\d\s\u2026_]+", clean) is not None

def extract_page_rows_fitz(doc: fitz.Document, page_number: int) -> tuple[list[dict[str, Any]], float]:
    if page_number > len(doc): return [], 595
    page = doc[page_number - 1]
    page_width = page.rect.width
    blocks = page.get_text("dict")["blocks"]
    all_spans = []
    for b in blocks:
        if "lines" not in b: continue
        for l in b["lines"]:
            for s in l["spans"]:
                txt = s["text"].strip()
                if txt:
                    all_spans.append({"text": txt, "x": s["bbox"][0], "y": s["bbox"][1], "w": s["bbox"][2]-s["bbox"][0], "h": s["bbox"][3]-s["bbox"][1], "font_size": s["size"], "is_bold": bool(s["flags"] & 16)})
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
        raw_rows.append({"text": full_text, "clean": normalize_text(full_text), "x": min(s["x"] for s in content_spans), "y": bucket["y"], "font_size": mean(s["font_size"] for s in content_spans), "is_bold": any(s["is_bold"] for s in content_spans)})
    return raw_rows, page_width

def analyze_zmist(rows: list[dict[str, Any]], page_width: float) -> dict[str, Any]:
    findings = []
    if not rows: return {"summary": "Текст не знайдено", "findings": ["Порожня сторінка"], "is_success": False}
    if not any("ЗМІСТ" in r["clean"].upper() for r in rows): findings.append('Не знайдено заголовок "ЗМІСТ".')
    major_pattern = r"^(ВСТУП|РОЗДІЛ\s+\d+|ВИСНОВКИ|ДОДАТКИ|СПИСОК ВИКОРИСТАНИХ ДЖЕРЕЛ)"
    for row in rows:
        c = row["clean"].upper()
        if re.match(major_pattern, c):
            if not row["is_bold"]: findings.append(f'Розділ "{row["clean"][:30]}" має бути ЖИРНИМ.')
        elif re.match(r"^\d+\.\d+", c):
            if row["is_bold"]: findings.append(f'Підпункт "{row["clean"][:30]}" не повинен бути жирним.')
    return {"summary": "Перевірку змісту завершено.", "findings": findings, "is_success": len(findings) == 0}

def analyze_page_numbers(doc: fitz.Document) -> dict[str, Any]:
    findings = []
    highlights = []
    for page_num in [1, 2, 3]:
        if page_num > len(doc): continue
        page = doc[page_num - 1]
        rect = page.rect
        search_rect = fitz.Rect(rect.width - 100, 0, rect.width, 100)
        words = page.get_text("words")
        digits = "".join(re.sub(r"[^\d]", "", w[4]) for w in words if fitz.Rect(w[:4]).intersects(search_rect))
        if page_num == 3 and digits != "3":
            findings.append(f"Стор. 3: знайдено '{digits}' замість '3'.")
            highlights.append({"page": 3, "x": rect.width - 100, "y": 0, "w": 100, "h": 100})
    return {"summary": "Нумерація перевірена.", "findings": findings, "is_success": len(findings) == 0, "highlights": highlights}

def analyze_general_text(doc: fitz.Document) -> dict[str, Any]:
    findings = []
    highlights = []
    pages_with_errors = set()
    CM = 28.346
    MARGIN_TOLERANCE = 0.5 * CM
    TARGET_LEFT, TARGET_RIGHT, TARGET_TOP = 2.5*CM, 1.0*CM, 2.0*CM
    
    def get_page_first_line_info(page) -> dict | None:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" not in b: continue
            for l in b["lines"]:
                text = "".join(s["text"] for s in l["spans"]).strip()
                if text and not (text.isdigit() and l["bbox"][1] < 60): return {"text": text, "is_bold": bool(l["spans"][0]["flags"] & 16), "starts_with_digit": text[0].isdigit()}
        return None

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
        actual_bottom_y = max(l["bbox"][3] for l in all_elements)
        actual_bottom = height - actual_bottom_y
        actual_left = min(l["bbox"][0] for l in all_elements)
        actual_right = width - max(l["bbox"][2] for l in all_elements)
        
        p_f = []
        if abs(actual_left - TARGET_LEFT) > MARGIN_TOLERANCE:
            p_f.append("Ліве поле"); highlights.append({"page": page_num, "x": 0, "y": 0, "w": actual_left, "h": height})
        if actual_bottom > 4.5 * CM:
            is_v = False
            if page_num < len(doc):
                n_i = get_page_first_line_info(doc[page_num])
                if n_i and n_i["is_bold"] and not n_i["starts_with_digit"]: is_v = True
            if not is_v:
                p_f.append("Порожнє місце знизу"); highlights.append({"page": page_num, "x": 0, "y": actual_bottom_y, "w": width, "h": actual_bottom})
        
        if p_f:
            pages_with_errors.add(page_num)
            for f in p_f: findings.append(f"Стор. {page_num}: {f}")
            
    return {"summary": "Текст перевірено.", "findings": findings, "is_success": len(findings) == 0, "pages_with_errors": sorted(list(pages_with_errors)), "highlights": highlights}

def analyze_chapters(doc: fitz.Document) -> dict[str, Any]:
    findings = []
    highlights = []
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
                if line["bbox"][1] > 120:
                    p_f.append("РОЗДІЛ не з нової сторінки"); highlights.append({"page": page_num, "x": 0, "y": 0, "w": page.rect.width, "h": line["bbox"][1]})
                if abs((line["bbox"][0]+line["bbox"][2])/2 - page.rect.width/2) > 45:
                    p_f.append("РОЗДІЛ не по центру"); highlights.append({"page": page_num, "x": line["bbox"][0], "y": line["bbox"][1], "w": line["bbox"][2]-line["bbox"][0], "h": line["bbox"][3]-line["bbox"][1]})
                
                title_lines = []
                c_idx = idx + 1
                while c_idx < len(lines):
                    l = lines[c_idx]
                    l_text = "".join(s["text"] for s in l["spans"]).strip()
                    # Якщо наступний рядок починається з цифри (підрозділ), назва розділу закінчилася
                    if re.match(r"^\d+", l_text): break
                    
                    if bool(l["spans"][0]["flags"] & 16) and abs((l["bbox"][0]+l["bbox"][2])/2 - page.rect.width/2) < 60:
                        title_lines.append(l); c_idx += 1
                    else: break
                
                if not title_lines: p_f.append("Не знайдено назву розділу")
                else:
                    for tl in title_lines:
                        txt = "".join(s["text"] for s in tl["spans"]).strip()
                        if txt != txt.upper() or not bool(tl["spans"][0]["flags"] & 16):
                            p_f.append(f"Назва '{txt[:10]}' має бути ВЕЛИКИМИ ЖИРНИМИ"); highlights.append({"page": page_num, "x": tl["bbox"][0], "y": tl["bbox"][1], "w": tl["bbox"][2]-tl["bbox"][0], "h": tl["bbox"][3]-tl["bbox"][1]})
                    if c_idx < len(lines) and (lines[c_idx]["bbox"][1] - title_lines[-1]["bbox"][3]) < 20:
                        p_f.append("Відсутній рядок після назви"); highlights.append({"page": page_num, "x": 0, "y": title_lines[-1]["bbox"][3], "w": page.rect.width, "h": 20})
                if p_f:
                    pages_with_errors.add(page_num)
                    for f in p_f: findings.append(f"Стор. {page_num}: {f}")
    return {"summary": "Розділи перевірено.", "findings": findings, "is_success": len(findings) == 0, "pages_with_errors": sorted(list(pages_with_errors)), "highlights": highlights}

def analyze_subchapters(doc: fitz.Document) -> dict[str, Any]:
    findings, highlights, pages_with_errors, CM = [], [], set(), 28.346
    for page_num in range(3, len(doc) + 1):
        page = doc[page_num-1]
        blocks = page.get_text("dict")["blocks"]
        lines = [l for b in blocks if "lines" in b for l in b["lines"] if "".join(s["text"] for s in l["spans"]).strip()]
        
        skip_until = -1
        for idx, line in enumerate(lines):
            if idx <= skip_until: continue
            
            text = "".join(s["text"] for s in line["spans"]).strip()
            # Патерн виключно 1.1 - 3.3
            if re.match(r"^[1-3]\.[1-3]\.?\s+", text):
                p_f = []
                is_bold = bool(line["spans"][0]["flags"] & 16)
                if not is_bold:
                    p_f.append(f"Підрозділ '{text[:20]}...' має бути жирним")
                    highlights.append({"page": page_num, "x": line["bbox"][0], "y": line["bbox"][1], "w": line["bbox"][2]-line["bbox"][0], "h": line["bbox"][3]-line["bbox"][1]})
                
                # 2. Абзацний відступ (1.5 см від лівого поля 2.5 см = 4.0 см від краю)
                # Допускаємо невелику похибку 0.2 см
                current_x = line["bbox"][0]
                expected_x = (2.5 + 1.5) * CM
                if abs(current_x - expected_x) > 0.2 * CM:
                    p_f.append(f"Відступ має бути 1.5 см (зараз { (current_x/CM - 2.5):.1f} см)")
                    highlights.append({"page": page_num, "x": 0, "y": line["bbox"][1], "w": current_x, "h": line["bbox"][3]-line["bbox"][1]})
                
                if idx > 0 and (line["bbox"][1] - lines[idx-1]["bbox"][3]) < 20:
                    p_f.append("Відсутній рядок зверху")
                    highlights.append({"page": page_num, "x": 0, "y": line["bbox"][1]-20, "w": page.rect.width, "h": 20})
                
                # Збираємо назву
                sub_ls = [line]
                curr = idx + 1
                while curr < len(lines):
                    l = lines[curr]
                    if bool(l["spans"][0]["flags"] & 16) and abs(l["bbox"][0] - line["bbox"][0]) < 10:
                        sub_ls.append(l); curr += 1
                    else: break
                
                skip_until = curr - 1 # Пропускаємо рядки назви в наступних ітераціях
                
                if curr < len(lines) and (lines[curr]["bbox"][1] - sub_ls[-1]["bbox"][3]) < 20:
                    p_f.append("Відсутній рядок знизу")
                    highlights.append({"page": page_num, "x": 0, "y": sub_ls[-1]["bbox"][3], "w": page.rect.width, "h": 20})
                
                if p_f:
                    pages_with_errors.add(page_num)
                    for f in p_f: findings.append(f"Стор. {page_num}: {f}")
                    
    return {"summary": "Підрозділи перевірено.", "findings": findings, "is_success": len(findings) == 0, "pages_with_errors": sorted(list(pages_with_errors)), "highlights": highlights}

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
        else: res = {"summary": "Unknown", "findings": [], "is_success": False}
        res["analysis_type"] = request.analysis_type
        return res
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
    finally: doc.close()
