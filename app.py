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

CM = 28.346 # 1 см в пунктах

def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()

def analyze_page_numbers(doc: fitz.Document) -> dict[str, Any]:
    findings, highlights = [], []
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
    findings, highlights, pages_with_errors = [], [], set()
    TARGET_LEFT = 2.5 * CM
    for p_num in range(3, len(doc) + 1):
        p = doc[p_num-1]
        w, h = p.rect.width, p.rect.height
        els = [l for b in p.get_text("dict")["blocks"] if "lines" in b for l in b["lines"] if "".join(s["text"] for s in l["spans"]).strip() and not ("".join(s["text"] for s in l["spans"]).strip().isdigit() and l["bbox"][1] < 60)]
        if not els: continue
        actual_left = min(l["bbox"][0] for l in els)
        actual_bottom_y = max(l["bbox"][3] for l in els)
        p_f = []
        if abs(actual_left - TARGET_LEFT) > 0.5 * CM:
            p_f.append("Ліве поле"); highlights.append({"page": p_num, "x": 0, "y": 0, "w": actual_left, "h": h})
        if (h - actual_bottom_y) > 4.5 * CM:
            p_f.append("Порожнє місце знизу"); highlights.append({"page": p_num, "x": 0, "y": actual_bottom_y, "w": w, "h": h - actual_bottom_y})
        if p_f:
            pages_with_errors.add(p_num)
            for f in p_f: findings.append(f"Стор. {p_num}: {f}")
    return {"summary": "Текст перевірено.", "findings": findings, "is_success": len(findings) == 0, "pages_with_errors": sorted(list(pages_with_errors)), "highlights": highlights}

def analyze_chapters(doc: fitz.Document) -> dict[str, Any]:
    findings, highlights, pages_with_errors = [], [], set()
    for p_num in range(3, len(doc) + 1):
        p = doc[p_num-1]
        ls = [l for b in p.get_text("dict")["blocks"] if "lines" in b for l in b["lines"] if "".join(s["text"] for s in l["spans"]).strip() and not ("".join(s["text"] for s in l["spans"]).strip().isdigit() and l["bbox"][1] < 60)]
        for idx, line in enumerate(ls):
            txt = "".join(s["text"] for s in line["spans"]).strip()
            if re.match(r"^РОЗДІЛ\s+\d+", txt.upper()):
                p_f = []
                if line["bbox"][1] > 120: p_f.append("РОЗДІЛ не з нової сторінки"); highlights.append({"page": p_num, "x": 0, "y": 0, "w": p.rect.width, "h": line["bbox"][1]})
                title_ls, curr = [], idx + 1
                while curr < len(ls):
                    l = ls[curr]
                    lt = "".join(s["text"] for s in l["spans"]).strip()
                    if re.match(r"^\d+", lt): break
                    if bool(l["spans"][0]["flags"] & 16) and abs((l["bbox"][0]+l["bbox"][2])/2 - p.rect.width/2) < 60: title_ls.append(l); curr += 1
                    else: break
                if p_f:
                    pages_with_errors.add(p_num)
                    for f in p_f: findings.append(f"Стор. {p_num}: {f}")
    return {"summary": "Розділи перевірено.", "findings": findings, "is_success": len(findings) == 0, "pages_with_errors": sorted(list(pages_with_errors)), "highlights": highlights}

def analyze_subchapters(doc: fitz.Document) -> dict[str, Any]:
    findings, highlights, pages_with_errors = [], [], set()
    for p_num in range(3, len(doc) + 1):
        p = doc[p_num-1]
        ls = [l for b in p.get_text("dict")["blocks"] if "lines" in b for l in b["lines"] if "".join(s["text"] for s in l["spans"]).strip()]
        l_bound = min([l["bbox"][0] for l in ls if not re.match(r"^[1-3]\.[1-3]", "".join(s["text"] for s in l["spans"]).strip()) and abs((l["bbox"][0]+l["bbox"][2])/2-p.rect.width/2)>50] or [2.5*CM])
        skip = -1
        for idx, line in enumerate(ls):
            if idx <= skip: continue
            txt = "".join(s["text"] for s in line["spans"]).strip()
            if re.match(r"^[1-3]\.[1-3]\.?\s+", txt):
                p_f = []
                current_x = line["bbox"][0]
                expected_x = l_bound + 1.5 * CM
                if abs(current_x - expected_x) > 0.2 * CM:
                    p_f.append(f"Відступ {(current_x - l_bound)/CM:.1f} см замість 1.5"); highlights.append({"page": p_num, "x": l_bound, "y": line["bbox"][1], "w": current_x - l_bound, "h": line["bbox"][3]-line["bbox"][1]})
                if p_f:
                    pages_with_errors.add(p_num)
                    for f in p_f: findings.append(f"Стор. {p_num}: {f}")
    return {"summary": "Підрозділи перевірено.", "findings": findings, "is_success": len(findings) == 0, "pages_with_errors": sorted(list(pages_with_errors)), "highlights": highlights}

def analyze_perelik(doc: fitz.Document) -> dict[str, Any]:
    findings, highlights, pages_with_errors = [], [], set()
    wrong = ["•", "●", "○", "▪", "▫", "", "√", "*", ""]
    allowed = ["-", "–", "—"]
    for p_num in range(3, len(doc) + 1):
        p = doc[p_num-1]
        for b in p.get_text("dict")["blocks"]:
            if "lines" not in b: continue
            for l in b["lines"]:
                txt = "".join(s["text"] for s in l["spans"]).strip()
                if not txt: continue
                if not txt[0].isalnum() and txt[0] not in allowed and (txt[0] in wrong or not any(txt.startswith(d) for d in allowed)):
                    if txt[0] in [",", ".", ";", ":", '"', "(", ")", "«", "»"]: continue
                    findings.append(f"Стор. {p_num}: Маркер '{txt[0]}' замість тире.")
                    pages_with_errors.add(p_num); highlights.append({"page": p_num, "x": l["spans"][0]["bbox"][0], "y": l["spans"][0]["bbox"][1], "w": l["spans"][0]["bbox"][2]-l["spans"][0]["bbox"][0], "h": l["spans"][0]["bbox"][3]-l["spans"][0]["bbox"][1]})
    return {"summary": "Переліки перевірено.", "findings": findings, "is_success": len(findings) == 0, "pages_with_errors": sorted(list(pages_with_errors)), "highlights": highlights}

def analyze_tables(doc: fitz.Document) -> dict[str, Any]:
    findings, highlights, pages_with_errors = [], [], set()
    for page_num in range(3, len(doc) + 1):
        page = doc[page_num-1]
        tabs = page.find_tables()
        all_lines = [l for b in page.get_text("dict")["blocks"] if "lines" in b for l in b["lines"]]
        
        for tab in tabs:
            p_f = []
            t_bbox = tab.bbox # [x0, y0, x1, y1]
            t_center = (t_bbox[0] + t_bbox[2]) / 2
            p_center = page.rect.width / 2
            
            # 1. Перевірка розташування таблиці
            if abs(t_center - p_center) > 15: # Допуск ~0.5 см
                p_f.append("Таблиця не по центру")
                highlights.append({"page": page_num, "x": t_bbox[0], "y": t_bbox[1], "w": t_bbox[2]-t_bbox[0], "h": t_bbox[3]-t_bbox[1]})
            
            # 2. Пошук назви над таблицею
            caption_line = None
            for l in reversed(all_lines):
                # Рядок має бути вище таблиці (не більше ніж на 40pt)
                if l["bbox"][3] < t_bbox[1] and (t_bbox[1] - l["bbox"][3]) < 40:
                    caption_line = l
                    break
            
            if not caption_line:
                p_f.append("Не знайдено назву над таблицею")
            else:
                txt = "".join(s["text"] for s in caption_line["spans"]).strip()
                # Формат: Таблиця 1.2 – Назва
                if not re.match(r"^Таблиця\s+\d+(\.\d+)*\s+[–-]\s+.+", txt):
                    p_f.append(f"Невірний формат назви: '{txt[:20]}...' (має бути 'Таблиця X.Y – Назва')")
                
                # Шрифт та стиль
                span = caption_line["spans"][0]
                if abs(span["size"] - 14) > 1: p_f.append(f"Розмір шрифту назви {span['size']:.1f} замість 14")
                if bool(span["flags"] & 16): p_f.append("Назва таблиці не повинна бути жирною")
                if bool(span["flags"] & 2): p_f.append("Назва таблиці не повинна бути курсивом")
                
                # Абзацний відступ (1.5 см від лівого поля 2.5 см = 4.0 см)
                if abs(caption_line["bbox"][0] - 4.0 * CM) > 0.3 * CM:
                    p_f.append(f"Назва має починатись з абзацу 1.5 см (зараз { (caption_line['bbox'][0]/CM - 2.5):.1f} см)")
                
                if p_f:
                    highlights.append({"page": page_num, "x": caption_line["bbox"][0], "y": caption_line["bbox"][1], "w": caption_line["bbox"][2]-caption_line["bbox"][0], "h": caption_line["bbox"][3]-caption_line["bbox"][1]})

            if p_f:
                pages_with_errors.add(page_num)
                for f in p_f: findings.append(f"Стор. {page_num}: {f}")
                
    return {"summary": "Таблиці перевірено.", "findings": findings, "is_success": len(findings) == 0, "pages_with_errors": sorted(list(pages_with_errors)), "highlights": highlights}

@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    try:
        pdf_bytes = base64.b64decode(request.pdf_base64)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        t = request.analysis_type
        if t == "page_numbers": res = analyze_page_numbers(doc)
        elif t == "general_text": res = analyze_general_text(doc)
        elif t == "chapters": res = analyze_chapters(doc)
        elif t == "subchapters": res = analyze_subchapters(doc)
        elif t == "perelik": res = analyze_perelik(doc)
        elif t == "tables": res = analyze_tables(doc)
        else: res = {"summary": "Unknown", "findings": [], "is_success": False}
        res["analysis_type"] = t
        return res
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
    finally: doc.close()
