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
    findings, highlights, pages_with_errors, CM = [], [], set(), 28.346
    MARGIN_TOLERANCE, TARGET_LEFT = 0.5 * CM, 2.5 * CM
    def get_p_info(p):
        for b in p.get_text("dict")["blocks"]:
            if "lines" not in b: continue
            for l in b["lines"]:
                t = "".join(s["text"] for s in l["spans"]).strip()
                if t and not (t.isdigit() and l["bbox"][1] < 60): return {"is_bold": bool(l["spans"][0]["flags"] & 16), "starts_digit": t[0].isdigit()}
        return None
    for p_num in range(3, len(doc) + 1):
        p = doc[p_num-1]
        w, h = p.rect.width, p.rect.height
        els = [l for b in p.get_text("dict")["blocks"] if "lines" in b for l in b["lines"] if "".join(s["text"] for s in l["spans"]).strip() and not ("".join(s["text"] for s in l["spans"]).strip().isdigit() and l["bbox"][1] < 60)]
        if not els: continue
        actual_top, actual_bottom_y = min(l["bbox"][1] for l in els), max(l["bbox"][3] for l in els)
        actual_left, actual_right = min(l["bbox"][0] for l in els), w - max(l["bbox"][2] for l in els)
        p_f = []
        if abs(actual_left - TARGET_LEFT) > MARGIN_TOLERANCE:
            p_f.append("Ліве поле"); highlights.append({"page": p_num, "x": 0, "y": 0, "w": actual_left, "h": h})
        if (h - actual_bottom_y) > 4.5 * CM:
            is_v = False
            if p_num < len(doc):
                ni = get_p_info(doc[p_num])
                if ni and ni["is_bold"] and not ni["starts_digit"]: is_v = True
            if not is_v: p_f.append("Порожнє місце знизу"); highlights.append({"page": p_num, "x": 0, "y": actual_bottom_y, "w": w, "h": h - actual_bottom_y})
        if p_f:
            pages_with_errors.add(p_num)
            for f in p_f: findings.append(f"Стор. {p_num}: {f}")
    return {"summary": "Текст перевірено.", "findings": findings, "is_success": len(findings) == 0, "pages_with_errors": sorted(list(pages_with_errors)), "highlights": highlights}

def analyze_chapters(doc: fitz.Document) -> dict[str, Any]:
    findings, highlights, pages_with_errors = [], [], set()
    for p_num in range(3, len(doc) + 1):
        p = doc[p_num-1]
        ls = [l for b in p.get_text("dict")["blocks"] if "lines" in b for l in b["lines"] if "".join(s["text"] for s in l["spans"]).strip() and not ("".join(s["text"] for s in l["spans"]).strip().isdigit() and l["bbox"][1] < 60)]
        if not ls: continue
        for idx, line in enumerate(ls):
            txt = "".join(s["text"] for s in line["spans"]).strip()
            if re.match(r"^РОЗДІЛ\s+\d+", txt.upper()):
                p_f = []
                if line["bbox"][1] > 120: p_f.append("РОЗДІЛ не з нової сторінки"); highlights.append({"page": p_num, "x": 0, "y": 0, "w": p.rect.width, "h": line["bbox"][1]})
                if abs((line["bbox"][0]+line["bbox"][2])/2 - p.rect.width/2) > 45: p_f.append("РОЗДІЛ не по центру"); highlights.append({"page": p_num, "x": line["bbox"][0], "y": line["bbox"][1], "w": line["bbox"][2]-line["bbox"][0], "h": line["bbox"][3]-line["bbox"][1]})
                title_ls, curr = [], idx + 1
                while curr < len(ls):
                    l = ls[curr]
                    lt = "".join(s["text"] for s in l["spans"]).strip()
                    if re.match(r"^\d+", lt): break
                    if bool(l["spans"][0]["flags"] & 16) and abs((l["bbox"][0]+l["bbox"][2])/2 - p.rect.width/2) < 60: title_ls.append(l); curr += 1
                    else: break
                if not title_ls: p_f.append("Не знайдено назву розділу")
                else:
                    for tl in title_ls:
                        t_txt = "".join(s["text"] for s in tl["spans"]).strip()
                        if t_txt != t_txt.upper() or not bool(tl["spans"][0]["flags"] & 16): p_f.append(f"Назва '{t_txt[:10]}' не вірна"); highlights.append({"page": p_num, "x": tl["bbox"][0], "y": tl["bbox"][1], "w": tl["bbox"][2]-tl["bbox"][0], "h": tl["bbox"][3]-tl["bbox"][1]})
                    if curr < len(ls) and (ls[curr]["bbox"][1] - title_ls[-1]["bbox"][3]) < 20: p_f.append("Відсутній рядок після назви"); highlights.append({"page": p_num, "x": 0, "y": title_ls[-1]["bbox"][3], "w": p.rect.width, "h": 20})
                if p_f:
                    pages_with_errors.add(p_num)
                    for f in p_f: findings.append(f"Стор. {p_num}: {f}")
    return {"summary": "Розділи перевірено.", "findings": findings, "is_success": len(findings) == 0, "pages_with_errors": sorted(list(pages_with_errors)), "highlights": highlights}

def analyze_subchapters(doc: fitz.Document) -> dict[str, Any]:
    findings, highlights, pages_with_errors, CM = [], [], set(), 28.346
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
                if not bool(line["spans"][0]["flags"] & 16): p_f.append("Підрозділ не жирний"); highlights.append({"page": p_num, "x": line["bbox"][0], "y": line["bbox"][1], "w": line["bbox"][2]-line["bbox"][0], "h": line["bbox"][3]-line["bbox"][1]})
                if abs(line["bbox"][0] - l_bound - 1.5*CM) > 0.2*CM: p_f.append(f"Відступ {(line['bbox'][0]-l_bound)/CM:.1f} см замість 1.5"); highlights.append({"page": p_num, "x": l_bound, "y": line["bbox"][1], "w": line["bbox"][0]-l_bound, "h": line["bbox"][3]-line["bbox"][1]})
                if idx > 0 and (line["bbox"][1] - ls[idx-1]["bbox"][3]) < 20: p_f.append("Відсутній рядок зверху"); highlights.append({"page": p_num, "x": 0, "y": line["bbox"][1]-20, "w": p.rect.width, "h": 20})
                sub_ls, curr = [line], idx + 1
                while curr < len(ls):
                    l = ls[curr]
                    lt = "".join(s["text"] for s in l["spans"]).strip()
                    if re.match(r"^\d+", lt): break
                    if bool(l["spans"][0]["flags"] & 16) and (abs(l["bbox"][0]-line["bbox"][0])<10 or abs(l["bbox"][0]-l_bound)<10): sub_ls.append(l); curr += 1
                    else: break
                skip = curr - 1
                if curr < len(ls) and (ls[curr]["bbox"][1] - sub_ls[-1]["bbox"][3]) < 20: p_f.append("Відсутній рядок знизу"); highlights.append({"page": p_num, "x": 0, "y": sub_ls[-1]["bbox"][3], "w": p.rect.width, "h": 20})
                if p_f:
                    pages_with_errors.add(p_num)
                    for f in p_f: findings.append(f"Стор. {p_num}: {f}")
    return {"summary": "Підрозділи перевірено.", "findings": findings, "is_success": len(findings) == 0, "pages_with_errors": sorted(list(pages_with_errors)), "highlights": highlights}

def analyze_perelik(doc: fitz.Document) -> dict[str, Any]:
    findings, highlights, pages_with_errors = [], [], set()
    # Заборонені символи (буліти, крапки, зірочки тощо)
    wrong_symbols = ["•", "●", "○", "▪", "▫", "", "√", "*", ""]
    # Дозволені символи (тире)
    allowed_dashes = ["-", "–", "—"]
    
    for page_num in range(3, len(doc) + 1):
        page = doc[page_num-1]
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" not in b: continue
            for l in b["lines"]:
                spans = l["spans"]
                if not spans: continue
                text = "".join(s["text"] for s in spans).strip()
                if not text: continue
                
                # Перевіряємо перші символи рядка
                first_char = text[0]
                
                # Якщо рядок починається з будь-якого символу, крім літери, цифри або дозволеного тире
                if not first_char.isalnum() and first_char not in allowed_dashes:
                    # Якщо це один із заборонених символів або просто не тире
                    if first_char in wrong_symbols or not any(text.startswith(d) for d in allowed_dashes):
                        # Спеціальна перевірка, щоб не чіпляти пунктуацію (наприклад, лапки)
                        if first_char in [",", ".", ";", ":", '"', "(", ")", "«", "»"]:
                            continue
                            
                        findings.append(f"Стор. {page_num}: Використано маркер '{first_char}' замість тире.")
                        pages_with_errors.add(page_num)
                        # Підсвічуємо сам символ
                        highlights.append({
                            "page": page_num,
                            "x": spans[0]["bbox"][0],
                            "y": spans[0]["bbox"][1],
                            "w": spans[0]["bbox"][2] - spans[0]["bbox"][0],
                            "h": spans[0]["bbox"][3] - spans[0]["bbox"][1]
                        })
    
    return {"summary": "Перевірку переліків завершено.", "findings": findings, "is_success": len(findings) == 0, "pages_with_errors": sorted(list(pages_with_errors)), "highlights": highlights}

@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    try:
        pdf_bytes = base64.b64decode(request.pdf_base64)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if request.analysis_type == "page_numbers": res = analyze_page_numbers(doc)
        elif request.analysis_type == "general_text": res = analyze_general_text(doc)
        elif request.analysis_type == "chapters": res = analyze_chapters(doc)
        elif request.analysis_type == "subchapters": res = analyze_subchapters(doc)
        elif request.analysis_type == "perelik": res = analyze_perelik(doc)
        elif request.analysis_type == "zmist":
            rows, pw = extract_page_rows_fitz(doc, request.page_number)
            res = analyze_zmist(rows, pw)
        else: res = {"summary": "Unknown", "findings": [], "is_success": False}
        res["analysis_type"] = request.analysis_type
        return res
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
    finally: doc.close()
