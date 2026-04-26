from __future__ import annotations

import base64
import io
import re
import logging
from statistics import mean, median
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
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

class AnalyzeRequest(BaseModel):
    pdf_base64: str = Field(..., min_length=100)
    analysis_type: str = Field(default="zmist")
    page_number: int = Field(default=2, ge=1)

class AnalyzeAllRequest(BaseModel):
    pdf_base64: str = Field(..., min_length=100)


def open_pdf_document(pdf_base64: str) -> fitz.Document:
    pdf_bytes = base64.b64decode(pdf_base64)
    return fitz.open(stream=pdf_bytes, filetype="pdf")

def is_page_number_line(line: dict[str, Any], page_height: float) -> bool:
    txt = "".join(s["text"] for s in line["spans"] ).strip()
    return bool(txt) and txt.isdigit() and line["bbox"][1] > page_height - 60


def _point_xy(point: Any) -> tuple[float, float] | None:
    if hasattr(point, "x") and hasattr(point, "y"):
        return float(point.x), float(point.y)
    if isinstance(point, (tuple, list)) and len(point) >= 2:
        return float(point[0]), float(point[1])
    return None


def estimate_table_bottom_from_horizontal_rule(page: fitz.Page, table_bbox: Any) -> float:
    x0, y0, x1, y1 = [float(v) for v in table_bbox]
    min_overlap = max(40.0, (x1 - x0) * 0.35)
    best_y = None

    try:
        drawings = page.get_drawings()
    except Exception:
        return y1

    for drawing in drawings:
        for item in drawing.get("items", []):
            if not item:
                continue

            op = item[0]
            line_y = None
            line_left = None
            line_right = None

            if op == "l" and len(item) >= 3:
                p1 = _point_xy(item[1])
                p2 = _point_xy(item[2])
                if not p1 or not p2 or abs(p1[1] - p2[1]) > 1.5:
                    continue
                line_y = (p1[1] + p2[1]) / 2
                line_left = min(p1[0], p2[0])
                line_right = max(p1[0], p2[0])
            elif op == "re" and len(item) >= 2:
                rect = item[1]
                if not hasattr(rect, "x0") or not hasattr(rect, "y0") or not hasattr(rect, "x1") or not hasattr(rect, "y1"):
                    continue
                if abs(float(rect.y1) - float(rect.y0)) > 2.5:
                    continue
                line_y = (float(rect.y0) + float(rect.y1)) / 2
                line_left = float(rect.x0)
                line_right = float(rect.x1)

            if line_y is None or line_left is None or line_right is None:
                continue

            overlap = min(line_right, x1) - max(line_left, x0)
            if overlap < min_overlap:
                continue
            if line_y < y0 - 2 or line_y > y1 + 6:
                continue
            if best_y is None or line_y > best_y:
                best_y = line_y

    return best_y if best_y is not None else y1


def has_visual_anchor_above(page: fitz.Page, caption_bbox: Any, max_gap: float = 140) -> bool:
    _, caption_top, _, _ = [float(v) for v in caption_bbox]
    anchors: list[tuple[float, float, float, float]] = []

    try:
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type", 0) != 0 and "bbox" in block:
                bx0, by0, bx1, by1 = [float(v) for v in block["bbox"]]
                anchors.append((bx0, by0, bx1, by1))
    except Exception:
        pass

    try:
        for drawing in page.get_drawings():
            rect = drawing.get("rect")
            if rect is not None:
                anchors.append((float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)))
    except Exception:
        pass

    for _, _, _, anchor_bottom in anchors:
        gap = caption_top - anchor_bottom
        if -5 <= gap <= max_gap:
            return True

    return False


def is_first_text_line_below_visual_anchor(
    page: fitz.Page,
    all_lines: list[dict[str, Any]],
    line_index: int,
    max_gap: float = 140
) -> bool:
    target_line = all_lines[line_index]
    _, caption_top, _, _ = [float(v) for v in target_line["bbox"]]
    anchor_bottom = None

    anchors: list[tuple[float, float, float, float]] = []
    try:
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type", 0) != 0 and "bbox" in block:
                bx0, by0, bx1, by1 = [float(v) for v in block["bbox"]]
                anchors.append((bx0, by0, bx1, by1))
    except Exception:
        pass

    try:
        for drawing in page.get_drawings():
            rect = drawing.get("rect")
            if rect is not None:
                anchors.append((float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)))
    except Exception:
        pass

    for _, _, _, bottom in anchors:
        gap = caption_top - bottom
        if -5 <= gap <= max_gap:
            if anchor_bottom is None or bottom > anchor_bottom:
                anchor_bottom = bottom

    if anchor_bottom is None:
        return False

    for prev_line in all_lines[:line_index]:
        prev_text = "".join(s["text"] for s in prev_line["spans"]).strip()
        if not prev_text or is_page_number_line(prev_line, page.rect.height):
            continue
        if prev_line["bbox"][1] >= anchor_bottom and prev_line["bbox"][3] <= caption_top + 1:
            return False

    return True


def is_formula_candidate_text(text: str) -> bool:
    clean = normalize_text(text)
    tail_normalized = clean
    trailing_marks = " \t\r\n.,;:₴$€£¥"
    while True:
        previous = tail_normalized
        tail_normalized = tail_normalized.rstrip(trailing_marks)
        lowered = tail_normalized.lower()
        for suffix in ("грн", "дол", "тис"):
            if not lowered.endswith(suffix):
                continue
            boundary_index = len(tail_normalized) - len(suffix) - 1
            if boundary_index >= 0 and tail_normalized[boundary_index].isalnum():
                continue
            tail_normalized = tail_normalized[:-len(suffix)].rstrip(trailing_marks)
            break
        if tail_normalized == previous:
            break
    if "=" not in clean:
        return False
    if re.match(r"^\s*(?:https|http|htpp)\b", clean, flags=re.IGNORECASE):
        return False
    if re.match(r"^\s*\d", clean):
        return False
    if '"' in clean or "«" in clean:
        return False
    if re.match(r"^\s*[a-zа-яіїєґ]", clean, flags=re.IGNORECASE) and len(clean.split()) > 4:
        return False
    if not re.search(r"[A-Za-zА-Яа-яІіЇїЄє\d]", clean):
        return False
    if not re.search(r"[+\-*/^()]", clean):
        return False
    if re.search(r"[\d%]\s*$", tail_normalized):
        return False
    if len(clean.split()) > 18:
        return False
    return True


def analyze_formulas(doc: fitz.Document) -> dict[str, Any]:
    findings, highlights, pages_with_errors = [], [], set()
    formula_number_pattern = re.compile(r"\((\d+\.\d+)\)\s*$")

    for page_num in range(3, len(doc) + 1):
        page = doc[page_num - 1]
        page_width = page.rect.width
        lines = [
            l for b in page.get_text("dict")["blocks"]
            if "lines" in b
            for l in b["lines"]
        ]
        lines.sort(key=lambda x: x["bbox"][1])
        valid_lines = [
            l for l in lines
            if "".join(s["text"] for s in l["spans"]).strip()
            and not is_page_number_line(l, page.rect.height)
        ]

        for idx, line in enumerate(valid_lines):
            text = "".join(s["text"] for s in line["spans"]).strip()
            if not is_formula_candidate_text(text):
                continue

            p_f = []
            line_center = (line["bbox"][0] + line["bbox"][2]) / 2
            if abs(line_center - page_width / 2) > 70:
                p_f.append("Формула або рівняння має бути розміщене по центру сторінки")

            if not formula_number_pattern.search(text):
                p_f.append("Праворуч від формули має бути номер у дужках у форматі (розділ.порядковий_номер)")
            else:
                if (page_width - line["bbox"][2]) > 120:
                    p_f.append("Номер формули має бути у крайньому правому положенні на рядку")

            if idx > 0:
                gap_above = line["bbox"][1] - valid_lines[idx - 1]["bbox"][3]
                if gap_above < 18:
                    p_f.append("Вище формули має бути залишено щонайменше один вільний рядок")
            if idx < len(valid_lines) - 1:
                gap_below = valid_lines[idx + 1]["bbox"][1] - line["bbox"][3]
                if gap_below < 18:
                    p_f.append("Нижче формули має бути залишено щонайменше один вільний рядок")

            if p_f:
                pages_with_errors.add(page_num)
                findings.extend(f"Стор. {page_num}: {msg}" for msg in p_f)
                highlights.append({
                    "page": page_num,
                    "x": line["bbox"][0],
                    "y": line["bbox"][1],
                    "w": line["bbox"][2] - line["bbox"][0],
                    "h": line["bbox"][3] - line["bbox"][1]
                })

    return {
        "summary": "Формули перевірено.",
        "findings": findings,
        "is_success": len(findings) == 0,
        "pages_with_errors": sorted(list(pages_with_errors)),
        "highlights": highlights
    }

CM = 28.346 # 1 см в пунктах

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
                    all_spans.append({"text": txt, "x": s["bbox"][0], "y": s["bbox"][1], "w": s["bbox"][2]-s["bbox"][0], "h": s["bbox"][3]-s["bbox"][1], "font_size": s["size"], "is_bold": bool(s["flags"] & 16), "bbox": s["bbox"]})
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
        raw_rows.append({
            "text": full_text, 
            "clean": normalize_text(full_text), 
            "x": min(s["x"] for s in content_spans), 
            "y": bucket["y"], 
            "font_size": mean(s["font_size"] for s in content_spans), 
            "is_bold": any(s["is_bold"] for s in content_spans),
            "bbox": [min(s["bbox"][0] for s in content_spans), min(s["bbox"][1] for s in content_spans), max(s["bbox"][2] for s in content_spans), max(s["bbox"][3] for s in content_spans)]
        })
    return raw_rows, page_width

def analyze_zmist(rows: list[dict[str, Any]], page_width: float, page_num: int, page_height: float) -> dict[str, Any]:
    findings, highlights, pages_with_errors = [], [], set()
    if page_num != 2:
        return {
            "summary": "Зміст перевірено.",
            "findings": ['Перевірка вкладки "Зміст" виконується лише на 2 сторінці.'],
            "is_success": False,
            "pages_with_errors": [page_num],
            "highlights": [],
        }
    if not rows:
        return {
            "summary": "Текст не знайдено",
            "findings": ["Порожня сторінка"],
            "is_success": False,
            "pages_with_errors": [page_num],
            "highlights": [],
        }

    target_left = 2.5 * CM
    target_right = 1.0 * CM
    target_top = 2.0 * CM
    target_bottom = 2.0 * CM
    margin_tolerance = 0.35 * CM
    font_tolerance = 0.8

    found_title = False
    content_rows = [row for row in rows if row["clean"].upper() != "ЗМІСТ"]
    for r in rows:
        if "ЗМІСТ" == r["clean"].upper():
            found_title = True
            break
    if not found_title:
        findings.append('Не знайдено заголовок "ЗМІСТ".')
        pages_with_errors.add(page_num)

    if content_rows:
        actual_left = min(row["bbox"][0] for row in content_rows)
        actual_top = min(row["bbox"][1] for row in rows)
        actual_right = max(row["bbox"][2] for row in content_rows)
        actual_bottom = max(row["bbox"][3] for row in content_rows)

        if abs(actual_left - target_left) > margin_tolerance:
            findings.append(f"Стор. {page_num}: Ліве поле змісту має бути 2,5 см.")
            pages_with_errors.add(page_num)
            highlights.append({"page": page_num, "x": 0, "y": 0, "w": actual_left, "h": page_height})
        if abs(actual_top - target_top) > margin_tolerance:
            findings.append(f"Стор. {page_num}: Верхнє поле змісту має бути 2 см.")
            pages_with_errors.add(page_num)
            highlights.append({"page": page_num, "x": 0, "y": 0, "w": page_width, "h": actual_top})
        if abs((page_width - actual_right) - target_right) > margin_tolerance:
            findings.append(f"Стор. {page_num}: Праве поле змісту має бути 1 см.")
            pages_with_errors.add(page_num)
            highlights.append({"page": page_num, "x": actual_right, "y": 0, "w": max(1, page_width - actual_right), "h": page_height})
        if (page_height - actual_bottom) < (target_bottom - margin_tolerance):
            findings.append(f"Стор. {page_num}: Нижнє поле змісту має бути не менше 2 см.")
            pages_with_errors.add(page_num)
            highlights.append({"page": page_num, "x": 0, "y": actual_bottom, "w": page_width, "h": max(1, page_height - actual_bottom)})

    # Для змісту службові елементи на кшталт "ВСТУП", "ДОДАТКИ",
    # "СПИСОК ВИКОРИСТАНИХ ДЖЕРЕЛ" перевіряємо як назви розділів:
    # великими літерами, жирним шрифтом, без вимоги абзацного відступу 1,5 см.
    major_pattern = r"^(ВСТУП|РОЗДІЛ\s+\d+|ВИСНОВКИ|ДОДАТКИ|СПИСОК ВИКОРИСТАНИХ ДЖЕРЕЛ)"
    subpoint_pattern = r"^\d+(\.\d+)+"
    for row in content_rows:
        c = row["clean"].upper()
        bbox = row["bbox"]
        if abs(row["font_size"] - 14) > font_tolerance:
            findings.append(f'Стор. {page_num}: Рядок "{row["clean"][:40]}" у змісті має бути шрифтом 14 пт.')
            pages_with_errors.add(page_num)
            highlights.append({"page": page_num, "x": bbox[0], "y": bbox[1], "w": bbox[2] - bbox[0], "h": bbox[3] - bbox[1]})

        if re.match(major_pattern, c):
            if row["clean"] != c:
                findings.append(f'Стор. {page_num}: Елемент "{row["clean"][:40]}" у змісті має бути великими літерами.')
                pages_with_errors.add(page_num)
                highlights.append({"page": page_num, "x": bbox[0], "y": bbox[1], "w": bbox[2] - bbox[0], "h": bbox[3] - bbox[1]})
            if not row["is_bold"]:
                findings.append(f'Стор. {page_num}: Елемент "{row["clean"][:40]}" у змісті має бути жирним шрифтом.')
                pages_with_errors.add(page_num)
                highlights.append({"page": page_num, "x": bbox[0], "y": bbox[1], "w": bbox[2] - bbox[0], "h": bbox[3] - bbox[1]})
        elif re.match(subpoint_pattern, c):
            if row["is_bold"]:
                findings.append(f'Стор. {page_num}: Підпункт "{row["clean"][:40]}" у змісті не повинен бути жирним.')
                pages_with_errors.add(page_num)
                highlights.append({"page": page_num, "x": bbox[0], "y": bbox[1], "w": bbox[2] - bbox[0], "h": bbox[3] - bbox[1]})

    return {
        "summary": "Зміст перевірено.",
        "findings": findings,
        "is_success": len(findings) == 0,
        "pages_with_errors": sorted(list(pages_with_errors)),
        "highlights": highlights,
    }

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
            findings.append(f"Стор. 3: Нумерація сторінок (знайдено '{digits}' замість '3').")
            highlights.append({"page": 3, "x": rect.width - 100, "y": 0, "w": 100, "h": 100})
    return {"summary": "Нумерація перевірена.", "findings": findings, "is_success": len(findings) == 0, "highlights": highlights}

def analyze_general_text(doc: fitz.Document) -> dict[str, Any]:
    findings, highlights, pages_with_errors = [], [], set()
    TARGET_LEFT = 2.5 * CM
    TARGET_RIGHT = 1.0 * CM
    INDENT = 1.5 * CM
    INDENT_TOLERANCE = 0.2 * CM

    appendices_start_page = None
    for page_num in range(3, len(doc) + 1):
        page = doc[page_num - 1]
        lines = [
            "".join(s["text"] for s in l["spans"]).strip()
            for b in page.get_text("dict")["blocks"] if "lines" in b
            for l in b["lines"]
            if "".join(s["text"] for s in l["spans"]).strip()
            and not is_page_number_line(l, page.rect.height)
        ]
        if any(re.match(r"^ДОДАТКИ\b", line, re.IGNORECASE) for line in lines):
            appendices_start_page = page_num
            break

    def page_starts_with_new_section(page_num: int) -> bool:
        if page_num < 1 or page_num > len(doc):
            return False
        page = doc[page_num - 1]
        lines = [
            l for b in page.get_text("dict")["blocks"] if "lines" in b
            for l in b["lines"]
            if "".join(s["text"] for s in l["spans"]).strip()
            and not is_page_number_line(l, page.rect.height)
            and not "".join(s["text"] for s in l["spans"]).strip().isdigit()
        ]
        if not lines:
            return False

        top_line = min(lines, key=lambda item: (item["bbox"][1], item["bbox"][0]))
        top_text = "".join(s["text"] for s in top_line["spans"]).strip().upper()
        if top_line["bbox"][1] > 120:
            return False

        return bool(re.match(r"^(РОЗДІЛ\s+\d+|ВСТУП|ВИСНОВКИ|СПИСОК ВИКОРИСТАНИХ ДЖЕРЕЛ|ДОДАТКИ|ДОДАТОК\b)", top_text))

    for p_num in range(3, len(doc) + 1):
        if appendices_start_page is not None and p_num >= appendices_start_page:
            break
        p = doc[p_num-1]
        page_dict = p.get_text("dict")
        w, h = p.rect.width, p.rect.height
        els = [
            l for b in page_dict["blocks"] if "lines" in b
            for l in b["lines"]
            if "".join(s["text"] for s in l["spans"]).strip()
            and not ("".join(s["text"] for s in l["spans"]).strip().isdigit() and l["bbox"][1] < 60)
        ]
        if not els:
            continue

        raw_text_lines = []
        for line in els:
            text = "".join(s["text"] for s in line["spans"]).strip()
            x0, y0, x1, y1 = line["bbox"]
            raw_text_lines.append({
                "line": line,
                "text": text,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "width": x1 - x0,
            })

        figure_caption_tops = [
            item["y0"]
            for item in raw_text_lines
            if re.match(r"^Рисунок\s+\d", item["text"], re.IGNORECASE)
        ]
        first_figure_caption_top = min(figure_caption_tops) if figure_caption_tops else None
        table_caption_tops = [
            item["y0"]
            for item in raw_text_lines
            if re.match(r"^Таблиця\s+\d", item["text"], re.IGNORECASE)
        ]
        first_table_caption_top = min(table_caption_tops) if table_caption_tops else None

        # Основний текст для цієї перевірки беремо лише з основного текстового потоку:
        # відсікаємо підпис рисунка/таблиці і текст, що належить самим об'єктам нижче.
        flow_candidates = []
        for item in raw_text_lines:
            text = item["text"]
            if re.match(r"^(Таблиця|Рисунок|Джерело)\b", text, re.IGNORECASE):
                continue
            if first_figure_caption_top is not None and item["y0"] >= first_figure_caption_top - 6:
                continue
            if first_table_caption_top is not None and item["y0"] >= first_table_caption_top - 6:
                continue
            flow_candidates.append(item)

        if not flow_candidates:
            continue

        left_candidates = [
            item["x0"] for item in flow_candidates
            if abs(item["x0"] - TARGET_LEFT) <= 0.8 * CM
        ]
        body_left = median(left_candidates) if left_candidates else median(item["x0"] for item in flow_candidates)

        text_flow = [
            item for item in flow_candidates
            if item["x0"] <= body_left + INDENT + 0.4 * CM
        ]
        if not text_flow:
            text_flow = flow_candidates

        actual_left = min(item["x0"] for item in text_flow)
        actual_bottom_y = max(item["y1"] for item in text_flow)
        occupied_bottom_candidates = [item["y1"] for item in raw_text_lines]
        for block in page_dict["blocks"]:
            bbox = block.get("bbox")
            if not bbox:
                continue
            if "lines" not in block or block.get("type") != 0:
                occupied_bottom_candidates.append(float(bbox[3]))
        try:
            for drawing in p.get_drawings():
                rect = drawing.get("rect")
                if rect is not None:
                    occupied_bottom_candidates.append(float(rect.y1))
        except Exception:
            pass
        occupied_bottom_y = max(occupied_bottom_candidates) if occupied_bottom_candidates else actual_bottom_y
        first_top_line = min(raw_text_lines, key=lambda item: (item["y0"], item["x0"]))
        has_appendix_heading_at_top = bool(
            re.match(r"^ДОДАТОК\b", first_top_line["text"].strip(), re.IGNORECASE)
            and first_top_line["y0"] < 120
        )
        next_page_starts_new_section = page_starts_with_new_section(p_num + 1)
        p_f = []
        if abs(actual_left - TARGET_LEFT) > 0.5 * CM:
            p_f.append("Ліве поле")
            highlights.append({"page": p_num, "x": 0, "y": 0, "w": actual_left, "h": h})
        if not has_appendix_heading_at_top and not next_page_starts_new_section and (h - occupied_bottom_y) > 4.5 * CM:
            p_f.append("Порожнє місце знизу")
            highlights.append({"page": p_num, "x": 0, "y": occupied_bottom_y, "w": w, "h": h - occupied_bottom_y})

        def is_regular_text_line(line: dict[str, Any]) -> bool:
            text = "".join(s["text"] for s in line["spans"]).strip()
            if not text:
                return False
            upper = text.upper()
            center = (line["bbox"][0] + line["bbox"][2]) / 2
            if abs(center - w / 2) < 55 and len(text) < 60:
                return False
            if upper == text and len(text) < 80:
                return False
            if re.match(r"^(РОЗДІЛ\s+\d+|ВСТУП|ВИСНОВКИ|ДОДАТКИ|СПИСОК ВИКОРИСТАНИХ ДЖЕРЕЛ)\b", upper):
                return False
            if re.match(r"^\d+(\.\d+)*\.?\s+", text):
                return False
            if re.match(r"^(Таблиця|Рисунок|Джерело|Формула|Рівняння)\b", text, re.IGNORECASE):
                return False
            if "=" in text:
                return False
            if re.search(r"\b\d+\s*[+\-*/=]\s*\d+", text):
                return False
            if re.search(r"[+\-*/^=()]", text) and re.search(r"\d", text):
                return False
            return True

        def starts_with_capital_word(text: str) -> bool:
            match = re.match(r'^[«"(\[]*([A-ZА-ЯІЇЄҐ][A-Za-zА-Яа-яІіЇїЄєҐґ\'’`\-]*)', text)
            return match is not None

        candidate_lines = [item["line"] for item in text_flow]
        regular_lines = [line for line in sorted(candidate_lines, key=lambda item: item["bbox"][1]) if is_regular_text_line(line)]
        expected_indent_x = body_left + INDENT

        paragraph_indents = []
        indent_highlight = None
        for idx in range(1, len(regular_lines)):
            line = regular_lines[idx]
            prev_line = regular_lines[idx - 1]
            line_text = "".join(s["text"] for s in line["spans"]).strip()

            prev_right_gap = (w - TARGET_RIGHT) - prev_line["bbox"][2]
            vertical_gap = line["bbox"][1] - prev_line["bbox"][3]
            starts_new_paragraph = prev_right_gap > 0.9 * CM or vertical_gap > 6
            if starts_new_paragraph and not starts_with_capital_word(line_text):
                continue
            if not starts_new_paragraph:
                continue

            current_x = line["bbox"][0]
            indent_size = current_x - body_left
            if indent_size <= 0.35 * CM:
                continue
            paragraph_indents.append(indent_size)
            if indent_highlight is None:
                indent_highlight = {"page": p_num, "x": current_x, "y": 0, "w": 0, "h": h, "kind": "indent"}

        if paragraph_indents:
            median_indent = median(paragraph_indents)
            if abs(median_indent - INDENT) > INDENT_TOLERANCE:
                actual_indent_cm = median_indent / CM
                p_f.append(f"Абзацний відступ на сторінці має бути 1,5 см. Фактично: {actual_indent_cm:.1f} см")
                indent_x = body_left + median_indent
                indent_highlight = {"page": p_num, "x": indent_x, "y": 0, "w": 0, "h": h, "kind": "indent"}
                highlights.append(indent_highlight)

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
        ls.sort(key=lambda x: x["bbox"][1])
        for idx, line in enumerate(ls):
            txt = "".join(s["text"] for s in line["spans"]).strip()
            if re.match(r"^РОЗДІЛ\s+\d+", txt.upper()):
                p_f = []
                if line["bbox"][1] > 120: 
                    p_f.append("РОЗДІЛ не з нової сторінки"); highlights.append({"page": p_num, "x": 0, "y": 0, "w": p.rect.width, "h": line["bbox"][1]})
                if abs((line["bbox"][0]+line["bbox"][2])/2 - p.rect.width/2) > 45: 
                    p_f.append("РОЗДІЛ не по центру"); highlights.append({"page": p_num, "x": line["bbox"][0], "y": line["bbox"][1], "w": line["bbox"][2]-line["bbox"][0], "h": line["bbox"][3]-line["bbox"][1]})
                
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
                        if t_txt != t_txt.upper() or not bool(tl["spans"][0]["flags"] & 16): 
                            p_f.append(f"Назва '{t_txt[:10]}' має бути ВЕЛИКИМИ ЖИРНИМИ"); highlights.append({"page": p_num, "x": tl["bbox"][0], "y": tl["bbox"][1], "w": tl["bbox"][2]-tl["bbox"][0], "h": tl["bbox"][3]-tl["bbox"][1]})
                    if curr < len(ls) and (ls[curr]["bbox"][1] - title_ls[-1]["bbox"][3]) < 18: 
                        p_f.append("Відсутній рядок після назви"); highlights.append({"page": p_num, "x": 0, "y": title_ls[-1]["bbox"][3], "w": p.rect.width, "h": 18})
                
                if p_f:
                    pages_with_errors.add(p_num)
                    for f in p_f: findings.append(f"Стор. {p_num}: {f}")
    return {"summary": "Розділи перевірено.", "findings": findings, "is_success": len(findings) == 0, "pages_with_errors": sorted(list(pages_with_errors)), "highlights": highlights}

def analyze_subchapters(doc: fitz.Document) -> dict[str, Any]:
    findings, highlights, pages_with_errors = [], [], set()
    for p_num in range(3, len(doc) + 1):
        p = doc[p_num-1]
        ls = [l for b in p.get_text("dict")["blocks"] if "lines" in b for l in b["lines"] if "".join(s["text"] for s in l["spans"]).strip()]
        if not ls: continue
        ls.sort(key=lambda x: x["bbox"][1])
        
        left_candidates = [l["bbox"][0] for l in ls if abs(l["bbox"][0] - 2.5 * CM) <= 0.8 * CM]
        l_bound = min(left_candidates) if left_candidates else 2.5 * CM
        
        skip = -1
        for idx, line in enumerate(ls):
            if idx <= skip: continue
            txt = "".join(s["text"] for s in line["spans"]).strip()
            if re.match(r"^[1-3]\.[1-3]\.?\s+", txt):
                p_f = []
                if not bool(line["spans"][0]["flags"] & 16):
                    p_f.append("Підрозділ не жирний"); highlights.append({"page": p_num, "x": line["bbox"][0], "y": line["bbox"][1], "w": line["bbox"][2]-line["bbox"][0], "h": line["bbox"][3]-line["bbox"][1]})
                
                current_x = line["bbox"][0]
                expected_x = l_bound + 1.5 * CM
                if abs(current_x - expected_x) > 0.2 * CM:
                    p_f.append(f"Відступ {(current_x - l_bound)/CM:.1f} см замість 1.5"); highlights.append({"page": p_num, "x": l_bound, "y": line["bbox"][1], "w": current_x - l_bound, "h": line["bbox"][3]-line["bbox"][1]})
                
                if idx > 0 and (line["bbox"][1] - ls[idx-1]["bbox"][3]) < 18: 
                    p_f.append("Відсутній рядок зверху"); highlights.append({"page": p_num, "x": 0, "y": line["bbox"][1]-18, "w": p.rect.width, "h": 18})
                
                sub_ls, curr = [line], idx + 1
                while curr < len(ls):
                    l = ls[curr]
                    lt = "".join(s["text"] for s in l["spans"]).strip()
                    if re.match(r"^\d+", lt): break
                    gap = l["bbox"][1] - sub_ls[-1]["bbox"][3]
                    is_bold = bool(l["spans"][0]["flags"] & 16)
                    if is_bold and gap < 18: 
                        sub_ls.append(l)
                        curr += 1
                    else: 
                        break
                
                skip = curr - 1
                if curr < len(ls) and (ls[curr]["bbox"][1] - sub_ls[-1]["bbox"][3]) < 18: 
                    p_f.append("Відсутній рядок знизу"); highlights.append({"page": p_num, "x": 0, "y": sub_ls[-1]["bbox"][3], "w": p.rect.width, "h": 18})
                
                if p_f:
                    pages_with_errors.add(p_num)
                    for f in p_f: findings.append(f"Стор. {p_num}: {f}")
    return {"summary": "Підрозділи перевірено.", "findings": findings, "is_success": len(findings) == 0, "pages_with_errors": sorted(list(pages_with_errors)), "highlights": highlights}

def analyze_references_section(doc: fitz.Document) -> dict[str, Any]:
    findings, highlights, pages_with_errors = [], [], set()
    target_title = "СПИСОК ВИКОРИСТАНИХ ДЖЕРЕЛ"
    matches: list[tuple[int, dict[str, Any]]] = []
    ignored_pages = {2, 3, 4}

    for page_num in range(1, len(doc) + 1):
        if page_num in ignored_pages:
            continue
        rows, page_width = extract_page_rows_fitz(doc, page_num)
        for row in rows:
            clean_upper = row["clean"].upper()
            if target_title not in clean_upper:
                continue
            matches.append((page_num, row | {"page_width": page_width}))

    if not matches:
        findings.append('Не знайдено заголовок "Список використаних джерел" у тексті документа (сторінки 2, 3 і 4 ігноруються).')
        return {
            "summary": "Список використаних джерел перевірено.",
            "findings": findings,
            "is_success": False,
            "pages_with_errors": [],
            "highlights": [],
        }

    if len(matches) > 1:
        pages = ", ".join(str(page_num) for page_num, _ in matches)
        findings.append(f'Заголовок "Список використаних джерел" знайдено кілька разів: стор. {pages}. Має бути один окремий розділ.')
        pages_with_errors.update(page_num for page_num, _ in matches)

    for page_num, row in matches:
        page_width = row["page_width"]
        row_text = row["clean"]
        row_upper = row_text.upper()
        row_errors = []

        if row_upper != target_title:
            row_errors.append('Заголовок має бути саме "Список використаних джерел" без зайвого тексту в цьому рядку')
        if not row["is_bold"]:
            row_errors.append('Заголовок "Список використаних джерел" має бути жирним')
        row_center = (row["bbox"][0] + row["bbox"][2]) / 2
        if abs(row_center - page_width / 2) > 45:
            row_errors.append('Заголовок "Список використаних джерел" має бути розміщений по центру')

        if row_errors:
            pages_with_errors.add(page_num)
            highlights.append({
                "page": page_num,
                "x": row["bbox"][0],
                "y": row["bbox"][1],
                "w": row["bbox"][2] - row["bbox"][0],
                "h": row["bbox"][3] - row["bbox"][1],
            })
            findings.extend(f"Стор. {page_num}: {error}" for error in row_errors)

    return {
        "summary": "Список використаних джерел перевірено.",
        "findings": findings,
        "is_success": len(findings) == 0,
        "pages_with_errors": sorted(list(pages_with_errors)),
        "highlights": highlights,
    }

def analyze_appendices(doc: fitz.Document) -> dict[str, Any]:
    findings, highlights, pages_with_errors = [], [], set()
    appendix_pattern = re.compile(r"^ДОДАТОК\s+([А-Я])$", re.IGNORECASE)
    found_any = False

    for page_num in range(1, len(doc) + 1):
        rows, page_width = extract_page_rows_fitz(doc, page_num)
        page_height = doc[page_num - 1].rect.height

        for row in rows:
            clean = row["clean"]
            if not appendix_pattern.match(clean):
                continue

            found_any = True
            row_errors = []
            x0, y0, x1, y1 = row["bbox"]

            # Верхній правий кут: рядок має бути достатньо високо і праворуч.
            if y0 > 120:
                row_errors.append('Позначення "Додаток X" має бути у верхньому правому куті сторінки')
            if x1 < page_width - 70:
                row_errors.append('Позначення "Додаток X" має бути вирівняне до правого краю сторінки')

            if row_errors:
                pages_with_errors.add(page_num)
                highlights.append({
                    "page": page_num,
                    "x": x0,
                    "y": y0,
                    "w": x1 - x0,
                    "h": y1 - y0,
                })
                findings.extend(f"Стор. {page_num}: {error}" for error in row_errors)

    if not found_any:
        return {
            "summary": "Додатки перевірено.",
            "findings": [],
            "is_success": True,
            "pages_with_errors": [],
            "highlights": [],
        }

    return {
        "summary": "Додатки перевірено.",
        "findings": findings,
        "is_success": len(findings) == 0,
        "pages_with_errors": sorted(list(pages_with_errors)),
        "highlights": highlights,
    }

def analyze_perelik(doc: fitz.Document) -> dict[str, Any]:
    findings, highlights, pages_with_errors = [], [], set()
    allowed_dashes = ["-", "–", "—"]
    excluded_starts = [",", ".", ";", ":", '"', "(", ")", "«", "»", "{", "["]
    
    for page_num in range(3, len(doc) + 1):
        page = doc[page_num-1]
        lines = [l for b in page.get_text("dict")["blocks"] if "lines" in b for l in b["lines"]]
        
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            text = "".join(s["text"] for s in line["spans"]).strip()
            if not text:
                idx += 1; continue
                
            first_char = text[0]
            
            # Якщо це потенційний "неправильний" маркер
            if not first_char.isalnum() and first_char not in allowed_dashes and first_char not in excluded_starts:
                # Перевіряємо, чи є такий самий символ у наступних рядках (формуємо блок списку)
                list_block = [line]
                next_idx = idx + 1
                while next_idx < len(lines):
                    next_line = lines[next_idx]
                    next_text = "".join(s["text"] for s in next_line["spans"]).strip()
                    if next_text and next_text[0] == first_char:
                        list_block.append(next_line)
                        next_idx += 1
                    elif not next_text: # Пропускаємо порожні рядки всередині списку
                        next_idx += 1
                    else:
                        break
                
                # Якщо знайдено 2 або більше однакових символи - це перелік
                if len(list_block) >= 2:
                    findings.append(f"Стор. {page_num}: Виявлено перелік з використанням '{first_char}' (має бути тире).")
                    pages_with_errors.add(page_num)
                    for l_item in list_block:
                        highlights.append({
                            "page": page_num,
                            "x": l_item["spans"][0]["bbox"][0],
                            "y": l_item["spans"][0]["bbox"][1],
                            "w": l_item["spans"][0]["bbox"][2] - l_item["spans"][0]["bbox"][0],
                            "h": l_item["spans"][0]["bbox"][3] - l_item["spans"][0]["bbox"][1]
                        })
                    idx = next_idx # Пропускаємо весь блок
                else:
                    idx += 1
            else:
                idx += 1
    
    return {"summary": "Перевірку переліків завершено.", "findings": findings, "is_success": len(findings) == 0, "pages_with_errors": sorted(list(pages_with_errors)), "highlights": highlights}

def analyze_tables(doc: fitz.Document) -> dict[str, Any]:
    findings, highlights, pages_with_errors = [], [], set()
    for page_num in range(3, len(doc) + 1):
        page = doc[page_num-1]
        try:
            tabs = page.find_tables()
        except Exception:
            continue
        if not tabs.tables: continue
        
        all_lines = [l for b in page.get_text("dict")["blocks"] if "lines" in b for l in b["lines"]]
        all_lines.sort(key=lambda x: x["bbox"][1]) # Сортуємо зверху вниз
        
        # Фактичне ліве поле тексту студента
        l_bound = min([l["bbox"][0] for l in all_lines if l["bbox"][0] < 150] or [2.5*CM])
        
        for tab in tabs:
            p_f = []
            t_bbox = tab.bbox
            
            # Перевірка: чи не є ця "таблиця" насправді графіком/рисунком?
            # Часто графіки з сіткою розпізнаються як таблиці.
            # Якщо під об'єктом є підпис "Рисунок", ми його ігноруємо тут.
            is_figure = False
            for l in all_lines:
                if l["bbox"][1] > t_bbox[3] and (l["bbox"][1] - t_bbox[3]) < 60:
                    bottom_txt = "".join(s["text"] for s in l["spans"]).strip().lower()
                    if bottom_txt.startswith("рис.") or bottom_txt.startswith("рисунок"):
                        is_figure = True
                        break
            if is_figure: continue
            
            t_center = (t_bbox[0] + t_bbox[2]) / 2
            # Центр розраховується з урахуванням полів (ліве 2.5 см, праве 1 см)
            expected_center = (2.5 * CM + (page.rect.width - 1.0 * CM)) / 2
            
            if abs(t_center - expected_center) > 15: 
                p_f.append("Таблиця не по центру (відносно полів)"); highlights.append({"page": page_num, "x": t_bbox[0], "y": t_bbox[1], "w": t_bbox[2]-t_bbox[0], "h": t_bbox[3]-t_bbox[1]})
            
            lines_above = [l for l in all_lines if l["bbox"][3] < t_bbox[1]]
            lines_above.reverse() # Знизу вгору від таблиці
            
            caption_lines = []
            found_idx = -1
            for i, l in enumerate(lines_above):
                if (t_bbox[1] - l["bbox"][3]) > 80: break # Не шукаємо надто високо
                txt = "".join(s["text"] for s in l["spans"]).strip().lower()
                if txt.startswith("таблиця") or txt.startswith("продовження") or txt.startswith("кінец"):
                    found_idx = i
                    break
            
            if found_idx != -1:
                # Беремо всі текстові блоки від слова "Таблиця" до самої таблиці
                caption_lines = [lines_above[j] for j in range(found_idx, -1, -1)]
            else:
                # Fallback, якщо не знайшли слова "Таблиця"
                for l in lines_above:
                    if not caption_lines:
                        if (t_bbox[1] - l["bbox"][3]) < 40: caption_lines.append(l)
                    else:
                        if (caption_lines[-1]["bbox"][1] - l["bbox"][3]) < 30:
                            caption_lines.append(l)
                            if len(caption_lines) == 2: break
                        else: break
                caption_lines.reverse()
                
            if not caption_lines:
                p_f.append("Не знайдено назву над таблицею")
                highlights.append({"page": page_num, "x": t_bbox[0], "y": t_bbox[1], "w": t_bbox[2]-t_bbox[0], "h": 20})
            else:
                txt = " ".join("".join(s["text"] for s in l["spans"]).strip() for l in caption_lines)
                first_line = caption_lines[0]
                lower_txt = txt.lower()
                
                # Ігноруємо перевірку формату, якщо це перенесення таблиці
                if lower_txt.startswith("продовження") or lower_txt.startswith("кінец"):
                    pass 
                elif not re.match(r"^Таблиця\s+\d+(\.\d+)*\s*[^\w\s]+\s*.+", txt, re.IGNORECASE): 
                    p_f.append(f"Невірний формат назви: '{txt[:20]}...'")
                    for cl in caption_lines:
                        highlights.append({"page": page_num, "x": cl["bbox"][0], "y": cl["bbox"][1], "w": cl["bbox"][2]-cl["bbox"][0], "h": cl["bbox"][3]-cl["bbox"][1]})
                else:
                    span = first_line["spans"][0]
                    if abs(span["size"] - 14) > 1: 
                        p_f.append(f"Розмір шрифту {span['size']:.1f} замість 14")
                        for cl in caption_lines: highlights.append({"page": page_num, "x": cl["bbox"][0], "y": cl["bbox"][1], "w": cl["bbox"][2]-cl["bbox"][0], "h": cl["bbox"][3]-cl["bbox"][1]})
                    if bool(span["flags"] & 16) or bool(span["flags"] & 2): 
                        p_f.append("Стиль шрифту (жирний/курсив) не дозволено")
                        for cl in caption_lines: highlights.append({"page": page_num, "x": cl["bbox"][0], "y": cl["bbox"][1], "w": cl["bbox"][2]-cl["bbox"][0], "h": cl["bbox"][3]-cl["bbox"][1]})
                    if abs(first_line["bbox"][0] - (l_bound + 1.5 * CM)) > 0.3 * CM: 
                        p_f.append(f"Відступ назви {(first_line['bbox'][0] - l_bound)/CM:.1f} см від краю тексту (має бути 1.5 см)")
                        highlights.append({"page": page_num, "x": l_bound, "y": first_line["bbox"][1], "w": first_line["bbox"][0]-l_bound, "h": first_line["bbox"][3]-first_line["bbox"][1]})
            
            if p_f:
                pages_with_errors.add(page_num)
                for f in p_f: findings.append(f"Стор. {page_num}: {f}")
    return {"summary": "Таблиці перевірено.", "findings": findings, "is_success": len(findings) == 0, "pages_with_errors": sorted(list(pages_with_errors)), "highlights": highlights}

def analyze_figures(doc: fitz.Document) -> dict[str, Any]:
    findings, highlights, pages_with_errors = [], [], set()
    for page_num in range(3, len(doc) + 1):
        page = doc[page_num-1]
        all_lines = [l for b in page.get_text("dict")["blocks"] if "lines" in b for l in b["lines"]]
        all_lines.sort(key=lambda x: x["bbox"][1])
        
        l_bound = min([l["bbox"][0] for l in all_lines if l["bbox"][0] < 150] or [2.5*CM])
        
        idx = 0
        while idx < len(all_lines):
            l = all_lines[idx]
            txt = "".join(s["text"] for s in l["spans"]).strip()
            if txt.startswith("Рис.") or txt.startswith("Рисунок"):
                if not has_visual_anchor_above(page, l["bbox"]) or not is_first_text_line_below_visual_anchor(page, all_lines, idx):
                    idx += 1
                    continue
                caption_lines = [l]
                next_idx = idx + 1
                if next_idx < len(all_lines):
                    next_l = all_lines[next_idx]
                    if (next_l["bbox"][1] - l["bbox"][3]) < 25:
                        caption_lines.append(next_l)
                        idx += 1
                
                full_txt = " ".join("".join(s["text"] for s in cl["spans"]).strip() for cl in caption_lines)
                first_line = caption_lines[0]
                p_f = []
                
                if full_txt.lower().startswith("рис."):
                    p_f.append("Має бути 'Рисунок', а не 'Рис.'")
                    for cl in caption_lines:
                        highlights.append({"page": page_num, "x": cl["bbox"][0], "y": cl["bbox"][1], "w": cl["bbox"][2]-cl["bbox"][0], "h": cl["bbox"][3]-cl["bbox"][1]})
                elif not re.match(r"^Рисунок\s+\d+(\.\d+)*\s*[^\w\s]+\s*.+", full_txt, re.IGNORECASE):
                    p_f.append(f"Невірний формат підпису рисунка: '{full_txt[:30]}...'")
                    for cl in caption_lines:
                        highlights.append({"page": page_num, "x": cl["bbox"][0], "y": cl["bbox"][1], "w": cl["bbox"][2]-cl["bbox"][0], "h": cl["bbox"][3]-cl["bbox"][1]})
                else:
                    span = first_line["spans"][0]
                    if abs(span["size"] - 14) > 1:
                        p_f.append(f"Розмір шрифту підпису {span['size']:.1f} замість 14")
                        for cl in caption_lines: highlights.append({"page": page_num, "x": cl["bbox"][0], "y": cl["bbox"][1], "w": cl["bbox"][2]-cl["bbox"][0], "h": cl["bbox"][3]-cl["bbox"][1]})
                    if bool(span["flags"] & 16) or bool(span["flags"] & 2):
                        p_f.append("Стиль шрифту підпису (жирний/курсив) не дозволено")
                        for cl in caption_lines: highlights.append({"page": page_num, "x": cl["bbox"][0], "y": cl["bbox"][1], "w": cl["bbox"][2]-cl["bbox"][0], "h": cl["bbox"][3]-cl["bbox"][1]})
                    
                    if abs(first_line["bbox"][0] - (l_bound + 1.5 * CM)) > 0.3 * CM:
                        p_f.append(f"Відступ підпису {(first_line['bbox'][0] - l_bound)/CM:.1f} см від краю тексту (має бути 1.5 см)")
                        highlights.append({"page": page_num, "x": l_bound, "y": first_line["bbox"][1], "w": first_line["bbox"][0]-l_bound, "h": first_line["bbox"][3]-first_line["bbox"][1]})
                            
                if p_f:
                    pages_with_errors.add(page_num)
                    for f in p_f: findings.append(f"Стор. {page_num}: {f}")
            idx += 1
            
    return {"summary": "Рисунки перевірено.", "findings": findings, "is_success": len(findings) == 0, "pages_with_errors": sorted(list(pages_with_errors)), "highlights": highlights}

def analyze_table_breaks(doc: fitz.Document) -> dict[str, Any]:
    findings, highlights, pages_with_errors = [], [], set()
    for page_num in range(3, len(doc) + 1):
        page = doc[page_num-1]
        try:
            tabs = page.find_tables()
        except Exception:
            continue
        if not tabs.tables: continue
        
        all_lines = [l for b in page.get_text("dict")["blocks"] if "lines" in b for l in b["lines"]]
        all_lines.sort(key=lambda x: x["bbox"][1])
        
        # Відфільтровуємо номери сторінок (цифри зверху)
        valid_lines = [l for l in all_lines if not ("".join(s["text"] for s in l["spans"]).strip().isdigit() and l["bbox"][1] < 60)]
        if not valid_lines: continue
        
        first_tab = tabs[0]
        t_bbox = first_tab.bbox
        
        # Перевіряємо, чи таблиця знаходиться на початку аркуша (верхня межа < 150 pt, тобто ~5.3 см)
        if t_bbox[1] < 150:
            lines_above = [l for l in valid_lines if l["bbox"][3] < t_bbox[1]]
            
            caption_l = None
            for l in lines_above:
                txt = "".join(s["text"] for s in l["spans"]).strip().lower()
                if txt.startswith("таблиця") or txt.startswith("продовження") or txt.startswith("кінец"):
                    caption_l = l
                    break
                    
            if caption_l:
                txt = "".join(s["text"] for s in caption_l["spans"]).strip()
                lower_txt = txt.lower()
                if lower_txt.startswith("продовження") or lower_txt.startswith("кінец"):
                    # Перевіряємо вирівнювання по правому краю
                    expected_right = page.rect.width - 1.0 * CM
                    actual_right = caption_l["bbox"][2]
                    # Допуск близько 1 см (30 pt)
                    if abs(expected_right - actual_right) > 30:
                        p_f = f"Текст '{txt[:20]}...' не вирівняний по правому краю"
                        findings.append(f"Стор. {page_num}: {p_f}")
                        pages_with_errors.add(page_num)
                        highlights.append({"page": page_num, "x": caption_l["bbox"][0], "y": caption_l["bbox"][1], "w": caption_l["bbox"][2]-caption_l["bbox"][0], "h": caption_l["bbox"][3]-caption_l["bbox"][1]})
            else:
                # Назви немає. Перевіряємо, чи це дійсно початок аркуша, а не просто абзац тексту перед таблицею
                # Якщо рядків над таблицею мало (< 3), вважаємо, що таблиця починає сторінку
                if len(lines_above) < 3:
                    p_f = "Сторінка починається з таблиці без вказівки 'Продовження/Кінець таблиці' (або 'Таблиця')"
                    findings.append(f"Стор. {page_num}: {p_f}")
                    pages_with_errors.add(page_num)
                    highlights.append({"page": page_num, "x": t_bbox[0], "y": t_bbox[1], "w": t_bbox[2]-t_bbox[0], "h": 20})

    return {"summary": "Розриви таблиць перевірено.", "findings": findings, "is_success": len(findings) == 0, "pages_with_errors": sorted(list(pages_with_errors)), "highlights": highlights}

def analyze_table_sources(doc: fitz.Document) -> dict[str, Any]:
    findings, highlights, pages_with_errors = [], [], set()
    source_search_gap = 90
    source_touch_tolerance = 8
    for page_num in range(3, len(doc) + 1):
        page = doc[page_num-1]
        try:
            tabs = page.find_tables()
        except Exception:
            continue
        if not tabs.tables: continue
        
        all_lines = [l for b in page.get_text("dict")["blocks"] if "lines" in b for l in b["lines"]]
        all_lines.sort(key=lambda x: x["bbox"][1])
        
        for tab in tabs:
            t_bbox = tab.bbox
            table_bottom = estimate_table_bottom_from_horizontal_rule(page, t_bbox)
            
            is_figure = False
            for l in all_lines:
                if l["bbox"][1] > table_bottom and (l["bbox"][1] - table_bottom) < 60:
                    bottom_txt = "".join(s["text"] for s in l["spans"]).strip().lower()
                    if bottom_txt.startswith("рис.") or bottom_txt.startswith("рисунок"):
                        is_figure = True
                        break
            if is_figure: continue
            
            lines_above = [l for l in all_lines if l["bbox"][3] < t_bbox[1]]
            lines_above.reverse()
            
            found_idx = -1
            for i, l in enumerate(lines_above):
                if (t_bbox[1] - l["bbox"][3]) > 80: break
                txt = "".join(s["text"] for s in l["spans"]).strip().lower()
                if txt.startswith("таблиця") or txt.startswith("продовження") or txt.startswith("кінец"):
                    found_idx = i
                    break
            
            is_continuation = False
            if found_idx != -1:
                top_txt = "".join(s["text"] for s in lines_above[found_idx]["spans"]).strip().lower()
                if top_txt.startswith("продовження") or top_txt.startswith("кінец"):
                    is_continuation = True
            if is_continuation: continue
            
            ends_with_bracket = False
            caption_lines = []
            if found_idx != -1:
                caption_lines = [lines_above[j] for j in range(found_idx, -1, -1)]
            else:
                for l in lines_above:
                    if not caption_lines:
                        if (t_bbox[1] - l["bbox"][3]) < 40: caption_lines.append(l)
                    else:
                        if (caption_lines[-1]["bbox"][1] - l["bbox"][3]) < 30:
                            caption_lines.append(l)
                            if len(caption_lines) == 2: break
                        else: break
                caption_lines.reverse()
                
            if caption_lines:
                full_txt = " ".join("".join(s["text"] for s in cl["spans"]).strip() for cl in caption_lines).strip()
                if re.search(r"\]\.?\s*$", full_txt): ends_with_bracket = True
                    
            has_source_below = False
            lines_below = [l for l in all_lines if l["bbox"][3] > table_bottom - source_touch_tolerance]
            source_lines = []
            source_line = None
            for l in lines_below:
                gap = max(0, l["bbox"][1] - table_bottom)
                if gap > source_search_gap:
                    break
                txt = "".join(s["text"] for s in l["spans"]).strip()
                if not txt or is_page_number_line(l, page.rect.height):
                    continue
                source_lines.append(l)

            if source_lines:
                source_block_text = " ".join(
                    "".join(s["text"] for s in l["spans"]).strip().lower()
                    for l in source_lines
                )
                source_line = next(
                    (
                        l for l in source_lines
                        if "джерело" in "".join(s["text"] for s in l["spans"]).strip().lower()
                    ),
                    None
                )
                if "джерело" in source_block_text:
                    has_source_below = True

            if source_line:
                span = source_line["spans"][0]
                if abs(span["size"] - 10) > 1 or not bool(span["flags"] & 2):
                    p_f = "Слово 'Джерело' та зміст має бути шрифтом 10 пт та курсивом"
                    findings.append(f"Стор. {page_num}: {p_f}")
                    pages_with_errors.add(page_num)
                    highlights.append({"page": page_num, "x": source_line["bbox"][0], "y": source_line["bbox"][1], "w": source_line["bbox"][2]-source_line["bbox"][0], "h": source_line["bbox"][3]-source_line["bbox"][1]})

            if not ends_with_bracket and not has_source_below:
                p_f = "Не вказано джерело таблиці (немає '[...]' в кінці назви або 'Джерело' під таблицею)"
                findings.append(f"Стор. {page_num}: {p_f}")
                pages_with_errors.add(page_num)
                highlights.append({"page": page_num, "x": t_bbox[0], "y": t_bbox[1], "w": t_bbox[2]-t_bbox[0], "h": t_bbox[3]-t_bbox[1]})

    return {"summary": "Джерела таблиць перевірено.", "findings": findings, "is_success": len(findings) == 0, "pages_with_errors": sorted(list(pages_with_errors)), "highlights": highlights}

def analyze_figure_sources(doc: fitz.Document) -> dict[str, Any]:
    findings, highlights, pages_with_errors = [], [], set()
    source_search_gap = 90
    for page_num in range(3, len(doc) + 1):
        page = doc[page_num-1]
        all_lines = [l for b in page.get_text("dict")["blocks"] if "lines" in b for l in b["lines"]]
        all_lines.sort(key=lambda x: x["bbox"][1])
        
        idx = 0
        while idx < len(all_lines):
            l = all_lines[idx]
            txt = "".join(s["text"] for s in l["spans"]).strip()
            if txt.startswith("Рис.") or txt.startswith("Рисунок"):
                if not has_visual_anchor_above(page, l["bbox"]) or not is_first_text_line_below_visual_anchor(page, all_lines, idx):
                    idx += 1
                    continue
                caption_lines = [l]
                next_idx = idx + 1
                if next_idx < len(all_lines):
                    next_l = all_lines[next_idx]
                    next_txt = "".join(s["text"] for s in next_l["spans"]).strip()
                    if next_txt and (next_l["bbox"][1] - l["bbox"][3]) < 25:
                        caption_lines.append(next_l)
                        next_idx += 1
                        
                full_txt = " ".join("".join(s["text"] for s in cl["spans"]).strip() for cl in caption_lines).strip()
                ends_with_bracket = bool(re.search(r"\]\.?\s*$", full_txt))
                
                has_source_below = False
                last_caption_line = caption_lines[-1]
                source_lines = []
                source_line = None
                for source_l in all_lines[next_idx:]:
                    gap = source_l["bbox"][1] - last_caption_line["bbox"][3]
                    if gap > source_search_gap:
                        break
                    source_txt = "".join(s["text"] for s in source_l["spans"]).strip()
                    if not source_txt or is_page_number_line(source_l, page.rect.height):
                        continue
                    source_lines.append(source_l)

                if source_lines:
                    source_block_text = " ".join(
                        "".join(s["text"] for s in line["spans"]).strip().lower()
                        for line in source_lines
                    )
                    source_line = next(
                        (
                            line for line in source_lines
                            if "джерело" in "".join(s["text"] for s in line["spans"]).strip().lower()
                        ),
                        None
                    )
                    if "джерело" in source_block_text:
                        has_source_below = True

                if source_line:
                    span = source_line["spans"][0]
                    if abs(span["size"] - 10) > 1 or not bool(span["flags"] & 2):
                        p_f = "Слово 'Джерело' та зміст має бути шрифтом 10 пт та курсивом"
                        findings.append(f"Стор. {page_num}: {p_f}")
                        pages_with_errors.add(page_num)
                        highlights.append({"page": page_num, "x": source_line["bbox"][0], "y": source_line["bbox"][1], "w": source_line["bbox"][2]-source_line["bbox"][0], "h": source_line["bbox"][3]-source_line["bbox"][1]})
                            
                if not ends_with_bracket and not has_source_below:
                    p_f = "Не вказано джерело рисунку (немає '[...]' в кінці назви або 'Джерело' під нею)"
                    findings.append(f"Стор. {page_num}: {p_f}")
                    pages_with_errors.add(page_num)
                    first_caption_line = caption_lines[0]
                    highlights.append({"page": page_num, "x": first_caption_line["bbox"][0], "y": first_caption_line["bbox"][1], "w": first_caption_line["bbox"][2]-first_caption_line["bbox"][0], "h": first_caption_line["bbox"][3]-first_caption_line["bbox"][1]})
                    
            idx += 1
            
    return {"summary": "Джерела рисунків перевірено.", "findings": findings, "is_success": len(findings) == 0, "pages_with_errors": sorted(list(pages_with_errors)), "highlights": highlights}

@app.get("/")
def root() -> dict[str, Any]:
    return {"service": "PDF Analyzer", "status": "ok", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    doc: fitz.Document | None = None
    try:
        doc = open_pdf_document(request.pdf_base64)
        t = request.analysis_type
        if t == "page_numbers":
            res = analyze_page_numbers(doc)
        elif t == "general_text":
            res = analyze_general_text(doc)
        elif t == "chapters":
            res = analyze_chapters(doc)
        elif t == "subchapters":
            res = analyze_subchapters(doc)
        elif t == "perelik":
            res = analyze_perelik(doc)
        elif t == "tables":
            res = analyze_tables(doc)
        elif t == "table_breaks":
            res = analyze_table_breaks(doc)
        elif t == "table_sources":
            res = analyze_table_sources(doc)
        elif t == "figures":
            res = analyze_figures(doc)
        elif t == "figure_sources":
            res = analyze_figure_sources(doc)
        elif t == "formulas":
            res = analyze_formulas(doc)
        elif t == "references":
            res = analyze_references_section(doc)
        elif t == "appendices":
            res = analyze_appendices(doc)
        elif t == "zmist":
            rows, pw = extract_page_rows_fitz(doc, request.page_number)
            res = analyze_zmist(rows, pw, request.page_number, doc[request.page_number - 1].rect.height)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown analysis_type: {t}")
        res["analysis_type"] = t
        return res
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Analyze request failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if doc is not None:
            doc.close()

@app.post("/analyze_all")
def analyze_all(request: AnalyzeAllRequest) -> dict[str, Any]:
    doc: fitz.Document | None = None
    try:
        doc = open_pdf_document(request.pdf_base64)
        
        results = {}
        
        def safe_analyze(key: str, func, *args):
            try:
                results[key] = func(*args)
            except Exception as e:
                logger.error(f"Error in {key}: {e}")
                results[key] = {"summary": "Помилка аналізу", "findings": [str(e)], "is_success": False}

        safe_analyze("page_numbers", analyze_page_numbers, doc)
        safe_analyze("general_text", analyze_general_text, doc)
        safe_analyze("chapters", analyze_chapters, doc)
        safe_analyze("subchapters", analyze_subchapters, doc)
        safe_analyze("perelik", analyze_perelik, doc)
        safe_analyze("tables", analyze_tables, doc)
        safe_analyze("table_breaks", analyze_table_breaks, doc)
        safe_analyze("table_sources", analyze_table_sources, doc)
        safe_analyze("figures", analyze_figures, doc)
        safe_analyze("figure_sources", analyze_figure_sources, doc)
        safe_analyze("formulas", analyze_formulas, doc)
        safe_analyze("references", analyze_references_section, doc)
        safe_analyze("appendices", analyze_appendices, doc)
        
        try:
            rows, pw = extract_page_rows_fitz(doc, 2)
            results["zmist"] = analyze_zmist(rows, pw, 2, doc[1].rect.height)
        except Exception as e:
            logger.error(f"Error in zmist: {e}")
            results["zmist"] = {"summary": "Помилка аналізу змісту", "findings": [str(e)], "is_success": False}
        
        return results
    except Exception as e: 
        logger.error(f"Global error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally: 
        if doc is not None:
            doc.close()



