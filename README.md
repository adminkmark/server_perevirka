# PDF Analyzer For GAS

Сервер для Render, який приймає PDF у `base64` і повертає JSON-аналіз.

## Локальний запуск

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

## Render

- репозиторій можна деплоїти напряму через `render.yaml`
- endpoint здоров'я: `GET /health`
- основний endpoint: `POST /analyze`

## Формат запиту

```json
{
  "pdf_base64": "JVBERi0xLjc...",
  "analysis_type": "zmist",
  "page_number": 2
}
```

## Формат відповіді

```json
{
  "analysis_type": "zmist",
  "page_number": 2,
  "summary": "Виявлено 1 відхилення(нь) у змісті.",
  "findings": [
    "Основні пункти змісту не визначені як жирні: \"ВСТУП\"."
  ],
  "rows": [],
  "metrics": {
    "page_width": 595.2,
    "major_rows": 5,
    "sub_rows": 8
  }
}
```
