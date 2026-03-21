# FootAI

Desktop football AI + Toto engine built with Python, LightGBM, and PyQt6.

## Implemented v1 foundations
- FootyStats API client with retry/backoff, rate limiting, and JSON cache.
- SQLite schema and DB adapter.
- Feature engineering module and decision engine.
- LightGBM multiclass trainer/predictor with calibration.
- Toto coupon generator (16/32 style limits).
- Auto-train scheduler helper.
- Basic PyQt6 4-tab shell UI.
- Unit tests for decision and Toto generation logic.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
pytest football_ai/tests
python main.py
```

## Project structure
See `football_ai/` for modules split by API/core/toto/database/scheduler/ui.
