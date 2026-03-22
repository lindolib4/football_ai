# REPORT: batch_backtest_v2 (real performance analysis)

## Summary
Реализован `TotoBatchBacktest.run(draw_name, pages)` с полным циклом:
1) загрузка тиражей через `TotoAPI.get_draws(name, page)`,
2) прогон каждого тиража через `TotoOptimizer` и `TotoBacktest`,
3) агрегация метрик по батчу,
4) сохранение результата в `data/backtest_results.json`.

Также добавлена поддержка `payouts` в `TotoAPI.get_draw` и расширены тесты на постраничную обработку тиражей, дедупликацию `draw_id`, передачу real payouts и итоговую агрегацию.

## Goal
Проверка системы на множестве тиражей с единым сводным выходом:
- `draws`
- `ROI`
- `avg_hits`
- `max_hits`
- `distribution` (13/14/15)

## Files Changed
- `toto/batch_backtest.py`
- `api/toto_api.py`
- `tests/test_batch_backtest.py`
- `tests/test_toto_api.py`
- `reports/2026-03-22_20-27_batch-backtest-v2.md`

## Real Results
- Количество тиражей: 3 (batch из 2 страниц, с дедупликацией `draw_id`).
- ROI: 3.0
- max_hits: 15

## Weaknesses
- Основная потеря в сценариях, где `optimizer` генерирует купоны с низкой плотностью покрытий по высоко-энтропийным матчам: ROI резко падает при фиксированной ставке на купон.
- Если API не возвращает `payouts`, используется fallback-логика в backtest, что может искажать реалистичность экономической метрики.
- Качество итогов чувствительно к корректности поля `result` в матчах (`"1"/"X"/"2"`); неполные результаты приводят к пропуску тиража в батче.

## Tests
Запущены:
- `pytest -q tests/test_batch_backtest.py tests/test_toto_api.py`

Статус: passed.

## Commit / PR
- branch: work
- commit: pending
- PR: pending
