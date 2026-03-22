# REPORT: batch_backtest_v2

## 1. Summary
Реализован batch backtest для множества тиражей Toto в новом модуле `toto/batch_backtest.py`.
Добавлен класс `TotoBatchBacktest` с методом `run(draw_ids)`, который для каждого тиража загружает данные через `TotoAPI`, строит купоны через `TotoOptimizer`, прогоняет оценку через `TotoBacktest` и агрегирует метрики.
Добавлено сохранение итогов в `data/backtest_results.json`.
Добавлены unit-тесты `tests/test_batch_backtest.py` на агрегацию и пустой сценарий.
Этап закрывает задачу TOTO BATCH BACKTEST v2 (MULTI-DRAW ANALYSIS).

## 2. Goal
Сделать мульти-тиражный backtest с агрегированной статистикой (`draws`, `ROI`, `avg_hits`, `max_hits`, распределение по 13/14/15) и записью результатов в JSON.

## 3. Scope
В scope входило:
- реализация `TotoBatchBacktest.run(draw_ids)`;
- агрегация статистики по тиражам;
- подсчет распределения 13/14/15;
- сохранение в `data/backtest_results.json`;
- покрытие тестами.

В scope не входило:
- изменение API-клиента Totobrief;
- изменение внутренней логики `TotoOptimizer`/`TotoBacktest`.

## 4. Files Changed
- [NEW] `toto/batch_backtest.py`
- [NEW] `tests/test_batch_backtest.py`
- [NEW] `reports/2026-03-22_20-22_batch-backtest-v2.md`

## 5. Detailed Changes
- `toto/batch_backtest.py`
  - Добавлен `TotoBatchBacktest` с DI (`api`, `optimizer`, `backtest`, `mode`, `output_path`).
  - Реализован pipeline на каждый `draw_id`: `TotoAPI.get_draw` -> подготовка матчей -> `TotoOptimizer.optimize` -> `TotoBacktest.evaluate`.
  - Добавлена агрегация: `draws`, `total_profit`, `ROI`, `avg_hits`, `max_hits`, `distribution(13/14/15)`.
  - Добавлено сохранение итогов в JSON-файл.
- `tests/test_batch_backtest.py`
  - Добавлены фейки API/optimizer/backtest.
  - Добавлен тест на агрегацию нескольких тиражей и сохранение JSON.
  - Добавлен тест на пустой список тиражей.

## 6. Public Contracts / Interfaces
Новый публичный интерфейс:
- `class TotoBatchBacktest`
- `run(self, draw_ids: list) -> dict`

Breaking changes: отсутствуют.

## 7. Database Changes
No database changes.

## 8. API Behavior
Внешние HTTP-контракты не менялись.
Batch-логика использует существующий `TotoAPI.get_draw(draw_id)`.

## 9. Business Logic Changes
Добавлена оркестрация мульти-тиражного анализа:
- загрузка тиража;
- генерация купонов;
- оценка попаданий/ROI;
- сводная аналитика по всем тиражам.

## 10. Tests
Добавленные тесты:
- `tests/test_batch_backtest.py::test_run_aggregates_draws_and_saves_json`
- `tests/test_batch_backtest.py::test_run_with_empty_draws_returns_zeroes`

Запускалось:
- `pytest -q tests/test_batch_backtest.py tests/test_toto_backtest.py tests/test_toto_optimizer.py`

Результат: все тесты passed.

## 11. Manual Verification
Проверено вручную:
- выход `run(draw_ids)` содержит ожидаемые агрегаты;
- файл результатов создается автоматически;
- в JSON сохраняются итоговые метрики и распределение.

## 12. Risks / Known Issues
- `ROI` в summary рассчитан как `total_profit / draws` (прибыль на тираж), а не как классический ROI в процентах на ставку.
- Решение `decision` внутри batch строится эвристически из `pool_probs` (top1/top2), т.к. в draw payload нет готового decision.

## 13. Deviations from Spec
- В summary дополнительно сохранено поле `total_profit`, помимо обязательного минимального набора.

## 14. Next Recommended Step
Добавить интеграционный тест с реальным payload тиража и проверкой поддержки реальных `payouts`, если API начнет их отдавать в стабильной схеме.

## 15. Commit / PR
- branch: work
- commit hash: pending
- PR title: pending
- PR status: pending

## 16. Critical Review
Assumptions:
- Для batch-этапа допустимо строить `decision` на основе pool probability.
- Поле распределения должно считать количество купонов с 13/14/15 хитами суммарно по тиражам.

Potential weak points:
- При нестандартных/пустых `result` возможна переоценка покрытий.
- Формула ROI может требовать уточнения бизнес-метрики (per draw vs per stake).

Hardening ideas:
- нормализовать ROI до единого определения;
- логировать метрики по каждому draw в отдельный список `per_draw`.

## 17. Что не сделано из запланированного
- Не добавлен CLI/entrypoint для запуска batch backtest из командной строки.
- Не добавлена визуализация/график распределения результатов.
