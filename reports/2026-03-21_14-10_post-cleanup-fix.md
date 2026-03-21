## 1. Summary
Выполнен post-cleanup fix после ошибочного упрощения структуры: восстановлены удаленные модули `core/features/transformers.py` и `core/evaluation/backtest.py`, возвращен пакет `core/evaluation`, добавлен недостающий `core/features/__init__.py`, проверены импорты и тесты.

## 2. Goal
Вернуть целостность модульной структуры без изменения `FeatureBuilder` и без повторной архитектурной миграции.

## 3. Scope
- Восстановление удаленных файлов.
- Исправление package-инициализации (`__init__.py`).
- Проверка импортов и тестов.
- Возврат обязательной отчетности в `reports/`.

## 4. Files Changed
- `core/features/__init__.py` (new)
- `core/features/transformers.py` (restored)
- `core/evaluation/__init__.py` (new)
- `core/evaluation/backtest.py` (restored)
- `tests/test_transformers_backtest.py` (new)
- `reports/2026-03-21_14-10_post-cleanup-fix.md` (new)

## 5. Detailed Changes
1) Восстановлен `core/features/transformers.py`:
- добавлены функции `handle_missing`, `normalize_features` (basic scaling), `safe_mean`, `calc_form_index`.

2) Восстановлен слой `core/evaluation`:
- создан `core/evaluation/__init__.py`;
- создан `core/evaluation/backtest.py` с классом `BacktestEngine` и методом `evaluate(predictions, results)`.

3) Исправлена package-структура:
- создан `core/features/__init__.py` и экспортированы `FeatureBuilder` + функции transformers.

4) Добавлены тесты на восстановленные модули:
- `tests/test_transformers_backtest.py`.

## 6. Public Contracts / Interfaces
Добавлены/восстановлены публичные контракты:
- `core.features.transformers.handle_missing(features, default=0.0) -> dict[str, float]`
- `core.features.transformers.normalize_features(features) -> dict[str, float]`
- `core.evaluation.backtest.BacktestEngine.evaluate(predictions, results) -> dict[str, float]`

## 7. Database Changes
Изменений схемы/миграций БД нет.

## 8. API Behavior
Внешние API-контракты не изменялись.

## 9. Business Logic Changes
Возвращена базовая функциональность preprocessing и backtest-оценки, ранее удаленная в cleanup.

## 10. Tests
- `pytest -q` — успешно.
- Дополнительно покрыты восстановленные modules (`transformers`, `backtest`).

## 11. Manual Verification
Проверено вручную:
- наличие файлов `core/features/transformers.py` и `core/evaluation/backtest.py`;
- наличие `__init__.py` в пакетах: `core/`, `core/features/`, `core/model/`, `core/decision/`, `core/evaluation/`, `database/`, `ingestion/`, `scheduler/`, `toto/`, `ui/`, `tests/`;
- отсутствие импортов `from football_ai...`.

## 12. Risks / Known Issues
- Реализация `BacktestEngine` и transformers минимальная (по ТЗ), не production-grade backtesting framework.

## 13. Deviations from Spec
Отклонений от ТЗ нет.

## 14. Next Recommended Step
Добавить расширенные сценарии backtesting (stake sizing, drawdown, stratified metrics) и отдельные тесты на крайние случаи.

## 15. Commit / PR
Коммит включает восстановление удаленных модулей и отчет по этапу.

## 16. Critical Review
Причина инцидента: в cleanup-этапе было выполнено агрессивное перемещение/удаление файлов, что привело к потере `transformers` и `evaluation`. Исправление выполнено адресно, без повторного изменения архитектуры.

## 17. Что не сделано из запланированного
- Не добавлены продвинутые метрики backtest (Sharpe, max drawdown, hit-rate per market).
- Не добавлены интеграционные тесты против реальных исторических данных.
