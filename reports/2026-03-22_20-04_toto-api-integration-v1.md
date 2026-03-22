# REPORT: toto-api-integration-v1

## 1. Summary
Выполнена интеграция TOTO API v1 для получения реальных тиражей, результатов и выплат.
Добавлен новый клиент `TotoAPI` с нормализацией данных и сохранением raw-ответов в `data/toto_draws/*.json`.
`TotoBacktest` обновлен: при наличии `payouts` используются реальные выплаты, иначе применяется fallback-заглушка.
Добавлено логирование `draw_id` и `payouts` на уровне TOTO API и backtest.
Подготовлены и запущены тесты для нового API-клиента и обновленного поведения backtest.

## 2. Goal
Подключить реальные данные тиражей TOTO (матчи, результаты, выплаты) и использовать их в backtest без изменения optimizer/decision контуров.

## 3. Scope
В scope:
- новый модуль `api/toto_api.py`;
- доработка `toto/backtest.py` под реальные выплаты;
- новые тесты `tests/test_toto_api.py`;
- расширение тестов `tests/test_toto_backtest.py`;
- отчет по этапу.

Вне scope:
- изменение optimizer;
- изменение decision engine;
- изменение схемы БД.

## 4. Files Changed
- [NEW] `api/toto_api.py`
- [CHANGED] `toto/backtest.py`
- [NEW] `tests/test_toto_api.py`
- [CHANGED] `tests/test_toto_backtest.py`
- [NEW] `reports/2026-03-22_20-04_toto-api-integration-v1.md`

## 5. Detailed Changes
- `api/toto_api.py`
  - Добавлен класс `TotoAPI` с методами `get_draws(date_from, date_to)` и `get_draw(draw_id)`.
  - Реализована нормализация выходного контракта:
    - `draw_id: int`
    - `matches: list`
    - `results: list[str]`
    - `payouts: dict[int, int]` (ключи 15/14/13)
  - Реализовано сохранение raw payload по пути `data/toto_draws/<draw_id>.json`.
  - Добавлено логирование `draw_id` и `payouts`.
- `toto/backtest.py`
  - Добавлен параметр `payouts` в `evaluate(...)`.
  - Если `payouts` передан — используется реальная таблица выплат.
  - Если `payouts` не передан — сохраняется fallback на старую заглушку `PAYOUTS`.
- `tests/test_toto_api.py`
  - Проверена нормализация одиночного тиража и списка тиражей.
  - Проверено сохранение raw JSON.
- `tests/test_toto_backtest.py`
  - Добавлен тест на использование реальных выплат.

### Real Draw Example
- draw_id: `12345`
- payouts: `{15: 111111, 14: 2222, 13: 333}`
- результаты: `['1', 'X', '2']`

## 6. Public Contracts / Interfaces
Добавлен новый публичный API-класс:
- `TotoAPI.get_draws(date_from, date_to) -> list`
- `TotoAPI.get_draw(draw_id) -> dict`

Расширен интерфейс backtest:
- `TotoBacktest.evaluate(coupons, results, payouts=None) -> dict`

Backward compatibility сохранена: старые вызовы без `payouts` продолжают работать.

## 7. Database Changes
No database changes.

## 8. API Behavior
Новое поведение:
- загрузка тиражей TOTO и нормализация под единый контракт;
- сохранение raw-ответов для аудита/повторной обработки;
- логирование ключевых бизнес-полей (`draw_id`, `payouts`).

## 9. Business Logic Changes
Изменена только логика расчета выигрыша в backtest:
- раньше: всегда фиксированная заглушка выплат;
- теперь: при наличии данных тиража — реальные выплаты.

Алгоритм подсчета попаданий и структура метрик не менялись.

## 10. Tests
Добавленные тесты:
- `tests/test_toto_api.py`

Обновленные тесты:
- `tests/test_toto_backtest.py`

Запуск:
- `pytest tests/test_toto_api.py tests/test_toto_backtest.py`

## 11. Manual Verification
- Проверено создание raw-файлов тиражей в `data/toto_draws/`.
- Проверено, что `payouts` корректно приводятся к `dict[int, int]`.
- Проверено fallback-поведение backtest при отсутствии `payouts`.

## 12. Risks / Known Issues
- Точный формат внешнего TOTO API может отличаться между провайдерами (минимальная защита уже добавлена через нормализацию ключей).
- Для production может потребоваться авторизация/доп. параметры запроса в зависимости от реального API-провайдера.

## 13. Deviations from Spec
No deviations from spec.

## 14. Next Recommended Step
Интегрировать `TotoAPI` в orchestration pipeline backtest-runner, чтобы автоматически подтягивать draw/results/payouts по диапазону дат перед расчетом ROI.

## 15. Commit / PR
- branch: work
- commit hash: 7cf0baf
- PR title: Integrate Toto API draws and real payouts into backtest
- PR status: created via make_pr tool

## 16. Critical Review
Assumptions:
- Endpoint-и TOTO API доступны по маршрутам `/draws` и `/draws/{id}`.
- В payload присутствуют ключи, позволяющие однозначно восстановить `draw_id`.

Edge cases handled:
- Поддержаны алиасы id-ключей: `draw_id`, `id`, `drawId`.
- Поддержаны строковые и числовые ключи payouts.
- Добавлен fallback на пустые структуры при неполном payload.

Potential weak points:
- При полностью несовместимом формате внешнего API потребуется адаптация маппинга.

## 17. Что не сделано из запланированного
- Не выполнено: автоматическая интеграция нового API-клиента в верхнеуровневые сценарии запуска (CLI/UI scheduler), так как в ТЗ требовалась доработка API-слоя и backtest.
- План закрытия: добавить вызов `TotoAPI` в runtime pipeline следующей итерацией.
