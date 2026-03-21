# Backtest Engine v1 (ROI + Accuracy)

## Что сделано

- Реализован `BacktestEngine.evaluate(matches, results)` в `core/evaluation/backtest.py`.
- Для каждого матча вызывается `FinalDecisionEngine.decide(...)`.
- Ставки `None` пропускаются.
- Добавлена проверка выигрыша для одиночных (`1`, `X`, `2`) и двойных (`1X`, `X2`, `12`) исходов.
- Добавлен расчет прибыли при фиксированной ставке `1`:
  - win: `odds - 1`
  - lose: `-1`
- Добавлены метрики:
  - `total_bets`
  - `wins`
  - `losses`
  - `accuracy`
  - `ROI`
- Добавлено логирование агрегатов: `bets`, `wins`, `roi`.

## Тесты

Добавлен файл `tests/test_backtest.py` с проверками:

1. Одиночные ставки (`1/X/2`) и корректность `accuracy + ROI`.
2. Двойные ставки (`1X/X2/12`) и корректный зачет побед.
3. Пропуск `None`-ставок и корректный расчет `ROI`.

## Backtest Example

Пример:

- Матч 1: final bet = `1`, odds = `2.0`, result = `1` → profit `+1.0`
- Матч 2: final bet = `2`, odds = `5.7`, result = `1` → profit `-1.0`

Итог:

- total_bets = 2
- wins = 1
- losses = 1
- accuracy = 1/2 = 0.5
- ROI = (1.0 - 1.0) / 2 = 0.0

## Weaknesses

- Сильная зависимость от качества `FinalDecisionEngine`: ошибки выбора ставки напрямую ухудшают ROI.
- Для двойных ставок нужен корректный коэффициент в `odds` (`O1X`, `OX2`, `O12`) — при отсутствии расчет невозможен.
- Текущая версия использует фиксированный stake=1 и не учитывает bankroll/risk-management.
- Нет разбиения метрик по сегментам (лиги, диапазоны odds, confidence), что затрудняет диагностику просадок.

## Проверки

- `pytest -q`
- `python scripts/validate_reports.py --mode staged`
