# REPORT: final-decision-engine-v1

## 1. Summary
Добавлен `FinalDecisionEngine v1`, который объединяет решения `DecisionEngine` и `ValueEngine` в единый финальный betting-выбор с учетом value-флага, конфликтов и порога EV.

## 2. Goal
Свести два независимых решения (`decision` и `value`) в единый, прозрачный и детерминированный результат для ставки.

## 3. Scope
- Добавлен новый модуль `core/decision/final_decision_engine.py`.
- Добавлены unit-тесты `tests/test_final_decision_engine.py`.
- Добавлен обязательный отчет этапа.

## 4. Files Changed
- `core/decision/final_decision_engine.py`
- `tests/test_final_decision_engine.py`
- `reports/2026-03-21_17-20_final-decision-engine-v1.md`

## 5. Detailed Changes
1. Реализован `FinalDecisionEngine.decide(probs, features, odds)`.
2. Шаг 1: вызов `DecisionEngine.decide()`.
3. Шаг 2: вызов `ValueEngine.calculate()`.
4. Шаг 3: согласование по правилам:
   - при `value_flag=False` итоговая ставка `None`;
   - при совпадении `decision` и `value_bet` выбирается `value_bet`;
   - при конфликте одиночного `decision` и `value_bet`:
     - `EV < 0.10` → оставить `decision`;
     - `EV >= 0.10` → взять `value_bet`;
   - для двойного `decision` (`1X`, `X2`, `12`):
     - если `value_bet` входит в двойной исход — выбрать `value_bet`;
     - иначе оставить двойной `decision`.
5. Добавлено логирование ключевых полей: `decision`, `value`, `conflict`, `final_bet`.

## 6. Output Contract
Возвращается словарь:
- `decision: str`
- `value_bet: str`
- `final_bet: str | None`
- `confidence: float`
- `EV: float`
- `value_flag: bool`

## 7. Tests
Добавлены проверки:
- совпадение `decision` и `value`;
- конфликт и переключение по порогу `EV=0.10`;
- `value_flag=False` и отсутствие ставки.

## 8. Risks / Notes
- В текущей версии `features` прокидываются в `DecisionEngine` без дополнительной логики, чтобы не усложнять слой merge.
- Порог `EV=0.10` жестко задан по ТЗ.

## 9. Deviations from Spec
Отклонений нет: реализованы все указанные правила согласования без изменения старых модулей и без ML.
