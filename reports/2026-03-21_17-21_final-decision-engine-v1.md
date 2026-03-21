# REPORT: final-decision-engine-v1

## 1. Summary
Добавлен `FinalDecisionEngine v1`, который объединяет результаты `DecisionEngine` и `ValueEngine` в единый финальный betting-выбор с учетом `value_flag`, конфликтов исходов и порога `EV`.

## 2. Goal
Свести два независимых решения (`decision` и `value_bet`) в один прозрачный, детерминированный и воспроизводимый итог для размещения ставки.

## 3. Scope
В scope этапа входило:
- добавление `core/decision/final_decision_engine.py`;
- добавление unit-тестов `tests/test_final_decision_engine.py`;
- фиксация отчета в `reports/`.

В scope не входило:
- изменение логики `DecisionEngine`;
- изменение логики `ValueEngine`;
- изменение данных, обучения и калибровки.

## 4. Files Changed
- `core/decision/final_decision_engine.py`
- `tests/test_final_decision_engine.py`
- `reports/2026-03-21_17-20_final-decision-engine-v1.md`

## 5. Detailed Changes
1. Реализован `FinalDecisionEngine.decide(probs, features, odds)`.
2. Добавлен шаг получения базового решения через `DecisionEngine.decide()`.
3. Добавлен шаг вычисления value-метрик через `ValueEngine.calculate()`.
4. Добавлены правила merge:
   - при `value_flag=False` итог `final_bet=None`;
   - при совпадении `decision` и `value_bet` выбирается `value_bet`;
   - при конфликте одиночных исходов:
     - `EV < 0.10` → сохранить `decision`;
     - `EV >= 0.10` → переключиться на `value_bet`;
   - для двойных исходов (`1X`, `X2`, `12`):
     - если `value_bet` входит в двойной исход — выбрать `value_bet`;
     - иначе сохранить двойной `decision`.
5. Добавлено логирование: `decision`, `value_bet`, `conflict`, `final_bet`.

## 6. Public Contracts / Interfaces
Возвращается словарь с полями:
- `decision: str`
- `value_bet: str`
- `final_bet: str | None`
- `confidence: float`
- `EV: float`
- `value_flag: bool`

Публичные интерфейсы существующих `DecisionEngine` и `ValueEngine` не менялись.

## 7. Database Changes
Изменений базы данных нет: миграции, схемы и SQL-слой не затрагивались.

## 8. API Behavior
Изменений внешнего API нет.
Внутренне добавлен новый слой финального решения, который использует уже существующие выходы decision/value без изменения их контрактов.

## 9. Business Logic Changes
Изменена только логика финального объединения результатов decision/value:
- старое поведение: отсутствовал отдельный единый детерминированный слой выбора `final_bet`;
- новое поведение: итоговая ставка формируется по формальным merge-правилам с порогом `EV=0.10`.

## 10. Tests
Добавлены unit-тесты для сценариев:
- совпадение `decision` и `value_bet`;
- конфликт и ветвление по порогу `EV=0.10`;
- отключение ставки при `value_flag=False`.

## 11. Manual Verification
Проверено вручную:
- файл отчета находится в `reports/`;
- имя файла соответствует формату `YYYY-MM-DD_HH-MM_<stage>.md`;
- структура отчета содержит 17 секций;
- секции заполнены текстом.

## 12. Risks / Known Issues
- Порог `EV=0.10` зафиксирован константно и может требовать параметризации при расширении системы.
- Для сложных конфликтных кейсов возможна потребность в дополнительных explainability-полях.

## 13. Deviations from Spec
Отклонений от заявленной спецификации merge-правил не зафиксировано.

## 14. Next Recommended Step
Следующий шаг: вынести порог `EV` в конфигурацию и добавить расширенные тесты на пограничные значения (`EV` около 0.10).

## 15. Commit / PR
- branch: `work`
- commit: будет создан после обновления отчета и прохождения `validate_reports`
- PR: будет создан через `make_pr` после коммита

## 16. Critical Review
Предположения:
- входные данные `decision` и `value_bet` всегда принадлежат допустимому множеству исходов.

Что обработано:
- конфликт одиночных исходов;
- двойные исходы;
- выключенный value-флаг.

Что стоит усилить:
- добавить формализацию причин выбора (`reason_code`) в результате `FinalDecisionEngine`;
- покрыть property-based тестами комбинации исходов и граничные значения EV.

## 17. Что не сделано из запланированного
- Не выполнено: параметризация порога `EV` через конфиг.
- Не выполнено: расширенные explainability-поля в финальном ответе.
- Причина: этап ограничен поставкой `FinalDecisionEngine v1` и базового тестового покрытия.
