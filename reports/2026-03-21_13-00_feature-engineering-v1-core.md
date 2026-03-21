# REPORT: feature_engineering_v1_critical_core

## 1. Summary
Реализован production-ready `FeatureBuilder` для prematch feature engineering в `football_ai/core/features/builder.py`.
Новая реализация строит чистый вектор признаков `dict[str, float]` по группам MATCH LEVEL, TEAM DIFF, HOME/AWAY, LEAGUE и RISK.
Добавлены обязательные helper-методы `safe_div`, `calc_diff`, `normalize_odds`, `calc_entropy` и безопасная обработка `-1/-2/None/NaN`.
Добавлены unit-тесты `tests/test_features.py` для проверки структуры, санитизации и helper-логики.
Изменения ограничены слоем features и тестами, без правок API, DB, ML training или decision engine.

## 2. Goal
Цель этапа — создать надежный prematch feature builder, пригодный для downstream ML, без postmatch data leakage.

## 3. Scope
В scope вошло:
- переписка `football_ai/core/features/builder.py` под новый контракт `build_features(match, home_stats, away_stats, league_stats)`;
- внедрение безопасных вычислительных helper-функций;
- добавление тестов `tests/test_features.py`.

В scope не входило:
- изменение API-клиентов и API-слоя;
- изменение DB-схемы, миграций и persistence;
- обучение моделей;
- изменение decision engine.

## 4. Files Changed
- [MOD] `football_ai/core/features/builder.py`
- [NEW] `tests/test_features.py`
- [NEW] `reports/2026-03-21_13-00_feature-engineering-v1-core.md`

## 5. Detailed Changes
- `FeatureBuilder` теперь строит строго prematch признаки из переданных статистик и odds.
- Добавлены нормализация коэффициентов в implied probabilities и risk-признаки (`entropy`, `gap`, `volatility`).
- Добавлена жесткая санитизация входов: `-1/-2` трактуются как отсутствующие значения, `None/NaN/inf` заменяются safe default.
- Добавлен `build()` как backward-compatible adapter к старому однопараметрическому интерфейсу.
- Все выходные значения форсируются к типу `float`.
- Ошибки конверсии и аварийные ситуации логируются через `logging`.

## 6. Public Contracts / Interfaces
Публичный контракт feature-модуля расширен методом:
- `FeatureBuilder.build_features(match, home_team_stats, away_team_stats, league_stats) -> dict[str, float]`.

Старый интерфейс `build(match_details)` сохранен через адаптер для совместимости.

## 7. Database Changes
Изменений БД, SQL-схем и миграций нет.

## 8. API Behavior
Поведение API не изменялось. Изменения выполнены только внутри feature engineering слоя.

## 9. Business Logic Changes
Добавлена новая логика построения prematch признаков:
- market-level probability features на основе odds;
- дифференциальные признаки команд;
- сплитовые home/away признаки;
- league-level context признаки;
- risk/meta признаки uncertainty.

Postmatch данные не используются.

## 10. Tests
Добавлены и выполнены тесты:
- `tests/test_features.py::test_build_features_contains_required_groups`
- `tests/test_features.py::test_build_features_sanitizes_invalid_values`
- `tests/test_features.py::test_helpers_behave_as_expected`

Также выполнен запуск `pytest` по новому тестовому модулю.

## 11. Manual Verification
Проверено вручную:
- все обязательные группы фич присутствуют;
- сумма implied probabilities ≈ 1.0;
- sentinel значения `-1/-2` не попадают в итоговый вектор как сырые значения;
- значения выходного словаря имеют тип `float`.

## 12. Risks / Known Issues
- Риск: при нестандартных схемах имен полей у внешних провайдеров часть признаков может приходить нулевой из-за fallback.
- Риск: текущие safe defaults (0.0) могут влиять на распределения фич для слабозаполненных лиг.
- Техдолг: возможна дальнейшая калибровка правил fallback/alias-ключей под конкретные источники статистики.

## 13. Deviations from Spec
Отклонений от спецификации этапа нет.

## 14. Next Recommended Step
Следующий шаг — добавить feature scaling/validation на уровне preprocessing pipeline (без обучения) и расширить тесты edge-case сценариями по разным провайдерам данных.

## 15. Commit / PR
Изменения подготовлены отдельным коммитом в текущей ветке.
После коммита создается PR через `make_pr` с описанием реализованного FEATURE ENGINEERING v1 этапа.

## 16. Critical Review
- Допущение: входные `home_team_stats/away_team_stats/league_stats` содержат prematch aggregate данные.
- Потенциальная зона ошибки: если upstream ошибочно передаст postmatch поля под совпадающими ключами, они не фильтруются на уровне схемы и это нужно контролировать upstream-валидаторами.
- Усиление: добавить schema-based whitelist входных полей для feature builder.

## 17. Что не сделано из запланированного
- Не добавлены интеграционные тесты полного pipeline (api->features->model) для этого этапа.
- Причина: scope этапа ограничен core feature builder + unit tests.
- План закрытия: добавить integration checks отдельным этапом после стабилизации форматов входной статистики.
