# REPORT: model_training_v1

## 1. Summary
Реализован production-ready trainer на базе одной модели LightGBM для предматчевого multiclass-прогноза P(1)/P(X)/P(2), добавлены обязательные helper-методы, логирование, сохранение модели и unit-тесты на обучение и `predict_proba`.

## 2. Goal
Построить единый training pipeline для предсказания исходов матча в классах `0/1/2` с хронологическим split и проверяемой валидацией по `accuracy` и `log_loss`.

## 3. Scope
Входило:
- `core/model/trainer.py` (pipeline обучения/оценки/сохранения).
- `tests/test_trainer.py` (покрытие базового сценария обучения).
- отчет этапа в `reports/`.

Не входило:
- изменение `FeatureBuilder`.
- любые альтернативные модели, auto-ML, random split.
- изменения API/DB схемы.

## 4. Files Changed
- `core/model/trainer.py`
- `tests/test_trainer.py`
- `reports/2026-03-21_10-52_model-training-v1.md`

## 5. Detailed Changes
- Удалена прежняя схема с `train_test_split`, `cross_val_score`, калибровкой и `pickle`.
- Реализован класс `ModelTrainer` с:
  - `clean_data()` для фильтрации `None/NaN/inf` и невалидного target;
  - `prepare_dataset()` для валидации и матриц `X/y`;
  - `split_chronological()` для split 80/20 строго по времени (или текущему порядку, если дата не дана);
  - `train()` с `LGBMClassifier(objective='multiclass', num_class=3, learning_rate=0.05, n_estimators=300, max_depth=-1, random_state=42)`;
  - `evaluate()` с `accuracy`, `log_loss` и распределением предсказаний;
  - `save()` через `joblib` в `data/models/model.pkl`.
- Добавлена защита от постматчевых полей (простая проверка ключей row).
- Добавлены логи размера датасета, числа фичей, train/valid size, accuracy/log_loss.

## 6. Public Contracts / Interfaces
```python
class ModelTrainer:
    def train(self, dataset: list[dict[str, Any]]) -> LGBMClassifier: ...
    def evaluate(self, model: LGBMClassifier, x_valid: np.ndarray, y_valid: np.ndarray) -> dict[str, float | dict[str, int]]: ...
    def save(self, model: LGBMClassifier, path: str | Path | None = None) -> None: ...

    def prepare_dataset(self, dataset: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]: ...
    def split_chronological(self, dataset: list[dict[str, Any]], train_ratio: float = 0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
    def clean_data(self, dataset: list[dict[str, Any]]) -> list[dict[str, Any]]: ...
```

## Dataset Info
- Количество матчей в тестовом прогоне: `600`.
- Количество фичей (из `FeatureBuilder`): `20`.

## Model Output Example
Пример `predict_proba` (одна строка):
`[0.4121, 0.3018, 0.2861]` (shape батча: `(n, 3)`).

## 7. Database Changes
Нет. Миграции и SQL не изменялись.

## 8. API Behavior
Нет. Контракты API и роутинг не менялись.

## 9. Business Logic Changes
- Модель обучается только на prematch-фичах.
- Split выполняется хронологически (первые 80% train, последние 20% valid).
- Оценка модели смещена на вероятностное качество (`log_loss`) плюс базовая точность (`accuracy`).

## 10. Tests
- `pytest -q` → `14 passed`.
- Добавлен `tests/test_trainer.py`:
  - модель обучается;
  - `predict_proba` возвращает shape `(n, 3)`;
  - в вероятностях нет `NaN`.

## 11. Manual Verification
- Ручной осмотр `trainer.py` на соответствие ограничениям ТЗ:
  - одна модель (LightGBM);
  - нет random split;
  - нет FeatureBuilder-изменений;
  - сохранение в `data/models/model.pkl` через `joblib`.

## 12. Risks / Known Issues
- Детектор post-match полей основан на шаблонах имен ключей и может не поймать экзотические названия.
- При малом или сильно несбалансированном датасете метрики могут быть нестабильны.
- В synthetic-тесте даты повторяются по дням месяца, что допустимо для unit-валидации формы/пайплайна, но не заменяет реальный исторический датасет.

## 13. Deviations from Spec
- Источник данных из БД в этом этапе не реализован: используется интерфейс `list[dict]` (как временно разрешено в ТЗ).

## 14. Next Recommended Step
Подключить слой загрузки prematch-выборки из `database` (matches + features) и запустить тренировку на реальной исторической выборке с фиксацией baseline-метрик.

## 15. Commit / PR
Commit: см. текущий `git rev-parse --short HEAD` для этой ветки.
PR: будет создан через `make_pr`.

## 16. Critical Review
Assumptions:
- Входной `dataset` содержит все 20 prematch-фичей из `FeatureBuilder` и `target` в кодировке `0/1/2`.

Edge cases:
- Пустой датасет после очистки.
- Датасет без валидных target-меток.
- Датасет <500 матчей (срабатывает warning).
- Отсутствие даты матча: split идёт по текущему порядку списка.

Слабые места модели:
- Базовые гиперпараметры без тюнинга.
- Нет калибровки вероятностей в этой версии.
- Нет feature importance / drift-мониторинга в текущем этапе.

## 17. Что не сделано из запланированного
- Не реализован прямой адаптер загрузки train-датасета из БД внутри `ModelTrainer`.
- Не добавлен CLI-скрипт запуска обучения отдельной командой.
- Не добавлен e2e-тест с реальными историческими данными из проекта.
