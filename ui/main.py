from __future__ import annotations

import json
import hashlib
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from api.footystats import FootyStatsClient
from api.toto_api import TotoAPI
from config import settings
from ingestion.loaders import IngestionLoader
from ingestion.normalizers import normalize_match
from PyQt6.QtCore import QObject, QDate, QEvent, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QTextCursor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QFrame,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolTip,
    QVBoxLayout,
    QWidget,
)
from toto.optimizer import TotoOptimizer

from scheduler.auto_train import AutoTrainer
from core.features.builder import FeatureBuilder

logger = logging.getLogger("ui")


class MatchesBatchPredictWorker(QObject):
    """Run heavy batch prediction outside the UI thread and report compact progress."""

    progress = pyqtSignal(int, int, int, int, int, str)
    finished = pyqtSignal(dict)

    def __init__(
        self,
        *,
        trainer: AutoTrainer,
        tasks: list[dict[str, Any]],
        progress_every: int = 25,
    ) -> None:
        super().__init__()
        self.trainer = trainer
        self.tasks = tasks
        self.progress_every = max(1, int(progress_every))
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        total = len(self.tasks)
        processed = 0
        predicted = 0
        skipped = 0
        failed = 0
        no_odds_predicted = 0
        no_odds_skipped = 0
        results: list[dict[str, Any]] = []

        predictor = getattr(self.trainer, "predictor", None)
        feature_columns: list[str] = list(getattr(predictor, "feature_columns", []) or [])

        for task in self.tasks:
            if self._cancel_requested:
                break

            row_idx = int(task.get("row_idx", -1))
            odds_status = str(task.get("odds_status") or "")
            row_meta = task.get("row_meta") if isinstance(task.get("row_meta"), dict) else {}

            try:
                if odds_status == "odds_partial":
                    skipped += 1
                    results.append(
                        {
                            "row_idx": row_idx,
                            "status": "skipped",
                            "reason": "Коэффициенты заполнены не полностью",
                            "source": "odds_partial",
                        }
                    )
                elif odds_status == "odds_missing":
                    feature_snapshot = self.trainer.build_runtime_feature_snapshot(
                        match=row_meta,
                        required_columns=feature_columns,
                    )
                    features = feature_snapshot.get("features", {}) if isinstance(feature_snapshot, dict) else {}
                    if not isinstance(features, dict) or not features:
                        no_odds_skipped += 1
                        skipped += 1
                        results.append(
                            {
                                "row_idx": row_idx,
                                "status": "skipped",
                                "reason": "Нет кф, fallback-прогноз невозможен",
                                "source": "no_odds",
                            }
                        )
                    else:
                        diag_result = predictor.predict_with_diagnostics(features, allow_no_odds_fallback=True)
                        if isinstance(diag_result, dict) and diag_result.get("status") == "predicted":
                            probs = diag_result.get("probs", {}) if isinstance(diag_result.get("probs"), dict) else {}
                            p1 = float(probs.get("P1", 0.0))
                            px = float(probs.get("PX", 0.0))
                            p2 = float(probs.get("P2", 0.0))
                            predicted += 1
                            no_odds_predicted += 1
                            results.append(
                                {
                                    "row_idx": row_idx,
                                    "status": "predicted",
                                    "source": "no_odds_fallback",
                                    "p1": p1,
                                    "px": px,
                                    "p2": p2,
                                    "feature_snapshot": feature_snapshot if isinstance(feature_snapshot, dict) else {},
                                    "feature_diagnostics": diag_result.get("feature_diagnostics", {}),
                                    "prediction_quality": diag_result.get("prediction_quality", {}),
                                    "reason": "Прогноз без кф (fallback)",
                                }
                            )
                        else:
                            no_odds_skipped += 1
                            skipped += 1
                            reason = "Нет кф, fallback-прогноз невозможен"
                            if isinstance(diag_result, dict):
                                reason = str(diag_result.get("reason") or reason)
                            results.append(
                                {
                                    "row_idx": row_idx,
                                    "status": "skipped",
                                    "reason": reason,
                                    "source": "no_odds",
                                }
                            )
                else:
                    odds_1 = float(task.get("odds_ft_1", 0.0))
                    odds_x = float(task.get("odds_ft_x", 0.0))
                    odds_2 = float(task.get("odds_ft_2", 0.0))
                    if odds_1 <= 1.0 or odds_x <= 1.0 or odds_2 <= 1.0:
                        skipped += 1
                        results.append(
                            {
                                "row_idx": row_idx,
                                "status": "skipped",
                                "reason": "коэффициенты отсутствуют или невалидны",
                                "source": "feature_error",
                            }
                        )
                    else:
                        runtime_payload: dict[str, Any] = {
                            **row_meta,
                            "odds_ft_1": odds_1,
                            "odds_ft_x": odds_x,
                            "odds_ft_2": odds_2,
                        }
                        if isinstance(runtime_payload.get("odds"), dict):
                            runtime_payload["odds"] = {
                                **runtime_payload.get("odds", {}),
                                "odds_ft_1": odds_1,
                                "odds_ft_x": odds_x,
                                "odds_ft_2": odds_2,
                            }

                        feature_snapshot = self.trainer.build_runtime_feature_snapshot(
                            match=runtime_payload,
                            required_columns=feature_columns,
                        )
                        features = feature_snapshot.get("features", {}) if isinstance(feature_snapshot, dict) else {}
                        if not isinstance(features, dict) or not features:
                            skipped += 1
                            results.append(
                                {
                                    "row_idx": row_idx,
                                    "status": "skipped",
                                    "reason": "runtime feature snapshot is empty",
                                    "source": "feature_error",
                                }
                            )
                        else:
                            prediction = predictor.predict({name: float(features[name]) for name in feature_columns})
                            p1 = float(prediction.get("P1", 0.0))
                            px = float(prediction.get("PX", 0.0))
                            p2 = float(prediction.get("P2", 0.0))
                            predicted += 1
                            results.append(
                                {
                                    "row_idx": row_idx,
                                    "status": "predicted",
                                    "source": "model_runtime",
                                    "p1": p1,
                                    "px": px,
                                    "p2": p2,
                                    "feature_snapshot": feature_snapshot if isinstance(feature_snapshot, dict) else {},
                                    "reason": "",
                                }
                            )

            except Exception as exc:
                failed += 1
                results.append(
                    {
                        "row_idx": row_idx,
                        "status": "failed",
                        "reason": str(exc),
                        "source": "predict_error",
                    }
                )

            processed += 1
            if processed % self.progress_every == 0 or processed == total:
                self.progress.emit(processed, total, predicted, skipped, failed, "batch_running")

        self.finished.emit(
            {
                "processed": processed,
                "total": total,
                "predicted": predicted,
                "skipped": skipped,
                "failed": failed,
                "no_odds_predicted": no_odds_predicted,
                "no_odds_skipped": no_odds_skipped,
                "cancelled": self._cancel_requested,
                "results": results,
            }
        )


class MainWindow(QMainWindow):
    MATCH_COL_SELECTED = 0
    MATCH_COL_NAME = 1
    MATCH_COL_LEAGUE = 2
    MATCH_COL_DATE = 3
    MATCH_COL_ODDS_1 = 4
    MATCH_COL_ODDS_X = 5
    MATCH_COL_ODDS_2 = 6
    MATCH_COL_P1 = 7
    MATCH_COL_PX = 8
    MATCH_COL_P2 = 9
    MATCH_COL_DECISION = 10
    MATCH_COL_STATUS = 11
    MATCH_COL_REASON = 12
    MATCH_COL_SOURCE = 13
    MATCH_COL_DB_SAVED = 14
    MATCH_COL_PREDICTABLE = 15

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FootAI")
        self.resize(1200, 800)
        self.setMinimumSize(980, 680)
        self.trainer = AutoTrainer()
        self.footy_api = FootyStatsClient()
        self.ingestion_loader = IngestionLoader(api=self.footy_api)
        # Wire DB to trainer so runtime feature assembly can query team_season_stats / match history.
        self.trainer.db = self.ingestion_loader.db
        self.toto_api = TotoAPI()
        self.toto_optimizer = TotoOptimizer()
        
        self.matches_table = QTableWidget()
        self.matches_input = QTextEdit()
        self.matches_status_label = QLabel()
        self.matches_summary_label = QLabel()
        self.matches_progress_stage_label = QLabel()
        self.matches_progress_bar = QProgressBar()
        self.matches_date_edit = QDateEdit()
        self.matches_start_date_edit = QDateEdit()
        self.matches_end_date_edit = QDateEdit()
        self.matches_search_input = QLineEdit()
        self.matches_only_completed_checkbox = QCheckBox("Только completed")
        self.matches_only_with_odds_checkbox = QCheckBox("Только с odds")
        self.matches_auto_save_checkbox = QCheckBox("Автосохранение в SQLite")
        self.matches_update_existing_checkbox = QCheckBox("Обновлять существующие")
        self.matches_load_mode_combo = QComboBox()
        self.predict_selected_button = QPushButton()
        self.predict_all_button = QPushButton()
        self.model_diagnostic_report_button = QPushButton()
        self.load_live_button = QPushButton()
        self.load_history_button = QPushButton()
        self.load_sqlite_button = QPushButton()
        self.refresh_visible_button = QPushButton()
        self.stop_operation_button = QPushButton()
        self.clear_matches_button = QPushButton()
        self.load_to_db_button = QPushButton()
        self.load_rows_button = QPushButton()
        self.current_matches_rows: list[dict[str, Any]] = []
        self._runtime_feature_context_by_row: dict[int, dict[str, Any]] = {}
        self._stop_requested = False
        self.status_label = QLabel()
        self.training_info_label = QLabel()
        self.training_progress_stage_label = QLabel()
        self.training_progress_bar = QProgressBar()
        self.output_box = QTextEdit()
        self.analysis_summary_box: QTextEdit | None = None
        self.last_train_dataset_summary: dict[str, Any] = {}
        self.last_train_weak_features: list[dict[str, Any]] = []
        self.last_smoke_status: str = "NOT_RUN"
        self.train_input = QTextEdit()
        self.predict_single_input = QTextEdit()
        self.predict_batch_input = QTextEdit()
        self.calibrate_checkbox = QCheckBox("Enable calibration after train")
        self.train_from_db_button = QPushButton()
        self.train_button = QPushButton()
        self.predict_one_button = QPushButton()
        self.predict_batch_button = QPushButton()
        
        # TOTO UI state
        self.current_draws: list[dict[str, Any]] = []
        self.current_draw_id: int | None = None
        self.current_coupons: list[list[str]] = []
        self.current_toto_draw_matches: list[dict[str, Any]] = []
        self.global_toto_history: dict | None = None
        self.toto_draws_combo = QComboBox()
        self.toto_status_label = QLabel()
        self.toto_history_status_label = QLabel("История Baltbet: не загружена")
        self.toto_matches_table = QTableWidget()
        self.toto_coupons_table = QTableWidget()
        self.toto_coupon_lines_box = QTextEdit()
        self.toto_copy_selected_button = QPushButton()
        self.toto_copy_all_button = QPushButton()
        self.toto_copy_base_button = QPushButton()
        self.toto_copy_insurance_button = QPushButton()
        self.toto_copy_changed_button = QPushButton()
        self.toto_coupon_stake_input = QLineEdit()
        self.toto_mode_selector = QComboBox()
        self.toto_input_mode_selector = QComboBox()
        self.toto_insurance_checkbox = QCheckBox("Enable insurance")
        self.toto_insurance_strength_selector = QComboBox()
        self.toto_summary_box = QTextEdit()
        self.toto_summary_verbose_box = QTextEdit()
        self.toto_summary_raw_box = QTextEdit()
        self.toto_selected_coupon_detail_box = QTextEdit()
        self.toto_detail_only_changed_checkbox = QCheckBox("Detail: only changed positions")
        self.toto_compact_coupon_view_checkbox = QCheckBox("Compact coupon list")
        self.toto_show_advanced_checkbox = QCheckBox("Show details/debug")
        self.toto_show_matches_panel_checkbox = QCheckBox("Show matches panel")
        self.toto_load_draws_button = QPushButton()
        self.toto_history_refresh_button = QPushButton()
        self.toto_draw_tools_group = QGroupBox()
        self.toto_generation_intent_label = QLabel()
        self.toto_clear_matches_button = QPushButton()
        self.toto_base_coupons_label = QLabel()
        self.toto_base_coupon_lines_box = QTextEdit()
        self.toto_insurance_coupons_label = QLabel()
        self.toto_insurance_coupon_lines_box = QTextEdit()
        self.toto_unknown_coupons_label = QLabel()
        self.toto_unknown_coupon_lines_box = QTextEdit()
        self.toto_layer_view_box = QTextEdit()
        self.current_toto_summary: dict[str, Any] | None = None
        self.current_coupon_entries: list[dict[str, Any]] = []
        self.current_insured_coupon_indices: set[int] = set()
        self.toto_coupon_display_line_map: dict[int, int] = {}
        self.last_smoke_payload: dict[str, Any] = {}
        self._tooltip_texts: dict[QWidget, str] = {}
        self._tooltip_target: QWidget | None = None
        self._tooltip_delay_ms = 1200
        self._tooltip_timer = QTimer(self)
        self._tooltip_timer.setSingleShot(True)
        self._tooltip_timer.timeout.connect(self._show_delayed_tooltip)
        self._model_fingerprint_cache: tuple[str, float, int, str] | None = None
        self._predict_worker_thread: QThread | None = None
        self._predict_worker: MatchesBatchPredictWorker | None = None
        self._predict_inflight: bool = False
        self._predict_run_context: dict[str, Any] = {}

        tabs = QTabWidget()
        tabs.addTab(self._matches_tab(), "MATCHES")
        tabs.addTab(self._analysis_tab(), "ANALYSIS")
        tabs.addTab(self._toto_tab(), "TOTO")
        tabs.addTab(self._training_tab(), "TRAINING")
        tabs.addTab(self._help_tab(), "СПРАВКА")
        self.setCentralWidget(tabs)
        self._safe_startup_load()

    def _help_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        title = QLabel("Как пользоваться FootAI")
        title.setStyleSheet("font-weight: bold; font-size: 15px;")
        layout.addWidget(title)

        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setPlainText(
              "================================\n"
              "ПОЛНОЕ РУКОВОДСТВО ПО FOOTAI\n"
              "================================\n\n"
              "1. ОСНОВНОЙ ПОРЯДОК РАБОТЫ\n"
              "================================\n"
              "Шаг 1: Загрузить матчи\n"
              "  - Перейти на вкладку MATCHES\n"
              "  - Выбрать дату или диапазон дат\n"
              "  - Нажать 'Загрузить матчи из API' (Live) или выбрать другой режим\n"
              "  - Проверить, что матчи загружены в таблицу\n\n"
              "Шаг 2: Убедиться, что матчи сохранены в БД\n"
              "  - Проверить строку статуса: 'В БД: X' показывает сохраненные матчи\n"
              "  - Если матчи не там, нажать 'Сохранить в БД' (опционально с автосохранением)\n\n"
              "Шаг 3: Загрузить модель (если первый запуск)\n"
              "  - Перейти на вкладку TRAINING\n"
              "  - Нажать 'Загрузить модель'\n"
              "  - Это загружает ранее сохраненные model artifacts (если есть)\n\n"
              "Шаг 4: Обучить модель (если нужно свежее обучение)\n"
              "  - На вкладке TRAINING нажать 'Обучить модель из SQLite'\n"
              "  - Это использует все завершенные (completed) матчи из БД\n"
              "  - Ждать завершения (с прогресс-баром)\n\n"
              "Шаг 5: Проверить готовность модели\n"
              "  - На TRAINING нажать 'Проверить готовность модели'\n"
              "  - Если статус 'готова', можно начинать прогноз\n\n"
              "Шаг 6: Выполнить прогноз\n"
              "  - На MATCHES либо выбрать отдельные матчи и 'Прогноз по выбранным'\n"
              "  - Либо нажать 'Прогноз по всем' для всех видимых матчей\n"
              "  - Результаты появятся в колонке 'Прогноз' и вероятности (P1/PX/P2)\n\n"
              "Шаг 7: (Опционально) Передать матчи в TOTO\n"
              "  - Выбрать матчи на MATCHES (галки)\n"
              "  - Нажать 'Добавить выбранные в TOTO'\n"
              "  - Перейти на TOTO и отредактировать список\n\n"
              "Шаг 8: (Опционально) Сформировать купоны\n"
              "  - На TOTO выбрать draw и режим (16 или 32 варианта)\n"
              "  - Нажать 'Generate coupons'\n"
              "  - Результаты появятся в таблице Coupons\n\n\n"
              "2. РЕЖИМЫ ЗАГРУЗКИ МАТЧЕЙ\n"
              "================================\n"
              "Есть 4 режима для разных сценариев:\n\n"
              "Live (текущая дата):\n"
              "  - Загружает только матчи с API за выбранную дату\n"
              "  - Лучше всего для ежедневной работы\n"
              "  - Матчи еще не завершены, odds могут обновляться\n\n"
              "Historical (диапазон дат):\n"
              "  - Загружает матчи за весь диапазон дат\n"
              "  - Медленнее, чем live, но полезно для накопления истории\n"
              "  - Можно фильтровать (только completed, только с odds)\n"
              "  - Лучше всего для обучения модели\n\n"
              "SQLite (сохраненные локально):\n"
              "  - Читает уже загруженные и сохраненные матчи из локальной БД\n"
              "  - Не требует интернета (быстро)\n"
              "  - Полезно для работы с исторической выборкой\n\n"
              "JSON (технический режим):\n"
              "  - Ручная загрузка матчей из JSON текста\n"
              "  - Для разработчиков, обычно не требуется\n\n\n"
              "3. ЧТО ДЕЛАЕТ 'ЗАГРУЗИТЬ МОДЕЛЬ'\n"
              "================================\n"
              "Эта кнопка загружает ранее сохраненные model artifacts с диска:\n"
              "- Параметры обученной нейросети\n"
              "- Схему признаков (feature columns)\n"
              "- Калибровку вероятностей (если была)\n\n"
              "КОГДА ИСПОЛЬЗОВАТЬ:\n"
              "- После первого обучения, в следующих запусках приложения\n"
              "- Если модель уже была обучена ранее и artifacts целы\n"
              "- Для инферентивного режима (только прогнозы, без обучения)\n\n"
              "ЕСЛИ ARTIFACTS НЕ НАЙДЕНА:\n"
              "- Нажать 'Обучить модель из SQLite'\n"
              "- Убедиться, что в БД достаточно завершенных матчей\n\n\n"
              "4. ЧТО ДЕЛАЕТ 'ОБУЧИТЬ МОДЕЛЬ ИЗ SQLITE'\n"
              "================================\n"
              "Это основной режим обучения модели:\n"
              "- Берет все completed matches из SQLite\n"
              "- Строит feature-row для каждого матча\n"
              "- Обучает LightGBM классификатор (1/X/2)\n"
              "- Сохраняет model artifacts на диск\n\n"
              "КОГДА ИСПОЛЬЗОВАТЬ:\n"
              "- Когда загружено значительное количество исторических матчей\n"
              "- Для улучшения модели свежими данными\n"
              "- После работы с historical load диапазоном дат\n\n"
              "МИНИМАЛЬНЫЕ ТРЕБОВАНИЯ:\n"
              "- Минимум 50-100 completed matches\n"
              "- Каждый матч должен иметь оды (1/X/2)\n"
              "- Данные должны быть сбалансированы по исходам\n\n\n"
              "5. ЧТО ДЕЛАЕТ 'ПРОВЕРИТЬ ГОТОВНОСТЬ МОДЕЛИ'\n"
              "================================\n"
              "Проверяет 4 условия готовности модели:\n"
              "1. Файл модели существует на диске (model artifacts loaded)\n"
              "2. Схема признаков загружена (feature schema)\n"
              "3. Модели LightGBM загружены (predictor trained)\n"
              "4. Калибровщик доступен (если был использован)\n\n"
              "РЕЗУЛЬТАТ:\n"
              "- 'Готова': все условия выполнены, можно прогнозировать\n"
              "- 'Принципиально не обучена': нужно обучить из SQLite\n"
              "- 'Частичная готовность': что-то упущено, нужна диагностика\n\n\n"
              "6. ЧТО ДЕЛАЕТ КНОПКА 'ОСТАНОВИТЬ'\n"
              "================================\n"
              "Запрашивает остановку текущей длительной операции:\n"
              "- Загрузка из API (live, historical)\n"
              "- Обновление видимых матчей\n"
              "- Прогноз по всем матчам\n"
              "- На некоторых операциях остановка может быть задержана\n\n"
              "ВАЖНО:\n"
              "- Не гарантирует мгновенную остановку\n"
              "- Может занять время, пока завершится текущий шаг\n"
              "- Операция может быть в несогласованном состоянии\n\n\n"
              "7. КОЛОНКА 'ПРОГНОЗ' (DECISION)\n"
              "================================\n"
              "В этой колонке находится итоговый прогноз модели.\n\n"
              "ВОЗМОЖНЫЕ ЗНАЧЕНИЯ:\n"
              "- '1' = победа домашней команды (home win)\n"
              "- 'X' = ничья (draw)\n"
              "- '2' = победа гостевой команды (away win)\n"
              "- '1X' = двойной исход (home win or draw)\n"
              "- 'X2' = двойной исход (draw or away win)\n"
              "- '12' = двойной исход (home win or away win, без ничьи)\n\n"
              "КАК ФОРМИРУЕТСЯ ПРОГНОЗ:\n"
              "1. Модель рассчитывает P1, PX, P2 (вероятности)\n"
              "2. Если два исхода очень близки (разница < 15%), берутся оба\n"
              "3. Иначе выбирается исход с максимальной вероятностью\n\n"
              "ИСПОЛЬЗОВАНИЕ:\n"
              "- Одиночный прогноз: выбрать исход с наибольшей уверенностью\n"
              "- Двойной прогноз: подстраховка, если два исхода примерно равновероятны\n\n\n"
              "8. ВКЛАДКА TOTO\n"
              "================================\n"
              "Сценарий работы:\n"
              "1. На MATCHES выбрать матчи галками\n"
              "2. Нажать 'Добавить выбранные в TOTO'\n"
              "3. Матчи переместятся во вкладку TOTO\n"
              "4. Выбрать draw (если нужна жеребьевка)\n"
              "5. Если нужно, отредактировать порядок и включение матчей\n"
              "6. Нажать 'Generate coupons'\n"
              "7. Купоны появятся в таблице ниже\n\n"
              "DRAWs И ОТДЕЛЬНЫЙ TOTO API:\n"
              "- Кнопка Load draws использует отдельный TOTO API (TOTO_API_BASE_URL).\n"
              "- Если TOTO_API_BASE_URL не настроен, Draws недоступны, это нормально.\n"
              "- Базовый рабочий сценарий не ломается: MATCHES -> Добавить выбранные в TOTO -> Generate coupons.\n\n"
              "РЕЖИМЫ КУПОНОВ:\n"
              "- 16: 16 вариантов на купон (классический TOTO)\n"
              "- 32: 32 варианта на купон (расширенный)\n\n"
              "INSURANCE (Страховка):\n"
              "- При включении добавляет дополнительные комбинации\n"
              "- Увеличивает количество купонов, но повышает шансы на выигрыш\n"
              "- Strength: 0.3 = минимум, 0.9 = максимум доп. комбинаций\n\n\n"
              "9. ВКЛАДКА ANALYSIS\n"
              "================================\n"
              "Текущий статус: содержит базовую сводку по БД и модели.\n\n"
              "ОТОБРАЖАЕТСЯ:\n"
              "- Всего матчей в БД\n"
              "- Завершенных матчей\n"
              "- Матчей с коэффициентами (odds)\n"
              "- Готовых для обучения строк\n"
              "- Дата последнего обучения\n"
              "- Статус модели\n"
              "- Баланс классов (1/X/2 проценты)\n"
              "- Предупреждения по качеству признаков\n\n"
              "ПЛАНЫ РАСШИРЕНИЯ:\n"
              "- Подробные графики и метрики модели\n"
              "- Анализ качества прогнозов\n"
              "- Сравнение моделей\n"
              "- Импорт/экспорт данных для анализа\n\n\n"
              "10. ЧАСТЫЕ ОШИБКИ И РЕШЕНИЯ\n"
              "================================\n"
              "\"Модель не готова\" при попытке прогноза:\n"
              "  Решение: нажать 'Загрузить модель' или 'Обучить из SQLite'\n\n"
              "\"Нет коэффициентов\" (пропущено много строк):\n"
              "  Решение: использовать только матчи с заполненными odds\n\n"
              "\"Недостаточно истории\" при обучении:\n"
              "  Решение: загрузить historical диапазон на несколько месяцев\n\n"
              "\"Очень медленно загружается historical\":\n"
              "  Решение: использовать фильтры (только completed, только odds)\n\n"
              "\n================================\n"
              "Версия: 1.0 (2026-03-24)\n"
              "Телеметрия выключена. Все данные хранятся локально.\n"
              "================================\n"
        )
        layout.addWidget(help_text)
        return widget

    def eventFilter(self, watched: object, event: QEvent) -> bool:
        if isinstance(watched, QWidget) and watched in self._tooltip_texts:
            if event.type() == QEvent.Type.Enter:
                self._tooltip_target = watched
                self._tooltip_timer.start(self._tooltip_delay_ms)
            elif event.type() in (QEvent.Type.Leave, QEvent.Type.Hide, QEvent.Type.FocusOut):
                if self._tooltip_target is watched:
                    self._tooltip_target = None
                    self._tooltip_timer.stop()
                    QToolTip.hideText()
        return super().eventFilter(watched, event)

    def _register_delayed_tooltip(self, widget: QWidget, text: str) -> None:
        self._tooltip_texts[widget] = text
        widget.setToolTip("")
        widget.installEventFilter(self)

    def _show_delayed_tooltip(self) -> None:
        if self._tooltip_target is None:
            return
        text = self._tooltip_texts.get(self._tooltip_target)
        if not text:
            return
        if not self._tooltip_target.isVisible():
            return
        pos = self._tooltip_target.mapToGlobal(self._tooltip_target.rect().bottomLeft())
        QToolTip.showText(pos, text, self._tooltip_target, self._tooltip_target.rect(), 6000)

    def _selected_matches_date(self) -> str:
        return self.matches_date_edit.date().toString("yyyy-MM-dd")

    def _on_matches_date_changed(self) -> None:
        selected_date = self._selected_matches_date()
        self.matches_status_label.setText(
            f"Матчи: выбрана дата {selected_date}. Нажмите 'Загрузить матчи из API' для обновления списка."
        )
        self._update_matches_summary()

    def _apply_matches_filter(self) -> None:
        query = self.matches_search_input.text().strip().lower()
        visible_rows = 0
        total_rows = self.matches_table.rowCount()
        for row_idx in range(total_rows):
            name_item = self.matches_table.item(row_idx, self.MATCH_COL_NAME)
            league_item = self.matches_table.item(row_idx, self.MATCH_COL_LEAGUE)
            date_item = self.matches_table.item(row_idx, self.MATCH_COL_DATE)
            reason_item = self.matches_table.item(row_idx, self.MATCH_COL_REASON)
            name_text = name_item.text().lower() if name_item else ""
            league_text = league_item.text().lower() if league_item else ""
            date_text = date_item.text().lower() if date_item else ""
            reason_text = reason_item.text().lower() if reason_item else ""
            match_visible = not query or any(query in text for text in (name_text, league_text, date_text, reason_text))
            self.matches_table.setRowHidden(row_idx, not match_visible)
            if match_visible:
                visible_rows += 1

        if query:
            self.matches_status_label.setText(
                f"Матчи: фильтр '{query}' -> показано {visible_rows} из {total_rows}"
            )

    def _analysis_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        title = QLabel("ANALYSIS: сводка данных и качества признаков")
        title.setStyleSheet("font-weight: bold; font-size: 15px;")
        layout.addWidget(title)

        refresh_button = QPushButton("Обновить анализ")
        refresh_button.clicked.connect(self._refresh_analysis_summary)
        layout.addWidget(refresh_button)

        self.analysis_summary_box = QTextEdit()
        self.analysis_summary_box.setReadOnly(True)
        self.analysis_summary_box.setPlainText(
            "Нажмите «Обновить анализ» для загрузки сводки.\n\n"
            "Назначение вкладки ANALYSIS:\n"
            "- Быстрый обзор состояния данных и модели.\n"
            "- Контроль готовности перед train/predict.\n"
            "- Раздел отображает weak features после Train from DB.\n"
        )
        layout.addWidget(self.analysis_summary_box)
        layout.addStretch()
        return widget

    def _simple_tab(self, text: str) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel(text))
        return widget

    def _toto_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        section_title_style = "font-weight: 600; color: #E6EDF3; margin: 2px 0 4px 0;"
        panel_style = (
            "QGroupBox {"
            "  font-weight: 600;"
            "  color: #E6EDF3;"
            "  background: #18222E;"
            "  border: 1px solid #2A3A4C;"
            "  border-radius: 6px;"
            "  margin-top: 8px;"
            "  padding-top: 10px;"
            "}"
            "QGroupBox::title {"
            "  subcontrol-origin: margin;"
            "  left: 10px;"
            "  color: #C9D6E2;"
            "  padding: 0 4px 0 4px;"
            "}"
        )
        compact_text_box_style = (
            "QTextEdit {"
            "  border: 1px solid #2A3A4C;"
            "  border-radius: 6px;"
            "  padding: 6px;"
            "  background: #101822;"
            "  color: #E6EDF3;"
            "  selection-background-color: #2E5C88;"
            "  selection-color: #FFFFFF;"
            "}"
        )
        monospace_text_box_style = (
            "QTextEdit {"
            "  border: 1px solid #2A3A4C;"
            "  border-radius: 6px;"
            "  padding: 6px;"
            "  background: #0F1720;"
            "  color: #E6EDF3;"
            "  font-family: Consolas, 'Courier New', monospace;"
            "  font-size: 12px;"
            "  line-height: 1.25;"
            "  selection-background-color: #2E5C88;"
            "  selection-color: #FFFFFF;"
            "}"
        )
        button_style = (
            "QPushButton {"
            "  background: #213042;"
            "  color: #E6EDF3;"
            "  border: 1px solid #2F4257;"
            "  border-radius: 6px;"
            "  padding: 4px 10px;"
            "}"
            "QPushButton:hover {"
            "  background: #2A3E55;"
            "}"
            "QPushButton:pressed {"
            "  background: #1A2836;"
            "}"
            "QPushButton:disabled {"
            "  color: #8FA2B5;"
            "  background: #1A2430;"
            "  border-color: #2A3644;"
            "}"
        )
        primary_button_style = (
            "QPushButton {"
            "  background: #2D5F8B;"
            "  color: #FFFFFF;"
            "  border: 1px solid #3B79AE;"
            "  border-radius: 6px;"
            "  padding: 4px 12px;"
            "  font-weight: 600;"
            "}"
            "QPushButton:hover {"
            "  background: #3A73A3;"
            "}"
            "QPushButton:pressed {"
            "  background: #254F74;"
            "}"
        )

        self.toto_status_label.setText("TOTO status: ready")
        self.toto_status_label.setStyleSheet("font-weight: 700; color: #E6EDF3; padding: 2px 0 4px 0;")
        layout.addWidget(self.toto_status_label)

        # ── History block ────────────────────────────────────────────────────
        history_group = QGroupBox("История Baltbet (global context)")
        history_group.setStyleSheet(panel_style)
        history_group_layout = QHBoxLayout(history_group)
        history_group_layout.setContentsMargins(10, 8, 10, 8)
        history_group_layout.setSpacing(8)
        self.toto_history_refresh_button.setText("Загрузить историю Baltbet")
        self.toto_history_refresh_button.setMinimumHeight(30)
        self.toto_history_refresh_button.setStyleSheet(button_style)
        self.toto_history_refresh_button.clicked.connect(self._on_load_toto_baltbet_history)
        history_group_layout.addWidget(self.toto_history_refresh_button)
        self.toto_history_status_label.setStyleSheet("color: #AFC0CF;")
        history_group_layout.addWidget(self.toto_history_status_label, 1)
        self._register_delayed_tooltip(
            self.toto_history_refresh_button,
            "Загружает историю тиражей Baltbet через TOTO API и передаёт её в оптимизатор. "
            "После загрузки все генерации в ручном режиме (MATCHES → TOTO) учитывают исторические сигналы.",
        )
        layout.addWidget(history_group)
        # ── /History block ───────────────────────────────────────────────────

        top_controls = QGroupBox("TOTO: основной сценарий")
        top_controls.setStyleSheet(panel_style)
        top_controls_layout = QVBoxLayout(top_controls)
        top_controls_layout.setContentsMargins(10, 8, 10, 8)
        top_controls_layout.setSpacing(8)

        scenario_note = QLabel("Сценарий: MATCHES -> TOTO -> Generate coupons -> Copy final coupons")
        scenario_note.setStyleSheet("color: #C1CFDB;")
        top_controls_layout.addWidget(scenario_note)

        # Keep draw controls available in code, but hide them from main product UI.
        self.toto_input_mode_selector.clear()
        self.toto_input_mode_selector.addItem("Manual matches", userData="manual_matches")
        self.toto_input_mode_selector.addItem("Draw mode", userData="draw_mode")
        self.toto_input_mode_selector.setCurrentIndex(0)
        self.toto_input_mode_selector.currentIndexChanged.connect(self._on_change_toto_input_mode)
        self.toto_input_mode_selector.setVisible(False)
        self.toto_load_draws_button.setText("Загрузить тиражи (TOTO API)")
        self.toto_load_draws_button.setStyleSheet(button_style)
        self.toto_load_draws_button.clicked.connect(self._on_load_toto_draws)
        self.toto_draws_combo.currentIndexChanged.connect(self._on_select_toto_draw)
        self.toto_draw_tools_group.setVisible(False)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        mode_row.addWidget(QLabel("Coupon mode:"))
        self.toto_mode_selector.clear()
        self.toto_mode_selector.addItems(["16", "32"])
        self.toto_mode_selector.setMinimumHeight(30)
        self.toto_mode_selector.setMinimumWidth(70)
        self.toto_mode_selector.currentIndexChanged.connect(self._update_toto_generation_intent_label)
        mode_row.addWidget(self.toto_mode_selector)

        self.toto_insurance_checkbox.setChecked(False)
        self.toto_insurance_checkbox.setMinimumHeight(30)
        mode_row.addWidget(self.toto_insurance_checkbox)

        mode_row.addWidget(QLabel("Insurance strength:"))
        self.toto_insurance_strength_selector.clear()
        self.toto_insurance_strength_selector.addItems(["0.3", "0.5", "0.7", "0.9"])
        self.toto_insurance_strength_selector.setCurrentText("0.7")
        self.toto_insurance_strength_selector.setEnabled(False)
        self.toto_insurance_strength_selector.setMinimumHeight(30)
        self.toto_insurance_strength_selector.setMinimumWidth(70)
        self.toto_insurance_checkbox.toggled.connect(self.toto_insurance_strength_selector.setEnabled)
        self.toto_insurance_checkbox.toggled.connect(self._update_toto_generation_intent_label)
        self.toto_insurance_checkbox.toggled.connect(self._update_toto_copy_actions_state)
        self.toto_insurance_strength_selector.currentIndexChanged.connect(self._update_toto_generation_intent_label)
        mode_row.addWidget(self.toto_insurance_strength_selector)

        mode_row.addWidget(QLabel("Coupon stake:"))
        self.toto_coupon_stake_input.setText("30")
        self.toto_coupon_stake_input.setMaximumWidth(80)
        self.toto_coupon_stake_input.setMinimumHeight(30)
        self.toto_coupon_stake_input.setPlaceholderText("30")
        mode_row.addWidget(self.toto_coupon_stake_input)

        self.toto_compact_coupon_view_checkbox.setChecked(True)
        self.toto_compact_coupon_view_checkbox.toggled.connect(self._refresh_coupon_views)
        self.toto_show_advanced_checkbox.setChecked(False)
        self.toto_show_matches_panel_checkbox.setChecked(False)

        generate_button = QPushButton("Generate coupons")
        generate_button.setMinimumHeight(32)
        generate_button.setStyleSheet(primary_button_style)
        generate_button.clicked.connect(self._on_generate_toto_coupons)
        mode_row.addWidget(generate_button)
        top_controls_layout.addLayout(mode_row)

        self.toto_generation_intent_label.setStyleSheet("color: #CFE0F0; font-weight: 600;")
        top_controls_layout.addWidget(self.toto_generation_intent_label)

        layout.addWidget(top_controls)

        result_zone = QWidget()
        result_layout = QVBoxLayout(result_zone)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.setSpacing(8)

        summary_group = QGroupBox("TOTO summary")
        summary_group.setStyleSheet(panel_style)
        summary_group_layout = QVBoxLayout(summary_group)
        summary_group_layout.setContentsMargins(10, 8, 10, 8)
        summary_group_layout.setSpacing(6)
        summary_hint = QLabel("Сначала итог прогона, затем купоны, потом слои и только после этого debug.")
        summary_hint.setStyleSheet("color: #AFC0CF;")
        summary_group_layout.addWidget(summary_hint)
        self.toto_summary_box.setReadOnly(True)
        self.toto_summary_box.setMinimumHeight(120)
        self.toto_summary_box.setMaximumHeight(150)
        self.toto_summary_box.setStyleSheet(compact_text_box_style)
        self.toto_summary_box.setPlaceholderText(
            "base coupons, insurance coupons, strategy-adjusted matches, insurance-target matches, insurance-changed matches"
        )
        summary_group_layout.addWidget(self.toto_summary_box)
        result_layout.addWidget(summary_group)

        base_group = QGroupBox("BASE COUPONS")
        base_group.setStyleSheet(panel_style)
        base_group_layout = QVBoxLayout(base_group)
        base_group_layout.setContentsMargins(10, 8, 10, 8)
        base_group_layout.setSpacing(6)
        self.toto_base_coupons_label.setStyleSheet("color: #C9D6E2; font-weight: 600;")
        base_group_layout.addWidget(self.toto_base_coupons_label)
        self.toto_base_coupon_lines_box.setReadOnly(True)
        self.toto_base_coupon_lines_box.setMinimumHeight(150)
        self.toto_base_coupon_lines_box.setStyleSheet(monospace_text_box_style)
        self.toto_base_coupon_lines_box.setPlaceholderText("Base coupons will appear here.")
        base_group_layout.addWidget(self.toto_base_coupon_lines_box)
        result_layout.addWidget(base_group)

        insurance_group = QGroupBox("INSURANCE COUPONS")
        insurance_group.setStyleSheet(panel_style)
        insurance_group_layout = QVBoxLayout(insurance_group)
        insurance_group_layout.setContentsMargins(10, 8, 10, 8)
        insurance_group_layout.setSpacing(6)
        self.toto_insurance_coupons_label.setStyleSheet("color: #C9D6E2; font-weight: 600;")
        insurance_group_layout.addWidget(self.toto_insurance_coupons_label)
        self.toto_insurance_coupon_lines_box.setReadOnly(True)
        self.toto_insurance_coupon_lines_box.setMinimumHeight(150)
        self.toto_insurance_coupon_lines_box.setStyleSheet(monospace_text_box_style)
        self.toto_insurance_coupon_lines_box.setPlaceholderText("Insurance coupons will appear here when insurance is enabled.")
        insurance_group_layout.addWidget(self.toto_insurance_coupon_lines_box)
        result_layout.addWidget(insurance_group)

        unknown_group = QGroupBox("UNKNOWN COUPON TYPE")
        unknown_group.setStyleSheet(panel_style)
        unknown_group_layout = QVBoxLayout(unknown_group)
        unknown_group_layout.setContentsMargins(10, 8, 10, 8)
        unknown_group_layout.setSpacing(6)
        self.toto_unknown_coupons_label.setStyleSheet("color: #C9D6E2; font-weight: 600;")
        unknown_group_layout.addWidget(self.toto_unknown_coupons_label)
        self.toto_unknown_coupon_lines_box.setReadOnly(True)
        self.toto_unknown_coupon_lines_box.setMinimumHeight(110)
        self.toto_unknown_coupon_lines_box.setStyleSheet(monospace_text_box_style)
        self.toto_unknown_coupon_lines_box.setPlaceholderText("Coupons with incomplete type metadata will appear here.")
        unknown_group_layout.addWidget(self.toto_unknown_coupon_lines_box)
        unknown_group.setVisible(False)
        result_layout.addWidget(unknown_group)

        coupon_export_controls = QHBoxLayout()
        self.toto_copy_all_button.setText("Copy final coupons")
        self.toto_copy_all_button.setStyleSheet(button_style)
        self.toto_copy_all_button.clicked.connect(self._on_copy_all_coupon_lines)
        coupon_export_controls.addWidget(self.toto_copy_all_button)
        self.toto_copy_base_button.setText("Copy base")
        self.toto_copy_base_button.setStyleSheet(button_style)
        self.toto_copy_base_button.clicked.connect(self._on_copy_base_coupon_lines)
        coupon_export_controls.addWidget(self.toto_copy_base_button)
        self.toto_copy_insurance_button.setText("Copy insurance")
        self.toto_copy_insurance_button.setStyleSheet(button_style)
        self.toto_copy_insurance_button.clicked.connect(self._on_copy_insurance_coupon_lines)
        coupon_export_controls.addWidget(self.toto_copy_insurance_button)
        coupon_export_controls.addStretch()
        result_layout.addLayout(coupon_export_controls)

        layer_group = QGroupBox("Compact per-match layer view")
        layer_group.setStyleSheet(panel_style)
        layer_group_layout = QVBoxLayout(layer_group)
        layer_group_layout.setContentsMargins(10, 8, 10, 8)
        layer_group_layout.setSpacing(6)
        layer_hint = QLabel("Match | model | strategy | insurance added | final coverage")
        layer_hint.setStyleSheet("color: #AFC0CF;")
        layer_group_layout.addWidget(layer_hint)
        self.toto_layer_view_box.setReadOnly(True)
        self.toto_layer_view_box.setMinimumHeight(180)
        self.toto_layer_view_box.setStyleSheet(compact_text_box_style)
        self.toto_layer_view_box.setPlaceholderText("Compact per-match layer view appears here after coupon generation.")
        layer_group_layout.addWidget(self.toto_layer_view_box)
        result_layout.addWidget(layer_group)

        view_controls = QHBoxLayout()
        view_controls.addWidget(self.toto_show_matches_panel_checkbox)
        view_controls.addWidget(self.toto_show_advanced_checkbox)
        view_controls.addStretch()
        result_layout.addLayout(view_controls)

        advanced_wrap = QWidget()
        advanced_layout = QVBoxLayout(advanced_wrap)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setSpacing(6)

        advanced_tabs = QTabWidget()
        advanced_tabs.setDocumentMode(True)
        advanced_tabs.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #2A3A4C; background: #121B26; }"
            "QTabBar::tab { background: #1C2A39; color: #BFD0DF; padding: 5px 10px; border: 1px solid #2A3A4C; }"
            "QTabBar::tab:selected { background: #27394D; color: #E6EDF3; }"
        )

        coupon_grid_tab = QWidget()
        coupon_grid_layout = QVBoxLayout(coupon_grid_tab)
        coupon_grid_layout.setContentsMargins(0, 0, 0, 0)
        coupon_grid_layout.addWidget(QLabel("Coupon grid (detailed row view):"))
        self.toto_coupons_table.setColumnCount(15)
        self.toto_coupons_table.setHorizontalHeaderLabels([f"M{i+1}" for i in range(15)])
        self.toto_coupons_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.toto_coupons_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.toto_coupons_table.setMinimumHeight(220)
        self.toto_coupons_table.setAlternatingRowColors(True)
        self.toto_coupons_table.verticalHeader().setDefaultSectionSize(24)
        self.toto_coupons_table.setStyleSheet(
            "QTableWidget {"
            "  background: #0F1720;"
            "  color: #E6EDF3;"
            "  border: 1px solid #2A3A4C;"
            "  gridline-color: #263648;"
            "  alternate-background-color: #141F2A;"
            "}"
            "QTableWidget::item:selected {"
            "  background: #2D5F8B;"
            "  color: #FFFFFF;"
            "}"
            "QHeaderView::section {"
            "  background: #1B2A39;"
            "  color: #D7E2ED;"
            "  border: 1px solid #2A3A4C;"
            "  padding: 4px;"
            "  font-weight: 600;"
            "}"
        )
        self.toto_coupons_table.itemSelectionChanged.connect(self._on_select_toto_coupon_row)
        coupon_grid_layout.addWidget(self.toto_coupons_table)
        advanced_tabs.addTab(coupon_grid_tab, "Coupon Grid")

        combined_lines_tab = QWidget()
        combined_lines_layout = QVBoxLayout(combined_lines_tab)
        combined_lines_layout.setContentsMargins(0, 0, 0, 0)
        combined_lines_layout.addWidget(QLabel("Combined coupon list (BASE / INSURANCE / UNKNOWN):"))
        self.toto_coupon_lines_box.setReadOnly(True)
        self.toto_coupon_lines_box.setMinimumHeight(180)
        self.toto_coupon_lines_box.setStyleSheet(monospace_text_box_style)
        self.toto_coupon_lines_box.setPlaceholderText(
            "Combined coupon list appears here for detailed review and exact row mapping."
        )
        combined_lines_layout.addWidget(self.toto_coupon_lines_box)
        advanced_tabs.addTab(combined_lines_tab, "Coupon Lines")

        selected_diff_tab = QWidget()
        selected_diff_layout = QVBoxLayout(selected_diff_tab)
        selected_diff_layout.setContentsMargins(0, 0, 0, 0)
        selected_diff_layout.addWidget(QLabel("Selected coupon detail / diff:"))
        self.toto_detail_only_changed_checkbox.setChecked(False)
        self.toto_detail_only_changed_checkbox.toggled.connect(self._on_select_toto_coupon_row)
        selected_diff_layout.addWidget(self.toto_detail_only_changed_checkbox)
        self.toto_selected_coupon_detail_box.setReadOnly(True)
        self.toto_selected_coupon_detail_box.setMinimumHeight(120)
        self.toto_selected_coupon_detail_box.setStyleSheet(monospace_text_box_style)
        self.toto_selected_coupon_detail_box.setPlaceholderText(
            "Select a coupon row to inspect exact BASE vs INSURANCE diff."
        )
        selected_diff_layout.addWidget(self.toto_selected_coupon_detail_box)
        advanced_tabs.addTab(selected_diff_tab, "Selected Diff")

        diagnostics_tab = QWidget()
        diagnostics_layout = QVBoxLayout(diagnostics_tab)
        diagnostics_layout.setContentsMargins(0, 0, 0, 0)
        diagnostics_layout.addWidget(QLabel("Diagnostics / explainability:"))
        self.toto_summary_verbose_box.setReadOnly(True)
        self.toto_summary_verbose_box.setMinimumHeight(150)
        self.toto_summary_verbose_box.setStyleSheet(compact_text_box_style)
        self.toto_summary_verbose_box.setPlaceholderText("Secondary diagnostics and explainability details.")
        diagnostics_layout.addWidget(self.toto_summary_verbose_box)
        advanced_tabs.addTab(diagnostics_tab, "Diagnostics")

        raw_tab = QWidget()
        raw_layout = QVBoxLayout(raw_tab)
        raw_layout.setContentsMargins(0, 0, 0, 0)
        raw_layout.addWidget(QLabel("Raw JSON summary (debug):"))
        self.toto_summary_raw_box.setReadOnly(True)
        self.toto_summary_raw_box.setMinimumHeight(150)
        self.toto_summary_raw_box.setStyleSheet(monospace_text_box_style)
        self.toto_summary_raw_box.setPlaceholderText("Raw JSON summary for debugging.")
        raw_layout.addWidget(self.toto_summary_raw_box)
        advanced_tabs.addTab(raw_tab, "Raw JSON")

        advanced_layout.addWidget(advanced_tabs)
        advanced_wrap.setVisible(False)
        result_layout.addWidget(advanced_wrap)

        matches_wrap = QWidget()
        matches_layout = QVBoxLayout(matches_wrap)
        matches_layout.setContentsMargins(0, 0, 0, 0)
        matches_layout.setSpacing(6)
        matches_section_label = QLabel("TOTO matches (manual include + order):")
        matches_section_label.setStyleSheet(section_title_style)
        matches_layout.addWidget(matches_section_label)
        self.toto_matches_table.setColumnCount(4)
        self.toto_matches_table.setHorizontalHeaderLabels(["Order", "Include", "Original", "Match"])
        self.toto_matches_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.toto_matches_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.toto_matches_table.setMinimumHeight(220)
        self.toto_matches_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.toto_matches_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        matches_layout.addWidget(self.toto_matches_table)

        match_order_buttons = QHBoxLayout()
        move_up_button = QPushButton("Move up")
        move_up_button.setStyleSheet(button_style)
        move_up_button.clicked.connect(self._on_move_toto_match_up)
        match_order_buttons.addWidget(move_up_button)

        move_down_button = QPushButton("Move down")
        move_down_button.setStyleSheet(button_style)
        move_down_button.clicked.connect(self._on_move_toto_match_down)
        match_order_buttons.addWidget(move_down_button)
        self.toto_clear_matches_button.setText("Очистить список матчей TOTO")
        self.toto_clear_matches_button.setStyleSheet(button_style)
        self.toto_clear_matches_button.clicked.connect(self._on_clear_toto_matches)
        match_order_buttons.addWidget(self.toto_clear_matches_button)
        match_order_buttons.addStretch()
        matches_layout.addLayout(match_order_buttons)
        matches_wrap.setVisible(False)
        result_layout.addWidget(matches_wrap)

        self.toto_show_advanced_checkbox.toggled.connect(advanced_wrap.setVisible)
        self.toto_show_matches_panel_checkbox.toggled.connect(matches_wrap.setVisible)

        layout.addWidget(result_zone, 1)

        self._refresh_toto_draws_availability()

        self._register_delayed_tooltip(
            self.toto_load_draws_button,
            "Загружает тиражи через внешний TOTO API (требует TOTO_API_BASE_URL в config). "
            "Нужна только для режима тиражей. "
            "Если вы переносите матчи из вкладки MATCHES вручную — эта кнопка не нужна.",
        )
        self._register_delayed_tooltip(
            self.toto_input_mode_selector,
            "Manual matches: основной сценарий MATCHES -> TOTO без draw controls. Draw mode: отдельный сценарий с загрузкой тиражей и Selected draw.",
        )
        self._register_delayed_tooltip(
            self.toto_clear_matches_button,
            "Очищает текущий список матчей TOTO (ручной и draw-список).",
        )
        self._register_delayed_tooltip(
            self.toto_draws_combo,
            "Selected draw используется только для draw-mode (TotoBrief/TOTO API). В manual mode (MATCHES -> TOTO) генерация работает из текущего ручного списка матчей.",
        )
        self._register_delayed_tooltip(
            self.toto_copy_selected_button,
            "Копирует только выбранный купон в чистом формате stake;M1;M2;...;M15.",
        )
        self._register_delayed_tooltip(
            self.toto_copy_all_button,
            "Копирует итоговые купоны в чистом формате stake;M1;...;M15 без служебного текста.",
        )
        self._register_delayed_tooltip(
            self.toto_copy_base_button,
            "Копирует только BASE купоны в чистом формате, без diff и служебных пояснений.",
        )
        self._register_delayed_tooltip(
            self.toto_copy_insurance_button,
            "Копирует только INSURANCE купоны в чистом формате, без diff и служебных пояснений.",
        )
        self._register_delayed_tooltip(
            self.toto_copy_changed_button,
            "Копирует только изменения (M#: base -> insurance) для выбранного insurance купона.",
        )
        self._register_delayed_tooltip(
            self.toto_detail_only_changed_checkbox,
            "Переключает detail-view: полный diff (base/insurance/changed) или только changed позиции.",
        )
        self._register_delayed_tooltip(
            self.toto_compact_coupon_view_checkbox,
            "Compact coupon list: в основном списке показываются только чистые coupon lines без verbose diff на каждую строку.",
        )
        self._register_delayed_tooltip(
            self.toto_show_advanced_checkbox,
            "Показывает/скрывает secondary блок: selected diff, explainability и raw JSON.",
        )
        self._register_delayed_tooltip(
            self.toto_show_matches_panel_checkbox,
            "Показывает/скрывает панель выбора матчей, чтобы освободить место под купоны.",
        )
        self._register_delayed_tooltip(
            self.toto_coupon_stake_input,
            "Сумма/ставка купона для copy-ready формата. Не связана с режимом 16/32.",
        )

        self._update_toto_history_status_label()
        self._update_toto_generation_intent_label()
        self._refresh_toto_mode_ui()
        self._update_toto_copy_actions_state()
        self.toto_base_coupons_label.setText("BASE COUPONS (0)")
        self.toto_insurance_coupons_label.setText("INSURANCE COUPONS (0)")
        self.toto_unknown_coupons_label.setText("UNKNOWN COUPON TYPE (0)")

        return widget

    def _matches_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        control_group = QGroupBox("Управление загрузкой")
        control_layout = QVBoxLayout(control_group)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Дата:"))
        self.matches_date_edit.setCalendarPopup(True)
        self.matches_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.matches_date_edit.setDate(QDate.currentDate())
        self.matches_date_edit.dateChanged.connect(lambda _: self._on_matches_date_changed())
        top_row.addWidget(self.matches_date_edit)

        top_row.addWidget(QLabel("Start:"))
        self.matches_start_date_edit.setCalendarPopup(True)
        self.matches_start_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.matches_start_date_edit.setDate(QDate.currentDate().addDays(-7))
        top_row.addWidget(self.matches_start_date_edit)

        top_row.addWidget(QLabel("End:"))
        self.matches_end_date_edit.setCalendarPopup(True)
        self.matches_end_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.matches_end_date_edit.setDate(QDate.currentDate())
        top_row.addWidget(self.matches_end_date_edit)

        top_row.addWidget(QLabel("Поиск:"))
        self.matches_search_input.setPlaceholderText("Поиск: команда, лига, дата")
        self.matches_search_input.textChanged.connect(lambda _: self._apply_matches_filter())
        top_row.addWidget(self.matches_search_input, 1)

        control_layout.addLayout(top_row)

        second_row = QHBoxLayout()
        # matches_load_mode_combo инициализируется, но не добавляется в layout
        # (кнопки загрузки ниже явно разделены по назначению)
        self.matches_load_mode_combo.clear()
        self.matches_load_mode_combo.addItems([
            "Live за выбранную дату",
            "Historical за диапазон дат",
            "SQLite фильтр",
        ])

        self.matches_only_completed_checkbox.setChecked(False)
        second_row.addWidget(self.matches_only_completed_checkbox)
        self.matches_only_with_odds_checkbox.setChecked(False)
        second_row.addWidget(self.matches_only_with_odds_checkbox)
        self.matches_auto_save_checkbox.setChecked(True)
        second_row.addWidget(self.matches_auto_save_checkbox)
        second_row.addWidget(self.matches_update_existing_checkbox)
        second_row.addStretch()
        control_layout.addLayout(second_row)

        action_row = QHBoxLayout()
        self.load_live_button.setText("Загрузить из API (за дату)")
        self.load_live_button.clicked.connect(self._on_load_live_matches)
        action_row.addWidget(self.load_live_button)

        self.load_history_button.setText("Загрузить из API (Start→End)")
        self.load_history_button.clicked.connect(self._on_load_historical_matches)
        action_row.addWidget(self.load_history_button)

        self.load_sqlite_button.setText("Из SQLite (Start→End)")
        self.load_sqlite_button.clicked.connect(self._on_load_matches_from_sqlite)
        action_row.addWidget(self.load_sqlite_button)

        self.refresh_visible_button.setText("Обновить видимые")
        self.refresh_visible_button.clicked.connect(self._on_refresh_visible_matches)
        action_row.addWidget(self.refresh_visible_button)

        self.stop_operation_button.setText("Остановить")
        self.stop_operation_button.clicked.connect(self._on_stop_operation)
        action_row.addWidget(self.stop_operation_button)

        control_layout.addLayout(action_row)
        layout.addWidget(control_group)

        self.matches_summary_label.setText("Дата: - | UI: 0 | В БД: 0 | С odds: 0 | Готово к прогнозу: 0 | Модель: не готова")
        self.matches_summary_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.matches_summary_label)

        self.matches_status_label.setText("Матчи: строки ещё не загружены")
        layout.addWidget(self.matches_status_label)

        self.matches_progress_stage_label.setText("Этап: ожидание")
        layout.addWidget(self.matches_progress_stage_label)
        self.matches_progress_bar.setRange(0, 100)
        self.matches_progress_bar.setValue(0)
        layout.addWidget(self.matches_progress_bar)

        splitter = QSplitter(Qt.Orientation.Vertical)

        self.matches_table.setColumnCount(16)
        self.matches_table.setHorizontalHeaderLabels(
            [
                "Selected",
                "Match",
                "League",
                "Date",
                "Кф 1",
                "Кф X",
                "Кф 2",
                "P1",
                "PX",
                "P2",
                "Прогноз",
                "Status",
                "Причина",
                "Источник",
                "DB saved",
                "Predictable",
            ]
        )
        self.matches_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.matches_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.matches_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.matches_table.horizontalHeader().setSectionResizeMode(self.MATCH_COL_NAME, QHeaderView.ResizeMode.Stretch)
        self.matches_table.horizontalHeader().setSectionResizeMode(self.MATCH_COL_REASON, QHeaderView.ResizeMode.Stretch)
        self.matches_table.itemChanged.connect(self._on_matches_table_item_changed)

        table_wrap = QWidget()
        table_layout = QVBoxLayout(table_wrap)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.addWidget(self.matches_table)
        splitter.addWidget(table_wrap)

        technical_group = QGroupBox("Технический JSON-режим (для разработчика)")
        technical_group.setCheckable(True)
        technical_group.setChecked(False)
        technical_layout = QVBoxLayout(technical_group)

        self.matches_input.setPlaceholderText(
            "[\n"
            "  {\"match\": \"Team A vs Team B\", \"odds_ft_1\": 1.9, \"odds_ft_x\": 3.2, \"odds_ft_2\": 4.1},\n"
            "  {\"match\": \"Team C vs Team D\", \"odds_ft_1\": 2.1, \"odds_ft_x\": 3.1, \"odds_ft_2\": 3.7}\n"
            "]"
        )
        self.matches_input.setMinimumHeight(110)
        technical_layout.addWidget(self.matches_input)

        self.load_rows_button.setText("Загрузить строки из JSON")
        self.load_rows_button.clicked.connect(self._on_load_matches_rows)
        technical_layout.addWidget(self.load_rows_button)

        technical_group.toggled.connect(self.matches_input.setVisible)
        technical_group.toggled.connect(self.load_rows_button.setVisible)
        self.matches_input.setVisible(False)
        self.load_rows_button.setVisible(False)
        splitter.addWidget(technical_group)

        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        select_controls_row = QHBoxLayout()
        select_all_button = QPushButton("Выбрать все")
        select_all_button.clicked.connect(self._on_select_all_matches)
        self._register_delayed_tooltip(
            select_all_button,
            "Выбирает все видимые матчи в таблице.",
        )
        select_controls_row.addWidget(select_all_button)

        deselect_all_button = QPushButton("Снять все")
        deselect_all_button.clicked.connect(self._on_deselect_all_matches)
        self._register_delayed_tooltip(
            deselect_all_button,
            "Снимает выделение со всех матчей.",
        )
        select_controls_row.addWidget(deselect_all_button)
        
        select_controls_row.addStretch()
        layout.addLayout(select_controls_row)

        bottom_row = QHBoxLayout()
        self.predict_selected_button.setText("Прогноз по выбранным")
        self.predict_selected_button.clicked.connect(self._on_predict_selected_matches)
        bottom_row.addWidget(self.predict_selected_button)

        self.predict_all_button.setText("Прогноз по всем")
        self.predict_all_button.clicked.connect(self._on_predict_all_matches)
        bottom_row.addWidget(self.predict_all_button)

        self.model_diagnostic_report_button.setText("Диагностика модели")
        self.model_diagnostic_report_button.clicked.connect(self._on_generate_model_diagnostic_report)
        bottom_row.addWidget(self.model_diagnostic_report_button)

        self.load_to_db_button.setText("Сохранить в БД")
        self.load_to_db_button.clicked.connect(self._on_ingest_todays_matches_to_db)
        bottom_row.addWidget(self.load_to_db_button)

        add_to_toto_button = QPushButton("Добавить выбранные в TOTO")
        add_to_toto_button.clicked.connect(self._on_add_selected_to_toto)
        self._register_delayed_tooltip(
            add_to_toto_button,
            "Передает все выбранные матчи во вкладку TOTO. Можно отредактировать порядок и исключить матчи перед генерацией купонов.",
        )
        bottom_row.addWidget(add_to_toto_button)

        self.clear_matches_button.setText("Очистить список")
        self.clear_matches_button.clicked.connect(self._on_clear_matches)
        bottom_row.addWidget(self.clear_matches_button)

        layout.addLayout(bottom_row)

        self._register_delayed_tooltip(
            self.load_live_button,
            "Загружает матчи из FootyStats API за выбранную дату (поле 'Дата'). Работает для любой даты — прошлой, сегодня, будущей. При включённом автосохранении сразу пишет в SQLite.",
        )
        self._register_delayed_tooltip(
            self.load_history_button,
            "Загружает матчи из FootyStats API за диапазон дат Start→End. Работает для прошлых И будущих дат (например, Start=сегодня, End=завтра). Нажмите Stop для отмены.",
        )
        self._register_delayed_tooltip(
            self.load_sqlite_button,
            "Читает матчи из локальной БД за диапазон Start→End. Не требует API. Полезно для работы с уже сохранёнными данными.",
        )
        self._register_delayed_tooltip(
            self.refresh_visible_button,
            "Обновляет статус видимых строк на основе текущих данных таблицы и модели.",
        )
        self._register_delayed_tooltip(
            self.stop_operation_button,
            "Запрашивает остановку текущей операции (загрузка, исторические загрузки, прогноз). Не гарантирует мгновенную остановку.",
        )
        self._register_delayed_tooltip(
            self.load_to_db_button,
            "Сохраняет текущие строки вручную в SQLite. Нужен API-ключ FootyStats для live-загрузки.",
        )
        self._register_delayed_tooltip(
            self.load_rows_button,
            "Технический режим: загрузка матчей из JSON вручную. Основной режим - FootyStats + SQLite.",
        )
        self._register_delayed_tooltip(
            self.predict_selected_button,
            "Строит прогноз только по отмеченным строкам, если модель загружена и готова.",
        )
        self._register_delayed_tooltip(
            self.predict_all_button,
            "Запускает прогноз по всем видимым строкам. Пропущенные строки получат явную причину в колонке Reason.",
        )
        self._register_delayed_tooltip(
            self.model_diagnostic_report_button,
            "Формирует диагностическую карту по реальным resolved прогнозам модели. Не относится к backtest/smoke/TOTO.",
        )
        self._register_delayed_tooltip(
            self.matches_search_input,
            "Поиск по командам, лиге, дате или части названия матча.",
        )
        self._register_delayed_tooltip(
            self.matches_date_edit,
            "Дата матчей для live-загрузки или фильтрации данных.",
        )
        self._register_delayed_tooltip(
            self.matches_start_date_edit,
            "Начало диапазона дат для массовой historical-загрузки.",
        )
        self._register_delayed_tooltip(
            self.matches_end_date_edit,
            "Конец диапазона дат для массовой historical-загрузки.",
        )
        self._register_delayed_tooltip(
            self.matches_only_completed_checkbox,
            "Если включено, в historical/sqlite режиме оставляются только завершенные матчи.",
        )
        self._register_delayed_tooltip(
            self.matches_only_with_odds_checkbox,
            "Если включено, оставляются только матчи с заполненными коэффициентами 1/X/2.",
        )
        self._register_delayed_tooltip(
            self.matches_auto_save_checkbox,
            "Если включено, после загрузки из API матчи автоматически сохраняются в SQLite.",
        )
        self._register_delayed_tooltip(
            self.calibrate_checkbox,
            "Включает калибровку вероятностей после обучения модели. Может увеличить время обучения.",
        )

        self._reset_matches_view_state()
        return widget

    def _reset_matches_view_state(self) -> None:
        self.current_matches_rows = []
        self._runtime_feature_context_by_row = {}
        self.matches_table.setRowCount(0)
        self.matches_status_label.setText("Матчи: рабочий список пуст, загрузите реальные строки")
        self._set_matches_progress("Ожидание", 0)
        self._update_matches_summary()
        self._refresh_predict_buttons()

    def _resolve_odds_status(self, odds_1: Any, odds_x: Any, odds_2: Any) -> str:
        def _is_valid(value: Any) -> bool:
            try:
                return float(value) > 1.0
            except (TypeError, ValueError):
                return False

        ok_1 = _is_valid(odds_1)
        ok_x = _is_valid(odds_x)
        ok_2 = _is_valid(odds_2)
        if ok_1 and ok_x and ok_2:
            return "odds_loaded"
        if ok_1 or ok_x or ok_2:
            return "odds_partial"
        return "odds_missing"

    def _table_odds_status(self, row_idx: int) -> str:
        odds_1_item = self.matches_table.item(row_idx, self.MATCH_COL_ODDS_1)
        odds_x_item = self.matches_table.item(row_idx, self.MATCH_COL_ODDS_X)
        odds_2_item = self.matches_table.item(row_idx, self.MATCH_COL_ODDS_2)
        odds_1 = odds_1_item.text().strip() if odds_1_item else ""
        odds_x = odds_x_item.text().strip() if odds_x_item else ""
        odds_2 = odds_2_item.text().strip() if odds_2_item else ""
        return self._resolve_odds_status(odds_1, odds_x, odds_2)

    @staticmethod
    def _source_label(code: str) -> str:
        """Translate internal source code to a human-readable label."""
        _map = {
            "model_runtime": "Модель",
            "trusted_precomputed": "Предрасчет",
            "implied_only": "Кф (расчет)",
            "feature_error": "Ошибка признаков",
            "predict_error": "Ошибка прогноза",
            "no_odds": "Нет кф",
            "odds_partial": "Неполные кф",
            "odds_loaded": "",
            "no_odds_fallback": "Модель без кф",
        }
        return _map.get(code, code)

    def _compose_match_reason(self, odds_status: str, reason: str | None = None) -> str:
        _status_labels = {
            "odds_loaded": "Кф загружены",
            "odds_recovered_from_alias": "Кф загружены",
            "odds_partial": "Неполные кф",
            "odds_missing": "Нет кф",
        }
        status_text = _status_labels.get(odds_status, odds_status)
        base_reason = str(reason or "").strip()
        if base_reason:
            return f"{status_text}; {base_reason}"
        return status_text

    def _set_matches_rows(self, rows: list[dict[str, Any]]) -> None:
        self.current_matches_rows = rows
        self.matches_table.blockSignals(True)
        self.matches_table.setRowCount(len(rows))
        for idx, row in enumerate(rows):
            select_box = QCheckBox()
            select_box.setChecked(True)
            select_box.stateChanged.connect(lambda _: self._refresh_predict_buttons())
            self.matches_table.setCellWidget(idx, self.MATCH_COL_SELECTED, select_box)

            match_name = str(row.get("match") or f"Match #{idx + 1}")
            league_name = str(row.get("league") or row.get("league_name") or "-")
            match_date = str(row.get("match_date") or row.get("match_date_iso") or self._selected_matches_date())
            status = str(row.get("status_ui") or row.get("status") or "loaded")
            reason = str(row.get("reason") or "")
            source = str(row.get("source") or "")

            odds_1 = row.get("odds_ft_1", "")
            odds_x = row.get("odds_ft_x", "")
            odds_2 = row.get("odds_ft_2", "")

            odds_status = str(row.get("odds_status") or self._resolve_odds_status(odds_1, odds_x, odds_2)).strip()
            row["odds_status"] = odds_status

            has_odds = odds_status in {"odds_loaded", "odds_recovered_from_alias"}
            db_saved = bool(row.get("db_saved", False))
            predictable = "Да" if has_odds and self._is_model_ready_for_predict() else "Нет"
            reason_display = self._compose_match_reason(odds_status, reason)

            _source_code = source
            if _source_code not in {
                "model_runtime",
                "trusted_precomputed",
                "implied_only",
                "feature_error",
                "predict_error",
                "no_odds",
                "odds_partial",
            }:
                if odds_status == "odds_missing":
                    _source_code = "no_odds"
                elif odds_status == "odds_partial":
                    _source_code = "odds_partial"
                else:
                    _source_code = ""
            source_display = self._source_label(_source_code) if _source_code else ""

            self.matches_table.setItem(idx, self.MATCH_COL_NAME, QTableWidgetItem(match_name))
            self.matches_table.setItem(idx, self.MATCH_COL_LEAGUE, QTableWidgetItem(league_name))
            self.matches_table.setItem(idx, self.MATCH_COL_DATE, QTableWidgetItem(match_date))
            self.matches_table.setItem(idx, self.MATCH_COL_ODDS_1, QTableWidgetItem(str(odds_1)))
            self.matches_table.setItem(idx, self.MATCH_COL_ODDS_X, QTableWidgetItem(str(odds_x)))
            self.matches_table.setItem(idx, self.MATCH_COL_ODDS_2, QTableWidgetItem(str(odds_2)))
            self.matches_table.setItem(idx, self.MATCH_COL_P1, QTableWidgetItem(""))
            self.matches_table.setItem(idx, self.MATCH_COL_PX, QTableWidgetItem(""))
            self.matches_table.setItem(idx, self.MATCH_COL_P2, QTableWidgetItem(""))
            self.matches_table.setItem(idx, self.MATCH_COL_DECISION, QTableWidgetItem(""))
            self.matches_table.setItem(idx, self.MATCH_COL_STATUS, QTableWidgetItem(status))
            self.matches_table.setItem(idx, self.MATCH_COL_REASON, QTableWidgetItem(reason_display))
            self.matches_table.setItem(idx, self.MATCH_COL_SOURCE, QTableWidgetItem(source_display))
            self.matches_table.setItem(idx, self.MATCH_COL_DB_SAVED, QTableWidgetItem("Да" if db_saved else "Нет"))
            self.matches_table.setItem(idx, self.MATCH_COL_PREDICTABLE, QTableWidgetItem(predictable))

        self.matches_table.blockSignals(False)
        self.matches_status_label.setText(f"Матчи: загружено {len(rows)} строк")
        self._set_matches_progress("Готово", 100)
        self._update_matches_summary()
        self._apply_matches_filter()
        self._refresh_predict_buttons()

    def _set_matches_progress(self, stage: str, value: int | None = None) -> None:
        self.matches_progress_stage_label.setText(f"Этап: {stage}")
        if value is None:
            self.matches_progress_bar.setRange(0, 0)
            return
        self.matches_progress_bar.setRange(0, 100)
        self.matches_progress_bar.setValue(max(0, min(value, 100)))

    def _set_training_progress(self, stage: str, value: int | None = None) -> None:
        self.training_progress_stage_label.setText(f"Этап: {stage}")
        if value is None:
            self.training_progress_bar.setRange(0, 0)
            return
        self.training_progress_bar.setRange(0, 100)
        self.training_progress_bar.setValue(max(0, min(value, 100)))

    def _get_model_readiness_snapshot(self) -> dict[str, Any]:
        if hasattr(self.trainer, "get_model_readiness"):
            snapshot = self.trainer.get_model_readiness()
            if isinstance(snapshot, dict):
                return snapshot
        model_file_exists = bool(self.trainer.model_file_exists)
        feature_schema_loaded = bool(self.trainer.feature_schema_loaded)
        models_loaded = bool(self.trainer.models_loaded)
        predictor_trained = bool(self.trainer.predictor_trained)
        return {
            "model_file_exists": model_file_exists,
            "feature_schema_loaded": feature_schema_loaded,
            "models_loaded": models_loaded,
            "predictor_trained": predictor_trained,
            "ready": bool(model_file_exists and feature_schema_loaded and models_loaded and predictor_trained),
        }

    def _refresh_global_model_state_ui(self) -> None:
        self._refresh_predict_buttons()
        self._refresh_training_buttons()
        self._refresh_analysis_summary()
        self._update_matches_summary()

    def _refresh_analysis_summary(self) -> None:
        if self.analysis_summary_box is None:
            return

        total_matches = 0
        completed_matches = 0
        matches_with_odds = 0
        training_rows = 0
        unique_completed_matches = 0
        final_training_rows = int(self.last_train_dataset_summary.get("final_train_rows", 0) or 0)
        dropped_rows = int(self.last_train_dataset_summary.get("dropped_rows", 0) or 0)
        duplicate_rows_removed = int(self.last_train_dataset_summary.get("duplicate_rows_removed", 0) or 0)
        model_readiness = self._get_model_readiness_snapshot()
        model_status = "готова" if bool(model_readiness.get("ready", False)) else "не обучена"

        weak_features: list[str] = []
        weak_details: list[str] = []

        try:
            conn = self.ingestion_loader.db.conn
            total_matches = int(conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0])
            completed_matches = int(
                conn.execute(
                    "SELECT COUNT(*) FROM matches WHERE status IN ('completed', 'complete', 'finished', 'full-time', 'ft')"
                ).fetchone()[0]
            )
            unique_completed_matches = int(
                conn.execute(
                    "SELECT COUNT(DISTINCT match_id) FROM matches WHERE status IN ('completed', 'complete', 'finished', 'full-time', 'ft')"
                ).fetchone()[0]
            )
            matches_with_odds = int(
                conn.execute(
                    "SELECT COUNT(*) FROM matches WHERE odds_ft_1 IS NOT NULL AND odds_ft_x IS NOT NULL AND odds_ft_2 IS NOT NULL"
                ).fetchone()[0]
            )

            dataset = self.ingestion_loader.db.build_training_dataset_from_db()
            training_rows = len(dataset)
            if final_training_rows <= 0:
                final_training_rows = int(training_rows)

            if not self.last_train_dataset_summary:
                debug = getattr(self.ingestion_loader.db, "last_training_dataset_debug", {})
                if isinstance(debug, dict):
                    duplicate_rows_removed = int(debug.get("duplicate_rows_removed", 0) or 0)

        except Exception as exc:
            logger.warning(f"ANALYSIS refresh failed: {exc}")

        weak_raw = self.last_train_weak_features
        for item in weak_raw[:5]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("feature", "?"))
            weak_features.append(name)
            fill = float(item.get("fill_rate", 0.0) or 0.0) * 100
            zero = float(item.get("zero_rate", 0.0) or 0.0) * 100
            fallback = float(item.get("fallback_rate", 0.0) or 0.0) * 100
            weak_details.append(f"- {name}: fill={fill:.1f}%, zero={zero:.1f}%, fallback={fallback:.1f}%")

        if not weak_details:
            weak_details.append("- Нет детализированных данных")

        weak_details_joined = "\n".join(weak_details)
        weak_text = ", ".join(weak_features) if weak_features else "Нет данных"
        smoke_display = self.last_smoke_status
        if smoke_display == "NOT_RUN" and bool(model_readiness.get("ready", False)):
            smoke_display = "NOT_RUN (в этой сессии; runtime model ready)"
        self.analysis_summary_box.setPlainText(
            "Текущая сводка ANALYSIS:\n\n"
            f"- Всего матчей в БД: {total_matches}\n"
            f"- Завершенных матчей: {completed_matches}\n"
            f"- Уникальных completed матчей: {unique_completed_matches}\n"
            f"- Матчей с коэффициентами: {matches_with_odds}\n"
            f"- Training rows (из SQLite): {training_rows}\n"
            f"- Final training rows: {final_training_rows}\n"
            f"- Dropped rows: {dropped_rows}\n"
            f"- Duplicate rows removed: {duplicate_rows_removed}\n"
            f"- Статус модели: {model_status}\n"
            f"- Smoke result: {smoke_display}\n\n"
            "Качество признаков:\n"
            f"- Weak/default-heavy features: {weak_text}\n\n"
            "Детализация weak features:\n"
            f"{weak_details_joined}\n"
        )

    def _update_matches_summary(self) -> None:
        total_rows = self.matches_table.rowCount()
        with_odds = 0
        predictable_rows = 0
        for row_idx in range(total_rows):
            o1 = self.matches_table.item(row_idx, self.MATCH_COL_ODDS_1)
            ox = self.matches_table.item(row_idx, self.MATCH_COL_ODDS_X)
            o2 = self.matches_table.item(row_idx, self.MATCH_COL_ODDS_2)
            has_odds = bool(o1 and ox and o2 and o1.text().strip() and ox.text().strip() and o2.text().strip())
            if has_odds:
                with_odds += 1

            pred_item = self.matches_table.item(row_idx, self.MATCH_COL_PREDICTABLE)
            if pred_item and pred_item.text().strip().lower() in ("да", "yes", "true"):
                predictable_rows += 1

        model_status = "готова" if self._is_model_ready_for_predict() else "не обучена"
        selected_date = self._selected_matches_date()
        db_count = self._db_matches_count()
        self.matches_summary_label.setText(
            f"Дата: {selected_date} | Загружено в UI: {total_rows} | В БД: {db_count} | "
            f"С odds: {with_odds} | Готово к прогнозу: {predictable_rows} | Модель: {model_status}"
        )
        self._refresh_model_diagnostic_button_state()

    def _model_diagnostic_thresholds(self) -> dict[str, int]:
        return {
            "min_resolved": 300,
            "normal_reliability": 500,
            "high_reliability": 1000,
        }

    def _resolved_prediction_count(self) -> int:
        try:
            row = self.ingestion_loader.db.conn.execute(
                "SELECT COUNT(*) FROM model_prediction_history WHERE actual_outcome IS NOT NULL"
            ).fetchone()
            if row is None:
                return 0
            return int(row[0] or 0)
        except Exception:
            return 0

    def _refresh_model_diagnostic_button_state(self) -> None:
        thresholds = self._model_diagnostic_thresholds()
        resolved = self._resolved_prediction_count()
        min_required = int(thresholds["min_resolved"])
        enabled = resolved >= min_required
        self.model_diagnostic_report_button.setEnabled(enabled)
        if enabled:
            self.model_diagnostic_report_button.setToolTip(
                f"Доступно: resolved={resolved}. Нормальная надежность с {thresholds['normal_reliability']}, высокая с {thresholds['high_reliability']}."
            )
        else:
            self.model_diagnostic_report_button.setToolTip(
                "Недостаточно завершенных прогнозов для надежного отчета. Нужно минимум 300."
            )

    def _on_generate_model_diagnostic_report(self) -> None:
        thresholds = self._model_diagnostic_thresholds()
        min_required = int(thresholds["min_resolved"])

        try:
            self.ingestion_loader.db.resolve_model_prediction_history(limit=20000)
        except Exception:
            logger.exception("Resolve prediction history before diagnostic report failed")

        resolved = self._resolved_prediction_count()
        if resolved < min_required:
            self.matches_status_label.setText(
                "Недостаточно завершенных прогнозов для надежного отчета. Нужно минимум 300."
            )
            self._show_warning(
                "Отчет пока недоступен",
                "Недостаточно завершенных прогнозов для надежного отчета. Нужно минимум 300.",
            )
            self._refresh_model_diagnostic_button_state()
            return

        try:
            result = self.ingestion_loader.db.write_model_diagnostic_report(
                report_dir="reports",
                min_resolved=min_required,
                league_min_count=20,
            )
        except Exception as exc:
            logger.exception("Model diagnostic report generation failed")
            self.matches_status_label.setText(f"Ошибка генерации диагностического отчета: {exc}")
            self._show_error("Ошибка генерации отчета", str(exc))
            return

        if not bool(result.get("generated", False)):
            reason_text = str(result.get("reason") or "Отчет недоступен")
            self.matches_status_label.setText(reason_text)
            self._show_warning("Отчет пока недоступен", reason_text)
            self._refresh_model_diagnostic_button_state()
            return

        report = result.get("report") if isinstance(result.get("report"), dict) else {}
        metadata = report.get("metadata") if isinstance(report.get("metadata"), dict) else {}
        resolved_count = int(metadata.get("resolved_predictions") or resolved)
        reliability = str(metadata.get("reliability_level") or "low")

        json_path = str(result.get("json_path") or "reports/model_diagnostic_report.json")
        md_path = str(result.get("md_path") or "reports/model_diagnostic_report.md")
        self.matches_status_label.setText(
            f"Диагностический отчет модели создан: resolved={resolved_count}, reliability={reliability}. "
            f"JSON={json_path}; MD={md_path}"
        )
        self._refresh_model_diagnostic_button_state()

    def _on_matches_table_item_changed(self, _item: QTableWidgetItem) -> None:
        self._update_matches_summary()

    def _on_stop_operation(self) -> None:
        self._stop_requested = True
        if self._predict_worker is not None:
            self._predict_worker.cancel()
        self.matches_status_label.setText("Матчи: запрошена остановка текущей операции")

    def _on_clear_matches(self) -> None:
        self.current_matches_rows = []
        self._runtime_feature_context_by_row = {}
        self.matches_table.setRowCount(0)
        self.matches_status_label.setText("Матчи: список очищен")
        self._set_matches_progress("Ожидание", 0)
        self._update_matches_summary()

    @staticmethod
    def _status_is_completed(status: Any) -> bool:
        value = str(status or "").strip().lower()
        return value in {
            "complete",
            "completed",
            "finished",
            "full-time",
            "full_time",
            "ft",
            "aet",
            "after extra time",
        }

    def _normalize_ui_rows_from_api(
        self,
        raw_rows: list[dict[str, Any]],
        selected_date: str,
        source: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        prepared_rows: list[dict[str, Any]] = []
        db_rows: list[dict[str, Any]] = []

        for raw in raw_rows:
            if not isinstance(raw, dict):
                continue

            normalized = normalize_match(raw)
            match_name = f"{normalized.get('home_team_name') or 'Home'} vs {normalized.get('away_team_name') or 'Away'}"
            league_name = str(normalized.get("league_name") or normalized.get("competition") or "").strip()
            if not league_name:
                league_name = "-"

            odds_1 = normalized.get("odds_ft_1")
            odds_x = normalized.get("odds_ft_x")
            odds_2 = normalized.get("odds_ft_2")
            odds_status = str(normalized.get("odds_status") or self._resolve_odds_status(odds_1, odds_x, odds_2))
            has_odds = odds_status in {"odds_loaded", "odds_recovered_from_alias"}

            status_raw = normalized.get("status")
            status_ui = str(status_raw or "unknown")
            reason = ""
            if self.matches_only_completed_checkbox.isChecked() and not self._status_is_completed(status_raw):
                continue
            if self.matches_only_with_odds_checkbox.isChecked() and not has_odds:
                continue
            if odds_status == "odds_missing":
                reason = "Нет валидных коэффициентов"
            elif odds_status == "odds_partial":
                reason = "Коэффициенты частично заполнены"

            prepared_rows.append(
                {
                    "match_id": normalized.get("match_id"),
                    "season_id": normalized.get("season_id"),
                    "home_team_id": normalized.get("home_team_id"),
                    "away_team_id": normalized.get("away_team_id"),
                    "match": match_name,
                    "league": league_name,
                    "match_date": normalized.get("match_date_iso") or selected_date,
                    "odds_ft_1": "" if odds_1 is None else odds_1,
                    "odds_ft_x": "" if odds_x is None else odds_x,
                    "odds_ft_2": "" if odds_2 is None else odds_2,
                    "home_ppg": normalized.get("home_ppg"),
                    "away_ppg": normalized.get("away_ppg"),
                    "pre_match_home_ppg": normalized.get("pre_match_home_ppg"),
                    "pre_match_away_ppg": normalized.get("pre_match_away_ppg"),
                    "avg_potential": normalized.get("avg_potential"),
                    "raw": normalized.get("raw"),
                    "odds_status": odds_status,
                    "status_ui": status_ui,
                    "reason": reason,
                    "source": "no_odds" if odds_status == "odds_missing" else ("odds_partial" if odds_status == "odds_partial" else ""),
                    "data_source": source,
                    "db_saved": False,
                }
            )

            status_for_db = "completed" if self._status_is_completed(status_raw) else status_raw
            normalized["status"] = status_for_db
            db_rows.append(normalized)

        return prepared_rows, db_rows

    def _save_matches_to_db(self, db_rows: list[dict[str, Any]]) -> dict[str, int]:
        if not db_rows:
            return {
                "loaded": 0,
                "added": 0,
                "updated": 0,
                "duplicates": 0,
                "db_total": self._db_matches_count(),
            }

        before_count = self._db_matches_count()
        self.ingestion_loader.db.upsert_matches(db_rows)
        after_count = self._db_matches_count()
        loaded_count = len(db_rows)
        added_count = max(after_count - before_count, 0)
        updated_count = max(loaded_count - added_count, 0)
        self._refresh_training_buttons()

        return {
            "loaded": loaded_count,
            "added": added_count,
            "updated": updated_count,
            "duplicates": 0,
            "db_total": after_count,
        }

    def _hydrate_stats_for_loaded_seasons(self, db_rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Load and persist team/league priors for season_ids present in saved match rows."""
        season_ids = sorted(
            {
                int(row.get("season_id"))
                for row in db_rows
                if isinstance(row, dict) and row.get("season_id") is not None
            }
        )
        if not season_ids:
            return {
                "seasons_targeted": 0,
                "teams_loaded": 0,
                "league_seasons_loaded": 0,
                "errors": [],
            }

        teams_loaded_total = 0
        league_loaded_total = 0
        errors: list[str] = []

        for idx, season_id in enumerate(season_ids, start=1):
            self._set_matches_progress(f"Historical: загрузка team/league stats {idx}/{len(season_ids)} (season {season_id})", None)
            QApplication.processEvents()
            try:
                teams_loaded = int(self.ingestion_loader.load_league_teams(season_id=season_id))
                teams_loaded_total += teams_loaded
            except Exception as exc:
                teams_loaded = 0
                errors.append(f"season {season_id}: load_league_teams failed ({exc})")
                logger.warning("Historical stats hydrate: load_league_teams failed season_id=%s error=%s", season_id, exc)

            try:
                league_loaded = bool(self.ingestion_loader.load_league_season(season_id=season_id))
                league_loaded_total += int(league_loaded)
            except Exception as exc:
                league_loaded = False
                errors.append(f"season {season_id}: load_league_season failed ({exc})")
                logger.warning("Historical stats hydrate: load_league_season failed season_id=%s error=%s", season_id, exc)

            logger.info(
                "Historical stats hydrate season_id=%s teams_loaded=%s league_loaded=%s",
                season_id,
                teams_loaded,
                league_loaded,
            )

        return {
            "seasons_targeted": len(season_ids),
            "teams_loaded": teams_loaded_total,
            "league_seasons_loaded": league_loaded_total,
            "errors": errors,
        }

    def _on_load_live_matches(self) -> None:
        if not settings.footystats_api_key:
            self.matches_status_label.setText("Матчи: API ключ FootyStats не настроен")
            self._show_warning("Live загрузка недоступна", "Не найден FOOTYSTATS_API_KEY. Проверьте .env.")
            return

        selected_date = self._selected_matches_date()
        self._set_matches_progress(f"Загрузка матчей за {selected_date}", None)

        try:
            raw_rows = self.footy_api.get_all_todays_matches(
                date=selected_date,
                timezone=settings.app_timezone,
                force_refresh=True,
            )
        except Exception as exc:
            logger.exception("Live match load failed")
            self.matches_status_label.setText(f"Матчи: ошибка загрузки live ({exc})")
            self._show_error("Ошибка live загрузки", str(exc))
            self._set_matches_progress("Ошибка", 0)
            return

        if not raw_rows:
            self.matches_status_label.setText(f"Матчи: на дату {selected_date} ничего не найдено")
            self._append_output({"event": "live_load", "date": selected_date, "rows": 0})
            self._set_matches_progress("Нет данных", 100)
            return

        prepared_rows, db_rows = self._normalize_ui_rows_from_api(
            raw_rows=[row for row in raw_rows if isinstance(row, dict)],
            selected_date=selected_date,
            source="api-live",
        )

        rows_with_odds = sum(
            1 for row in prepared_rows if row.get("odds_ft_1") != "" and row.get("odds_ft_x") != "" and row.get("odds_ft_2") != ""
        )

        if not prepared_rows:
            self.matches_status_label.setText("Матчи: в ответе API нет пригодных строк")
            self._append_output({"event": "live_load", "date": selected_date, "rows": 0, "usable": 0})
            self._set_matches_progress("Нет пригодных строк", 100)
            return

        db_summary: dict[str, int] = {
            "loaded": 0,
            "added": 0,
            "updated": 0,
            "duplicates": 0,
            "db_total": self._db_matches_count(),
        }
        if self.matches_auto_save_checkbox.isChecked():
            self._set_matches_progress("Сохранение строк в БД", None)
            db_summary = self._save_matches_to_db(db_rows)
            for row in prepared_rows:
                row["db_saved"] = True

        self._set_matches_rows(prepared_rows)
        self.matches_status_label.setText(
            f"Загружено {len(prepared_rows)} матчей за {selected_date}. "
            f"В БД добавлено: {db_summary['added']}, обновлено: {db_summary['updated']}, "
            f"дубликатов: {db_summary['duplicates']}, без коэффициентов: {len(prepared_rows) - rows_with_odds}."
            + ("" if self.matches_auto_save_checkbox.isChecked() else " Автосохранение выключено: строки только в UI.")
        )
        self._append_output(
            {
                "event": "live_load",
                "date": selected_date,
                "rows": len(prepared_rows),
                "ui_rows": len(prepared_rows),
                "with_odds": rows_with_odds,
                "missing_odds": len(prepared_rows) - rows_with_odds,
                "db_added": db_summary["added"],
                "db_updated": db_summary["updated"],
                "db_total": db_summary["db_total"],
                "auto_saved": self.matches_auto_save_checkbox.isChecked(),
            }
        )
        self._set_matches_progress("Завершено", 100)

    def _on_load_historical_matches(self) -> None:
        if not settings.footystats_api_key:
            self._show_warning("Historical загрузка недоступна", "Не найден FOOTYSTATS_API_KEY. Проверьте .env.")
            return

        start_qdate = self.matches_start_date_edit.date()
        end_qdate = self.matches_end_date_edit.date()
        if start_qdate > end_qdate:
            self._show_warning("Некорректный диапазон", "Start date не может быть больше End date.")
            return

        self._stop_requested = False
        start_date = datetime.strptime(start_qdate.toString("yyyy-MM-dd"), "%Y-%m-%d").date()
        end_date = datetime.strptime(end_qdate.toString("yyyy-MM-dd"), "%Y-%m-%d").date()
        total_days = (end_date - start_date).days + 1

        all_prepared_rows: list[dict[str, Any]] = []
        all_db_rows: list[dict[str, Any]] = []
        processed_days = 0

        current = start_date
        while current <= end_date:
            if self._stop_requested:
                self.matches_status_label.setText("Историческая загрузка остановлена пользователем")
                break

            day_str = current.isoformat()
            processed_days += 1
            progress = int((processed_days / max(total_days, 1)) * 100)
            self._set_matches_progress(f"Historical: {processed_days}/{total_days} ({day_str})", progress)
            QApplication.processEvents()

            try:
                raw_rows = self.footy_api.get_all_todays_matches(date=day_str, timezone=settings.app_timezone)
            except Exception as exc:
                logger.exception("Historical day load failed for %s", day_str)
                self._append_output({"event": "historical_day_error", "date": day_str, "error": str(exc)})
                current += timedelta(days=1)
                continue

            prepared_rows, db_rows = self._normalize_ui_rows_from_api(
                raw_rows=[row for row in raw_rows if isinstance(row, dict)],
                selected_date=day_str,
                source="api-historical",
            )
            all_prepared_rows.extend(prepared_rows)
            all_db_rows.extend(db_rows)

            current += timedelta(days=1)

        if not all_prepared_rows:
            self.matches_status_label.setText("Historical: не найдено пригодных строк за выбранный диапазон")
            self._set_matches_progress("Нет данных", 100)
            self._append_output(
                {
                    "event": "historical_load",
                    "status": "empty",
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days_processed": processed_days,
                    "rows_found": 0,
                }
            )
            return

        db_summary = {"loaded": 0, "added": 0, "updated": 0, "duplicates": 0, "db_total": self._db_matches_count()}
        stats_summary = {"seasons_targeted": 0, "teams_loaded": 0, "league_seasons_loaded": 0, "errors": []}
        if self.matches_auto_save_checkbox.isChecked():
            self._set_matches_progress("Historical: сохранение в БД", None)
            db_summary = self._save_matches_to_db(all_db_rows)
            stats_summary = self._hydrate_stats_for_loaded_seasons(all_db_rows)
            for row in all_prepared_rows:
                row["db_saved"] = True

        self._set_matches_rows(all_prepared_rows)
        completed_rows = sum(1 for row in all_prepared_rows if self._status_is_completed(row.get("status_ui")))
        rows_with_odds = sum(
            1
            for row in all_prepared_rows
            if row.get("odds_ft_1") != "" and row.get("odds_ft_x") != "" and row.get("odds_ft_2") != ""
        )
        self.matches_status_label.setText(
            f"Historical {start_date.isoformat()} -> {end_date.isoformat()}: дней={processed_days}, найдено={len(all_prepared_rows)}, "
            f"completed={completed_rows}, с odds={rows_with_odds}, добавлено в БД={db_summary['added']}, обновлено={db_summary['updated']}, "
            f"team_stats={stats_summary['teams_loaded']}, league_stats={stats_summary['league_seasons_loaded']}"
        )
        self._append_output(
            {
                "event": "historical_load",
                "status": "ok",
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days_processed": processed_days,
                "rows_found": len(all_prepared_rows),
                "completed_rows": completed_rows,
                "with_odds": rows_with_odds,
                "db_added": db_summary["added"],
                "db_updated": db_summary["updated"],
                "db_total": db_summary["db_total"],
                "stats_seasons_targeted": stats_summary["seasons_targeted"],
                "team_stats_rows_loaded": stats_summary["teams_loaded"],
                "league_stats_rows_loaded": stats_summary["league_seasons_loaded"],
                "stats_errors": stats_summary["errors"],
            }
        )
        self._set_matches_progress("Historical завершено", 100)

    def _on_load_matches_from_sqlite(self) -> None:
        self._set_matches_progress("Чтение из SQLite", None)
        rows = self._load_sqlite_rows()
        if not rows:
            self.matches_status_label.setText("SQLite: по выбранным фильтрам строки не найдены")
            self._set_matches_progress("Нет строк", 100)
            return

        self._set_matches_rows(rows)
        self.matches_status_label.setText(f"SQLite: загружено {len(rows)} строк по выбранным фильтрам")
        self._append_output({"event": "sqlite_load", "rows": len(rows)})
        self._set_matches_progress("SQLite загрузка завершена", 100)

    def _load_sqlite_rows(self) -> list[dict[str, Any]]:
        start_iso = self.matches_start_date_edit.date().toString("yyyy-MM-dd")
        end_iso = self.matches_end_date_edit.date().toString("yyyy-MM-dd")
        query = """
            SELECT
                m.match_id, m.match_date_iso, m.status,
                m.home_team_name, m.away_team_name,
                m.odds_ft_1, m.odds_ft_x, m.odds_ft_2,
                m.season_id, m.home_team_id, m.away_team_id,
                m.home_ppg, m.away_ppg, m.pre_match_home_ppg, m.pre_match_away_ppg,
                m.avg_potential, m.raw_json,
                COALESCE(l.league_name, json_extract(m.raw_json, '$.league_name'), json_extract(m.raw_json, '$.competition'), json_extract(m.raw_json, '$.league'), '') as league_name
            FROM matches m
            LEFT JOIN leagues l ON m.season_id = l.season_id
            WHERE (m.match_date_iso IS NULL OR DATE(m.match_date_iso) BETWEEN DATE(?) AND DATE(?))
            ORDER BY m.date_unix DESC
            LIMIT 2000
        """
        raw_rows = self.ingestion_loader.db.conn.execute(query, (start_iso, end_iso)).fetchall()

        prepared: list[dict[str, Any]] = []
        for row_raw in raw_rows:
            row = dict(row_raw)
            odds_1 = row.get("odds_ft_1")
            odds_x = row.get("odds_ft_x")
            odds_2 = row.get("odds_ft_2")
            odds_status = self._resolve_odds_status(odds_1, odds_x, odds_2)
            has_odds = odds_status in {"odds_loaded", "odds_recovered_from_alias"}

            if self.matches_only_completed_checkbox.isChecked() and not self._status_is_completed(row.get("status")):
                continue
            if self.matches_only_with_odds_checkbox.isChecked() and not has_odds:
                continue

            league = str(row.get("league_name") or "").strip()
            if not league:
                league = "-"

            raw_payload = {}
            raw_json = row.get("raw_json")
            if isinstance(raw_json, str) and raw_json.strip():
                try:
                    parsed = json.loads(raw_json)
                    if isinstance(parsed, dict):
                        raw_payload = parsed
                except Exception:
                    raw_payload = {}

            prepared.append(
                {
                    "match_id": row.get("match_id"),
                    "season_id": row.get("season_id"),
                    "home_team_id": row.get("home_team_id"),
                    "away_team_id": row.get("away_team_id"),
                    "match": f"{row.get('home_team_name') or 'Home'} vs {row.get('away_team_name') or 'Away'}",
                    "league": league,
                    "match_date": row.get("match_date_iso") or "",
                    "odds_ft_1": "" if odds_1 is None else odds_1,
                    "odds_ft_x": "" if odds_x is None else odds_x,
                    "odds_ft_2": "" if odds_2 is None else odds_2,
                    "home_ppg": row.get("home_ppg"),
                    "away_ppg": row.get("away_ppg"),
                    "pre_match_home_ppg": row.get("pre_match_home_ppg"),
                    "pre_match_away_ppg": row.get("pre_match_away_ppg"),
                    "avg_potential": row.get("avg_potential"),
                    "raw": raw_payload,
                    "odds_status": odds_status,
                    "status_ui": str(row.get("status") or "unknown"),
                    "reason": "" if has_odds else ("Коэффициенты частично заполнены" if odds_status == "odds_partial" else "Нет валидных коэффициентов"),
                    "source": "no_odds" if odds_status == "odds_missing" else ("odds_partial" if odds_status == "odds_partial" else ""),
                    "data_source": "sqlite",
                    "db_saved": True,
                }
            )
        return prepared

    def _on_refresh_visible_matches(self) -> None:
        total_rows = self.matches_table.rowCount()
        for row_idx in range(total_rows):
            odds_status = self._table_odds_status(row_idx)
            odds_ok = odds_status in {"odds_loaded", "odds_recovered_from_alias"}

            predictable_item = QTableWidgetItem("Да" if odds_ok and self._is_model_ready_for_predict() else "Нет")
            self.matches_table.setItem(row_idx, self.MATCH_COL_PREDICTABLE, predictable_item)

            reason_item = self.matches_table.item(row_idx, self.MATCH_COL_REASON)
            if not odds_ok:
                base_reason = "Коэффициенты частично заполнены" if odds_status == "odds_partial" else "Нет валидных коэффициентов"
                self.matches_table.setItem(
                    row_idx,
                    self.MATCH_COL_REASON,
                    QTableWidgetItem(self._compose_match_reason(odds_status, base_reason)),
                )
                self.matches_table.setItem(
                    row_idx,
                    self.MATCH_COL_SOURCE,
                    QTableWidgetItem(self._source_label("odds_partial") if odds_status == "odds_partial" else self._source_label("no_odds")),
                )
            elif reason_item is None:
                self.matches_table.setItem(row_idx, self.MATCH_COL_REASON, QTableWidgetItem(self._compose_match_reason(odds_status)))

        self.matches_status_label.setText("Матчи: видимые строки обновлены")
        self._update_matches_summary()

    def _db_matches_count(self) -> int:
        try:
            row = self.ingestion_loader.db.conn.execute("SELECT COUNT(*) AS cnt FROM matches").fetchone()
            if row is None:
                return 0
            return int(row[0])
        except Exception:
            return 0

    def _on_ingest_todays_matches_to_db(self) -> None:
        if not settings.footystats_api_key:
            self.matches_status_label.setText("Матчи: загрузка в БД пропущена, нет API ключа")
            self._show_warning("Загрузка в БД недоступна", "Не найден FOOTYSTATS_API_KEY. Проверьте .env.")
            return

        selected_date = self._selected_matches_date()
        self._set_matches_progress(f"Сохранение в БД за {selected_date}", None)

        before_count = self._db_matches_count()
        try:
            loaded_count = int(self.ingestion_loader.load_todays_matches(date=selected_date, timezone=settings.app_timezone))
        except Exception as exc:
            logger.exception("Ingest today's matches to DB failed")
            self.matches_status_label.setText(f"Матчи: ошибка записи в БД ({exc})")
            self._show_error("Ошибка загрузки в БД", str(exc))
            self._set_matches_progress("Ошибка", 0)
            return

        after_count = self._db_matches_count()
        added_count = max(after_count - before_count, 0)
        updated_estimate = max(loaded_count - added_count, 0)

        if loaded_count <= 0:
            self.matches_status_label.setText(f"Матчи: на дату {selected_date} API вернул 0 строк")
            self._append_output(
                {
                    "event": "ingest_todays_matches",
                    "date": selected_date,
                    "loaded": 0,
                    "added": 0,
                    "updated_estimate": 0,
                    "db_total_matches": after_count,
                    "status": "empty",
                }
            )
            self._set_matches_progress("Нет данных", 100)
            return

        self.matches_status_label.setText(
            f"Сохранение в БД завершено: дата {selected_date}, загружено={loaded_count}, "
            f"добавлено={added_count}, обновлено={updated_estimate}, всего в БД={after_count}"
        )
        self._append_output(
            {
                "event": "ingest_todays_matches",
                "date": selected_date,
                "loaded": loaded_count,
                "added": added_count,
                "updated_estimate": updated_estimate,
                "db_total_matches": after_count,
                "status": "ok",
            }
        )
        self._set_matches_progress("Сохранение завершено", 100)
        self._update_matches_summary()
        self._refresh_training_buttons()

    def _on_load_matches_rows(self) -> None:
        if not self.matches_input.toPlainText().strip():
            self.matches_status_label.setText("Матчи: поле JSON пустое")
            self._show_warning(
                "Пустой JSON",
                "Поле JSON пустое. Для ручного режима вставьте JSON или используйте загрузку из FootyStats.",
            )
            return
        try:
            payload = self._parse_json(self.matches_input.toPlainText())
            if not isinstance(payload, list):
                raise ValueError("Matches JSON must be a list.")
            if not payload:
                raise ValueError("Matches JSON list is empty.")
            if not all(isinstance(item, dict) for item in payload):
                raise ValueError("Each match row must be a JSON object.")

            prepared_rows: list[dict[str, Any]] = []
            for item in payload:
                prepared_rows.append(
                    {
                        "match": item.get("match") or item.get("name") or "JSON match",
                        "league": item.get("league") or item.get("league_name") or "JSON",
                        "match_date": item.get("match_date") or item.get("match_date_iso") or self._selected_matches_date(),
                        "odds_ft_1": item.get("odds_ft_1", ""),
                        "odds_ft_x": item.get("odds_ft_x", ""),
                        "odds_ft_2": item.get("odds_ft_2", ""),
                        "status_ui": item.get("status_ui") or "manual-json",
                        "reason": item.get("reason") or "",
                        "source": "json",
                        "db_saved": bool(item.get("db_saved", False)),
                    }
                )

            self._set_matches_rows(prepared_rows)
            self.matches_status_label.setText(f"JSON режим: загружено {len(prepared_rows)} строк")
        except Exception as exc:
            logger.exception("Load matches rows failed")
            self.matches_status_label.setText(f"Матчи: ошибка чтения JSON ({exc})")
            self._show_error("Ошибка загрузки JSON", str(exc))

    def _on_select_all_matches(self) -> None:
        """Select all visible matches."""
        for row_idx in range(self.matches_table.rowCount()):
            if not self.matches_table.isRowHidden(row_idx):
                widget = self.matches_table.cellWidget(row_idx, self.MATCH_COL_SELECTED)
                if isinstance(widget, QCheckBox):
                    widget.setChecked(True)
        self._refresh_predict_buttons()

    def _on_deselect_all_matches(self) -> None:
        """Deselect all matches."""
        for row_idx in range(self.matches_table.rowCount()):
            widget = self.matches_table.cellWidget(row_idx, self.MATCH_COL_SELECTED)
            if isinstance(widget, QCheckBox):
                widget.setChecked(False)
        self._refresh_predict_buttons()

    def _current_toto_input_mode(self) -> str:
        # Main product flow is MATCHES -> TOTO; draw-mode is hidden from primary UI.
        return "manual_matches"

    def _is_toto_manual_mode(self) -> bool:
        return self._current_toto_input_mode() == "manual_matches"

    def _on_change_toto_input_mode(self) -> None:
        self._refresh_toto_mode_ui()

    def _update_toto_history_status_label(self) -> None:
        draws_count = 0
        events_count = 0
        history_loaded = isinstance(self.global_toto_history, dict)
        if history_loaded:
            draws_count = int(self.global_toto_history.get("history_draws_loaded_count", 0) or 0)
            events_count = int(self.global_toto_history.get("history_events_loaded_count", 0) or 0)

        if history_loaded and (draws_count > 0 or events_count > 0):
            self.toto_history_status_label.setText(
                f"История Baltbet: loaded | recent draws in UI cache: {draws_count} | total historical events for strategy: {events_count}"
            )
            self.toto_history_status_label.setStyleSheet("color: #8ED1A1;")
            self.toto_history_refresh_button.setText("Обновить историю Baltbet")
            return

        self.toto_history_status_label.setText(
            "История Baltbet: not loaded | recent draws in UI cache: 0 | total historical events for strategy: 0"
        )
        self.toto_history_status_label.setStyleSheet("color: #AFC0CF;")
        self.toto_history_refresh_button.setText("Загрузить историю Baltbet")

    def _update_toto_generation_intent_label(self) -> None:
        mode = self.toto_mode_selector.currentText().strip() or "16"
        expected_lines = 16 if mode == "16" else 32
        insurance_enabled = self.toto_insurance_checkbox.isChecked()
        if insurance_enabled:
            self.toto_generation_intent_label.setText(
                f"Expected result: mode {mode} ({expected_lines} lines) -> base + insurance coupons"
            )
        else:
            self.toto_generation_intent_label.setText(
                f"Expected result: mode {mode} ({expected_lines} lines) -> only base coupons"
            )

    def _refresh_toto_mode_ui(self) -> None:
        self.toto_draw_tools_group.setVisible(False)
        self._refresh_toto_draws_availability()
        self._update_toto_generation_intent_label()
        self._update_toto_history_status_label()
        self._update_toto_copy_actions_state()

    def _refresh_toto_draws_availability(self) -> None:
        if not hasattr(self, "toto_load_draws_button"):
            return
        self.toto_load_draws_button.setEnabled(False)
        self.toto_draws_combo.setEnabled(False)
        self.toto_status_label.setText(
            f"TOTO status: manual mode | matches ready={len(self.current_toto_draw_matches)}"
        )

    def _on_clear_toto_matches(self) -> None:
        self.current_toto_draw_matches = []
        self.current_coupons = []
        self.current_toto_summary = None
        self.current_coupon_entries = []
        self.current_insured_coupon_indices = set()
        self.toto_coupon_display_line_map = {}
        self.toto_matches_table.setRowCount(0)
        self.toto_coupons_table.setRowCount(0)
        self.toto_coupon_lines_box.clear()
        self.toto_base_coupon_lines_box.clear()
        self.toto_insurance_coupon_lines_box.clear()
        self.toto_unknown_coupon_lines_box.clear()
        self.toto_layer_view_box.clear()
        self.toto_selected_coupon_detail_box.clear()
        self.toto_summary_box.clear()
        self.toto_summary_verbose_box.clear()
        self._clear_toto_summary_raw_box()
        self.toto_draws_combo.setEnabled(False)
        self.toto_status_label.setText("TOTO status: manual mode, список матчей очищен")
        self._update_toto_copy_actions_state()

    def _set_toto_summary_raw_text(self, text: str) -> None:
        box = getattr(self, "toto_summary_raw_box", None)
        if not isinstance(box, QTextEdit):
            return
        try:
            box.setPlainText(text)
        except RuntimeError:
            logger.warning("toto_summary_raw_box is deleted; skip raw summary write")

    def _clear_toto_summary_raw_box(self) -> None:
        box = getattr(self, "toto_summary_raw_box", None)
        if not isinstance(box, QTextEdit):
            return
        try:
            box.clear()
        except RuntimeError:
            logger.warning("toto_summary_raw_box is deleted; skip raw summary clear")

    def _on_predict_selected_matches(self) -> None:
        if not self._ensure_model_ready_for_predict():
            return
        if self._predict_inflight:
            self._show_warning("Прогноз уже выполняется", "Дождитесь завершения текущего batch-прогноза.")
            return
        selected_rows = [idx for idx in range(self.matches_table.rowCount()) if self._is_row_selected(idx)]
        if not selected_rows:
            self.matches_status_label.setText("Прогноз не запущен: не выбрано ни одной строки")
            self._show_warning("Нет выбранных строк", "Отметьте хотя бы одну строку для прогноза.")
            self._refresh_predict_buttons()
            return
        self._predict_matches_rows(selected_rows)

    def _on_predict_all_matches(self) -> None:
        if not self._ensure_model_ready_for_predict():
            return
        if self._predict_inflight:
            self._show_warning("Прогноз уже выполняется", "Дождитесь завершения текущего batch-прогноза.")
            return
        all_rows = [idx for idx in range(self.matches_table.rowCount()) if not self.matches_table.isRowHidden(idx)]
        self._predict_matches_rows(all_rows)

    def _is_model_ready_for_predict(self) -> bool:
        snapshot = self._get_model_readiness_snapshot()
        return bool(snapshot.get("ready", False))

    def _refresh_predict_buttons(self) -> None:
        ready = self._is_model_ready_for_predict()
        selected_rows_count = sum(1 for idx in range(self.matches_table.rowCount()) if self._is_row_selected(idx))
        has_visible_rows = any(not self.matches_table.isRowHidden(idx) for idx in range(self.matches_table.rowCount()))
        can_start = (not self._predict_inflight) and ready
        self.predict_selected_button.setEnabled(can_start and selected_rows_count > 0)
        self.predict_all_button.setEnabled(can_start and has_visible_rows)
        self._update_matches_summary()

    def _ensure_model_ready_for_predict(self) -> bool:
        if self._is_model_ready_for_predict():
            return True

        snapshot = self._get_model_readiness_snapshot()
        if bool(snapshot.get("model_file_exists", False)):
            try:
                self.trainer.load()
            except Exception:
                logger.exception("Auto-load before predict failed")

        self._refresh_predict_buttons()
        if self._is_model_ready_for_predict():
            return True

        self.matches_status_label.setText("Матчи: прогноз недоступен, модель не готова")
        self._show_warning(
            "Модель не готова",
            "Прогноз доступен только после обучения из SQLite или загрузки artifacts модели.",
        )
        return False

    def _predict_matches_rows(self, row_indexes: list[int]) -> dict[str, Any]:
        if not row_indexes:
            self.matches_status_label.setText("Прогноз не выполнен: нет строк для обработки")
            return {
                "total_rows": 0,
                "predicted_count": 0,
                "skipped_count": 0,
                "failed_count": 0,
            }

        tasks: list[dict[str, Any]] = []
        for row_idx in row_indexes:
            row_meta = self.current_matches_rows[row_idx] if row_idx < len(self.current_matches_rows) else {}
            odds_status = self._table_odds_status(row_idx)
            task: dict[str, Any] = {
                "row_idx": row_idx,
                "odds_status": odds_status,
                "row_meta": row_meta if isinstance(row_meta, dict) else {},
            }
            if odds_status == "odds_present":
                task["odds_ft_1"] = self._read_float_cell(row_idx, self.MATCH_COL_ODDS_1)
                task["odds_ft_x"] = self._read_float_cell(row_idx, self.MATCH_COL_ODDS_X)
                task["odds_ft_2"] = self._read_float_cell(row_idx, self.MATCH_COL_ODDS_2)
            tasks.append(task)
            self._set_row_status(row_idx, "queued")
            self.matches_table.setItem(
                row_idx,
                self.MATCH_COL_REASON,
                QTableWidgetItem("В очереди batch-прогноза"),
            )

        self._predict_run_context = {
            "row_indexes": list(row_indexes),
            "model_meta": self._current_model_meta(),
            "predicted_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        }

        self._predict_inflight = True
        self._refresh_predict_buttons()
        self._stop_requested = False
        self._set_matches_progress("Прогноз: запуск batch worker", 1)
        self.matches_status_label.setText(f"Прогноз запущен: строк={len(tasks)}")

        thread = QThread(self)
        worker = MatchesBatchPredictWorker(trainer=self.trainer, tasks=tasks, progress_every=20)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_predict_worker_progress)
        worker.finished.connect(self._on_predict_worker_finished)
        worker.finished.connect(thread.quit)
        thread.finished.connect(self._teardown_predict_worker)

        self._predict_worker_thread = thread
        self._predict_worker = worker
        thread.start()

        return {
            "total_rows": len(row_indexes),
            "predicted_count": 0,
            "skipped_count": 0,
            "failed_count": 0,
            "status": "started_async",
        }

    def _on_predict_worker_progress(
        self,
        processed: int,
        total: int,
        predicted: int,
        skipped: int,
        failed: int,
        _state: str,
    ) -> None:
        if total <= 0:
            return
        percent = min(99, max(1, int((processed / total) * 100)))
        self._set_matches_progress(f"Прогноз: обработано {processed}/{total}", percent)
        self.matches_status_label.setText(
            f"Прогноз в процессе: {processed}/{total}, predicted={predicted}, skipped={skipped}, failed={failed}"
        )

    def _on_predict_worker_finished(self, payload: dict[str, Any]) -> None:
        self._predict_inflight = False

        results = payload.get("results") if isinstance(payload.get("results"), list) else []
        model_meta = self._predict_run_context.get("model_meta") if isinstance(self._predict_run_context, dict) else {}
        predicted_at = str(self._predict_run_context.get("predicted_at") or datetime.utcnow().replace(microsecond=0).isoformat() + "Z")
        if not isinstance(model_meta, dict):
            model_meta = self._current_model_meta()

        journal_rows: list[dict[str, Any]] = []

        for item in results:
            if not isinstance(item, dict):
                continue
            row_idx = int(item.get("row_idx", -1))
            if row_idx < 0 or row_idx >= self.matches_table.rowCount():
                continue

            status = str(item.get("status") or "")
            source = str(item.get("source") or "")
            reason = str(item.get("reason") or "")

            if status == "predicted":
                p1 = self._safe_float(item.get("p1"))
                px = self._safe_float(item.get("px"))
                p2 = self._safe_float(item.get("p2"))
                decision = self._generate_decision(p1, px, p2)

                self.matches_table.setItem(row_idx, self.MATCH_COL_P1, QTableWidgetItem(f"{p1:.4f}"))
                self.matches_table.setItem(row_idx, self.MATCH_COL_PX, QTableWidgetItem(f"{px:.4f}"))
                self.matches_table.setItem(row_idx, self.MATCH_COL_P2, QTableWidgetItem(f"{p2:.4f}"))
                self.matches_table.setItem(row_idx, self.MATCH_COL_DECISION, QTableWidgetItem(decision))
                self.matches_table.setItem(
                    row_idx,
                    self.MATCH_COL_REASON,
                    QTableWidgetItem(reason or self._compose_match_reason(self._table_odds_status(row_idx))),
                )
                self.matches_table.setItem(row_idx, self.MATCH_COL_SOURCE, QTableWidgetItem(self._source_label(source or "model_runtime")))
                self.matches_table.setItem(row_idx, self.MATCH_COL_PREDICTABLE, QTableWidgetItem("Да"))
                self._set_row_status(row_idx, "predicted")

                row_meta = self.current_matches_rows[row_idx] if row_idx < len(self.current_matches_rows) else {}
                feature_snapshot = item.get("feature_snapshot") if isinstance(item.get("feature_snapshot"), dict) else {}
                diagnostics = item.get("feature_diagnostics") if isinstance(item.get("feature_diagnostics"), dict) else {}
                prediction_quality = item.get("prediction_quality") if isinstance(item.get("prediction_quality"), dict) else {}
                final_top1 = max(("1", p1), ("X", px), ("2", p2), key=lambda pair: pair[1])[0]
                no_odds_mode = source == "no_odds_fallback"

                journal_rows.append(
                    self._build_prediction_journal_row(
                        model_meta=model_meta,
                        predicted_at=predicted_at,
                        row_meta=row_meta if isinstance(row_meta, dict) else {},
                        p1=p1,
                        px=px,
                        p2=p2,
                        final_predicted_outcome=final_top1,
                        confidence=(
                            self._safe_float(prediction_quality.get("raw_confidence"))
                            if no_odds_mode
                            else max(p1, px, p2)
                        ) or max(p1, px, p2),
                        calibrated_confidence=(
                            self._safe_float(prediction_quality.get("calibrated_confidence"))
                            if prediction_quality.get("calibrated_confidence") is not None
                            else None
                        ),
                        feature_context_level=str(
                            diagnostics.get("context_level")
                            or feature_snapshot.get("context_level")
                            or "unknown"
                        ),
                        signal_strength=str(prediction_quality.get("signal_strength") or feature_snapshot.get("signal_strength") or "unknown"),
                        market_disagreement_flag=bool(prediction_quality.get("market_disagreement_flag", feature_snapshot.get("market_disagreement_flag", False))),
                        weak_favorite_flag=bool(prediction_quality.get("weak_favorite_flag", feature_snapshot.get("weak_favorite_flag", False))),
                        draw_risk_flag=bool(prediction_quality.get("draw_risk_flag", feature_snapshot.get("draw_risk_flag", False))),
                        stats_override_signal_flag=bool(feature_snapshot.get("stats_override_signal", False)),
                        no_odds_mode=no_odds_mode,
                        prediction_source=source or "model_runtime",
                        prediction_status="predicted",
                    )
                )
            elif status == "skipped":
                self.matches_table.setItem(row_idx, self.MATCH_COL_P1, QTableWidgetItem(""))
                self.matches_table.setItem(row_idx, self.MATCH_COL_PX, QTableWidgetItem(""))
                self.matches_table.setItem(row_idx, self.MATCH_COL_P2, QTableWidgetItem(""))
                self.matches_table.setItem(row_idx, self.MATCH_COL_DECISION, QTableWidgetItem(""))
                self.matches_table.setItem(row_idx, self.MATCH_COL_REASON, QTableWidgetItem(reason or "Пропущено"))
                self.matches_table.setItem(row_idx, self.MATCH_COL_SOURCE, QTableWidgetItem(self._source_label(source or "feature_error")))
                self.matches_table.setItem(row_idx, self.MATCH_COL_PREDICTABLE, QTableWidgetItem("Нет"))
                self._set_row_status(row_idx, "skipped")
            else:
                self.matches_table.setItem(row_idx, self.MATCH_COL_REASON, QTableWidgetItem(reason or "Ошибка прогноза"))
                self.matches_table.setItem(row_idx, self.MATCH_COL_SOURCE, QTableWidgetItem(self._source_label(source or "predict_error")))
                self.matches_table.setItem(row_idx, self.MATCH_COL_PREDICTABLE, QTableWidgetItem("Нет"))
                self._set_row_status(row_idx, "failed")

        if journal_rows:
            try:
                self.ingestion_loader.db.save_model_prediction_history_rows(journal_rows)
            except Exception:
                logger.exception("Prediction journal save failed")

        total = int(payload.get("total", 0) or 0)
        processed = int(payload.get("processed", 0) or 0)
        predicted = int(payload.get("predicted", 0) or 0)
        skipped = int(payload.get("skipped", 0) or 0)
        failed = int(payload.get("failed", 0) or 0)
        no_odds_predicted = int(payload.get("no_odds_predicted", 0) or 0)
        cancelled = bool(payload.get("cancelled", False))

        if cancelled or self._stop_requested:
            self.matches_status_label.setText(
                f"Прогноз остановлен: обработано={processed}/{total}, predicted={predicted}, skipped={skipped}, failed={failed}"
            )
        else:
            no_odds_info = f" (в т.ч. без кф: {no_odds_predicted})" if no_odds_predicted > 0 else ""
            self.matches_status_label.setText(
                f"Прогноз завершен: всего={total}, спрогнозировано={predicted}{no_odds_info}, пропущено={skipped}, ошибок={failed}"
            )

        self._set_matches_progress("Прогноз завершен", 100)
        self._refresh_predict_buttons()
        self._update_matches_summary()
        self._stop_requested = False
        self._predict_run_context = {}

    def _teardown_predict_worker(self) -> None:
        if self._predict_worker is not None:
            try:
                self._predict_worker.deleteLater()
            except RuntimeError:
                pass
        if self._predict_worker_thread is not None:
            try:
                self._predict_worker_thread.deleteLater()
            except RuntimeError:
                pass
        self._predict_worker = None
        self._predict_worker_thread = None

    def _current_model_meta(self) -> dict[str, str]:
        predictor = getattr(self.trainer, "predictor", None)
        model_path_raw = getattr(predictor, "model_path", None) if predictor is not None else None
        if model_path_raw is None:
            return {"model_version": "unknown", "model_fingerprint": "unknown", "model_mtime": "unknown"}

        try:
            model_path = str(model_path_raw)
            fingerprint = self._file_fingerprint(model_path)
            mtime = datetime.fromtimestamp(Path(model_path).stat().st_mtime).isoformat()
            version = Path(model_path).name
            return {
                "model_version": version,
                "model_fingerprint": fingerprint,
                "model_mtime": mtime,
            }
        except Exception:
            return {"model_version": str(model_path_raw), "model_fingerprint": "unknown", "model_mtime": "unknown"}

    def _file_fingerprint(self, file_path: str) -> str:
        path = Path(file_path)
        stat = path.stat()
        mtime = float(stat.st_mtime)
        size = int(stat.st_size)

        if self._model_fingerprint_cache is not None:
            cached_path, cached_mtime, cached_size, cached_fingerprint = self._model_fingerprint_cache
            if cached_path == str(path) and cached_mtime == mtime and cached_size == size:
                return cached_fingerprint

        digest = hashlib.sha1()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        fingerprint = digest.hexdigest()
        self._model_fingerprint_cache = (str(path), mtime, size, fingerprint)
        return fingerprint

    def _build_prediction_journal_row(
        self,
        *,
        model_meta: dict[str, str],
        predicted_at: str,
        row_meta: dict[str, Any],
        p1: float,
        px: float,
        p2: float,
        final_predicted_outcome: str,
        confidence: float | None,
        calibrated_confidence: float | None,
        feature_context_level: str,
        signal_strength: str,
        market_disagreement_flag: bool,
        weak_favorite_flag: bool,
        draw_risk_flag: bool,
        stats_override_signal_flag: bool,
        no_odds_mode: bool,
        prediction_source: str,
        prediction_status: str,
    ) -> dict[str, Any]:
        match_id = row_meta.get("match_id")
        season_id = row_meta.get("season_id")
        league_id = row_meta.get("league_id")
        match_date_iso = row_meta.get("match_date_iso") or row_meta.get("match_date")
        home_team = row_meta.get("home_team_name") or row_meta.get("home_team") or row_meta.get("home")
        away_team = row_meta.get("away_team_name") or row_meta.get("away_team") or row_meta.get("away")

        dedupe_key = "|".join(
            [
                str(match_id or ""),
                str(model_meta.get("model_fingerprint") or "unknown"),
                str(predicted_at),
                f"{p1:.4f}",
                f"{px:.4f}",
                f"{p2:.4f}",
                str(prediction_source),
            ]
        )

        return {
            "predicted_at": predicted_at,
            "model_version": model_meta.get("model_version"),
            "model_fingerprint": model_meta.get("model_fingerprint"),
            "model_mtime": model_meta.get("model_mtime"),
            "match_id": match_id,
            "match_date_iso": match_date_iso,
            "home_team": home_team,
            "away_team": away_team,
            "season_id": season_id,
            "league_id": league_id,
            "p1": p1,
            "px": px,
            "p2": p2,
            "final_predicted_outcome": final_predicted_outcome,
            "confidence": confidence,
            "calibrated_confidence": calibrated_confidence,
            "feature_context_level": feature_context_level,
            "signal_strength": signal_strength,
            "market_disagreement_flag": market_disagreement_flag,
            "weak_favorite_flag": weak_favorite_flag,
            "draw_risk_flag": draw_risk_flag,
            "stats_override_signal_flag": stats_override_signal_flag,
            "no_odds_mode": no_odds_mode,
            "prediction_source": prediction_source,
            "prediction_status": prediction_status,
            "dedupe_key": dedupe_key,
        }

    def _is_row_selected(self, row_idx: int) -> bool:
        widget = self.matches_table.cellWidget(row_idx, self.MATCH_COL_SELECTED)
        return bool(isinstance(widget, QCheckBox) and widget.isChecked())

    def _extract_match_features(self, row_idx: int) -> dict[str, float]:
        odds_status = self._table_odds_status(row_idx)
        if odds_status == "odds_missing":
            raise ValueError("odds_missing: коэффициенты отсутствуют")
        if odds_status == "odds_partial":
            raise ValueError("odds_partial: коэффициенты заполнены не полностью")

        odds_1 = self._table_float(row_idx, self.MATCH_COL_ODDS_1, "odds_ft_1")
        odds_x = self._table_float(row_idx, self.MATCH_COL_ODDS_X, "odds_ft_x")
        odds_2 = self._table_float(row_idx, self.MATCH_COL_ODDS_2, "odds_ft_2")

        # Validate bookmaker odds: must be > 1.0 (invalid/missing odds would produce
        # uniform implied probs and identical predictions for all such matches)
        bad_odds = []
        if odds_1 <= 1.0:
            bad_odds.append(f"1={odds_1}")
        if odds_x <= 1.0:
            bad_odds.append(f"X={odds_x}")
        if odds_2 <= 1.0:
            bad_odds.append(f"2={odds_2}")
        if bad_odds:
            raise ValueError(
                f"коэффициенты отсутствуют или невалидны ({', '.join(bad_odds)}); "
                "требуется > 1.0 для корректного прогноза"
            )

        loaded_columns = getattr(self.trainer.predictor, "feature_columns", None)
        row_meta = self.current_matches_rows[row_idx] if row_idx < len(self.current_matches_rows) else {}
        if not isinstance(row_meta, dict):
            row_meta = {}

        runtime_payload: dict[str, Any] = {
            **row_meta,
            "odds_ft_1": odds_1,
            "odds_ft_x": odds_x,
            "odds_ft_2": odds_2,
        }
        if isinstance(runtime_payload.get("odds"), dict):
            runtime_payload["odds"] = {
                **runtime_payload.get("odds", {}),
                "odds_ft_1": odds_1,
                "odds_ft_x": odds_x,
                "odds_ft_2": odds_2,
            }

        feature_snapshot = self.trainer.build_runtime_feature_snapshot(
            match=runtime_payload,
            required_columns=loaded_columns,
        )
        features = feature_snapshot.get("features", {}) if isinstance(feature_snapshot, dict) else {}
        if not isinstance(features, dict) or not features:
            reason = "runtime feature snapshot is empty"
            if isinstance(feature_snapshot, dict):
                reason = str(feature_snapshot.get("reason") or reason)
            raise ValueError(reason)

        self._runtime_feature_context_by_row[row_idx] = feature_snapshot

        if loaded_columns:
            missing = [name for name in loaded_columns if name not in features or features[name] is None]
            if missing:
                raise ValueError(f"missing required features: {missing[:3]}{'...' if len(missing) > 3 else ''}")
            return {name: float(features[name]) for name in loaded_columns}

        return {k: float(v) for k, v in features.items() if not str(k).startswith("__")}

    def _feature_context_suffix_for_row(self, row_idx: int) -> str:
        snapshot = self._runtime_feature_context_by_row.get(row_idx, {})
        if not isinstance(snapshot, dict):
            return ""
        context_level = str(snapshot.get("context_level") or "").strip()
        if not context_level:
            return ""
        return f" | ctx={context_level}"

    def _try_no_odds_predict_for_row(self, row_idx: int) -> dict[str, Any] | None:
        """Attempt no-odds predictor diagnostics fallback for a Matches-tab row.

        Routes through the runtime feature builder (same path as main predictor) to ensure
        consistent feature derivation for partial-context and degraded contexts.

        Returns predict_with_diagnostics result dict, or a skipped sentinel if
        the predictor or feature path is unavailable.
        """
        predictor = getattr(self.trainer, "predictor", None)
        if predictor is None or not getattr(predictor, "is_ready", False):
            return {"status": "skipped", "reason": "Нет кф, fallback-прогноз невозможен"}

        feature_columns: list[str] | None = getattr(predictor, "feature_columns", None)
        if not feature_columns:
            return {"status": "skipped", "reason": "Нет кф, fallback-прогноз невозможен"}

        # Get the raw match data for this row
        match_data: dict[str, Any] = {}
        current_rows = getattr(self, "current_matches_rows", None)
        if isinstance(current_rows, list) and 0 <= row_idx < len(current_rows):
            raw = current_rows[row_idx]
            if isinstance(raw, dict):
                match_data = raw

        # Use the trainer's runtime feature builder to derive features consistently
        # This ensures avg_goals, entropy, gap, volatility match the main model path
        try:
            feature_snapshot = self.trainer.build_runtime_feature_snapshot(
                match=match_data,
                required_columns=feature_columns,
            )
            if not isinstance(feature_snapshot, dict):
                return {"status": "skipped", "reason": "Нет кф, fallback-прогноз невозможен"}
            
            features = feature_snapshot.get("features", {})
            if not isinstance(features, dict) or not features:
                return {"status": "skipped", "reason": "Нет кф, fallback-прогноз невозможен"}
            
            # Cache the context for later display if predict succeeds
            self._runtime_feature_context_by_row[row_idx] = feature_snapshot
            
            # Call predict with diagnostics, allowing fallback when market features are missing
            return predictor.predict_with_diagnostics(features, allow_no_odds_fallback=True)
        except Exception as exc:
            logger.warning("_try_no_odds_predict_for_row[%d] failed: %s", row_idx, exc)
            return {"status": "skipped", "reason": "Нет кф, fallback-прогноз невозможен"}

    def _table_float(self, row_idx: int, col_idx: int, field: str) -> float:
        item = self.matches_table.item(row_idx, col_idx)
        if item is None:
            raise ValueError(f"{field} is empty")
        text = item.text().strip()
        if not text:
            raise ValueError(f"{field} is empty")
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"{field} is invalid") from exc
    def _generate_decision(self, p1: float, px: float, p2: float, threshold: float = 0.15) -> str:
        """Generate decision from probabilities using the shared decision engine.
        
        Delegates to core.decision.decision_engine.DecisionEngine for consistent
        decision logic across MATCHES and TOTO workflows.
        
        Args:
            p1: Probability of outcome 1 (home win)
            px: Probability of outcome X (draw)
            p2: Probability of outcome 2 (away win)
            threshold: Unused; kept for backwards compatibility. Actual logic is in DecisionEngine.
        
        Returns:
            Decision string: "1", "X", "2", "1X", "X2", or "12"
        """
        try:
            from core.decision.decision_engine import DecisionEngine
            
            engine = DecisionEngine()
            result = engine.decide(
                probs={"P1": p1, "PX": px, "P2": p2},
                features={},  # DecisionEngine currently doesn't depend on features
            )
            decision = result.get("decision", "X")
            return str(decision).strip() or "X"
        except Exception as exc:
            logger.warning("_generate_decision: DecisionEngine failed, falling back to local logic: %s", exc)
            # Fallback to simple logic if DecisionEngine is unavailable
            probs = [("1", p1), ("X", px), ("2", p2)]
            probs_sorted = sorted(probs, key=lambda x: x[1], reverse=True)
            
            highest = probs_sorted[0]
            second_highest = probs_sorted[1]
            
            # Simple fallback threshold logic
            if highest[1] - second_highest[1] <= threshold:
                outcomes = sorted([highest[0], second_highest[0]], key=lambda x: ["1", "X", "2"].index(x))
                return "".join(outcomes)
            else:
                return highest[0]


    def _set_row_status(self, row_idx: int, status: str) -> None:
        status_item = QTableWidgetItem(status)
        status_lower = status.strip().lower()
        bg_color = QColor("white")
        fg_color = QColor("black")
        
        if status_lower == "predicted":
            bg_color = QColor("#27ae60")  # Darker green
            fg_color = QColor("white")
        elif status_lower == "queued":
            bg_color = QColor("#f39c12")  # Darker orange/yellow
            fg_color = QColor("white")
        elif status_lower.startswith("skipped"):
            bg_color = QColor("#e67e22")  # Dark orange
            fg_color = QColor("white")
        elif status_lower.startswith("failed"):
            bg_color = QColor("#e74c3c")  # Dark red
            fg_color = QColor("white")

        status_item.setBackground(bg_color)
        status_item.setForeground(fg_color)
        self.matches_table.setItem(row_idx, self.MATCH_COL_STATUS, status_item)

    def _on_add_selected_to_toto(self) -> None:
        """Transfer selected matches from MATCHES tab to TOTO tab with full model context."""
        selected_rows = []
        for row_idx in range(self.matches_table.rowCount()):
            if self._is_row_selected(row_idx):
                selected_rows.append(row_idx)
        
        if not selected_rows:
            self._show_warning("Никаких матчей не выбрано", "Пожалуйста, отметьте матчи галками перед передачей в TOTO.")
            return
        
        # Prepare match data for TOTO with FULL context
        added_count = 0
        rejected_count = 0
        bridge_debug: list[dict[str, Any]] = []
        for row_idx in selected_rows:
            try:
                row_meta = self.current_matches_rows[row_idx] if row_idx < len(self.current_matches_rows) else {}
                match_name = self.matches_table.item(row_idx, self.MATCH_COL_NAME)
                league = self.matches_table.item(row_idx, self.MATCH_COL_LEAGUE)
                date = self.matches_table.item(row_idx, self.MATCH_COL_DATE)
                odds_1 = self.matches_table.item(row_idx, self.MATCH_COL_ODDS_1)
                odds_x = self.matches_table.item(row_idx, self.MATCH_COL_ODDS_X)
                odds_2 = self.matches_table.item(row_idx, self.MATCH_COL_ODDS_2)
                p1 = self.matches_table.item(row_idx, self.MATCH_COL_P1)
                px = self.matches_table.item(row_idx, self.MATCH_COL_PX)
                p2 = self.matches_table.item(row_idx, self.MATCH_COL_P2)
                decision = self.matches_table.item(row_idx, self.MATCH_COL_DECISION)
                source_item = self.matches_table.item(row_idx, self.MATCH_COL_SOURCE)
                reason_item = self.matches_table.item(row_idx, self.MATCH_COL_REASON)
                
                match_str = match_name.text() if match_name else f"Match {row_idx + 1}"
                league_str = league.text() if league else "-"
                date_str = date.text() if date else ""
                decision_str = decision.text() if decision else "-"
                source_str = source_item.text().strip() if source_item else ""
                reason_str = reason_item.text().strip() if reason_item else ""
                odds_status = str(row_meta.get("odds_status") or self._table_odds_status(row_idx)).strip()
                runtime_snapshot = self._runtime_feature_context_by_row.get(row_idx, {})
                if not isinstance(runtime_snapshot, dict):
                    runtime_snapshot = {}

                runtime_features = runtime_snapshot.get("features", {})
                if not isinstance(runtime_features, dict) or not runtime_features:
                    fallback_features = row_meta.get("features") if isinstance(row_meta.get("features"), dict) else {}
                    runtime_features = fallback_features if isinstance(fallback_features, dict) else {}
                feature_context_level = str(
                    runtime_snapshot.get("context_level")
                    or row_meta.get("feature_context_level")
                    or "unknown"
                )
                
                # Extract odds
                odds_ft_1_val = odds_1.text() if odds_1 else ""
                odds_ft_x_val = odds_x.text() if odds_x else ""
                odds_ft_2_val = odds_2.text() if odds_2 else ""
                
                # Extract model probabilities (if model has made predictions)
                p1_val = p1.text() if p1 else ""
                px_val = px.text() if px else ""
                p2_val = p2.text() if p2 else ""
                model_used = bool(p1_val and px_val and p2_val)
                p1_num = self._safe_float(p1_val) if model_used else 0.0
                px_num = self._safe_float(px_val) if model_used else 0.0
                p2_num = self._safe_float(p2_val) if model_used else 0.0

                bridge_skip_reason: str | None = None
                if odds_status == "odds_missing":
                    bridge_skip_reason = "odds_missing"
                elif odds_status == "odds_partial":
                    bridge_skip_reason = "odds_partial"
                elif source_str in {"no_odds", "feature_error", "predict_error"}:
                    bridge_skip_reason = source_str

                if bridge_skip_reason is not None:
                    rejected_count += 1
                    bridge_debug.append(
                        {
                            "row_index": row_idx,
                            "match_id": row_meta.get("match_id"),
                            "match": match_str,
                            "action": "rejected",
                            "reason": bridge_skip_reason,
                            "odds_status": odds_status,
                            "source": source_str,
                            "ui_reason": reason_str,
                        }
                    )
                    continue

                probabilities_source = "model_runtime" if source_str == "model_runtime" and model_used else "implied_only"
                real_pool_probs = self._extract_pool_probs(row_meta if isinstance(row_meta, dict) else {})
                payload_pool_quotes = row_meta.get("pool_quotes") if isinstance(row_meta.get("pool_quotes"), dict) else {}
                payload_bookmaker_quotes = row_meta.get("bookmaker_quotes") if isinstance(row_meta.get("bookmaker_quotes"), dict) else {}
                payload_bookmaker_probs = {}
                if isinstance(row_meta.get("norm_bookmaker_probs"), dict):
                    payload_bookmaker_probs = row_meta.get("norm_bookmaker_probs", {})
                elif isinstance(row_meta.get("bookmaker_probs"), dict):
                    payload_bookmaker_probs = row_meta.get("bookmaker_probs", {})
                
                toto_entry = {
                    "selected": True,
                    "original_index": len(self.current_toto_draw_matches) + 1,
                    "payload": {
                        "match_id": row_meta.get("match_id"),
                        "match": match_str,
                        "league": league_str,
                        "date": date_str,
                        "odds_status": odds_status,
                        "probabilities_source": probabilities_source,
                        "source_reason": reason_str,
                        "odds_ft_1": odds_ft_1_val,
                        "odds_ft_x": odds_ft_x_val,
                        "odds_ft_2": odds_ft_2_val,
                        "odds": {
                            "odds_ft_1": odds_ft_1_val,
                            "odds_ft_x": odds_ft_x_val,
                            "odds_ft_2": odds_ft_2_val,
                        },
                        "model_probs": {
                            "P1": p1_num,
                            "PX": px_num,
                            "P2": p2_num,
                        } if model_used else {},
                        "model_prob_1": p1_num if model_used else None,
                        "model_prob_x": px_num if model_used else None,
                        "model_prob_2": p2_num if model_used else None,
                        "pool_probs": real_pool_probs,
                        "pool_quotes": payload_pool_quotes,
                        "bookmaker_quotes": payload_bookmaker_quotes,
                        "bookmaker_probs": payload_bookmaker_probs,
                        "norm_bookmaker_probs": payload_bookmaker_probs,
                        "features": runtime_features,
                        "feature_context_level": feature_context_level,
                        "feature_context": {
                            "context_level": feature_context_level,
                            "available_non_market_count": int(runtime_snapshot.get("available_non_market_count", 0) or 0),
                            "total_non_market_count": int(runtime_snapshot.get("total_non_market_count", 0) or 0),
                            "fallback_non_market_count": int(runtime_snapshot.get("fallback_non_market_count", 0) or 0),
                            "status": str(runtime_snapshot.get("status") or "unknown"),
                            "reason": runtime_snapshot.get("reason"),
                        },
                        "prognosis": decision_str,
                        "source": "matches_tab",
                        "model_used": model_used,
                        "model_confidence": self._calc_confidence(p1_val, px_val, p2_val) if model_used else 0.0,
                        "data_source": row_meta.get("data_source") or row_meta.get("source") or "matches_tab",
                        "bridge_debug": {
                            "match_id": row_meta.get("match_id"),
                            "odds_status": odds_status,
                            "probabilities_source": probabilities_source,
                            "model_probs_present": model_used,
                            "pool_probs_present": bool(real_pool_probs),
                            "source_column": source_str,
                            "reason_column": reason_str,
                        },
                    },
                    "display_name": f"{match_str} ({league_str})",
                }
                
                # Check for duplicates - don't add if match already in TOTO
                is_duplicate = False
                for existing in self.current_toto_draw_matches:
                    if existing.get("display_name") == toto_entry["display_name"]:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    self.current_toto_draw_matches.append(toto_entry)
                    added_count += 1
                    bridge_debug.append(
                        {
                            "row_index": row_idx,
                            "match_id": row_meta.get("match_id"),
                            "match": match_str,
                            "action": "added",
                            "odds_status": odds_status,
                            "probabilities_source": probabilities_source,
                            "model_probs": toto_entry["payload"].get("model_probs"),
                            "pool_probs": toto_entry["payload"].get("pool_probs"),
                            "odds": toto_entry["payload"].get("odds"),
                            "reason": reason_str,
                        }
                    )
                else:
                    bridge_debug.append(
                        {
                            "row_index": row_idx,
                            "match_id": row_meta.get("match_id"),
                            "match": match_str,
                            "action": "duplicate_skipped",
                            "odds_status": odds_status,
                            "probabilities_source": probabilities_source,
                        }
                    )
            except Exception as exc:
                logger.warning(f"Failed to add row {row_idx} to TOTO: {exc}")
                rejected_count += 1
                bridge_debug.append(
                    {
                        "row_index": row_idx,
                        "action": "error",
                        "error": str(exc),
                    }
                )
                continue

        self._set_toto_summary_raw_text(
            json.dumps(
                {
                    "event": "matches_to_toto_bridge",
                    "selected_rows": len(selected_rows),
                    "added_count": added_count,
                    "rejected_count": rejected_count,
                    "entries": bridge_debug,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        
        if added_count > 0:
            self._set_toto_draw_matches_rows()
            self.toto_draws_combo.setEnabled(False)
            self._refresh_toto_mode_ui()
            msg = f"Передано в TOTO: {added_count} матчей"
            if rejected_count > 0:
                msg += f", отклонено={rejected_count}"
            self.matches_status_label.setText(f"Матчи: {msg}")
            self.toto_status_label.setText(
                f"TOTO status: добавлено {added_count} матчей из MATCHES, отклонено {rejected_count}"
            )
            logger.info(f"Added {added_count} matches to TOTO from MATCHES tab")
        else:
            self._show_warning("Ошибка при передаче", "Не удалось добавить матчи в TOTO. Проверьте raw summary: возможно, матчи отклонены из-за odds_missing / odds_partial / feature_error.")

    @staticmethod
    def _calc_confidence(p1_str: str, px_str: str, p2_str: str) -> float:
        """Calculate model confidence (max probability value)."""
        try:
            p1 = float(p1_str) if p1_str else 0.0
            px = float(px_str) if px_str else 0.0
            p2 = float(p2_str) if p2_str else 0.0
            return max(p1, px, p2)
        except (ValueError, TypeError):
            return 0.0

    # === TOTO HANDLERS ===

    def _on_load_toto_baltbet_history(self) -> None:
        self.toto_history_status_label.setText("История Baltbet: loading...")
        self.toto_history_status_label.setStyleSheet("color: #AFC0CF;")
        try:
            result = self.toto_api.get_draw_history(name="baltbet-main")
        except ValueError as exc:
            # Human-readable config/runtime message for the UI.
            self.toto_history_status_label.setText(f"История Baltbet: {exc}")
            self.toto_history_status_label.setStyleSheet("color: red;")
            return
        except Exception as exc:
            logger.exception("Failed to load Baltbet draw history")
            self.toto_history_status_label.setText(f"История Baltbet: ошибка загрузки ({exc})")
            self.toto_history_status_label.setStyleSheet("color: red;")
            return

        if not isinstance(result, dict):
            self.toto_history_status_label.setText("История Baltbet: получен пустой ответ")
            self.toto_history_status_label.setStyleSheet("color: orange;")
            return

        self.global_toto_history = result
        self.toto_optimizer.set_global_history_context(result)
        drawing_name = str(result.get("drawing_name", "baltbet-main"))
        self._update_toto_history_status_label()
        self._append_output(
            {
                "event": "toto_baltbet_history_loaded",
                "drawing_name": drawing_name,
                "history_draws_loaded_count": int(result.get("history_draws_loaded_count", 0) or 0),
                "history_events_loaded_count": int(result.get("history_events_loaded_count", 0) or 0),
            }
        )

    def _on_load_toto_draws(self) -> None:
        if self._is_toto_manual_mode():
            self.toto_status_label.setText(
                "TOTO status: draw tools hidden in main flow (MATCHES -> TOTO)"
            )
            return
        try:
            self.current_draws = self.toto_api.get_draws(name="main", page=1)
        except ValueError as exc:
            self.toto_status_label.setText(f"TOTO status: загрузка draws недоступна ({exc})")
            return
        except Exception as exc:
            logger.exception("Failed to load TOTO draws")
            self.toto_status_label.setText(f"TOTO status: draw load failed ({exc})")
            self._show_error("Draw load failed", str(exc))
            return

        if not self.current_draws:
            self.toto_status_label.setText("TOTO status: draw mode, no draws available")
            self._append_output({"event": "toto_draw_load", "rows": 0})
            return

        self.toto_draws_combo.clear()
        for draw in self.current_draws:
            draw_id = draw.get("id")
            number = draw.get("number", "N/A")
            ended_at = draw.get("ended_at", "")
            self.toto_draws_combo.addItem(f"Draw #{draw_id} - {number} ({ended_at})", userData=draw_id)

        self.toto_draws_combo.setEnabled(True)

        self.current_draw_id = self.current_draws[0].get("id")
        self._load_selected_toto_draw_matches()
        self.toto_status_label.setText(f"TOTO status: draw mode, loaded {len(self.current_draws)} draws")
        self._append_output({"event": "toto_draw_load", "rows": len(self.current_draws)})

    def _on_select_toto_draw(self) -> None:
        if self._is_toto_manual_mode():
            return
        if self.toto_draws_combo.count() == 0:
            return

        draw_id = self.toto_draws_combo.currentData()
        if draw_id is not None:
            self.current_draw_id = int(draw_id)
            self.toto_coupons_table.setRowCount(0)
            self.current_coupons = []
            self.current_toto_summary = None
            self.current_coupon_entries = []
            self.toto_coupon_display_line_map = {}
            self.toto_summary_box.clear()
            self.toto_summary_verbose_box.clear()
            self._clear_toto_summary_raw_box()
            self.toto_coupon_lines_box.clear()
            self.toto_selected_coupon_detail_box.clear()
            self._load_selected_toto_draw_matches()
            self.toto_status_label.setText(f"TOTO status: draw #{self.current_draw_id} selected")

    def _load_selected_toto_draw_matches(self) -> bool:
        if self.current_draw_id is None:
            self.current_toto_draw_matches = []
            self._set_toto_draw_matches_rows()
            return False

        try:
            draw = self.toto_api.get_draw(self.current_draw_id)
        except Exception as exc:
            logger.exception("Failed to load selected draw matches")
            self.current_toto_draw_matches = []
            self._set_toto_draw_matches_rows()
            self.toto_status_label.setText(f"TOTO status: draw details load failed ({exc})")
            self._show_error("Draw details load failed", str(exc))
            return False

        raw_matches = draw.get("matches")
        if not isinstance(raw_matches, list) or not raw_matches:
            self.current_toto_draw_matches = []
            self._set_toto_draw_matches_rows()
            self.toto_status_label.setText("TOTO status: selected draw has no matches")
            return False

        prepared: list[dict[str, Any]] = []
        for idx, raw_match in enumerate(raw_matches):
            if not isinstance(raw_match, dict):
                continue
            prepared.append(
                {
                    "selected": True,
                    "original_index": idx + 1,
                    "payload": raw_match,
                    "display_name": self._format_toto_match_name(raw_match, idx + 1),
                }
            )

        self.current_toto_draw_matches = prepared
        self._set_toto_draw_matches_rows()
        self.toto_status_label.setText(
            f"TOTO status: draw #{self.current_draw_id} selected, matches loaded={len(self.current_toto_draw_matches)}"
        )
        return bool(self.current_toto_draw_matches)

    def _set_toto_draw_matches_rows(self) -> None:
        self.toto_matches_table.setRowCount(len(self.current_toto_draw_matches))
        for row_idx, entry in enumerate(self.current_toto_draw_matches):
            order_item = QTableWidgetItem(str(row_idx + 1))
            order_item.setFlags(order_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.toto_matches_table.setItem(row_idx, 0, order_item)

            select_box = QCheckBox()
            select_box.setChecked(bool(entry.get("selected", True)))
            self.toto_matches_table.setCellWidget(row_idx, 1, select_box)

            original_item = QTableWidgetItem(str(entry.get("original_index", row_idx + 1)))
            original_item.setFlags(original_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.toto_matches_table.setItem(row_idx, 2, original_item)

            match_item = QTableWidgetItem(str(entry.get("display_name", f"Match {row_idx + 1}")))
            match_item.setFlags(match_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.toto_matches_table.setItem(row_idx, 3, match_item)

    def _sync_toto_match_selection_from_table(self) -> None:
        for row_idx, entry in enumerate(self.current_toto_draw_matches):
            widget = self.toto_matches_table.cellWidget(row_idx, 1)
            if isinstance(widget, QCheckBox):
                entry["selected"] = widget.isChecked()

    def _collect_selected_toto_matches_in_order(self) -> list[dict[str, Any]]:
        self._sync_toto_match_selection_from_table()
        selected_entries: list[dict[str, Any]] = []
        for row_idx, entry in enumerate(self.current_toto_draw_matches):
            if not entry.get("selected", True):
                continue
            payload = entry.get("payload")
            if not isinstance(payload, dict):
                continue
            selected_entries.append(
                {
                    "user_order_index": row_idx + 1,
                    "original_index": int(entry.get("original_index", row_idx + 1)),
                    "display_name": str(entry.get("display_name", f"Match {row_idx + 1}")),
                    "payload": payload,
                }
            )
        return selected_entries

    def _on_move_toto_match_up(self) -> None:
        self._move_toto_match_by_offset(-1)

    def _on_move_toto_match_down(self) -> None:
        self._move_toto_match_by_offset(1)

    def _move_toto_match_by_offset(self, offset: int) -> None:
        if not self.current_toto_draw_matches:
            self.toto_status_label.setText("TOTO status: no draw matches to reorder")
            return

        current_row = self.toto_matches_table.currentRow()
        if current_row < 0:
            self.toto_status_label.setText("TOTO status: select a match row to reorder")
            return

        new_row = current_row + offset
        if new_row < 0 or new_row >= len(self.current_toto_draw_matches):
            self.toto_status_label.setText("TOTO status: cannot move match beyond list bounds")
            return

        self._sync_toto_match_selection_from_table()
        self.current_toto_draw_matches[current_row], self.current_toto_draw_matches[new_row] = (
            self.current_toto_draw_matches[new_row],
            self.current_toto_draw_matches[current_row],
        )
        self._set_toto_draw_matches_rows()
        self.toto_matches_table.selectRow(new_row)
        self.toto_status_label.setText(
            f"TOTO status: reordered matches (row {current_row + 1} -> {new_row + 1})"
        )

    def _format_toto_match_name(self, match: dict[str, Any], fallback_index: int) -> str:
        if match.get("name"):
            return str(match["name"])

        home = match.get("home_team") or match.get("home") or match.get("home_name")
        away = match.get("away_team") or match.get("away") or match.get("away_name")
        if home or away:
            return f"{home or 'Home'} vs {away or 'Away'}"

        return f"Match {fallback_index}"

    def _on_generate_toto_coupons(self) -> None:
        self._update_toto_generation_intent_label()
        manual_mode = self._is_toto_manual_mode()

        if manual_mode and not self.current_toto_draw_matches:
            self.toto_status_label.setText("TOTO status: manual mode, no matches transferred from MATCHES")
            self._show_warning("No data", "Manual mode: сначала передайте матчи из вкладки MATCHES.")
            return

        if (not manual_mode) and self.current_draw_id is None and not self.current_toto_draw_matches:
            self.toto_status_label.setText("TOTO status: draw mode, no draw selected")
            self._show_warning("No data", "Draw mode: загрузите draws и выберите draw перед генерацией.")
            return

        if (not manual_mode) and (not self.current_toto_draw_matches) and self.current_draw_id is not None:
            loaded = self._load_selected_toto_draw_matches()
            if not loaded:
                self._append_output({"event": "toto_coupon_generate", "draw_id": self.current_draw_id, "matches": 0})
                return

        selected_match_entries = self._collect_selected_toto_matches_in_order()
        if not selected_match_entries:
            self.toto_status_label.setText("TOTO status: no matches selected for generation")
            self._show_warning("No matches selected", "Select at least one match in the TOTO match list.")
            self._append_output(
                {
                    "event": "toto_coupon_generate",
                    "draw_id": self.current_draw_id,
                    "matches": 0,
                    "selected_matches": 0,
                    "subset_used": True,
                }
            )
            return

        # Prepare matches for optimizer with model-assisted probabilities when possible.
        matches = []
        model_used_count = 0
        implied_prob_used_count = 0
        bookmaker_used_count = 0
        pool_used_count = 0
        odds_reconstructed_count = 0
        missing_market_inputs_count = 0
        fallback_reasons: dict[str, int] = {}
        match_summaries: list[dict[str, Any]] = []
        decision_distribution_pre = {
            "single_1": 0,
            "single_X": 0,
            "single_2": 0,
            "double_1X": 0,
            "double_X2": 0,
            "double_12": 0,
            "other": 0,
        }
        manual_source_count = 0
        global_history_available = isinstance(self.global_toto_history, dict)
        global_history_draws_count = (
            int(self.global_toto_history.get("history_draws_loaded_count", 0))
            if global_history_available
            else 0
        )
        global_history_events_count = (
            int(self.global_toto_history.get("history_events_loaded_count", 0))
            if global_history_available
            else 0
        )
        feature_context_counts = {
            "full_context": 0,
            "partial_context": 0,
            "degraded_context": 0,
            "odds_only_context": 0,
            "unknown": 0,
        }
        for entry in selected_match_entries:
            match_idx = int(entry["user_order_index"]) - 1
            match = entry["payload"]

            pool_probs = self._extract_pool_probs(match)
            inference_result = self.trainer.prepare_toto_match_for_inference(match=match, pool_probs=pool_probs)

            selected_probs = inference_result.get("probs", pool_probs)
            model_probs = inference_result.get("model_probs")
            source = str(inference_result.get("source", "pool_context_only"))
            mode_name = str(inference_result.get("mode_name", source))
            fallback_reason_raw = inference_result.get("fallback_reason")
            fallback_reason = str(fallback_reason_raw) if fallback_reason_raw else None
            runtime_features = inference_result.get("runtime_features", {})
            if not isinstance(runtime_features, dict):
                runtime_features = {}
            feature_context_level = str(inference_result.get("feature_context_level") or "unknown")
            if feature_context_level not in feature_context_counts:
                feature_context_level = "unknown"
            feature_context_counts[feature_context_level] = int(feature_context_counts.get(feature_context_level, 0)) + 1

            market_inputs = inference_result.get("market_inputs_available") or {}
            if isinstance(market_inputs, dict) and market_inputs.get("has_odds"):
                odds_reconstructed_from_bk = str(match.get("odds_source", "")) == "reconstructed_from_bookmaker_quotes"
                if odds_reconstructed_from_bk:
                    odds_reconstructed_count += 1

            if source == "model":
                model_used_count += 1
            elif source == "implied_from_odds":
                implied_prob_used_count += 1
                reason_key = fallback_reason or "model_unavailable"
                fallback_reasons[reason_key] = fallback_reasons.get(reason_key, 0) + 1
            elif source == "bookmaker_market":
                bookmaker_used_count += 1
                reason_key = fallback_reason or "no_odds_bk_market_used"
                fallback_reasons[reason_key] = fallback_reasons.get(reason_key, 0) + 1
            elif source == "provided_match_probs":
                reason_key = fallback_reason or "trusted_precomputed_used"
                fallback_reasons[reason_key] = fallback_reasons.get(reason_key, 0) + 1
            else:
                pool_used_count += 1
                if not (isinstance(market_inputs, dict) and (market_inputs.get("has_odds") or market_inputs.get("has_bookmaker_quotes"))):
                    missing_market_inputs_count += 1
                reason_key = fallback_reason or "unspecified_pool_fallback"
                fallback_reasons[reason_key] = fallback_reasons.get(reason_key, 0) + 1

            decision = self._build_toto_decision_from_probs(selected_probs)
            self._increment_decision_distribution(decision_distribution_pre, decision)
            if str(match.get("source", "")).strip().lower() == "matches_tab":
                manual_source_count += 1
            match_name = str(entry["display_name"])

            payload_history = match.get("toto_brief_history") if isinstance(match.get("toto_brief_history"), dict) else None
            resolved_history = payload_history
            if resolved_history is None and global_history_available:
                # Manual MATCHES -> TOTO bridge: inject loaded global history into each match.
                resolved_history = self.global_toto_history

            payload_pool_quotes = match.get("pool_quotes") if isinstance(match.get("pool_quotes"), dict) else {}
            payload_bookmaker_quotes = match.get("bookmaker_quotes") if isinstance(match.get("bookmaker_quotes"), dict) else {}
            payload_bookmaker_probs = {}
            if isinstance(match.get("norm_bookmaker_probs"), dict):
                payload_bookmaker_probs = match.get("norm_bookmaker_probs", {})
            elif isinstance(match.get("bookmaker_probs"), dict):
                payload_bookmaker_probs = match.get("bookmaker_probs", {})
            elif payload_bookmaker_quotes:
                payload_bookmaker_probs = {
                    "P1": self._safe_float(payload_bookmaker_quotes.get("bk_win_1")) / 100.0,
                    "PX": self._safe_float(payload_bookmaker_quotes.get("bk_draw")) / 100.0,
                    "P2": self._safe_float(payload_bookmaker_quotes.get("bk_win_2")) / 100.0,
                }

            matches.append(
                {
                    "name": match_name,
                    "probs": selected_probs,
                    "model_probs": model_probs,
                    "pool_probs": inference_result.get("pool_probs", pool_probs),
                    "pool_quotes": payload_pool_quotes,
                    "bookmaker_quotes": payload_bookmaker_quotes,
                    "bookmaker_probs": payload_bookmaker_probs,
                    "norm_bookmaker_probs": payload_bookmaker_probs,
                    "toto_brief_history": resolved_history,
                    "prob_source": source,
                    "fallback_reason": fallback_reason,
                    "features": runtime_features,
                    "feature_context_level": feature_context_level,
                    "decision": decision,
                }
            )
            match_summaries.append(
                {
                    "match_index": match_idx + 1,
                    "original_draw_index": int(entry["original_index"]),
                    "match_name": match_name,
                    "probability_source": source,
                    "mode_name": mode_name,
                    "fallback_reason": fallback_reason,
                    "decision": decision,
                    "used_probs": {
                        "P1": round(self._safe_float(selected_probs.get("P1")), 6),
                        "PX": round(self._safe_float(selected_probs.get("PX")), 6),
                        "P2": round(self._safe_float(selected_probs.get("P2")), 6),
                    },
                    "has_odds": bool(isinstance(market_inputs, dict) and market_inputs.get("has_odds")),
                    "has_bk_quotes": bool(isinstance(market_inputs, dict) and market_inputs.get("has_bookmaker_quotes")),
                    "odds_status": match.get("odds_status"),
                    "payload_probabilities_source": match.get("probabilities_source"),
                    "payload_source_reason": match.get("source_reason"),
                    "payload_model_probs": match.get("model_probs"),
                    "payload_pool_probs": match.get("pool_probs"),
                    "payload_pool_quotes": payload_pool_quotes,
                    "payload_bookmaker_quotes": payload_bookmaker_quotes,
                    "payload_bookmaker_probs": payload_bookmaker_probs,
                    "payload_toto_brief_history": bool(isinstance(payload_history, dict)),
                    "global_history_injected": bool(resolved_history is not None and payload_history is None),
                    "feature_context_level": feature_context_level,
                    "runtime_feature_count": len(runtime_features),
                    "runtime_features_non_empty": bool(runtime_features),
                    "bridge_debug": match.get("bridge_debug"),
                }
            )

        if not matches:
            self.toto_status_label.setText("TOTO status: draw has no valid matches")
            return

        # Get selected mode
        mode = self.toto_mode_selector.currentText()
        insurance_enabled = self.toto_insurance_checkbox.isChecked()

        strength_text = self.toto_insurance_strength_selector.currentText().strip()
        if not strength_text:
            strength_text = "0.7"
        try:
            insurance_strength = float(strength_text)
        except ValueError:
            insurance_strength = 0.7

        if insurance_strength < 0.0:
            insurance_strength = 0.0
        if insurance_strength > 1.0:
            insurance_strength = 1.0

        # Generate coupons
        try:
            if insurance_enabled:
                self.current_coupons = self.toto_optimizer.optimize_insurance(
                    matches=matches,
                    mode=mode,
                    insurance_strength=insurance_strength,
                    global_history_context=self.global_toto_history,
                )
            else:
                self.current_coupons = self.toto_optimizer.optimize(
                    matches=matches,
                    mode=mode,
                    global_history_context=self.global_toto_history,
                )
        except Exception as exc:
            logger.exception("Coupon generation failed")
            self.toto_status_label.setText(f"TOTO status: coupon generation failed ({exc})")
            self._show_error("Coupon generation failed", str(exc))
            return

        if not self.current_coupons:
            self.toto_status_label.setText("TOTO status: no coupons generated")
            return

        optimizer_summary = self.toto_optimizer.last_run_summary if isinstance(self.toto_optimizer.last_run_summary, dict) else {}
        insurance_rebalance_changes = 0
        insured_coupon_indices_raw = optimizer_summary.get("insured_coupon_indices", [])
        insured_coupon_indices = {
            int(idx)
            for idx in insured_coupon_indices_raw
            if isinstance(idx, int) or (isinstance(idx, str) and idx.isdigit())
        }

        coupon_entries_raw = optimizer_summary.get("coupon_entries", [])
        coupon_entries = coupon_entries_raw if isinstance(coupon_entries_raw, list) else []
        self.current_coupon_entries = coupon_entries
        self.current_insured_coupon_indices = set(insured_coupon_indices)

        # Display coupons (insured rows are visually marked when available).
        self._set_toto_coupons_rows(
            self.current_coupons,
            insured_indices=insured_coupon_indices,
            coupon_entries=coupon_entries,
        )
        self._refresh_coupon_views()
        if self.current_coupons:
            self.toto_coupons_table.selectRow(0)
            self._on_select_toto_coupon_row()

        coupon_lines = self._build_coupon_lines(coupons=self.current_coupons)

        # Calculate coverage score
        coverage_score = self.toto_optimizer.coverage_score(
            coupons=self.current_coupons,
            matches=matches,
        )

        generation_mode = "insurance" if insurance_enabled else "base"
        non_model = len(matches) - model_used_count
        # Use optimizer's computed probability_source — it accurately reflects
        # pool/history/insurance usage.  Fall back to local inference-level counts
        # only when the optimizer summary is missing.
        _opt_prob_source = optimizer_summary.get("probability_source", "")
        if _opt_prob_source:
            probability_source = _opt_prob_source
        elif model_used_count == len(matches):
            probability_source = "model_only"
        elif model_used_count > 0:
            probability_source = (
                f"partial (model={model_used_count}, implied={implied_prob_used_count},"
                f" bk={bookmaker_used_count}, pool={pool_used_count})"
            )
        elif implied_prob_used_count > 0 or bookmaker_used_count > 0:
            probability_source = (
                f"market-based (implied={implied_prob_used_count},"
                f" bk={bookmaker_used_count}, pool={pool_used_count})"
            )
        else:
            probability_source = "pool_fallback_only"

        if fallback_reasons:
            top_reason, top_reason_count = max(fallback_reasons.items(), key=lambda item: item[1])
            fallback_summary = f" | fallback_top={top_reason} ({top_reason_count})"
        else:
            fallback_summary = ""

        self.toto_status_label.setText(
            f"TOTO status: generated {len(self.current_coupons)} coupons | mode={mode} | insurance={'on' if insurance_enabled else 'off'} | coverage={coverage_score:.2%}"
        )

        if model_used_count > 0:
            model_runtime_ready = True
            model_reason = "model-assisted bridge active"
        else:
            model_runtime_ready = False
            if fallback_reasons:
                model_reason = max(fallback_reasons.items(), key=lambda item: item[1])[0]
            else:
                model_reason = "no model-driven matches"

        insurance_diag = optimizer_summary.get("insurance_diagnostics", {}) if isinstance(optimizer_summary, dict) else {}
        if isinstance(insurance_diag, dict):
            target_indexes = [
                int(idx)
                for idx in insurance_diag.get("target_match_indexes", [])
                if isinstance(idx, int) or (isinstance(idx, str) and str(idx).isdigit())
            ]
            expanded_matches: list[dict[str, Any]] = []
            match_layers = optimizer_summary.get("match_layer_diagnostics", [])
            for target_idx in target_indexes:
                if not isinstance(match_layers, list) or target_idx < 0 or target_idx >= len(match_layers):
                    continue
                row = match_layers[target_idx]
                if not isinstance(row, dict):
                    continue
                final_cov = row.get("final_effective_coverage", [])
                if not isinstance(final_cov, list):
                    final_cov = []
                expanded_matches.append(
                    {
                        "match_index": target_idx + 1,
                        "outcomes": [str(x) for x in final_cov if str(x).strip()],
                        "outcomes_count": len([x for x in final_cov if str(x).strip()]),
                    }
                )

            insurance_breakdown = {
                "generation_mode": generation_mode,
                "enabled": bool(insurance_diag.get("insurance_enabled", generation_mode == "insurance")),
                "insured_matches_count": int(insurance_diag.get("insured_matches_count", len(target_indexes)) or len(target_indexes)),
                "expanded_outcomes_count": int(insurance_diag.get("expanded_outcomes_count", 0) or 0),
                "expanded_matches": expanded_matches,
                "note": "sourced from optimizer insurance diagnostics",
            }
        else:
            insurance_breakdown = self._build_toto_insurance_breakdown(
                coupons=self.current_coupons,
                generation_mode=generation_mode,
            )

        coupon_distribution = self._collect_coupon_distribution(self.current_coupons)
        explicit_input_mode = self._current_toto_input_mode()
        if explicit_input_mode == "manual_matches":
            input_mode = "manual_matches"
            if global_history_events_count > 0:
                selected_draw_note = (
                    "Ручной режим: матчи из вкладки MATCHES. "
                    "Загруженная история Baltbet используется как global context."
                )
            else:
                selected_draw_note = "Ручной режим: матчи из вкладки MATCHES, тираж не используется."
        elif manual_source_count > 0:
            input_mode = "mixed"
            if global_history_events_count > 0:
                selected_draw_note = (
                    "Смешанный режим: часть матчей из MATCHES, часть из тиража; "
                    "история Baltbet подключена."
                )
            else:
                selected_draw_note = "Смешанный режим: часть матчей из MATCHES, часть из тиража."
        else:
            input_mode = "draw_mode"
            selected_draw_note = "Режим тиража: матчи из выбранного тиража."

        detailed_summary = {
            "draw_id": self.current_draw_id,
            "input_mode": input_mode,
            "selected_draw_note": selected_draw_note,
            "mode": mode,
            "generation_mode": generation_mode,
            "coupons_count": len(self.current_coupons),
            "coverage_score": float(coverage_score),
            "probability_source": probability_source,
            "model_used_count": model_used_count,
            "implied_prob_used_count": implied_prob_used_count,
            "bookmaker_used_count": bookmaker_used_count,
            "pool_used_count": pool_used_count,
            "odds_reconstructed_count": odds_reconstructed_count,
            "missing_market_inputs_count": missing_market_inputs_count,
            "selected_matches_count": len(selected_match_entries),
            "total_draw_matches_count": len(self.current_toto_draw_matches),
            "subset_used": len(selected_match_entries) != len(self.current_toto_draw_matches),
            "user_order_applied": True,
            "global_history_context_loaded": bool(global_history_available),
            "global_history_draws_loaded_count": global_history_draws_count,
            "global_history_events_loaded_count": global_history_events_count,
            "fallback_reasons": fallback_reasons,
            "insurance_breakdown": insurance_breakdown,
            "insurance_rebalance_changes": insurance_rebalance_changes,
            "decision_distribution_pre": decision_distribution_pre,
            "coupon_distribution_post": coupon_distribution,
            "coupon_lines": coupon_lines,
            "matches": match_summaries,
            "feature_context_counts": feature_context_counts,
            "optimizer_diagnostics": optimizer_summary,
            # Flattened high-value diagnostics for human summary rendering.
            "history_influenced_matches_count": optimizer_summary.get("history_influenced_matches_count", 0),
            "history_adjustment_summary": optimizer_summary.get("history_adjustment_summary", {}),
            "market_adjustment_summary": optimizer_summary.get("market_adjustment_summary", {}),
            "strategy_adjusted_distribution": optimizer_summary.get("strategy_adjusted_distribution", {}),
            "insurance_target_matches": optimizer_summary.get("insurance_target_matches", []),
            "insurance_target_matches_count": optimizer_summary.get("insurance_target_matches_count", 0),
            "insurance_cells_changed_count": optimizer_summary.get("insurance_cells_changed_count", 0),
            "affected_coupon_lines_count": optimizer_summary.get("affected_coupon_lines_count", 0),
            "generated_coupon_count": optimizer_summary.get("generated_coupon_count", len(self.current_coupons)),
            "unique_coupon_count": optimizer_summary.get("unique_coupon_count", len({tuple(c) for c in self.current_coupons})),
            "duplicate_coupon_count": optimizer_summary.get("duplicate_coupon_count", 0),
            "base_coupons_count": optimizer_summary.get("base_coupons_count", len(self.current_coupons)),
            "insured_coupons_count": optimizer_summary.get("insured_coupons_count", 0),
            "base_coupon_count": optimizer_summary.get("base_coupon_count", optimizer_summary.get("base_coupons_count", len(self.current_coupons))),
            "insurance_coupon_count": optimizer_summary.get("insurance_coupon_count", optimizer_summary.get("insured_coupons_count", 0)),
            "insured_coupon_indices": optimizer_summary.get("insured_coupon_indices", []),
            "coupon_entries": optimizer_summary.get("coupon_entries", []),
            "match_layer_diagnostics": optimizer_summary.get("match_layer_diagnostics", []),
            "strategy_adjusted_matches_count": optimizer_summary.get("strategy_adjusted_matches_count", 0),
            "insurance_changed_matches_count": optimizer_summary.get("insurance_changed_matches_count", 0),
            "insurance_changed_coupon_lines_count": optimizer_summary.get("insurance_changed_coupon_lines_count", optimizer_summary.get("affected_coupon_lines_count", 0)),
            "match_level_decision_diagnostics": optimizer_summary.get("match_level_decision_diagnostics", []),
        }
        self.current_toto_summary = detailed_summary
        self._render_toto_summary(detailed_summary)
        self._update_toto_history_status_label()
        self._update_toto_generation_intent_label()
        self._update_toto_copy_actions_state()

        self._append_output(
            {
                "event": "toto_coupon_generate",
                "draw_id": self.current_draw_id,
                "generation_mode": generation_mode,
                "mode": mode,
                "insurance_strength": float(insurance_strength),
                "coupons": len(self.current_coupons),
                "coverage_score": float(coverage_score),
                "matches": len(matches),
                "selected_matches": len(selected_match_entries),
                "total_draw_matches": len(self.current_toto_draw_matches),
                "subset_used": len(selected_match_entries) != len(self.current_toto_draw_matches),
                "user_order_applied": True,
                "probability_source": probability_source,
                "model_runtime_ready": model_runtime_ready,
                "model_runtime_reason": model_reason,
                "model_used": int(optimizer_summary.get("model_used_count", model_used_count) or model_used_count),
                "implied_prob_used": implied_prob_used_count,
                "bookmaker_used": int(optimizer_summary.get("bookmaker_used_count", bookmaker_used_count) or bookmaker_used_count),
                "pool_used": int(optimizer_summary.get("pool_used_count", pool_used_count) or pool_used_count),
                "odds_reconstructed": odds_reconstructed_count,
                "missing_market_inputs": missing_market_inputs_count,
                "fallback_reasons": fallback_reasons,
                "insurance_breakdown": insurance_breakdown,
                "insurance_rebalance_changes": insurance_rebalance_changes,
                "decision_distribution_pre": decision_distribution_pre,
                "coupon_distribution_post": coupon_distribution,
            }
        )

    def _increment_decision_distribution(self, distribution: dict[str, int], decision: str) -> None:
        normalized = "".join(ch for ch in decision.upper() if ch in {"1", "X", "2"})
        unique = "".join(sorted(set(normalized), key=lambda x: ["1", "X", "2"].index(x)))

        key: str
        if unique == "1":
            key = "single_1"
        elif unique == "X":
            key = "single_X"
        elif unique == "2":
            key = "single_2"
        elif unique == "1X":
            key = "double_1X"
        elif unique == "X2":
            key = "double_X2"
        elif unique == "12":
            key = "double_12"
        else:
            key = "other"

        distribution[key] = int(distribution.get(key, 0)) + 1

    def _collect_coupon_distribution(self, coupons: list[list[str]]) -> dict[str, int]:
        distribution = {
            "single_1": 0,
            "single_X": 0,
            "single_2": 0,
            "double_1X": 0,
            "double_X2": 0,
            "double_12": 0,
            "other": 0,
        }
        for coupon in coupons:
            for outcome in coupon:
                self._increment_decision_distribution(distribution, str(outcome))
        return distribution

    def _build_toto_decision_from_probs(self, probs: dict[str, Any]) -> str:
        p1 = self._safe_float(probs.get("P1"))
        px = self._safe_float(probs.get("PX"))
        p2 = self._safe_float(probs.get("P2"))

        ranked = sorted(
            [("1", p1), ("X", px), ("2", p2)],
            key=lambda item: item[1],
            reverse=True,
        )
        top_outcome, top_prob = ranked[0]
        second_outcome, second_prob = ranked[1]

        margin = top_prob - second_prob
        # Symmetric threshold: close outcomes become doubles regardless of 1/X/2 label.
        if margin <= 0.08:
            outcomes = sorted([top_outcome, second_outcome], key=lambda x: ["1", "X", "2"].index(x))
            return "".join(outcomes)

        # Away-win rescue: if P2 is near-top and non-trivial, avoid always collapsing to 1/X.
        if top_outcome != "2" and p2 >= 0.30 and (top_prob - p2) <= 0.04:
            return "2"

        return top_outcome

    def _rebalance_insurance_coupons(
        self,
        coupons: list[list[str]],
        insurance_breakdown: dict[str, Any],
    ) -> tuple[list[list[str]], int]:
        if not coupons:
            return coupons, 0
        if not isinstance(insurance_breakdown, dict) or not insurance_breakdown.get("enabled"):
            return coupons, 0

        expanded_matches = insurance_breakdown.get("expanded_matches", [])
        if not isinstance(expanded_matches, list) or not expanded_matches:
            return coupons, 0

        adjusted = [list(c) for c in coupons]
        total = len(adjusted)
        replacements = 0

        for item in expanded_matches:
            if not isinstance(item, dict):
                continue
            col_idx = int(item.get("match_index", 0) or 0) - 1
            outcomes_raw = item.get("outcomes", [])
            if col_idx < 0 or not isinstance(outcomes_raw, list):
                continue

            outcomes = [str(x).strip() for x in outcomes_raw if str(x).strip() in {"1", "X", "2", "1X", "X2", "12"}]
            if len(outcomes) <= 1:
                continue

            counts = {outcome: 0 for outcome in outcomes}
            for row in adjusted:
                if col_idx < len(row):
                    token = str(row[col_idx]).strip()
                    if token in counts:
                        counts[token] += 1

            primary_outcome = max(counts.items(), key=lambda kv: kv[1])[0]
            min_alt_count = max(1, int(round(total * 0.2)))

            for alt_outcome in outcomes:
                if alt_outcome == primary_outcome:
                    continue
                deficit = min_alt_count - counts.get(alt_outcome, 0)
                if deficit <= 0:
                    continue

                for row_idx in range(total - 1, -1, -1):
                    if deficit <= 0:
                        break
                    row = adjusted[row_idx]
                    if col_idx >= len(row):
                        continue
                    if str(row[col_idx]).strip() != primary_outcome:
                        continue
                    row[col_idx] = alt_outcome
                    deficit -= 1
                    replacements += 1

        return adjusted, replacements

    def _coupon_stake_prefix(self) -> str:
        raw_text = self.toto_coupon_stake_input.text().strip()
        if not raw_text:
            return "30"
        try:
            value = float(raw_text)
        except ValueError:
            return "30"
        if value <= 0:
            return "30"
        if value.is_integer():
            return str(int(value))
        return f"{value:.2f}".rstrip("0").rstrip(".")

    def _base_coupon_count(self, coupon_entries: list[dict[str, Any]] | None = None) -> int:
        entries = coupon_entries if isinstance(coupon_entries, list) and coupon_entries else self.current_coupon_entries
        if isinstance(entries, list) and entries:
            return sum(
                1
                for entry in entries
                if isinstance(entry, dict) and str(entry.get("coupon_type", "")).strip().lower() == "base"
            )

        if isinstance(self.current_toto_summary, dict):
            return int(self.current_toto_summary.get("base_coupon_count", 0) or 0)
        return 0

    def _paired_base_coupon_index(self, coupon_index: int, base_coupon_count: int) -> int | None:
        if base_coupon_count <= 0 or coupon_index < base_coupon_count:
            return None
        insurance_row_index = coupon_index - base_coupon_count
        return insurance_row_index % base_coupon_count

    @staticmethod
    def _coupon_changed_cells(base_coupon: list[str], compared_coupon: list[str]) -> list[tuple[int, str, str]]:
        changed: list[tuple[int, str, str]] = []
        max_len = min(len(base_coupon), len(compared_coupon))
        for col_idx in range(max_len):
            base_val = str(base_coupon[col_idx]).strip().upper()
            current_val = str(compared_coupon[col_idx]).strip().upper()
            if base_val != current_val:
                changed.append((col_idx, base_val, current_val))
        return changed

    def _coupon_lines_for_section(
        self,
        section: str,
        coupons: list[list[str]] | None = None,
        coupon_entries: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        source_coupons = coupons if coupons is not None else self.current_coupons
        entries = coupon_entries if coupon_entries is not None else self.current_coupon_entries
        if not source_coupons:
            return []

        if section == "all":
            return self._build_coupon_lines(source_coupons)

        base_count = self._base_coupon_count(entries)
        if section == "base":
            filtered = source_coupons[:base_count] if base_count > 0 else source_coupons
        elif section == "insurance":
            filtered = source_coupons[base_count:] if base_count > 0 else []
        else:
            filtered = source_coupons
        return self._build_coupon_lines(filtered)

    def _copy_coupon_section(self, section: str, status_label: str) -> None:
        lines = self._coupon_lines_for_section(section)
        if not lines:
            if section == "insurance":
                self.toto_status_label.setText("TOTO status: insurance coupons are not available")
            else:
                self._show_warning("Coupon copy", f"Нет купонов для секции: {status_label}.")
            return

        QApplication.clipboard().setText("\n".join(lines))
        if section == "all":
            self.toto_status_label.setText(f"TOTO status: copied {len(lines)} final coupon lines")
        else:
            self.toto_status_label.setText(f"TOTO status: copied {len(lines)} {status_label} coupon lines")

    def _format_coupon_diff_block(
        self,
        coupon_index: int,
        coupons: list[list[str]],
        coupon_lines: list[str],
        coupon_entries: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        if coupon_index < 0 or coupon_index >= len(coupons) or coupon_index >= len(coupon_lines):
            return []

        entries = coupon_entries if isinstance(coupon_entries, list) else []
        coupon_line = coupon_lines[coupon_index]
        coupon = coupons[coupon_index]
        base_count = self._base_coupon_count(entries)
        paired_base_index = self._paired_base_coupon_index(coupon_index, base_count)
        if paired_base_index is None or paired_base_index >= len(coupons):
            return [coupon_line]

        base_coupon = coupons[paired_base_index]
        base_line = coupon_lines[paired_base_index] if paired_base_index < len(coupon_lines) else self._build_coupon_lines([base_coupon])[0]
        changed_cells = self._coupon_changed_cells(base_coupon, coupon)
        changed_text = "; ".join(
            f"M{col_idx + 1}: {before} -> {after}"
            for col_idx, before, after in changed_cells
        ) or "none"
        return [
            f"base:      {base_line}",
            f"insurance: {coupon_line}",
            f"changed:   {changed_text}",
        ]

    @staticmethod
    def _format_changed_positions_lines(changed_cells: list[tuple[int, str, str]]) -> list[str]:
        if not changed_cells:
            return ["No insurance changes relative to base coupon"]
        return [
            f"- M{col_idx + 1}: base={before} -> insurance={after}"
            for col_idx, before, after in changed_cells
        ]

    def _selected_coupon_diff_payload(self, row_idx: int) -> dict[str, Any]:
        if row_idx < 0 or row_idx >= len(self.current_coupons):
            return {"valid": False}

        entries = self.current_coupon_entries
        coupon_lines = self._build_coupon_lines(self.current_coupons)
        base_count = self._base_coupon_count(entries)
        coupon_type = "base"
        if row_idx < len(entries) and isinstance(entries[row_idx], dict):
            coupon_type = str(entries[row_idx].get("coupon_type", "base")).strip().lower() or "base"

        payload: dict[str, Any] = {
            "valid": True,
            "row_idx": row_idx,
            "coupon_type": coupon_type,
            "coupon_line": coupon_lines[row_idx] if row_idx < len(coupon_lines) else "",
            "paired_base_idx": None,
            "base_line": "",
            "changed_cells": [],
        }
        if coupon_type != "insurance":
            return payload

        paired_base_idx = self._paired_base_coupon_index(row_idx, base_count)
        if paired_base_idx is None or paired_base_idx >= len(self.current_coupons):
            return payload

        base_coupon = self.current_coupons[paired_base_idx]
        insurance_coupon = self.current_coupons[row_idx]
        base_line = coupon_lines[paired_base_idx] if paired_base_idx < len(coupon_lines) else self._build_coupon_lines([base_coupon])[0]
        changed_cells = self._coupon_changed_cells(base_coupon, insurance_coupon)
        payload["paired_base_idx"] = paired_base_idx
        payload["base_line"] = base_line
        payload["changed_cells"] = changed_cells
        return payload

    def _scroll_coupon_display_to_row(self, row_idx: int) -> None:
        line_idx = self.toto_coupon_display_line_map.get(row_idx)
        if line_idx is None:
            return

        cursor = self.toto_coupon_lines_box.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        for _ in range(line_idx):
            cursor.movePosition(QTextCursor.MoveOperation.Down)
        self.toto_coupon_lines_box.setTextCursor(cursor)
        self.toto_coupon_lines_box.ensureCursorVisible()

    def _refresh_coupon_views(self) -> None:
        if not self.current_coupons:
            self.toto_coupon_lines_box.clear()
            self.toto_base_coupon_lines_box.clear()
            self.toto_insurance_coupon_lines_box.clear()
            self.toto_unknown_coupon_lines_box.clear()
            self.toto_base_coupons_label.setText("BASE COUPONS (0)")
            self.toto_insurance_coupons_label.setText("INSURANCE COUPONS (0)")
            self.toto_unknown_coupons_label.setText("UNKNOWN COUPON TYPE (0)")
            self.toto_coupon_display_line_map = {}
            self._update_toto_copy_actions_state()
            return

        coupon_lines = self._build_coupon_lines(coupons=self.current_coupons)
        grouped_coupon_lines, coupon_display_line_map = self._build_grouped_coupon_lines(
            coupons=self.current_coupons,
            coupon_lines=coupon_lines,
            coupon_entries=self.current_coupon_entries,
        )
        self.toto_coupon_display_line_map = coupon_display_line_map
        self.toto_coupon_lines_box.setPlainText("\n".join(grouped_coupon_lines))

        sections = self._classify_coupon_sections(
            coupon_count=len(self.current_coupons),
            coupon_entries=self.current_coupon_entries,
            insured_indices=self.current_insured_coupon_indices,
            base_coupon_count=self._base_coupon_count(self.current_coupon_entries),
        )
        self._set_coupon_section_box(
            self.toto_base_coupons_label,
            self.toto_base_coupon_lines_box,
            "BASE COUPONS",
            [coupon_lines[idx] for idx in sections["base"] if idx < len(coupon_lines)],
            empty_text="(none)",
        )
        self._set_coupon_section_box(
            self.toto_insurance_coupons_label,
            self.toto_insurance_coupon_lines_box,
            "INSURANCE COUPONS",
            [coupon_lines[idx] for idx in sections["insurance"] if idx < len(coupon_lines)],
            empty_text="(none)",
        )
        self._set_coupon_section_box(
            self.toto_unknown_coupons_label,
            self.toto_unknown_coupon_lines_box,
            "UNKNOWN COUPON TYPE",
            [coupon_lines[idx] for idx in sections["unknown"] if idx < len(coupon_lines)],
            empty_text="(none)",
        )
        unknown_parent = self.toto_unknown_coupon_lines_box.parentWidget()
        if isinstance(unknown_parent, QGroupBox):
            unknown_parent.setVisible(bool(sections["unknown"]))
        self._update_toto_copy_actions_state()

    @staticmethod
    def _classify_coupon_sections(
        coupon_count: int,
        coupon_entries: list[dict[str, Any]] | None = None,
        insured_indices: set[int] | None = None,
        base_coupon_count: int | None = None,
    ) -> dict[str, list[int]]:
        sections: dict[str, list[int]] = {"base": [], "insurance": [], "unknown": []}
        if coupon_count <= 0:
            return sections

        entries = coupon_entries if isinstance(coupon_entries, list) else []
        if entries:
            classified = False
            for row_idx in range(coupon_count):
                entry = entries[row_idx] if row_idx < len(entries) and isinstance(entries[row_idx], dict) else {}
                coupon_type = str(entry.get("coupon_type", "")).strip().lower()
                insurance_applied = bool(entry.get("insurance_applied_flag", False))
                if coupon_type == "base" and not insurance_applied:
                    sections["base"].append(row_idx)
                    classified = True
                elif coupon_type == "insurance" or insurance_applied:
                    sections["insurance"].append(row_idx)
                    classified = True
                else:
                    sections["unknown"].append(row_idx)
            if classified:
                return sections

        insured = sorted(idx for idx in (insured_indices or set()) if 0 <= idx < coupon_count)
        if insured:
            insured_set = set(insured)
            sections["insurance"] = insured
            sections["base"] = [idx for idx in range(coupon_count) if idx not in insured_set]
            return sections

        if base_coupon_count is not None and 0 <= int(base_coupon_count) <= coupon_count:
            resolved_base_count = int(base_coupon_count)
            sections["base"] = list(range(resolved_base_count))
            sections["insurance"] = list(range(resolved_base_count, coupon_count))
            return sections

        sections["unknown"] = list(range(coupon_count))
        return sections

    @staticmethod
    def _set_coupon_section_box(
        label: QLabel,
        box: QTextEdit,
        title: str,
        lines: list[str],
        empty_text: str,
    ) -> None:
        label.setText(f"{title} ({len(lines)})")
        box.setPlainText("\n".join(lines) if lines else empty_text)

    def _build_coupon_lines(self, coupons: list[list[str]]) -> list[str]:
        lines: list[str] = []
        prefix = self._coupon_stake_prefix()
        for coupon in coupons:
            normalized: list[str] = []
            for value in coupon:
                token = str(value).strip().upper()
                if token in {"1", "X", "2", "1X", "X2", "12"}:
                    normalized.append(token)
                else:
                    normalized.append("-")
            lines.append(";".join([prefix, *normalized]))
        return lines

    def _on_copy_selected_coupon_line(self) -> None:
        row_idx = self.toto_coupons_table.currentRow()
        if row_idx < 0:
            self._show_warning("Coupon copy", "Выберите строку купона в таблице перед копированием.")
            return
        if row_idx >= len(self.current_coupons):
            self._show_warning("Coupon copy", "Выбранный индекс купона вне текущего диапазона.")
            return

        lines = self._build_coupon_lines([self.current_coupons[row_idx]])
        if not lines:
            self._show_warning("Coupon copy", "Нет строк для копирования.")
            return

        QApplication.clipboard().setText(lines[0])
        self.toto_status_label.setText("TOTO status: selected clean coupon copied")

    def _on_copy_all_coupon_lines(self) -> None:
        if not self.current_coupons:
            self._show_warning("Coupon copy", "Сначала сгенерируйте купоны.")
            return

        self._copy_coupon_section("all", "all")

    def _on_copy_base_coupon_lines(self) -> None:
        if not self.current_coupons:
            self._show_warning("Coupon copy", "Сначала сгенерируйте купоны.")
            return

        self._copy_coupon_section("base", "base")

    def _on_copy_insurance_coupon_lines(self) -> None:
        if not self.current_coupons:
            self._show_warning("Coupon copy", "Сначала сгенерируйте купоны.")
            return

        self._copy_coupon_section("insurance", "insurance")

    def _on_copy_selected_coupon_changed_only(self) -> None:
        row_idx = self.toto_coupons_table.currentRow()
        if row_idx < 0:
            self._show_warning("Coupon copy", "Выберите insurance купон в таблице перед копированием diff.")
            return

        payload = self._selected_coupon_diff_payload(row_idx)
        if not bool(payload.get("valid", False)):
            self._show_warning("Coupon copy", "Невозможно получить детали выбранного купона.")
            return

        if str(payload.get("coupon_type", "base")) != "insurance":
            self._show_warning("Coupon copy", "Copy changed only доступен только для insurance купона.")
            return

        changed_cells = payload.get("changed_cells", [])
        if not isinstance(changed_cells, list):
            changed_cells = []
        if not changed_cells:
            self.toto_status_label.setText("TOTO status: selected insurance coupon has no changed positions")
            return

        lines = self._format_changed_positions_lines(changed_cells)
        QApplication.clipboard().setText("\n".join(lines))
        self.toto_status_label.setText(f"TOTO status: copied {len(lines)} changed diff lines")

    def _extract_pool_probs(self, match: dict[str, Any]) -> dict[str, float]:
        pool_probs_raw = match.get("pool_probs", {})
        if isinstance(pool_probs_raw, dict) and pool_probs_raw:
            probs = {
                "P1": self._safe_float(pool_probs_raw.get("P1")),
                "PX": self._safe_float(pool_probs_raw.get("PX")),
                "P2": self._safe_float(pool_probs_raw.get("P2")),
            }
            if probs["P1"] + probs["PX"] + probs["P2"] > 0:
                return self._normalize_probs(probs)

        public_probs_raw = match.get("public_probs", {})
        if isinstance(public_probs_raw, dict) and public_probs_raw:
            probs = {
                "P1": self._safe_float(public_probs_raw.get("P1")),
                "PX": self._safe_float(public_probs_raw.get("PX")),
                "P2": self._safe_float(public_probs_raw.get("P2")),
            }
            if probs["P1"] + probs["PX"] + probs["P2"] > 0:
                return self._normalize_probs(probs)

        probs = {
            "P1": self._safe_float(match.get("pool_prob_1", match.get("public_prob_1", 0.0))),
            "PX": self._safe_float(match.get("pool_prob_x", match.get("public_prob_x", 0.0))),
            "P2": self._safe_float(match.get("pool_prob_2", match.get("public_prob_2", 0.0))),
        }
        if probs["P1"] + probs["PX"] + probs["P2"] > 0:
            return self._normalize_probs(probs)
        return {}

    def _normalize_probs(self, probs: dict[str, Any]) -> dict[str, float]:
        p1 = self._safe_float(probs.get("P1"))
        px = self._safe_float(probs.get("PX"))
        p2 = self._safe_float(probs.get("P2"))

        total = p1 + px + p2
        if total <= 0:
            return {"P1": 1.0 / 3, "PX": 1.0 / 3, "P2": 1.0 / 3}

        return {
            "P1": p1 / total,
            "PX": px / total,
            "P2": p2 / total,
        }

    def _set_toto_coupons_rows(
        self,
        coupons: list[list[str]],
        insured_indices: set[int] | None = None,
        coupon_entries: list[dict[str, Any]] | None = None,
    ) -> None:
        self.toto_coupons_table.setRowCount(len(coupons))
        insured_set = insured_indices or set()
        entries = coupon_entries if isinstance(coupon_entries, list) else []
        base_coupon_count = self._base_coupon_count(entries)
        for row_idx, coupon in enumerate(coupons):
            row_entry = entries[row_idx] if row_idx < len(entries) and isinstance(entries[row_idx], dict) else {}
            coupon_type = str(row_entry.get("coupon_type", "")).strip().lower()
            source_stage = str(row_entry.get("source_stage", "")).strip()
            insurance_applied = bool(row_entry.get("insurance_applied_flag", False))
            is_insured_row = (row_idx in insured_set) or coupon_type == "insurance" or insurance_applied
            paired_base_idx = self._paired_base_coupon_index(row_idx, base_coupon_count)
            paired_base_coupon = coupons[paired_base_idx] if paired_base_idx is not None and paired_base_idx < len(coupons) else []
            changed_cells = {
                col_idx: (before, after)
                for col_idx, before, after in self._coupon_changed_cells(paired_base_coupon, coupon)
            }
            for col_idx, bet in enumerate(coupon):
                if col_idx < 15:
                    item = QTableWidgetItem(str(bet))
                    if is_insured_row and col_idx in changed_cells:
                        before, after = changed_cells[col_idx]
                        item.setBackground(QColor("#FFE8A3"))
                        item.setForeground(QColor("#111827"))
                        item.setToolTip(
                            f"insurance changed on M{col_idx + 1}: {before} -> {after}"
                        )
                    elif is_insured_row:
                        item.setBackground(QColor("#E6F4FF"))
                        item.setForeground(QColor("#0B2540"))
                        if source_stage:
                            item.setToolTip(f"insurance coupon ({source_stage})")
                        else:
                            item.setToolTip("insurance coupon")
                    else:
                        item.setForeground(QColor("#E6EDF3"))
                        if source_stage:
                            item.setToolTip(f"base coupon ({source_stage})")
                        else:
                            item.setToolTip("base coupon")
                    self.toto_coupons_table.setItem(row_idx, col_idx, item)
        self._update_toto_copy_actions_state()

    @staticmethod
    def _insurance_coupon_diff_note(
        insurance_coupon: list[str],
        base_coupons: list[list[str]],
    ) -> str:
        """Return a compact note of which match positions this insurance coupon differs from the base."""
        if not base_coupons:
            return ""
        n_cols = max((len(c) for c in base_coupons), default=0)
        reference: list[str] = []
        for col in range(n_cols):
            counts: dict[str, int] = {}
            for bc in base_coupons:
                if col < len(bc):
                    val = str(bc[col]).strip()
                    counts[val] = counts.get(val, 0) + 1
            reference.append(max(counts.items(), key=lambda kv: kv[1])[0] if counts else "")
        changed: list[str] = []
        for col, (ins_val, ref_val) in enumerate(zip(insurance_coupon, reference)):
            if str(ins_val).strip() != ref_val:
                changed.append(f"M{col + 1}")
        if not changed:
            return "  ← страховка: исходы совпадают с базовыми"
        return f"  ← страховка изменила: {', '.join(changed)}"

    def _build_grouped_coupon_lines(
        self,
        coupons: list[list[str]],
        coupon_lines: list[str],
        coupon_entries: list[dict[str, Any]] | None = None,
    ) -> tuple[list[str], dict[int, int]]:
        if not coupons or not coupon_lines:
            return [], {}

        entries = coupon_entries if isinstance(coupon_entries, list) else []
        sections = self._classify_coupon_sections(
            coupon_count=len(coupon_lines),
            coupon_entries=entries,
            insured_indices=self.current_insured_coupon_indices,
            base_coupon_count=self._base_coupon_count(entries),
        )
        base_rows = sections["base"]
        insurance_rows = sections["insurance"]
        unknown_rows = sections["unknown"]

        output: list[str] = [f"=== BASE COUPONS ({len(base_rows)}) ==="]
        line_map: dict[int, int] = {}
        if base_rows:
            for row_idx in base_rows:
                line_map[row_idx] = len(output)
                output.append(coupon_lines[row_idx])
        else:
            output.append("(none)")
        output.append("")
        output.append(f"=== INSURANCE COUPONS ({len(insurance_rows)}) ===")
        compact_view = self.toto_compact_coupon_view_checkbox.isChecked()
        if insurance_rows:
            for row_idx in insurance_rows:
                line_map[row_idx] = len(output)
                if compact_view:
                    output.append(coupon_lines[row_idx])
                else:
                    output.extend(self._format_coupon_diff_block(row_idx, coupons, coupon_lines, entries))
                    output.append("")
        else:
            output.append("(none)")

        if output and output[-1] == "":
            output.pop()

        if unknown_rows:
            output.append("")
            output.append("=== UNKNOWN COUPON TYPE ===")
            for row_idx in unknown_rows:
                line_map[row_idx] = len(output)
                output.append(coupon_lines[row_idx])

        return output, line_map

    def _on_select_toto_coupon_row(self) -> None:
        row_idx = self.toto_coupons_table.currentRow()
        if row_idx < 0 or row_idx >= len(self.current_coupons):
            self.toto_selected_coupon_detail_box.clear()
            self._update_toto_copy_actions_state()
            return

        payload = self._selected_coupon_diff_payload(row_idx)
        if not bool(payload.get("valid", False)):
            self.toto_selected_coupon_detail_box.clear()
            self._update_toto_copy_actions_state()
            return

        coupon_type = str(payload.get("coupon_type", "base"))
        coupon_line = str(payload.get("coupon_line", ""))
        changed_cells = payload.get("changed_cells", [])
        if not isinstance(changed_cells, list):
            changed_cells = []

        detail_lines = [f"type: {coupon_type.upper()}"]
        only_changed = self.toto_detail_only_changed_checkbox.isChecked()
        if coupon_type == "insurance":
            if only_changed:
                detail_lines.append("Changed matches only:")
                detail_lines.extend(self._format_changed_positions_lines(changed_cells))
            else:
                detail_lines.append("pairing: exact base pair via optimizer insurance line cycle")
                detail_lines.append(f"base:      {str(payload.get('base_line', ''))}")
                detail_lines.append(f"insurance: {coupon_line}")
                detail_lines.append("Changed matches only:")
                detail_lines.extend(self._format_changed_positions_lines(changed_cells))
        else:
            detail_lines.append(f"base:      {coupon_line}")
            detail_lines.append("Changed matches only:")
            detail_lines.append("No insurance changes relative to base coupon")

        self.toto_selected_coupon_detail_box.setPlainText("\n".join(detail_lines))
        self._scroll_coupon_display_to_row(row_idx)
        self._update_toto_copy_actions_state()

    def _update_toto_copy_actions_state(self) -> None:
        has_coupons = bool(self.current_coupons)
        self.toto_copy_all_button.setEnabled(has_coupons)

        sections = self._classify_coupon_sections(
            coupon_count=len(self.current_coupons),
            coupon_entries=self.current_coupon_entries,
            insured_indices=self.current_insured_coupon_indices,
            base_coupon_count=self._base_coupon_count(self.current_coupon_entries),
        )

        self.toto_copy_selected_button.setEnabled(False)
        self.toto_copy_base_button.setEnabled(bool(sections["base"]))
        self.toto_copy_insurance_button.setEnabled(bool(sections["insurance"]))
        self.toto_copy_changed_button.setEnabled(False)

    def _build_toto_insurance_breakdown(
        self,
        coupons: list[list[str]],
        generation_mode: str,
    ) -> dict[str, Any]:
        if generation_mode != "insurance":
            return {
                "generation_mode": generation_mode,
                "enabled": False,
                "insured_matches_count": 0,
                "expanded_outcomes_count": 0,
                "expanded_matches": [],
                "note": "insurance not enabled",
            }

        if not coupons:
            return {
                "generation_mode": generation_mode,
                "enabled": True,
                "insured_matches_count": 0,
                "expanded_outcomes_count": 0,
                "expanded_matches": [],
                "note": "no coupons generated",
            }

        max_cols = max(len(coupon) for coupon in coupons)
        expanded_matches: list[dict[str, Any]] = []
        expanded_outcomes_count = 0

        for col_idx in range(max_cols):
            unique_outcomes = sorted(
                {
                    str(coupon[col_idx]).strip()
                    for coupon in coupons
                    if col_idx < len(coupon) and str(coupon[col_idx]).strip()
                }
            )
            if len(unique_outcomes) > 1:
                expanded_outcomes_count += len(unique_outcomes)
                expanded_matches.append(
                    {
                        "match_index": col_idx + 1,
                        "outcomes": unique_outcomes,
                        "outcomes_count": len(unique_outcomes),
                    }
                )

        return {
            "generation_mode": generation_mode,
            "enabled": True,
            "insured_matches_count": len(expanded_matches),
            "expanded_outcomes_count": expanded_outcomes_count,
            "expanded_matches": expanded_matches,
            "note": "expanded matches/outcomes are derived from generated coupon set",
        }

    def _render_toto_summary(self, summary: dict[str, Any]) -> None:
        self.toto_summary_box.clear()
        self.toto_layer_view_box.clear()
        self.toto_summary_verbose_box.clear()
        self._clear_toto_summary_raw_box()
        insurance_info = summary.get("insurance_breakdown", {}) if isinstance(summary, dict) else {}
        decision_pre = summary.get("decision_distribution_pre", {}) if isinstance(summary, dict) else {}
        decision_post = summary.get("coupon_distribution_post", {}) if isinstance(summary, dict) else {}
        match_summaries = summary.get("matches", []) if isinstance(summary, dict) else []
        strategy_adjusted_dist = summary.get("strategy_adjusted_distribution", {}) if isinstance(summary, dict) else {}
        history_adj_summary = summary.get("history_adjustment_summary", {}) if isinstance(summary, dict) else {}
        market_adj_summary = summary.get("market_adjustment_summary", {}) if isinstance(summary, dict) else {}
        match_decision_diag = summary.get("match_level_decision_diagnostics", []) if isinstance(summary, dict) else []
        readiness = self._get_model_readiness_snapshot()
        smoke_status = self.last_smoke_status
        smoke_sample_ok = self.last_smoke_payload.get("sample_predict_ok") if isinstance(self.last_smoke_payload, dict) else None
        smoke_sample_text = "unknown"
        if smoke_sample_ok is True:
            smoke_sample_text = "OK"
        elif smoke_sample_ok is False:
            smoke_sample_text = "FAILED"
        runtime_ready_text = "yes" if bool(readiness.get("ready", False)) else "no"

        expanded_matches = insurance_info.get("expanded_matches", []) if isinstance(insurance_info, dict) else []
        expanded_labels = [f"M{int(item.get('match_index', 0))}" for item in expanded_matches if isinstance(item, dict)]
        expanded_matches_text = ", ".join(expanded_labels) if expanded_labels else "-"
        expanded_examples: list[str] = []
        for item in expanded_matches[:5]:
            if not isinstance(item, dict):
                continue
            m_idx = int(item.get("match_index", 0) or 0)
            outcomes = item.get("outcomes", [])
            outcomes_text = ",".join(str(o) for o in outcomes) if isinstance(outcomes, list) else "-"
            base = "-"
            if isinstance(match_summaries, list) and 1 <= m_idx <= len(match_summaries):
                row = match_summaries[m_idx - 1]
                if isinstance(row, dict):
                    base = str(row.get("decision") or "-")
            expanded_examples.append(f"M{m_idx}: base {base} -> insured [{outcomes_text}]")
        expanded_examples_text = "\n".join(expanded_examples) if expanded_examples else "-"

        generated_coupon_count = int(summary.get("generated_coupon_count", summary.get("coupons_count", 0)) or 0)
        unique_coupon_count = int(summary.get("unique_coupon_count", generated_coupon_count) or 0)
        duplicate_coupon_count = int(summary.get("duplicate_coupon_count", max(generated_coupon_count - unique_coupon_count, 0)) or 0)
        avg_hamming = float(summary.get("average_hamming_distance_between_coupons", 0.0) or 0.0)
        median_hamming = float(summary.get("median_hamming_distance", 0.0) or 0.0)
        min_hamming = float(summary.get("min_hamming_distance", 0.0) or 0.0)
        diversity_score = float(summary.get("coupon_diversity_score", 0.0) or 0.0)
        base_coupons_count = int(summary.get("base_coupon_count", summary.get("base_coupons_count", generated_coupon_count)) or 0)
        insured_coupons_count = int(summary.get("insurance_coupon_count", summary.get("insured_coupons_count", 0)) or 0)
        history_influenced_count = int(summary.get("history_influenced_matches_count", 0) or 0)
        strategy_adjusted_matches_count = int(summary.get("strategy_adjusted_matches_count", 0) or 0)
        insurance_target_matches_count = int(summary.get("insurance_target_matches_count", 0) or 0)
        insurance_changed_matches_count = int(summary.get("insurance_changed_matches_count", 0) or 0)
        insurance_cells_changed_count = int(summary.get("insurance_cells_changed_count", 0) or 0)
        affected_coupon_lines_count = int(summary.get("insurance_changed_coupon_lines_count", summary.get("affected_coupon_lines_count", 0)) or 0)
        feature_context_counts = summary.get("feature_context_counts", {}) if isinstance(summary, dict) else {}
        full_ctx = int(feature_context_counts.get("full_context", 0) or 0)
        partial_ctx = int(feature_context_counts.get("partial_context", 0) or 0)
        odds_only_ctx = int(feature_context_counts.get("odds_only_context", 0) or 0)
        degraded_ctx = int(feature_context_counts.get("degraded_context", 0) or 0)
        unknown_ctx = int(feature_context_counts.get("unknown", 0) or 0)

        insured_indices_raw = summary.get("insured_coupon_indices", [])
        insured_indices_text = "-"
        if isinstance(insured_indices_raw, list) and insured_indices_raw:
            shown = [str(int(idx) + 1) for idx in insured_indices_raw[:12] if isinstance(idx, int)]
            if shown:
                suffix = "..." if len(insured_indices_raw) > len(shown) else ""
                insured_indices_text = ", ".join(shown) + suffix

        global_history_draws_count = int(summary.get("global_history_draws_loaded_count", 0) or 0)
        global_history_events_count = int(summary.get("global_history_events_loaded_count", 0) or 0)
        history_status = "loaded" if (global_history_draws_count > 0 or global_history_events_count > 0) else "not loaded"
        if isinstance(insurance_info, dict) and "enabled" in insurance_info:
            insurance_enabled = bool(insurance_info.get("enabled"))
        elif insured_coupons_count > 0:
            insurance_enabled = True
        elif str(summary.get("generation_mode", "")).strip().lower() == "insurance":
            insurance_enabled = True
        else:
            insurance_enabled = bool(self.toto_insurance_checkbox.isChecked())
        expected_result = "base + insurance coupons" if insurance_enabled else "only base coupons"

        # Determine actual data source label for the summary header.
        opt_diag = summary.get("optimizer_diagnostics", {}) if isinstance(summary, dict) else {}
        pool_sigs_used = int(opt_diag.get("pool_signals_used_count", 0) or 0)
        hist_draws_loaded = int(opt_diag.get("history_draws_loaded_count", 0) or 0)
        hist_events_loaded = int(opt_diag.get("history_events_loaded_count", 0) or 0)
        model_used_count_for_label = int(summary.get("model_used_count", 0) or 0)
        if hist_events_loaded > 0:
            data_source_label = f"Модель + история Baltbet (события, {hist_events_loaded} событий)"
        elif hist_draws_loaded > 0:
            data_source_label = f"Модель + статистика Baltbet (агрегат, {hist_draws_loaded} тиражей)"
        elif pool_sigs_used > 0:
            data_source_label = f"Модель + public/pool сигналы ({pool_sigs_used} матчей)"
        elif model_used_count_for_label > 0:
            data_source_label = "Только модель"
        else:
            data_source_label = "Pool/public без модели"

        # History state readable labels.
        _hist_state_labels: dict[str, str] = {
            "history_events": "история Baltbet (события)",
            "stats_from_precomputed": "статистика Baltbet (агрегат)",
            "history_not_requested": "история не запрошена",
            "no_history_available": "история недоступна",
            "global_history_baltbet": "история Baltbet (загружена пользователем)",
        }

        # Per-match transparency using new layer contract when available.
        match_layer_diag = summary.get("match_layer_diagnostics", []) if isinstance(summary, dict) else []
        match_diag_lines: list[str] = []
        if isinstance(match_layer_diag, list) and match_layer_diag:
            for row in match_layer_diag[:20]:
                if not isinstance(row, dict):
                    continue

                match_label = str(row.get("match", f"Матч {row.get('match_index', '?')}"))
                base_decision = str(row.get("base_model_decision", "n/a"))
                strategy_decision = str(row.get("toto_strategy_adjusted_decision", "n/a"))
                insurance_added = row.get("insurance_added_outcomes", [])
                final_coverage = row.get("final_effective_coverage", [])
                strategy_reasons = row.get("strategy_reason_codes", [])
                insurance_reasons = row.get("insurance_reason_codes", [])
                strategy_human = str(row.get("strategy_human_explanation", "")).strip()
                insurance_human = str(row.get("insurance_human_explanation", "")).strip()
                strategy_strength = str(row.get("strategy_justification_strength", "n/a"))
                insurance_strength = str(row.get("insurance_justification_strength", "n/a"))

                insurance_added_text = ", ".join(str(x) for x in insurance_added) if isinstance(insurance_added, list) and insurance_added else "none"
                final_coverage_text = ", ".join(str(x) for x in final_coverage) if isinstance(final_coverage, list) and final_coverage else "n/a"
                strategy_reasons_text = ", ".join(str(x) for x in strategy_reasons) if isinstance(strategy_reasons, list) and strategy_reasons else "not available"
                insurance_reasons_text = ", ".join(str(x) for x in insurance_reasons) if isinstance(insurance_reasons, list) and insurance_reasons else "not available"
                strategy_human_text = strategy_human if strategy_human else strategy_reasons_text
                insurance_human_text = insurance_human if insurance_human else insurance_reasons_text

                line_parts = [
                    f"--- {match_label} ---",
                    f"  model:            {base_decision}",
                    f"  strategy:         {strategy_decision}",
                    f"  insurance added:  {insurance_added_text}",
                    f"  final coverage:   [{final_coverage_text}]",
                    f"  strategy why:     {strategy_human_text}",
                    f"  insurance why:    {insurance_human_text}",
                    f"  strategy strength:{strategy_strength}",
                    f"  insurance strength:{insurance_strength}",
                    f"  strategy codes:   {strategy_reasons_text}",
                    f"  insurance codes:  {insurance_reasons_text}",
                ]
                match_diag_lines.append("\n".join(line_parts))
        elif isinstance(match_decision_diag, list):
            for diag_row in match_decision_diag[:12]:
                if not isinstance(diag_row, dict):
                    continue
                m_idx = int(diag_row.get("match_index", 0) or 0)
                match_label = str(diag_row.get("match", f"Матч {m_idx}"))

                # What the user saw on the MATCHES tab before transferring to TOTO.
                ms_row = (
                    match_summaries[m_idx - 1]
                    if isinstance(match_summaries, list) and 1 <= m_idx <= len(match_summaries)
                    else {}
                )
                matches_pred = str(ms_row.get("decision", "-")) if isinstance(ms_row, dict) else "-"

                # What the optimizer chose as the base TOTO outcome.
                model_dec = str(diag_row.get("model_decision", "-"))
                adjusted_dec = str(diag_row.get("strategy_adjusted_decision", "-"))

                # Signal sources used.
                pool_used = bool(diag_row.get("pool_signals_used", False))
                hist_used = bool(diag_row.get("history_used", False))
                hist_state = str(diag_row.get("history_state", "history_not_requested"))
                reason = str(diag_row.get("reason", ""))

                if hist_used and hist_state not in ("history_not_requested", "no_history_available", ""):
                    src_info = _hist_state_labels.get(hist_state, hist_state)
                elif pool_used:
                    src_info = "public/pool сигналы"
                else:
                    src_info = "только модель"

                change_marker = " [изменено по сигналу]" if adjusted_dec != model_dec else " [без изменений]"

                line_parts = [
                    f"--- {match_label} ---",
                    f"  MATCHES прогноз:   {matches_pred}",
                    f"  TOTO база:          {adjusted_dec}{change_marker}",
                    f"  Источник сигналов:  {src_info}",
                ]
                if reason:
                    line_parts.append(f"  Причина:            {reason}")
                match_diag_lines.append("\n".join(line_parts))

        match_diag_text = "\n\n".join(match_diag_lines) if match_diag_lines else "(нет данных диагностики)"

        adjacent_added_count = int(summary.get("adjacent_alternatives_added_count", 0) or 0)
        strong_opposite_added_count = int(summary.get("strong_opposite_alternatives_added_count", 0) or 0)
        strong_opposite_with_strong_justification_count = int(
            summary.get("strong_opposite_with_strong_justification_count", 0) or 0
        )
        if isinstance(match_layer_diag, list):
            for row in match_layer_diag:
                if not isinstance(row, dict):
                    continue
                ins_codes = row.get("insurance_reason_codes", [])
                if not isinstance(ins_codes, list):
                    continue
                if "adjacent_safety" in ins_codes and adjacent_added_count <= 0:
                    adjacent_added_count += 1
                if "opposite_safety" in ins_codes and strong_opposite_added_count <= 0:
                    strong_opposite_added_count += 1
                if (
                    "opposite_safety" in ins_codes
                    and str(row.get("insurance_justification_strength", "weak")) == "strong"
                    and strong_opposite_with_strong_justification_count <= 0
                ):
                    strong_opposite_with_strong_justification_count += 1

        insurance_yn = "Да" if insurance_enabled else "Нет"
        mode_label = str(summary.get("mode", "-"))
        gen_mode = str(summary.get("generation_mode", "-"))
        selected_draw_note = str(summary.get("selected_draw_note", "-"))

        layer_lines: list[str] = []
        if isinstance(match_layer_diag, list) and match_layer_diag:
            for row in match_layer_diag[:20]:
                if not isinstance(row, dict):
                    continue
                match_label = str(row.get("match", f"Матч {row.get('match_index', '?')}"))
                model_decision = str(row.get("base_model_decision", "-"))
                strategy_decision = str(row.get("toto_strategy_adjusted_decision", "-"))
                insurance_added = row.get("insurance_added_outcomes", [])
                final_coverage = row.get("final_effective_coverage", [])
                insurance_text = ", ".join(str(x) for x in insurance_added) if isinstance(insurance_added, list) and insurance_added else "-"
                final_text = ", ".join(str(x) for x in final_coverage) if isinstance(final_coverage, list) and final_coverage else strategy_decision
                layer_lines.append(
                    f"{match_label} | model {model_decision} | strategy {strategy_decision} | insurance {insurance_text} | final [{final_text}]"
                )
        elif isinstance(match_decision_diag, list):
            for diag_row in match_decision_diag[:20]:
                if not isinstance(diag_row, dict):
                    continue
                match_label = str(diag_row.get("match", f"Матч {diag_row.get('match_index', '?')}"))
                model_decision = str(diag_row.get("model_decision", "-"))
                strategy_decision = str(diag_row.get("strategy_adjusted_decision", "-"))
                layer_lines.append(
                    f"{match_label} | model {model_decision} | strategy {strategy_decision} | insurance - | final [{strategy_decision}]"
                )
        if not layer_lines:
            layer_lines = ["(нет данных по слоям)"]

        diagnostics_lines = [
            "TOTO diagnostics",
            f"source: {data_source_label}",
            f"input mode: {selected_draw_note}",
            f"coupon size: {mode_label}",
            f"generation mode: {gen_mode}",
            f"coverage: {float(summary.get('coverage_score', 0.0) or 0.0):.2%}",
            f"stake: {self._coupon_stake_prefix()}",
            f"history influenced matches: {history_influenced_count}",
            f"insured rows: {insured_indices_text}",
            f"insurance enabled: {insurance_yn}",
            f"insured matches count: {int((insurance_info or {}).get('insured_matches_count', 0) or 0)}",
            f"expanded outcomes: {int((insurance_info or {}).get('expanded_outcomes_count', 0) or 0)}",
            f"cells changed: {insurance_cells_changed_count}",
            f"coupon lines changed: {affected_coupon_lines_count}",
            f"expanded matches: {expanded_matches_text}",
            f"expanded examples:\n{expanded_examples_text}",
            f"adjacent alternatives added: {adjacent_added_count}",
            f"strong opposite added: {strong_opposite_added_count}",
            f"strong opposite with strong justification: {strong_opposite_with_strong_justification_count}",
            f"unique coupons: {unique_coupon_count}",
            f"duplicates: {duplicate_coupon_count}",
            f"avg hamming: {avg_hamming:.4f} | median: {median_hamming:.4f} | min: {min_hamming:.4f} | diversity: {diversity_score:.4f}",
            f"feature context: full={full_ctx}, partial={partial_ctx}, odds_only={odds_only_ctx}, degraded={degraded_ctx}, unknown={unknown_ctx}",
            f"history adjustment mean/max: {float(history_adj_summary.get('mean_history_adjustment', 0.0) or 0.0):.4f} / {float(history_adj_summary.get('max_history_adjustment', 0.0) or 0.0):.4f}",
            f"market adjustment mean/max: {float(market_adj_summary.get('mean_market_adjustment', 0.0) or 0.0):.4f} / {float(market_adj_summary.get('max_market_adjustment', 0.0) or 0.0):.4f}",
            "decision distribution pre:",
            f"  1={int(decision_pre.get('single_1', 0) or 0)} X={int(decision_pre.get('single_X', 0) or 0)} 2={int(decision_pre.get('single_2', 0) or 0)} 1X={int(decision_pre.get('double_1X', 0) or 0)} X2={int(decision_pre.get('double_X2', 0) or 0)} 12={int(decision_pre.get('double_12', 0) or 0)}",
            "decision distribution post-strategy:",
            f"  1={int(strategy_adjusted_dist.get('single_1', 0) or 0)} X={int(strategy_adjusted_dist.get('single_X', 0) or 0)} 2={int(strategy_adjusted_dist.get('single_2', 0) or 0)} 1X={int(strategy_adjusted_dist.get('double_1X', 0) or 0)} X2={int(strategy_adjusted_dist.get('double_X2', 0) or 0)} 12={int(strategy_adjusted_dist.get('double_12', 0) or 0)}",
            "decision distribution final coupons:",
            f"  1={int(decision_post.get('single_1', 0) or 0)} X={int(decision_post.get('single_X', 0) or 0)} 2={int(decision_post.get('single_2', 0) or 0)} 1X={int(decision_post.get('double_1X', 0) or 0)} X2={int(decision_post.get('double_X2', 0) or 0)} 12={int(decision_post.get('double_12', 0) or 0)}",
            "match diagnostics:",
            match_diag_text,
            f"runtime model ready: {runtime_ready_text}",
            f"smoke: {smoke_status} | predict: {smoke_sample_text}",
        ]
        if insurance_enabled and insured_coupons_count > 0:
            strategy_source = "model+history+insurance"
        elif history_status == "loaded" and (pool_sigs_used > 0 or history_influenced_count > 0):
            strategy_source = "model+history"
        elif pool_sigs_used > 0:
            strategy_source = "model+pool"
        else:
            strategy_source = "model"

        quick_lines = [
            "TOTO summary",
            f"base coupons: {base_coupons_count}",
            f"insurance coupons: {insured_coupons_count}",
            f"strategy-adjusted matches: {strategy_adjusted_matches_count}",
            f"insurance-target matches: {insurance_target_matches_count}",
            f"insurance-changed matches: {insurance_changed_matches_count}",
            "",
            f"coverage: {float(summary.get('coverage_score', 0.0) or 0.0):.2%}",
            f"history: {history_status}",
            f"strategy source: {strategy_source}",
            f"quality: unique={unique_coupon_count}, duplicates={duplicate_coupon_count}, min-hamming={min_hamming:.2f}",
        ]
        self.toto_summary_box.setPlainText("\n".join(quick_lines))
        self.toto_layer_view_box.setPlainText("\n".join(layer_lines))
        self.toto_summary_verbose_box.setPlainText("\n".join(diagnostics_lines))
        self._set_toto_summary_raw_text(json.dumps(summary, ensure_ascii=False, indent=2))

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _training_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.status_label.setText("Runtime status: initializing...")
        layout.addWidget(self.status_label)

        self.training_info_label.setText("Модель: не проверена | Последнее обучение: нет данных")
        self.training_info_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.training_info_label)

        self.training_progress_stage_label.setText("Этап: ожидание")
        layout.addWidget(self.training_progress_stage_label)
        self.training_progress_bar.setRange(0, 100)
        self.training_progress_bar.setValue(0)
        layout.addWidget(self.training_progress_bar)

        user_group = QGroupBox("Основной режим")
        user_layout = QVBoxLayout(user_group)

        buttons_row = QHBoxLayout()

        load_button = QPushButton("Загрузить модель")
        load_button.clicked.connect(self._on_load_model)
        buttons_row.addWidget(load_button)

        check_ready_button = QPushButton("Проверить готовность модели")
        check_ready_button.clicked.connect(self._on_check_model_readiness)
        buttons_row.addWidget(check_ready_button)

        self.train_from_db_button = QPushButton("Обучить модель из SQLite")
        self.train_from_db_button.clicked.connect(self._on_train_model_from_db)
        buttons_row.addWidget(self.train_from_db_button)

        smoke_button = QPushButton("Показать журнал обучения (smoke)")
        smoke_button.clicked.connect(self._on_run_smoke_test)
        buttons_row.addWidget(smoke_button)

        user_layout.addLayout(buttons_row)
        user_layout.addWidget(self.calibrate_checkbox)
        layout.addWidget(user_group)

        technical_group = QGroupBox("Технический режим")
        technical_group.setCheckable(True)
        technical_group.setChecked(False)
        technical_layout = QVBoxLayout(technical_group)

        train_json_label = QLabel("Train dataset JSON (ручной режим):")
        technical_layout.addWidget(train_json_label)
        self.train_input.setPlaceholderText(
            "[\n"
            "  {\n"
            "    \"target\": 0,\n"
            "    \"odds_ft_1\": 1.9,\n"
            "    \"odds_ft_x\": 3.2,\n"
            "    \"odds_ft_2\": 4.1\n"
            "  }\n"
            "]"
        )
        self.train_input.setMinimumHeight(120)
        self.train_input.textChanged.connect(self._refresh_training_buttons)
        technical_layout.addWidget(self.train_input)

        self.train_button = QPushButton("Обучить по JSON (технический)")
        self.train_button.clicked.connect(self._on_train_model)
        technical_layout.addWidget(self.train_button)

        technical_layout.addWidget(QLabel("Predict single JSON (dict[str, float]):"))
        self.predict_single_input.setPlaceholderText("{\"odds_ft_1\": 1.9, \"odds_ft_x\": 3.2, \"odds_ft_2\": 4.1}")
        self.predict_single_input.setMinimumHeight(90)
        technical_layout.addWidget(self.predict_single_input)

        self.predict_one_button = QPushButton("Тестовый predict single")
        self.predict_one_button.clicked.connect(self._on_predict_single)
        technical_layout.addWidget(self.predict_one_button)

        technical_layout.addWidget(QLabel("Predict batch JSON (list[dict[str, float]]):"))
        self.predict_batch_input.setPlaceholderText(
            "[\n"
            "  {\"odds_ft_1\": 1.9, \"odds_ft_x\": 3.2, \"odds_ft_2\": 4.1},\n"
            "  {\"odds_ft_1\": 2.1, \"odds_ft_x\": 3.1, \"odds_ft_2\": 3.7}\n"
            "]"
        )
        self.predict_batch_input.setMinimumHeight(100)
        technical_layout.addWidget(self.predict_batch_input)

        self.predict_batch_button = QPushButton("Тестовый predict batch")
        self.predict_batch_button.clicked.connect(self._on_predict_batch)
        technical_layout.addWidget(self.predict_batch_button)

        technical_group.toggled.connect(self.train_input.setVisible)
        technical_group.toggled.connect(self.train_button.setVisible)
        technical_group.toggled.connect(train_json_label.setVisible)
        technical_group.toggled.connect(self.predict_single_input.setVisible)
        technical_group.toggled.connect(self.predict_one_button.setVisible)
        technical_group.toggled.connect(self.predict_batch_input.setVisible)
        technical_group.toggled.connect(self.predict_batch_button.setVisible)
        for w in (
            train_json_label,
            self.train_input,
            self.train_button,
            self.predict_single_input,
            self.predict_one_button,
            self.predict_batch_input,
            self.predict_batch_button,
        ):
            w.setVisible(False)

        layout.addWidget(technical_group)

        layout.addWidget(QLabel("Output:"))
        self.output_box.setReadOnly(True)
        self.output_box.setMinimumHeight(140)
        layout.addWidget(self.output_box)

        self._register_delayed_tooltip(
            self.train_from_db_button,
            "Основной режим обучения. Берет накопленные completed-матчи из SQLite и обучает модель.",
        )
        self._register_delayed_tooltip(
            self.train_button,
            "Технический режим. Обучает на вручную вставленном JSON-датасете.",
        )

        return widget

    def _safe_startup_load(self) -> None:
        try:
            self.trainer.load()
            self._set_status("Runtime status: model loaded and ready.")
        except FileNotFoundError:
            self._set_status("Runtime status: model artifacts not found (train required).")
        except Exception as exc:
            logger.exception("Failed to load model runtime on startup")
            self._set_status(f"Runtime status: load failed ({exc}).")
        finally:
            self._restore_toto_history_from_cache()
            self._refresh_global_model_state_ui()
            self._refresh_toto_draws_availability()

    def _restore_toto_history_from_cache(self) -> None:
        try:
            cached = self.toto_api.get_cached_draw_history(name="baltbet-main")
        except Exception:
            logger.exception("Failed to restore cached Baltbet history on startup")
            cached = None

        if isinstance(cached, dict):
            self.global_toto_history = cached
            self.toto_optimizer.set_global_history_context(cached)
            self._update_toto_history_status_label()
            self._append_output(
                {
                    "event": "toto_baltbet_history_restored",
                    "history_draws_loaded_count": int(cached.get("history_draws_loaded_count", 0) or 0),
                    "history_events_loaded_count": int(cached.get("history_events_loaded_count", 0) or 0),
                }
            )
            return

        self.global_toto_history = None
        self.toto_optimizer.set_global_history_context(None)
        self._update_toto_history_status_label()

    def _completed_matches_count(self) -> int:
        try:
            row = self.ingestion_loader.db.conn.execute(
                """
                SELECT COUNT(*)
                FROM matches
                WHERE status = 'completed'
                  AND (winning_team_id IS NOT NULL OR (home_goals IS NOT NULL AND away_goals IS NOT NULL))
                """
            ).fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return 0

    def _refresh_training_buttons(self) -> None:
        completed_count = self._completed_matches_count()
        if hasattr(self, "train_from_db_button"):
            self.train_from_db_button.setEnabled(completed_count > 0)
        json_ok = bool(self.train_input.toPlainText().strip())
        if hasattr(self, "train_button"):
            self.train_button.setEnabled(json_ok)

        snapshot = self._get_model_readiness_snapshot()
        final_rows = int(self.last_train_dataset_summary.get("final_train_rows", 0) or 0)

        self.training_info_label.setText(
            f"Модель: {'готова' if bool(snapshot.get('ready', False)) else 'не готова'} | "
            f"Completed в БД для train: {completed_count} | "
            f"Final train rows: {final_rows} | Smoke: {self.last_smoke_status}"
        )

    def _on_check_model_readiness(self) -> None:
        readiness = self._get_model_readiness_snapshot()
        readiness["feature_columns_count"] = len(getattr(self.trainer.predictor, "feature_columns", []) or [])
        ready = bool(readiness.get("ready", False))
        if ready:
            self._set_status("Модель готова к прогнозу")
        else:
            self._set_status("Модель не готова: проверьте artifacts/feature schema/обучение")

        self._append_output({"event": "model_readiness_check", "ready": ready, **readiness})
        self._refresh_global_model_state_ui()

    def _on_load_model(self) -> None:
        self._set_training_progress("Загрузка artifacts", None)
        try:
            self.trainer.load()
            self._set_status("Model load succeeded. Runtime is ready.")
            self._append_output(
                {
                    "event": "load",
                    **self._get_model_readiness_snapshot(),
                }
            )
            self._set_training_progress("Artifacts загружены", 100)
            self._refresh_global_model_state_ui()
        except FileNotFoundError:
            self._set_status("Model artifacts not found. Train first.")
            self._show_warning("Load failed", "Model artifacts not found. Train the model first.")
            self._set_training_progress("Artifacts не найдены", 0)
            self._refresh_global_model_state_ui()
        except Exception as exc:
            logger.exception("Model load failed")
            self._set_status(f"Model load failed: {exc}")
            self._show_error("Load failed", str(exc))
            self._set_training_progress("Ошибка загрузки", 0)
            self._refresh_global_model_state_ui()

    def _on_train_model(self) -> None:
        self._set_training_progress("Проверка JSON-датасета", 5)
        if not self.train_input.toPlainText().strip():
            self._set_status("Ручной JSON-датасет пуст. Для обычного обучения используйте 'Обучить модель из SQLite'.")
            self._show_warning(
                "Пустой JSON",
                "Поле Train dataset JSON пустое. Вставьте JSON вручную или используйте 'Обучить модель из БД'.",
            )
            self._set_training_progress("Ожидание", 0)
            self._refresh_training_buttons()
            return
        try:
            dataset = self._parse_json(self.train_input.toPlainText())
            if not isinstance(dataset, list) or not dataset:
                raise ValueError("Train dataset must be a non-empty JSON list.")
            if not all(isinstance(row, dict) for row in dataset):
                raise ValueError("Each train row must be a JSON object.")

            self._set_training_progress("Обучение модели", None)
            self.trainer.train(dataset=dataset, calibrate=self.calibrate_checkbox.isChecked())
            self._set_status("Training completed. Runtime is ready.")
            self._append_output(
                {
                    "event": "train",
                    "rows": len(dataset),
                    **self._get_model_readiness_snapshot(),
                }
            )
            self._set_training_progress("Обучение завершено", 100)
            self._refresh_global_model_state_ui()
        except Exception as exc:
            logger.exception("Training failed")
            self._set_status(f"Training failed: {exc}")
            self._show_error("Training failed", str(exc))
            self._set_training_progress("Ошибка обучения", 0)
            self._refresh_global_model_state_ui()

    def _run_post_train_smoke_check(self, sample_limit: int = 3) -> dict[str, Any]:
        snapshot = self._get_model_readiness_snapshot()
        model_file_ok = bool(snapshot.get("model_file_exists", False))
        schema_ok = bool(snapshot.get("feature_schema_loaded", False))

        model_load_ok = False
        sample_predict_ok = False
        sample_predict_count = 0
        sample_predict_errors: list[str] = []

        try:
            self.trainer.load()
            model_load_ok = True
        except Exception as exc:
            sample_predict_errors.append(f"reload failed: {exc}")

        if model_load_ok:
            sample_features: list[dict[str, float]] = []
            for row_idx in range(self.matches_table.rowCount()):
                if len(sample_features) >= sample_limit:
                    break
                try:
                    sample_features.append(self._extract_match_features(row_idx))
                except Exception:
                    continue

            if not sample_features:
                fallback = FeatureBuilder.build_features(
                    {"odds_ft_1": 2.0, "odds_ft_x": 3.2, "odds_ft_2": 3.6},
                    {},
                    {},
                    {},
                )
                required = getattr(self.trainer.predictor, "feature_columns", None) or list(fallback.keys())
                sample_features = [{name: float(fallback.get(name, 0.0)) for name in required}]

            try:
                for row in sample_features[:sample_limit]:
                    prediction = self.trainer.predict(row)
                    if not isinstance(prediction, dict):
                        raise ValueError("predict returned non-dict")
                    if not all(key in prediction for key in ("P1", "PX", "P2")):
                        raise ValueError("predict missing P1/PX/P2")
                    sample_predict_count += 1
                sample_predict_ok = sample_predict_count > 0
            except Exception as exc:
                sample_predict_errors.append(f"sample predict failed: {exc}")

        readiness_after = self._get_model_readiness_snapshot()
        ready = bool(
            model_file_ok
            and schema_ok
            and model_load_ok
            and sample_predict_ok
            and readiness_after.get("ready", False)
        )
        result = "READY" if ready else "FAILED"

        return {
            "model_file_ok": model_file_ok,
            "schema_ok": schema_ok,
            "model_load_ok": model_load_ok,
            "sample_predict_ok": sample_predict_ok,
            "sample_predict_count": sample_predict_count,
            "errors": sample_predict_errors,
            "result": result,
        }

    def _append_smoke_human_log(self, smoke: dict[str, Any]) -> None:
        lines = [
            f"Smoke: model file {'OK' if smoke.get('model_file_ok') else 'FAILED'}",
            f"Smoke: schema {'OK' if smoke.get('schema_ok') else 'FAILED'}",
            f"Smoke: model load {'OK' if smoke.get('model_load_ok') else 'FAILED'}",
            f"Smoke: sample predict {'OK' if smoke.get('sample_predict_ok') else 'FAILED'} (rows={int(smoke.get('sample_predict_count', 0) or 0)})",
            f"Smoke: {smoke.get('result', 'FAILED')}",
        ]
        self.output_box.append("\n".join(lines))

    def _on_train_model_from_db(self) -> None:
        self._set_training_progress("Построение train dataset из SQLite", None)
        try:
            result = self.trainer.train_from_db(
                database=self.ingestion_loader.db,
                calibrate=self.calibrate_checkbox.isChecked(),
            )

            if not isinstance(result, dict):
                raise ValueError("Train from DB returned invalid result format.")

            status = str(result.get("status", "error"))
            message = str(result.get("message", ""))
            dataset_summary = result.get("dataset_summary", {}) if isinstance(result.get("dataset_summary"), dict) else {}
            weak_features = result.get("weak_features", []) if isinstance(result.get("weak_features"), list) else []
            self.last_train_dataset_summary = dataset_summary
            self.last_train_weak_features = [item for item in weak_features if isinstance(item, dict)]

            readiness = self._get_model_readiness_snapshot()
            payload = {
                "event": "train_from_db",
                "status": status,
                "message": message,
                "completed_matches_found": int(result.get("completed_matches_found", 0) or 0),
                "training_rows": int(result.get("training_rows", 0) or 0),
                "raw_completed_rows": int(dataset_summary.get("raw_completed_rows", 0) or 0),
                "unique_completed_matches": int(dataset_summary.get("unique_completed_matches", 0) or 0),
                "rows_after_feature_build": int(dataset_summary.get("rows_after_feature_build", 0) or 0),
                "rows_after_dedup": int(dataset_summary.get("rows_after_dedup", 0) or 0),
                "duplicate_rows_removed": int(dataset_summary.get("duplicate_rows_removed", 0) or 0),
                "final_train_rows": int(dataset_summary.get("final_train_rows", result.get("training_rows", 0)) or 0),
                "dropped_rows": int(dataset_summary.get("dropped_rows", result.get("dropped_rows", 0)) or 0),
                **readiness,
            }
            if "error" in result:
                payload["error"] = str(result.get("error"))

            self._append_output(payload)
            self.output_box.append(self._format_quality_summary_ru(result, title="Качество после Train from DB"))

            if status == "success":
                smoke = self._run_post_train_smoke_check(sample_limit=3)
                self.last_smoke_status = str(smoke.get("result", "FAILED"))
                self.last_smoke_payload = dict(smoke)
                self._append_output({"event": "smoke_post_train", **smoke})
                self._append_smoke_human_log(smoke)
                self._set_status(
                    "Train from DB completed. Use MATCHES tab: load rows and run Predict selected/all for smoke check."
                )
                self._set_training_progress("Train from DB завершен", 100)
                self._refresh_global_model_state_ui()
                return

            if status in ("empty", "too_small"):
                self.last_smoke_status = "FAILED"
                self.last_smoke_payload = {"result": "FAILED", "sample_predict_ok": False}
                completed_found = int(result.get("completed_matches_found", 0) or 0)
                if completed_found <= 0:
                    friendly = (
                        "В базе пока нет завершённых матчей для обучения. "
                        "Сейчас загружены только live/незавершённые матчи. "
                        "Для Train from DB нужна historical/completed history."
                    )
                else:
                    friendly = message
                self._set_status(f"Train from DB skipped: {friendly}")
                self._show_warning("Train from DB: недостаточно данных", friendly)
                self._set_training_progress("Недостаточно данных", 100)
                self._refresh_global_model_state_ui()
                return

            self.last_smoke_status = "FAILED"
            self.last_smoke_payload = {"result": "FAILED", "sample_predict_ok": False}
            self._set_status(f"Train from DB failed: {message}")
            self._show_error("Train from DB failed", message)
            self._set_training_progress("Ошибка обучения", 0)
            self._refresh_global_model_state_ui()
        except Exception as exc:
            logger.exception("Train from DB failed")
            self.last_smoke_status = "FAILED"
            self.last_smoke_payload = {"result": "FAILED", "sample_predict_ok": False}
            self._set_status(f"Train from DB failed: {exc}")
            self._show_error("Train from DB failed", str(exc))
            self._set_training_progress("Ошибка обучения", 0)
            self._refresh_global_model_state_ui()
    
    def _on_run_smoke_test(self) -> None:
        self._set_training_progress("Smoke test: readiness + sample predict", None)
        try:
            smoke = self._run_post_train_smoke_check(sample_limit=3)
            self.last_smoke_status = str(smoke.get("result", "FAILED"))
            self.last_smoke_payload = dict(smoke)
            self._append_output({"event": "smoke_manual", **smoke})
            self._append_smoke_human_log(smoke)

            if self.last_smoke_status == "READY":
                self._set_status("Smoke test completed: READY")
                self._set_training_progress("Smoke test READY", 100)
            else:
                self._set_status("Smoke test completed: FAILED")
                self._set_training_progress("Smoke test FAILED", 100)

            self._refresh_global_model_state_ui()
        except Exception as exc:
            logger.exception("Smoke test failed")
            self.last_smoke_status = "FAILED"
            self.last_smoke_payload = {"result": "FAILED", "sample_predict_ok": False}
            self._set_status(f"Smoke test failed: {exc}")
            self._show_error("Smoke test failed", str(exc))
            self._set_training_progress("Ошибка smoke test", 0)
            self._refresh_global_model_state_ui()

    def _on_predict_single(self) -> None:
        try:
            features = self._parse_json(self.predict_single_input.toPlainText())
            if not isinstance(features, dict) or not features:
                raise ValueError("Predict single input must be a non-empty JSON object.")

            result = self.trainer.predict(features)
            self._set_status("Single prediction completed.")
            self._append_output({"event": "predict_single", "result": result})
        except Exception as exc:
            logger.exception("Single prediction failed")
            self._set_status(f"Single prediction failed: {exc}")
            self._show_error("Predict single failed", str(exc))

    def _on_predict_batch(self) -> None:
        try:
            rows = self._parse_json(self.predict_batch_input.toPlainText())
            if not isinstance(rows, list):
                raise ValueError("Predict batch input must be a JSON list.")
            if not rows:
                self._set_status("Predict batch skipped: empty input list.")
                self._append_output({"event": "predict_batch", "results": []})
                return
            if not all(isinstance(row, dict) for row in rows):
                raise ValueError("Each batch item must be a JSON object.")

            results = self.trainer.predict_batch(rows)
            self._set_status(f"Batch prediction completed for {len(results)} rows.")
            self._append_output({"event": "predict_batch", "count": len(results), "results": results})
        except Exception as exc:
            logger.exception("Batch prediction failed")
            self._set_status(f"Batch prediction failed: {exc}")
            self._show_error("Predict batch failed", str(exc))

    def _set_status(self, message: str) -> None:
        prefix = "Runtime status: "
        if message.startswith(prefix):
            self.status_label.setText(message)
            return
        self.status_label.setText(f"{prefix}{message}")

    def _append_output(self, payload: Any) -> None:
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        self.output_box.append(text)

    def _format_quality_summary_ru(self, payload: dict[str, Any], title: str) -> str:
        def _as_int(value: Any, default: int = 0) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def _as_float(value: Any, default: float = 0.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        event = str(payload.get("event", ""))
        if event == "smoke_e2e":
            dataset_summary = payload.get("dataset_summary", {}) if isinstance(payload.get("dataset_summary"), dict) else {}
            target_distribution = payload.get("target_distribution", {}) if isinstance(payload.get("target_distribution"), dict) else {}
            feature_quality_summary = (
                payload.get("feature_quality_summary", {}) if isinstance(payload.get("feature_quality_summary"), dict) else {}
            )
            weak_features_raw = payload.get("weak_features", []) if isinstance(payload.get("weak_features"), list) else []
            predict_quality = (
                payload.get("predict_quality_summary", {}) if isinstance(payload.get("predict_quality_summary"), dict) else {}
            )
            warnings_raw = payload.get("quality_hints", {}).get("warnings", []) if isinstance(payload.get("quality_hints"), dict) else []
            if not warnings_raw and isinstance(payload.get("warnings"), list):
                warnings_raw = payload.get("warnings", [])
        else:
            dataset_summary = payload.get("dataset_summary", {}) if isinstance(payload.get("dataset_summary"), dict) else {}
            target_distribution = payload.get("target_distribution", {}) if isinstance(payload.get("target_distribution"), dict) else {}
            feature_quality_summary = (
                payload.get("feature_quality_summary", {}) if isinstance(payload.get("feature_quality_summary"), dict) else {}
            )
            weak_features_raw = payload.get("weak_features", []) if isinstance(payload.get("weak_features"), list) else []
            predict_quality = (
                payload.get("predict_quality_summary", {}) if isinstance(payload.get("predict_quality_summary"), dict) else {}
            )
            warnings_raw = payload.get("warnings", []) if isinstance(payload.get("warnings"), list) else []

        completed = _as_int(
            dataset_summary.get("completed_matches_found", payload.get("completed_matches_found", 0)),
            0,
        )
        training_rows = _as_int(dataset_summary.get("training_rows", payload.get("training_rows", 0)), 0)
        raw_completed_rows = _as_int(dataset_summary.get("raw_completed_rows", completed), completed)
        unique_completed_matches = _as_int(dataset_summary.get("unique_completed_matches", completed), completed)
        rows_after_feature_build = _as_int(dataset_summary.get("rows_after_feature_build", training_rows), training_rows)
        rows_after_dedup = _as_int(dataset_summary.get("rows_after_dedup", training_rows), training_rows)
        duplicate_rows_removed = _as_int(dataset_summary.get("duplicate_rows_removed", 0), 0)
        final_train_rows = _as_int(dataset_summary.get("final_train_rows", training_rows), training_rows)
        dropped_rows = _as_int(dataset_summary.get("dropped_rows", payload.get("dropped_rows", completed - training_rows)), 0)
        if dropped_rows < 0:
            dropped_rows = 0

        home_count = _as_int(target_distribution.get("home_win_count", 0), 0)
        draw_count = _as_int(target_distribution.get("draw_count", 0), 0)
        away_count = _as_int(target_distribution.get("away_win_count", 0), 0)

        weak_names: list[str] = []
        for item in weak_features_raw[:5]:
            if isinstance(item, dict):
                feature_name = item.get("feature")
                if feature_name:
                    weak_names.append(str(feature_name))
            elif item:
                weak_names.append(str(item))

        top_healthy = feature_quality_summary.get("top_healthy_features", [])
        healthy_names: list[str] = []
        if isinstance(top_healthy, list):
            for item in top_healthy[:5]:
                if isinstance(item, dict) and item.get("feature"):
                    healthy_names.append(str(item.get("feature")))

        predicted = _as_int(predict_quality.get("predicted_count", 0), 0)
        skipped = _as_int(predict_quality.get("skipped_count", 0), 0)
        failed = _as_int(predict_quality.get("failed_count", 0), 0)
        total_eval = _as_int(predict_quality.get("total_evaluated", predicted + skipped + failed), predicted + skipped + failed)
        usable_rate = _as_float(predict_quality.get("usable_prediction_rate", 0.0), 0.0)
        if usable_rate <= 0 and total_eval > 0:
            usable_rate = predicted / total_eval

        backtest = payload.get("post_train_backtest", {}) if isinstance(payload.get("post_train_backtest"), dict) else {}
        backtest_run = bool(backtest.get("backtest_run", False))

        warning_map = {
            "dataset is very small (<50 completed matches)": "Очень мало завершённых матчей (<50), оценка качества нестабильна.",
            "dataset is below 500 completed matches; quality estimate is unstable": "Матчей меньше 500, качество модели пока нестабильно.",
            "high dropped_rows rate (>30%) between raw dataset and training rows": "Слишком много строк отбрасывается перед обучением (>30%).",
            "draw class is missing in training rows": "В обучении отсутствуют ничьи (класс X).",
            "draw class is underrepresented (<8%)": "Ничьи представлены слабо (<8%).",
            "class distribution is heavily imbalanced towards one side": "Баланс классов сильно перекошен в одну сторону.",
            "several features look weak/default-heavy; review weak_features list": "Есть слабые признаки: много нулей/заглушек.",
            "usable prediction rate is low (<60%)": "Низкий полезный процент прогнозов (<60%).",
            "feature contract is satisfied, but part of features may still be default-heavy": "Контракт фич формально закрыт, но часть признаков остаётся default-heavy.",
            "train from DB did not succeed; e2e cycle is incomplete": "Обучение из БД не завершилось успешно, e2e-цикл неполный.",
        }

        warnings_ru: list[str] = []
        for w in warnings_raw:
            w_text = str(w)
            warnings_ru.append(warning_map.get(w_text, w_text))
        if not warnings_ru:
            warnings_ru = ["Явных предупреждений не обнаружено."]

        weak_text = ", ".join(weak_names) if weak_names else "Нет данных"
        healthy_text = ", ".join(healthy_names) if healthy_names else "Нет данных"

        weak_diagnostics: list[str] = []
        for item in weak_features_raw[:5]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("feature", "?"))
            zero_rate = _as_float(item.get("zero_rate", 0.0), 0.0)
            fallback_rate = _as_float(item.get("fallback_rate", 0.0), 0.0)
            fill_rate = _as_float(item.get("fill_rate", 0.0), 0.0)
            weak_diagnostics.append(
                f"- {name}: fill={fill_rate * 100:.1f}%, zero={zero_rate * 100:.1f}%, fallback={fallback_rate * 100:.1f}%"
            )

        lines = [
            "----------------------------------------",
            title,
            "----------------------------------------",
            "",
            "Датасет:",
            f"- Raw completed rows: {raw_completed_rows}",
            f"- Unique completed matches: {unique_completed_matches}",
            f"- Completed matches (legacy counter): {completed}",
            f"- Rows after feature build: {rows_after_feature_build}",
            f"- Rows after dedup: {rows_after_dedup}",
            f"- Duplicate rows removed: {duplicate_rows_removed}",
            f"- Реально пошло в обучение: {training_rows}",
            f"- Final training rows: {final_train_rows}",
            f"- Отброшено строк: {dropped_rows}",
            (
                f"- Пояснение: training_rows ({training_rows}) может отличаться от completed ({completed}) "
                "из-за фильтрации/очистки и отброса невалидных строк."
            ),
            "",
            "Баланс исходов:",
            f"- Победа хозяев (1): {home_count}",
            f"- Ничья (X): {draw_count}",
            f"- Победа гостей (2): {away_count}",
            "",
            "Качество признаков:",
            f"- Слабые фичи: {weak_text}",
            f"- Более здоровые фичи: {healthy_text}",
            "",
            "Детализация слабых фич:",
        ]
        lines.extend(weak_diagnostics if weak_diagnostics else ["- Нет детализированных данных по слабым фичам"])

        if backtest_run:
            metrics = backtest.get("metrics", {}) if isinstance(backtest.get("metrics"), dict) else {}
            class_metrics = backtest.get("class_metrics", {}) if isinstance(backtest.get("class_metrics"), dict) else {}
            context_metrics = backtest.get("context_metrics", {}) if isinstance(backtest.get("context_metrics"), dict) else {}
            signal_quality = backtest.get("signal_quality", {}) if isinstance(backtest.get("signal_quality"), dict) else {}
            confidence_buckets = signal_quality.get("confidence_buckets", {}) if isinstance(signal_quality.get("confidence_buckets"), dict) else {}
            market_vs_stats = backtest.get("market_vs_stats", {}) if isinstance(backtest.get("market_vs_stats"), dict) else {}

            total_eval_bt = _as_int(backtest.get("total_eval_matches", 0), 0)
            evaluated_bt = _as_int(backtest.get("evaluated_matches", 0), 0)
            skipped_bt = _as_int(backtest.get("skipped_matches", 0), 0)

            acc = metrics.get("accuracy")
            ll = metrics.get("log_loss")
            avg_conf = metrics.get("average_confidence")
            avg_cal_conf = metrics.get("average_calibrated_confidence")
            brier = metrics.get("brier_score")

            hit_rate = class_metrics.get("hit_rate", {}) if isinstance(class_metrics.get("hit_rate"), dict) else {}
            full_ctx = context_metrics.get("full_context", {}) if isinstance(context_metrics.get("full_context"), dict) else {}
            partial_ctx = context_metrics.get("partial_context", {}) if isinstance(context_metrics.get("partial_context"), dict) else {}
            degraded_ctx = context_metrics.get("degraded_context", {}) if isinstance(context_metrics.get("degraded_context"), dict) else {}
            no_odds_ctx = context_metrics.get("no_odds_mode", {}) if isinstance(context_metrics.get("no_odds_mode"), dict) else {}

            strong_sig = signal_quality.get("strong_signal", {}) if isinstance(signal_quality.get("strong_signal"), dict) else {}
            medium_sig = signal_quality.get("medium_signal", {}) if isinstance(signal_quality.get("medium_signal"), dict) else {}
            weak_sig = signal_quality.get("weak_signal", {}) if isinstance(signal_quality.get("weak_signal"), dict) else {}

            lines.extend([
                "",
                "----------------------------------------",
                "Backtest после Train from DB",
                "----------------------------------------",
                "",
                "Общие метрики:",
                f"- Evaluated matches: {evaluated_bt}/{total_eval_bt}",
                f"- Skipped matches: {skipped_bt}",
                f"- Accuracy: {acc if acc is not None else 'n/a'}",
                f"- Log loss: {ll if ll is not None else 'n/a'}",
                f"- Average confidence: {avg_conf if avg_conf is not None else 'n/a'}",
                f"- Average calibrated confidence: {avg_cal_conf if avg_cal_conf is not None else 'n/a'}",
                f"- Brier score: {brier if brier is not None else 'n/a'}",
                "",
                "По классам:",
                f"- 1 hit rate: {hit_rate.get('1')}",
                f"- X hit rate: {hit_rate.get('X')}",
                f"- 2 hit rate: {hit_rate.get('2')}",
                "",
                "По контексту:",
                f"- full_context accuracy: {full_ctx.get('accuracy')}",
                f"- partial_context accuracy: {partial_ctx.get('accuracy')}",
                f"- degraded_context accuracy: {degraded_ctx.get('accuracy')}",
                f"- no_odds_mode accuracy: {no_odds_ctx.get('accuracy')}",
                "",
                "По силе сигнала:",
                f"- strong_signal accuracy: {strong_sig.get('accuracy')}",
                f"- medium_signal accuracy: {medium_sig.get('accuracy')}",
                f"- weak_signal accuracy: {weak_sig.get('accuracy')}",
                "",
                "По confidence buckets:",
                f"- high_confidence accuracy: {(confidence_buckets.get('high_confidence', {}) if isinstance(confidence_buckets.get('high_confidence'), dict) else {}).get('accuracy')}",
                f"- mid_confidence accuracy: {(confidence_buckets.get('mid_confidence', {}) if isinstance(confidence_buckets.get('mid_confidence'), dict) else {}).get('accuracy')}",
                f"- low_confidence accuracy: {(confidence_buckets.get('low_confidence', {}) if isinstance(confidence_buckets.get('low_confidence'), dict) else {}).get('accuracy')}",
                "",
                "Рынок vs модель:",
                f"- aligned accuracy: {market_vs_stats.get('accuracy_when_aligned')}",
                f"- disagreement accuracy: {market_vs_stats.get('accuracy_when_disagreeing')}",
                f"- suspicious disagreements: {market_vs_stats.get('suspicious_market_disagreement_count')}",
                f"- stats_override_signal count: {market_vs_stats.get('stats_override_signal_count')}",
                f"- weak favorite count: {market_vs_stats.get('weak_favorite_flag_count')}",
                f"- draw risk count: {market_vs_stats.get('draw_risk_flag_count')}",
            ])

            summary_lines = backtest.get("summary_lines", [])
            if isinstance(summary_lines, list) and summary_lines:
                lines.append("")
                lines.append("Краткая интерпретация backtest:")
                for item in summary_lines[:5]:
                    lines.append(f"- {item}")
        else:
            backtest_reason = str(backtest.get("reason") or "Проверка качества прогноза не выполнялась в этом режиме")
            lines.extend([
                "",
                "Качество прогноза:",
                f"- Всего матчей для проверки: {total_eval}",
                f"- Успешно предсказано: {predicted}",
                f"- Пропущено: {skipped}",
                f"- Ошибки/сбои: {failed}",
                (
                    f"- Причина: {backtest_reason}"
                    if backtest_reason
                    else (
                        "- Проверка качества прогноза не выполнялась в этом режиме"
                        if total_eval == 0
                        else f"- Полезный процент прогнозов: {usable_rate * 100:.1f}%"
                    )
                ),
            ])

        lines.extend([
            "",
            "Предупреждения:",
        ])
        for item in warnings_ru:
            lines.append(f"- {item}")

        return "\n".join(lines)

    @staticmethod
    def _parse_json(text: str) -> Any:
        stripped = text.strip()
        if not stripped:
            raise ValueError("JSON input is empty.")
        return json.loads(stripped)

    @staticmethod
    def _show_warning(title: str, message: str) -> None:
        QMessageBox.warning(None, title, message)

    @staticmethod
    def _show_error(title: str, message: str) -> None:
        QMessageBox.critical(None, title, message)


def run_ui() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
