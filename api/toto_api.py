from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from api.client import HttpClient
from config import settings

logger = logging.getLogger("toto")


class TotoAPI:
    """Client for Totobrief drawings and drawing details endpoints."""

    def __init__(self, base_url: str | None = None, data_dir: str | Path = "data/toto_draws") -> None:
        self.http = HttpClient(timeout_sec=settings.request_timeout_sec, retries=settings.request_retries)
        self.base_url = self._resolve_base_url(base_url)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.history_db_path = Path(getattr(settings, "db_path", "database/footai.sqlite3"))
        self.history_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_history_snapshot_table()

    def _ensure_history_snapshot_table(self) -> None:
        try:
            conn = sqlite3.connect(self.history_db_path)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS toto_history_snapshots (
                    community_name TEXT PRIMARY KEY,
                    provider TEXT,
                    source TEXT,
                    history_state TEXT,
                    history_empty_reason TEXT,
                    history_stats_ready INTEGER NOT NULL DEFAULT 0,
                    history_draws_loaded_count INTEGER NOT NULL DEFAULT 0,
                    history_events_loaded_count INTEGER NOT NULL DEFAULT 0,
                    pages_loaded INTEGER NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()
            conn.close()
        except Exception:
            logger.exception("toto_history_db_table_init_failed path=%s", self.history_db_path)

    @staticmethod
    def _event_key(event: dict[str, Any]) -> tuple[Any, ...]:
        return (
            event.get("drawing_id"),
            event.get("draw_number"),
            event.get("order"),
            event.get("name"),
            event.get("result"),
            event.get("score"),
        )

    @staticmethod
    def _draw_key(draw: dict[str, Any]) -> tuple[Any, ...]:
        return (
            draw.get("id"),
            draw.get("number"),
            draw.get("ended_at"),
        )

    def _merge_history_payloads(
        self,
        previous: dict[str, Any] | None,
        fresh: dict[str, Any],
    ) -> dict[str, Any]:
        prev_draws = previous.get("draws", []) if isinstance(previous, dict) else []
        new_draws = fresh.get("draws", []) if isinstance(fresh, dict) else []
        merged_draws_map: dict[tuple[Any, ...], dict[str, Any]] = {}
        for row in prev_draws:
            if isinstance(row, dict):
                merged_draws_map[self._draw_key(row)] = row
        for row in new_draws:
            if isinstance(row, dict):
                merged_draws_map[self._draw_key(row)] = row
        merged_draws = list(merged_draws_map.values())

        prev_events = previous.get("events", []) if isinstance(previous, dict) else []
        new_events = fresh.get("events", []) if isinstance(fresh, dict) else []
        merged_events_map: dict[tuple[Any, ...], dict[str, Any]] = {}
        for row in prev_events:
            if isinstance(row, dict):
                merged_events_map[self._event_key(row)] = row
        for row in new_events:
            if isinstance(row, dict):
                merged_events_map[self._event_key(row)] = row
        merged_events = list(merged_events_map.values())

        history_state, history_empty_reason = self._resolve_history_state(
            include_history=True,
            draws_count=len(merged_draws),
            events_count=len(merged_events),
        )
        stats = self._history_stats_from_events(merged_events)
        history_stats_ready = bool(self._safe_float(stats.get("events_count", 0.0)) > 0.0)

        requested_payload = {}
        if isinstance(previous, dict) and isinstance(previous.get("requested"), dict):
            requested_payload = dict(previous.get("requested", {}))
        if isinstance(fresh, dict) and isinstance(fresh.get("requested"), dict):
            requested_payload.update(fresh.get("requested", {}))

        return {
            "drawing_name": str(
                fresh.get("drawing_name")
                or (previous.get("drawing_name") if isinstance(previous, dict) else "baltbet-main")
                or "baltbet-main"
            ),
            "provider": str(fresh.get("provider") or (previous.get("provider") if isinstance(previous, dict) else "totobrief") or "totobrief"),
            "source": str(fresh.get("source") or (previous.get("source") if isinstance(previous, dict) else "draw_history_api") or "draw_history_api"),
            "history_state": history_state,
            "history_empty_reason": history_empty_reason,
            "history_stats_ready": history_stats_ready,
            "draws_loaded_count": len(merged_draws),
            "history_draws_loaded_count": len(merged_draws),
            "history_events_loaded_count": len(merged_events),
            "pages_loaded": int(fresh.get("pages_loaded", 0) or 0),
            "requested": requested_payload,
            "draws": merged_draws,
            "events": merged_events,
            "stats": stats,
        }

    def _save_history_snapshot_to_db(self, name: str, payload: dict[str, Any]) -> bool:
        try:
            conn = sqlite3.connect(self.history_db_path)
            conn.execute(
                """
                INSERT INTO toto_history_snapshots (
                    community_name,
                    provider,
                    source,
                    history_state,
                    history_empty_reason,
                    history_stats_ready,
                    history_draws_loaded_count,
                    history_events_loaded_count,
                    pages_loaded,
                    payload_json,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(community_name) DO UPDATE SET
                    provider=excluded.provider,
                    source=excluded.source,
                    history_state=excluded.history_state,
                    history_empty_reason=excluded.history_empty_reason,
                    history_stats_ready=excluded.history_stats_ready,
                    history_draws_loaded_count=excluded.history_draws_loaded_count,
                    history_events_loaded_count=excluded.history_events_loaded_count,
                    pages_loaded=excluded.pages_loaded,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    str(name),
                    str(payload.get("provider", "totobrief")),
                    str(payload.get("source", "draw_history_api")),
                    str(payload.get("history_state", "history_not_requested")),
                    payload.get("history_empty_reason"),
                    int(bool(payload.get("history_stats_ready", False))),
                    int(payload.get("history_draws_loaded_count", 0) or 0),
                    int(payload.get("history_events_loaded_count", 0) or 0),
                    int(payload.get("pages_loaded", 0) or 0),
                    json.dumps(payload, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
            conn.close()
            return True
        except Exception:
            logger.exception("toto_history_db_save_failed name=%s", name)
            return False

    def _load_history_snapshot_from_db(self, name: str) -> dict[str, Any] | None:
        try:
            conn = sqlite3.connect(self.history_db_path)
            row = conn.execute(
                """
                SELECT payload_json
                FROM toto_history_snapshots
                WHERE community_name = ?
                """,
                (str(name),),
            ).fetchone()
            conn.close()
        except Exception:
            logger.exception("toto_history_db_load_failed name=%s", name)
            return None

        if not row:
            return None
        try:
            payload = json.loads(str(row[0]))
        except Exception:
            logger.exception("toto_history_db_payload_parse_failed name=%s", name)
            return None
        return payload if isinstance(payload, dict) else None

    def get_cached_draw_history(self, name: str = "baltbet-main") -> dict[str, Any] | None:
        from_db = self._load_history_snapshot_from_db(name=name)
        if isinstance(from_db, dict):
            return from_db

        safe_name = str(name).strip().replace("/", "_") or "community"
        snapshot_path = self.data_dir / f"history_{safe_name}.json"
        if not snapshot_path.exists():
            return None
        try:
            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("toto_history_snapshot_load_failed name=%s path=%s", name, snapshot_path)
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    @staticmethod
    def _resolve_base_url(base_url: str | None = None) -> str:
        if isinstance(base_url, str) and base_url.strip():
            return base_url.strip().rstrip("/")
        configured = str(getattr(settings, "toto_api_base_url", "") or "").strip()
        if configured:
            return configured.rstrip("/")
        return ""

    def _require_base_url(self) -> str:
        resolved = self._resolve_base_url(self.base_url)
        if resolved:
            self.base_url = resolved
            return resolved
        raise ValueError(
            "TotoBrief API base URL is not configured. "
            "Set TOTO_API_BASE_URL (or TOTOBRIEF_API_BASE_URL / TOTOBRIEF_BASE_URL / TOTOBRIEF_URL) in .env."
        )

    @staticmethod
    def _safe_params(params: dict[str, Any]) -> dict[str, Any]:
        sensitive_keys = {"key", "api_key", "apikey", "token", "secret", "access_token", "authorization"}
        return {key: ("***" if key.lower() in sensitive_keys else value) for key, value in params.items()}

    @staticmethod
    def _response_error_name(response: Any) -> str:
        error_type = getattr(response, "error_type", None)
        if error_type is None:
            return "unknown"
        return str(getattr(error_type, "value", error_type)).lower()

    @staticmethod
    def _response_terminal(response: Any) -> bool:
        return bool(getattr(response, "terminal", False))

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _request_json(self, url: str, params: dict[str, Any] | None = None, operation: str = "request") -> dict[str, Any] | None:
        request_params = params if params is not None else None
        safe_params = self._safe_params(request_params or {})
        response = self.http.get(url=url, params=request_params)
        if response.payload is not None:
            return response.payload

        error_name = self._response_error_name(response)
        terminal = self._response_terminal(response)
        status_code = getattr(response, "status_code", None)
        log = logger.error if terminal else logger.warning
        log(
            "toto_request operation=%s url=%s params=%s status=%s error=%s terminal=%s",
            operation,
            url,
            safe_params,
            status_code,
            error_name,
            terminal,
        )
        return None

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _normalize_probs(cls, probs: dict[str, Any], default: float = 1.0 / 3.0) -> dict[str, float]:
        p1 = cls._safe_float(probs.get("P1"))
        px = cls._safe_float(probs.get("PX"))
        p2 = cls._safe_float(probs.get("P2"))
        total = p1 + px + p2
        if total <= 0:
            return {"P1": default, "PX": default, "P2": default}
        return {
            "P1": p1 / total,
            "PX": px / total,
            "P2": p2 / total,
        }

    @classmethod
    def _extract_quotes_triplet(
        cls,
        payload: dict[str, Any],
        candidates: list[tuple[str, str, str]],
    ) -> tuple[float, float, float, bool]:
        for k1, kx, k2 in candidates:
            v1 = payload.get(k1)
            vx = payload.get(kx)
            v2 = payload.get(k2)
            if v1 is None and vx is None and v2 is None:
                continue
            return cls._safe_float(v1), cls._safe_float(vx), cls._safe_float(v2), True
        return 0.0, 0.0, 0.0, False

    @classmethod
    def _extract_nested_quotes_triplet(
        cls,
        payload: dict[str, Any],
        container_keys: tuple[str, ...],
        candidates: list[tuple[str, str, str]],
    ) -> tuple[float, float, float, bool]:
        for container_key in container_keys:
            container = payload.get(container_key)
            if not isinstance(container, dict):
                continue
            v1, vx, v2, found = cls._extract_quotes_triplet(container, candidates)
            if found:
                return v1, vx, v2, True
        return 0.0, 0.0, 0.0, False

    @classmethod
    def _extract_pool_probs(cls, payload: dict[str, Any]) -> dict[str, float]:
        p1, px, p2, found_top = cls._extract_quotes_triplet(
            payload,
            candidates=[
                ("pool_win_1", "pool_draw", "pool_win_2"),
                ("pool_1", "pool_x", "pool_2"),
                ("pool_p1", "pool_px", "pool_p2"),
            ],
        )
        if not found_top:
            p1, px, p2, _ = cls._extract_nested_quotes_triplet(
                payload,
                container_keys=("quotes", "pool_quotes", "pool"),
                candidates=[
                    ("pool_win_1", "pool_draw", "pool_win_2"),
                    ("win_1", "draw", "win_2"),
                    ("P1", "PX", "P2"),
                ],
            )
        return {
            "P1": p1 / 100.0,
            "PX": px / 100.0,
            "P2": p2 / 100.0,
        }

    @classmethod
    def _extract_bookmaker_quotes(cls, payload: dict[str, Any]) -> dict[str, float]:
        bk1, bkx, bk2, found_top = cls._extract_quotes_triplet(
            payload,
            candidates=[
                ("bk_win_1", "bk_draw", "bk_win_2"),
                ("bookmaker_win_1", "bookmaker_draw", "bookmaker_win_2"),
            ],
        )
        if not found_top:
            bk1, bkx, bk2, _ = cls._extract_nested_quotes_triplet(
                payload,
                container_keys=("quotes", "bookmaker_quotes", "bookmaker", "bk_quotes"),
                candidates=[
                    ("bk_win_1", "bk_draw", "bk_win_2"),
                    ("win_1", "draw", "win_2"),
                    ("P1", "PX", "P2"),
                ],
            )
        return {
            "bk_win_1": bk1,
            "bk_draw": bkx,
            "bk_win_2": bk2,
        }

    @classmethod
    def _extract_pool_quotes(cls, payload: dict[str, Any]) -> dict[str, float]:
        probs = cls._extract_pool_probs(payload)
        return {
            "pool_win_1": probs["P1"] * 100.0,
            "pool_draw": probs["PX"] * 100.0,
            "pool_win_2": probs["P2"] * 100.0,
        }

    @classmethod
    def _bookmaker_probs_from_quotes(cls, quotes: dict[str, Any]) -> dict[str, float]:
        raw = {
            "P1": cls._safe_float(quotes.get("bk_win_1")) / 100.0,
            "PX": cls._safe_float(quotes.get("bk_draw")) / 100.0,
            "P2": cls._safe_float(quotes.get("bk_win_2")) / 100.0,
        }
        return cls._normalize_probs(raw)

    @classmethod
    def _extract_order(cls, payload: dict[str, Any], fallback_order: int) -> int:
        for key in ("order", "event_order", "position", "index"):
            if payload.get(key) is not None:
                return max(1, cls._safe_int(payload.get(key), default=fallback_order))
        return fallback_order

    @staticmethod
    def _extract_match_score(payload: dict[str, Any]) -> str:
        for key in ("score", "final_score", "result_score"):
            value = payload.get(key)
            if value:
                return str(value)
        home_score = payload.get("home_score")
        away_score = payload.get("away_score")
        if home_score is not None and away_score is not None:
            return f"{home_score}:{away_score}"
        return ""

    @classmethod
    def _extract_odds_aliases(cls, payload: dict[str, Any], bookmaker_probs: dict[str, float]) -> tuple[dict[str, float], str | None]:
        odds_payload = payload.get("odds")
        odds = odds_payload if isinstance(odds_payload, dict) else {}

        o1 = payload.get("odds_ft_1") or odds.get("odds_ft_1") or odds.get("O1")
        ox = payload.get("odds_ft_x") or odds.get("odds_ft_x") or odds.get("OX")
        o2 = payload.get("odds_ft_2") or odds.get("odds_ft_2") or odds.get("O2")

        if o1 is not None and ox is not None and o2 is not None:
            return {
                "O1": cls._safe_float(o1),
                "OX": cls._safe_float(ox),
                "O2": cls._safe_float(o2),
            }, None

        # Backward-compatible reconstruction path for model bridge.
        p1 = cls._safe_float(bookmaker_probs.get("P1"))
        px = cls._safe_float(bookmaker_probs.get("PX"))
        p2 = cls._safe_float(bookmaker_probs.get("P2"))
        if p1 > 0 and px > 0 and p2 > 0:
            return {
                "O1": 1.0 / p1,
                "OX": 1.0 / px,
                "O2": 1.0 / p2,
            }, "reconstructed_from_bookmaker_quotes"

        return {}, None

    @staticmethod
    def _extract_match_name(payload: dict[str, Any]) -> str:
        for key in ("name", "match_name", "label", "title"):
            value = payload.get(key)
            if value:
                return str(value)

        home = payload.get("home")
        away = payload.get("away")
        if home and away:
            return f"{home} - {away}"
        return ""

    @staticmethod
    def _extract_match_result(payload: dict[str, Any]) -> str:
        for key in ("result", "outcome", "final_result", "sign"):
            value = payload.get(key)
            if value in {"1", "X", "2"}:
                return str(value)
        return ""

    def _normalize_match(self, payload: dict[str, Any], fallback_order: int) -> dict[str, Any]:
        pool_probs = self._normalize_probs(self._extract_pool_probs(payload))
        pool_quotes = self._extract_pool_quotes(payload)
        bookmaker_quotes = self._extract_bookmaker_quotes(payload)
        bookmaker_probs = self._bookmaker_probs_from_quotes(bookmaker_quotes)
        odds_aliases, odds_source = self._extract_odds_aliases(payload, bookmaker_probs)

        has_explicit_order = any(payload.get(key) is not None for key in ("order", "event_order", "position", "index"))
        has_rich_context = any(
            payload.get(key) is not None
            for key in (
                "championship",
                "tournament",
                "start_at",
                "starts_at",
                "kickoff",
                "score",
                "final_score",
                "result_score",
                "home_score",
                "away_score",
                "bk_win_1",
                "bk_draw",
                "bk_win_2",
                "bookmaker_quotes",
                "bookmaker",
                "bk_quotes",
                "odds",
                "odds_ft_1",
                "odds_ft_x",
                "odds_ft_2",
            )
        )

        if not has_explicit_order and not has_rich_context:
            return {
                "name": self._extract_match_name(payload),
                "result": self._extract_match_result(payload),
                "pool_probs": pool_probs,
            }

        normalized = {
            "order": self._extract_order(payload, fallback_order=fallback_order),
            "name": self._extract_match_name(payload),
            "championship": str(payload.get("championship") or payload.get("tournament") or ""),
            "start_at": str(payload.get("start_at") or payload.get("starts_at") or payload.get("kickoff") or ""),
            "result": self._extract_match_result(payload),
            "score": self._extract_match_score(payload),
            "pool_quotes": pool_quotes,
            "bookmaker_quotes": bookmaker_quotes,
            "norm_pool_probs": pool_probs,
            "norm_bookmaker_probs": bookmaker_probs,
            # Backward compatibility for current UI/TOTO flow.
            "pool_probs": pool_probs,
        }

        if odds_aliases:
            normalized["odds"] = odds_aliases
            normalized["odds_ft_1"] = odds_aliases["O1"]
            normalized["odds_ft_x"] = odds_aliases["OX"]
            normalized["odds_ft_2"] = odds_aliases["O2"]
            if odds_source is not None:
                normalized["odds_source"] = odds_source

        home = payload.get("home") or payload.get("home_name") or payload.get("home_team")
        away = payload.get("away") or payload.get("away_name") or payload.get("away_team")
        if home:
            normalized["home"] = str(home)
        if away:
            normalized["away"] = str(away)

        # Do not include empty optional values to keep backward-compatible test assertions stable.
        if not normalized["championship"]:
            normalized.pop("championship")
        if not normalized["start_at"]:
            normalized.pop("start_at")
        if not normalized["result"]:
            normalized.pop("result")
        if not normalized["score"]:
            normalized.pop("score")

        return normalized

    def _normalize_payouts(self, payload: dict[str, Any]) -> dict[int, int] | None:
        payouts = payload.get("payouts")
        if not isinstance(payouts, dict):
            return None

        normalized: dict[int, int] = {}
        for key, value in payouts.items():
            try:
                hit_count = int(key)
                normalized[hit_count] = int(float(value))
            except (TypeError, ValueError):
                continue

        return normalized or None

    def _save_raw_draw(self, draw_id: int, payload: dict[str, Any]) -> None:
        path = self.data_dir / f"{draw_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _save_history_snapshot(self, name: str, payload: dict[str, Any]) -> bool:
        safe_name = str(name).strip().replace("/", "_") or "community"
        path = self.data_dir / f"history_{safe_name}.json"
        try:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return True
        except Exception:
            logger.exception("toto_history_snapshot_save_failed name=%s path=%s", name, path)
            return False

    @staticmethod
    def _resolve_history_state(
        *,
        include_history: bool,
        draws_count: int,
        events_count: int,
    ) -> tuple[str, str | None]:
        if not include_history:
            return "history_not_requested", "history_not_requested"
        if draws_count <= 0:
            return "history_requested_but_empty", "history_requested_but_empty"
        if events_count <= 0:
            return "draws_only_no_events", "draws_only_no_events"
        return "full_history_ready", None

    def _normalize_draw(self, draw_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        source_payload = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        if not isinstance(source_payload, dict):
            source_payload = {}

        matches_payload = source_payload.get("matches")
        if not isinstance(matches_payload, list):
            matches_payload = source_payload.get("events", [])
        if not isinstance(matches_payload, list):
            matches_payload = []

        draw = {
            "draw_id": int(draw_id),
            "matches": [
                self._normalize_match(match, fallback_order=index + 1)
                for index, match in enumerate(matches_payload)
                if isinstance(match, dict)
            ],
        }

        for key in ("id", "number", "status", "ended_at", "jackpot", "pool_sum"):
            value = source_payload.get(key)
            if value is not None and value != "":
                draw[key] = value

        payments = source_payload.get("payments")
        if isinstance(payments, dict) and payments:
            draw["payments"] = payments

        payouts = self._normalize_payouts(source_payload)
        if payouts is not None:
            draw["payouts"] = payouts

        logger.info("toto_draw draw_id=%s matches=%s", draw["draw_id"], len(draw["matches"]))
        return draw

    def _history_stats_from_events(self, events: list[dict[str, Any]]) -> dict[str, float]:
        if not events:
            return {
                "events_count": 0.0,
                "draw_results_rate": 0.0,
                "upset_rate": 0.0,
                "favorite_fail_rate": 0.0,
                "pool_vs_bookmaker_gap": 0.0,
            }

        draws = 0
        upsets = 0
        favorite_fails = 0
        pool_vs_bk_gap_sum = 0.0

        for event in events:
            result = str(event.get("result", "")).strip().upper()
            if result in ("DRAW", "D"):
                result = "X"
            if result == "X":
                draws += 1

            pool_1 = self._safe_float(event.get("pool_win_1"))
            pool_x = self._safe_float(event.get("pool_draw"))
            pool_2 = self._safe_float(event.get("pool_win_2"))
            bk_1 = self._safe_float(event.get("bk_win_1"))
            bk_x = self._safe_float(event.get("bk_draw"))
            bk_2 = self._safe_float(event.get("bk_win_2"))

            pool_total = pool_1 + pool_x + pool_2
            bk_total = bk_1 + bk_x + bk_2
            if pool_total <= 0 or bk_total <= 0:
                continue

            pool_probs = {
                "1": pool_1 / pool_total,
                "X": pool_x / pool_total,
                "2": pool_2 / pool_total,
            }
            bk_probs = {
                "1": bk_1 / bk_total,
                "X": bk_x / bk_total,
                "2": bk_2 / bk_total,
            }

            pool_favorite = max(("1", "X", "2"), key=lambda out: pool_probs[out])
            bk_favorite = max(("1", "X", "2"), key=lambda out: bk_probs[out])
            if result and result in ("1", "X", "2") and result != pool_favorite:
                upsets += 1
            if result and result in ("1", "X", "2") and result != bk_favorite:
                favorite_fails += 1

            pool_vs_bk_gap_sum += (
                abs(pool_probs["1"] - bk_probs["1"])
                + abs(pool_probs["X"] - bk_probs["X"])
                + abs(pool_probs["2"] - bk_probs["2"])
            )

        total = float(len(events))
        return {
            "events_count": total,
            "draw_results_rate": draws / total,
            "upset_rate": upsets / total,
            "favorite_fail_rate": favorite_fails / total,
            "pool_vs_bookmaker_gap": pool_vs_bk_gap_sum / total,
        }

    def _extract_history_event_rows(self, draw: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for match in draw.get("matches", []):
            if not isinstance(match, dict):
                continue
            rows.append(
                {
                    "drawing_id": draw.get("draw_id"),
                    "draw_number": draw.get("number"),
                    "draw_ended_at": draw.get("ended_at"),
                    "order": match.get("order"),
                    "name": match.get("name", ""),
                    "championship": match.get("championship", ""),
                    "start_at": match.get("start_at", ""),
                    "result": match.get("result", ""),
                    "score": match.get("score", ""),
                    "pool_win_1": self._safe_float(match.get("pool_quotes", {}).get("pool_win_1") if isinstance(match.get("pool_quotes"), dict) else match.get("pool_win_1")),
                    "pool_draw": self._safe_float(match.get("pool_quotes", {}).get("pool_draw") if isinstance(match.get("pool_quotes"), dict) else match.get("pool_draw")),
                    "pool_win_2": self._safe_float(match.get("pool_quotes", {}).get("pool_win_2") if isinstance(match.get("pool_quotes"), dict) else match.get("pool_win_2")),
                    "bk_win_1": self._safe_float(match.get("bookmaker_quotes", {}).get("bk_win_1") if isinstance(match.get("bookmaker_quotes"), dict) else match.get("bk_win_1")),
                    "bk_draw": self._safe_float(match.get("bookmaker_quotes", {}).get("bk_draw") if isinstance(match.get("bookmaker_quotes"), dict) else match.get("bk_draw")),
                    "bk_win_2": self._safe_float(match.get("bookmaker_quotes", {}).get("bk_win_2") if isinstance(match.get("bookmaker_quotes"), dict) else match.get("bk_win_2")),
                }
            )
        return rows

    def get_draw_history(
        self,
        name: str = "baltbet-main",
        start_page: int = 1,
        max_pages: int = 5,
        max_draws: int = 100,
        include_draw_payload: bool = True,
        expand_events: bool = True,
        persist_snapshot: bool = True,
    ) -> dict[str, Any]:
        """Load Baltbet draw history with pagination and optional event expansion.

        Returns a dictionary with:
        - drawing_name
        - draws (brief list)
        - events (flattened event rows)
        - stats (history diagnostics)
        """
        if max_pages < 1:
            max_pages = 1
        if max_draws < 1:
            max_draws = 1

        previous_snapshot = self.get_cached_draw_history(name=name)

        draws_brief: list[dict[str, Any]] = []
        page = max(1, int(start_page))
        pages_loaded = 0

        while pages_loaded < max_pages and len(draws_brief) < max_draws:
            page_rows = self.get_draws(name=name, page=page)
            pages_loaded += 1
            if not page_rows:
                break

            for row in page_rows:
                if len(draws_brief) >= max_draws:
                    break
                draw_id = self._safe_int(row.get("id"), default=0)
                if draw_id <= 0:
                    continue
                draws_brief.append(row)

            page += 1

        events: list[dict[str, Any]] = []
        expanded_draws: list[dict[str, Any]] = []
        should_expand_events = bool(include_draw_payload or expand_events)
        if should_expand_events:
            for row in draws_brief:
                draw_id = self._safe_int(row.get("id"), default=0)
                if draw_id <= 0:
                    continue
                draw = self.get_draw(draw_id)
                if include_draw_payload:
                    expanded_draws.append(draw)
                events.extend(self._extract_history_event_rows(draw))

        history_state, history_empty_reason = self._resolve_history_state(
            include_history=True,
            draws_count=len(draws_brief),
            events_count=len(events),
        )

        stats = self._history_stats_from_events(events)
        history_stats_ready = bool(stats.get("events_count", 0.0) > 0)

        live_snapshot_payload = {
            "drawing_name": str(name),
            "provider": "totobrief",
            "source": "draw_history_api",
            "history_state": history_state,
            "history_empty_reason": history_empty_reason,
            "history_stats_ready": history_stats_ready,
            "draws_loaded_count": len(draws_brief),
            "history_draws_loaded_count": len(draws_brief),
            "history_events_loaded_count": len(events),
            "pages_loaded": pages_loaded,
            "requested": {
                "start_page": int(start_page),
                "max_pages": int(max_pages),
                "max_draws": int(max_draws),
                "include_draw_payload": bool(include_draw_payload),
                "expand_events": bool(expand_events),
                "persist_snapshot": bool(persist_snapshot),
            },
            "draws": draws_brief,
            "events": events,
            "stats": stats,
        }

        snapshot_payload = self._merge_history_payloads(
            previous=previous_snapshot,
            fresh=live_snapshot_payload,
        )

        history_payload = {
            **snapshot_payload,
            "history_snapshot_saved": False,
            "history_db_saved": False,
        }
        if include_draw_payload:
            history_payload["expanded_draws"] = expanded_draws

        if persist_snapshot:
            history_payload["history_snapshot_saved"] = self._save_history_snapshot(name=name, payload=snapshot_payload)
            history_payload["history_db_saved"] = self._save_history_snapshot_to_db(name=name, payload=snapshot_payload)

        logger.info(
            "toto_history_loaded name=%s draws=%d events=%d pages=%d state=%s",
            name,
            int(snapshot_payload.get("history_draws_loaded_count", 0) or 0),
            int(snapshot_payload.get("history_events_loaded_count", 0) or 0),
            pages_loaded,
            str(snapshot_payload.get("history_state", history_state)),
        )
        return history_payload

    def get_draws(self, name: str, page: int = 1) -> list[dict[str, Any]]:
        base_url = self._require_base_url()

        payload = self._request_json(
            url=f"{base_url}/api/v1/community/{name}/drawings",
            params={"page": page},
            operation="get_draws",
        )
        if payload is None:
            return []

        rows = payload.get("drawings", payload.get("data", payload))
        if not isinstance(rows, list):
            logger.warning("toto_draws_invalid_payload name=%s page=%s payload_type=%s", name, page, type(rows).__name__)
            rows = []

        normalized: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            draw_id = self._safe_int(row.get("id"), default=0)
            if draw_id <= 0:
                logger.warning("toto_draws_skip_invalid_id name=%s page=%s raw_id=%s", name, page, row.get("id"))
                continue
            normalized.append(
                {
                    "id": draw_id,
                    "number": row.get("number"),
                    "ended_at": row.get("ended_at"),
                }
            )
            if row.get("status") is not None:
                normalized[-1]["status"] = row.get("status")
            if row.get("jackpot") is not None:
                normalized[-1]["jackpot"] = row.get("jackpot")
            if row.get("pool_sum") is not None:
                normalized[-1]["pool_sum"] = row.get("pool_sum")
        return normalized

    def get_draw(
        self,
        draw_id: int,
        include_history: bool = False,
        history_name: str = "baltbet-main",
        history_max_pages: int = 3,
        history_max_draws: int = 64,
    ) -> dict[str, Any]:
        base_url = self._require_base_url()

        payload = self._request_json(
            url=f"{base_url}/api/v1/community/drawing-info/{draw_id}",
            operation="get_draw",
        )
        if payload is None:
            logger.warning("toto_draw_unavailable draw_id=%s reason=request_failed", draw_id)
            return {"draw_id": int(draw_id), "matches": []}

        raw_draw = payload.get("drawing") if isinstance(payload.get("drawing"), dict) else payload
        if isinstance(payload.get("data"), dict):
            # Totobrief often wraps draw body in top-level data.
            raw_draw = payload.get("data")
        if not isinstance(raw_draw, dict):
            logger.warning("toto_draw_invalid_payload draw_id=%s payload_type=%s", draw_id, type(raw_draw).__name__)
            raw_draw = {}

        draw = self._normalize_draw(draw_id, raw_draw)
        self._save_raw_draw(draw_id, raw_draw)

        if include_history:
            history = self.get_draw_history(
                name=history_name,
                start_page=1,
                max_pages=history_max_pages,
                max_draws=history_max_draws,
                include_draw_payload=False,
                expand_events=True,
                persist_snapshot=True,
            )
            history_state, history_empty_reason = self._resolve_history_state(
                include_history=True,
                draws_count=self._safe_int(history.get("history_draws_loaded_count"), 0),
                events_count=self._safe_int(history.get("history_events_loaded_count"), 0),
            )
            stats = history.get("stats", {}) if isinstance(history.get("stats"), dict) else {}
            history_stats_ready = bool(self._safe_float(stats.get("events_count")) > 0)

            draw["toto_brief_history"] = {
                "drawings": history.get("draws", []),
                "events": history.get("events", []),
                "stats": stats,
                "provider": history.get("provider", "totobrief"),
                "source": history.get("source", "draw_history_api"),
                "history_draws_loaded_count": history.get("history_draws_loaded_count", 0),
                "history_events_loaded_count": history.get("history_events_loaded_count", 0),
                "history_stats_ready": history_stats_ready,
                "history_snapshot_saved": bool(history.get("history_snapshot_saved", False)),
                "history_state": history_state,
                "history_empty_reason": history_empty_reason,
            }

            # Attach global history context to each event row to support optimizer
            # in draw-mode and manual selected-draw bridge.
            history_events = draw["toto_brief_history"].get("events", [])
            history_stats = draw["toto_brief_history"].get("stats", {})
            for match in draw.get("matches", []):
                if not isinstance(match, dict):
                    continue
                match["toto_brief_history"] = {
                    "drawings": draw["toto_brief_history"].get("drawings", []),
                    "events": history_events,
                    "stats": history_stats,
                    "history_draws_loaded_count": draw["toto_brief_history"].get("history_draws_loaded_count", 0),
                    "history_events_loaded_count": draw["toto_brief_history"].get("history_events_loaded_count", 0),
                    "history_stats_ready": draw["toto_brief_history"].get("history_stats_ready", False),
                    "history_snapshot_saved": draw["toto_brief_history"].get("history_snapshot_saved", False),
                    "history_state": draw["toto_brief_history"].get("history_state"),
                    "history_empty_reason": draw["toto_brief_history"].get("history_empty_reason"),
                }
        return draw
