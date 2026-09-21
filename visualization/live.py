from __future__ import annotations

from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Sequence
from urllib.parse import urlparse

from evolution import EvaluationProgress, EvolutionConfig, GenerationStats
from world import EpisodeResult, FoodWorldConfig

from .export import RankedReplay, _ranked_replays_to_dicts, episode_to_dict


class LiveRunViewer:
    """A small loopback-only web server for watching an active experiment."""

    def __init__(
        self,
        evolution_config: EvolutionConfig,
        world_config: FoodWorldConfig,
        sample_limit: int,
        experiment: str = "find_food",
        label: str | None = None,
    ) -> None:
        if sample_limit < 1:
            raise ValueError("sample_limit must be at least one.")

        self._sample_limit = sample_limit
        self._lock = Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: Thread | None = None
        self._url: str | None = None
        self._final_snapshot_served = Event()
        self._current_scores: list[float] = []
        self._state = self._make_state(
            evolution_config,
            world_config,
            experiment,
            label,
        )
        template = Path(__file__).with_name("live_viewer_template.html").read_text(
            encoding="utf-8"
        )
        self._template = template.replace("__STATIC_STATE_JSON__", "null").encode(
            "utf-8"
        )

    @property
    def url(self) -> str:
        if self._url is None:
            raise RuntimeError("The live viewer has not been started.")
        return self._url

    def start(self) -> str:
        if self._server is not None:
            return self.url

        viewer = self

        class RequestHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
                path = urlparse(self.path).path
                if path in {"/", "/index.html"}:
                    viewer._send(self, HTTPStatus.OK, "text/html; charset=utf-8", viewer._template)
                    return
                if path == "/api/state":
                    snapshot, is_final = viewer._snapshot()
                    payload = snapshot.encode("utf-8")
                    viewer._send(self, HTTPStatus.OK, "application/json; charset=utf-8", payload)
                    if is_final:
                        viewer._final_snapshot_served.set()
                    return
                viewer._send(self, HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"Not found")

            def log_message(self, _format: str, *_args: object) -> None:
                # Browser polling is expected; keeping the terminal focused on evolution data.
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), RequestHandler)
        self._server.daemon_threads = True
        port = self._server.server_address[1]
        self._url = f"http://127.0.0.1:{port}/"
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self.url

    def begin_run(
        self,
        evolution_config: EvolutionConfig,
        world_config: FoodWorldConfig,
        *,
        experiment: str,
        label: str | None = None,
    ) -> None:
        """Reuse the open page for the next condition in a comparison."""
        with self._lock:
            self._current_scores = []
            self._final_snapshot_served.clear()
            self._state = self._make_state(
                evolution_config,
                world_config,
                experiment,
                label,
            )

    def record_evaluation(self, progress: EvaluationProgress) -> None:
        """Receive one finished candidate from the main evolution process."""
        with self._lock:
            current_generation = self._state["progress"]["generation"]
            if progress.generation != current_generation:
                self._begin_generation(progress.generation, progress.total)

            self._current_scores.append(progress.score)
            current = self._state["current"]
            if current["best_score"] is None or progress.score > current["best_score"]:
                current["best_score"] = progress.score
                current["best_eat_rate"] = progress.eat_rate
            current["average_score"] = sum(self._current_scores) / len(self._current_scores)
            self._state["phase"] = "evaluating"
            self._state["message"] = self._message(
                f"Evaluating generation {progress.generation}"
            )
            self._state["progress"] = {
                "generation": progress.generation,
                "completed": progress.completed,
                "total": progress.total,
                "workers": self._state["progress"]["workers"],
            }

            if progress.replay is not None:
                sample = {
                    "candidate_index": progress.candidate_index,
                    "worker_pid": progress.worker_pid,
                    "score": _round_float(progress.score),
                    "eat_rate": _round_float(progress.eat_rate),
                    "episode": episode_to_dict(progress.replay),
                }
                samples = [
                    item
                    for item in self._state["samples"]
                    if item["candidate_index"] != progress.candidate_index
                ]
                samples.append(sample)
                samples.sort(key=lambda item: item["candidate_index"])
                self._state["samples"] = samples[: self._sample_limit]

    def record_generation_best(
        self,
        stats: GenerationStats,
        episode: EpisodeResult,
    ) -> None:
        """Backward-compatible helper for callers that provide one winner."""
        self.record_generation_leaders(
            stats,
            [
                RankedReplay(
                    rank=1,
                    score=stats.best_score,
                    eat_rate=stats.best_eat_rate,
                    episode=episode,
                )
            ],
        )

    def record_generation_leaders(
        self,
        stats: GenerationStats,
        replays: Sequence[RankedReplay],
    ) -> None:
        """Show the highest-scoring brains after a generation is fully scored."""
        if not replays:
            raise ValueError("At least one ranked replay is required.")

        with self._lock:
            self._state["metrics"].append(asdict(stats))
            self._state["generation_leaders"] = _ranked_replays_to_dicts(
                replays[: self._sample_limit]
            )
            self._state["generation_best"] = {
                "generation": stats.generation,
                "score": _round_float(stats.best_score),
                "eat_rate": _round_float(stats.best_eat_rate),
                "episode": episode_to_dict(replays[0].episode),
            }
            self._state["phase"] = "selecting"
            self._state["message"] = self._message(
                f"Selected the top {len(self._state['generation_leaders'])} brains "
                f"from generation {stats.generation}"
            )

    def record_generation_diagnostics(
        self,
        stats: GenerationStats,
        replays: Sequence[RankedReplay],
        diagnostics: dict[str, Any],
        selected_generation: int | None = None,
    ) -> None:
        """Show one generation winner across a fixed validation room panel."""
        if not replays:
            raise ValueError("At least one validation replay is required.")

        with self._lock:
            self._state["metrics"].append(asdict(stats))
            self._state["diagnostics"].append(dict(diagnostics))
            self._state["selected_generation"] = selected_generation
            self._state["generation_leaders"] = _ranked_replays_to_dicts(
                replays[: self._sample_limit]
            )
            self._state["generation_best"] = {
                "generation": stats.generation,
                "score": _round_float(stats.best_score),
                "eat_rate": _round_float(stats.best_eat_rate),
                "episode": episode_to_dict(replays[0].episode),
            }
            self._state["phase"] = "diagnostic"
            self._state["message"] = self._message(
                f"Validating generation {stats.generation} winner in "
                f"{len(self._state['generation_leaders'])} balanced rooms"
            )

    def finish(
        self,
        best_stats: GenerationStats,
        best_episode: EpisodeResult,
        final_viewer_path: Path | None,
        *,
        ranked_replays: Sequence[RankedReplay] | None = None,
        final_label: str | None = None,
        message: str = "Training complete",
    ) -> None:
        """Leave a final snapshot in the live page before the process exits."""
        with self._lock:
            self._final_snapshot_served.clear()
            if ranked_replays is not None:
                leaders = _ranked_replays_to_dicts(
                    ranked_replays[: self._sample_limit]
                )
            else:
                leaders = list(self._state["generation_leaders"])
            if not leaders:
                leaders = _ranked_replays_to_dicts(
                    [
                        RankedReplay(
                            rank=1,
                            score=best_stats.best_score,
                            eat_rate=best_stats.best_eat_rate,
                            episode=best_episode,
                        )
                    ]
                )

            if final_label is not None:
                self._state["label"] = final_label
            self._state["phase"] = "complete"
            self._state["message"] = self._message(message)
            self._state["final"] = {
                "generation": best_stats.generation,
                "score": leaders[0]["score"],
                "eat_rate": leaders[0]["eat_rate"],
                "episode": leaders[0]["episode"],
                "leaders": leaders,
                "viewer_path": str(final_viewer_path) if final_viewer_path else None,
            }

    def wait_until_final_served(self, timeout: float = 2.0) -> bool:
        """Wait briefly so the browser receives the final state before Python exits."""
        return self._final_snapshot_served.wait(timeout)

    def close(self) -> None:
        """Stop the local server; useful for callers that keep Python alive."""
        server = self._server
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._server = None
        self._thread = None
        self._url = None

    def _begin_generation(self, generation: int, total: int) -> None:
        self._current_scores = []
        self._state["samples"] = []
        self._state["generation_leaders"] = []
        self._state["current"] = {
            "best_score": None,
            "average_score": None,
            "best_eat_rate": None,
        }
        self._state["progress"] = {
            "generation": generation,
            "completed": 0,
            "total": total,
            "workers": self._state["progress"]["workers"],
        }

    @staticmethod
    def _make_state(
        evolution_config: EvolutionConfig,
        world_config: FoodWorldConfig,
        experiment: str,
        label: str | None,
    ) -> dict[str, Any]:
        return {
            "experiment": experiment,
            "label": label,
            "phase": "starting",
            "message": _labelled_message(label, "Preparing generation 0"),
            "config": {
                "evolution": asdict(evolution_config),
                "world": asdict(world_config),
            },
            "progress": {
                "generation": 0,
                "completed": 0,
                "total": evolution_config.population_size,
                "workers": evolution_config.workers,
            },
            "current": {
                "best_score": None,
                "average_score": None,
                "best_eat_rate": None,
            },
            "metrics": [],
            "diagnostics": [],
            "selected_generation": None,
            "samples": [],
            "generation_leaders": [],
            "generation_best": None,
            "final": None,
        }

    def _message(self, message: str) -> str:
        return _labelled_message(self._state.get("label"), message)

    def _snapshot(self) -> tuple[str, bool]:
        with self._lock:
            return (
                json.dumps(self._state, separators=(",", ":")),
                self._state["phase"] == "complete",
            )

    @staticmethod
    def _send(
        handler: BaseHTTPRequestHandler,
        status: HTTPStatus,
        content_type: str,
        body: bytes,
    ) -> None:
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        handler.wfile.write(body)


def _round_float(value: float) -> float:
    return round(value, 5)


def _labelled_message(label: str | None, message: str) -> str:
    return f"{label}: {message}" if label else message
