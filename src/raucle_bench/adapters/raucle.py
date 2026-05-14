"""Adapter for the canonical ``raucle-detect`` Python library."""

from __future__ import annotations

from raucle_bench.adapter import Prediction


class RaucleAdapter:
    """Wraps :class:`raucle_detect.Scanner`.

    Constructor *mode* maps directly to raucle's sensitivity setting
    (``strict`` / ``standard`` / ``permissive``). The adapter's ``name``
    embeds the mode so a single benchmark run can compare configurations.
    """

    def __init__(self, mode: str = "standard") -> None:
        self.mode = mode
        self._scanner = None  # set in setup()
        # Late import so raucle-bench is installable without raucle-detect.
        try:
            import raucle_detect  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "raucle-detect is required for the RaucleAdapter. "
                "Install with: pip install 'raucle-bench[raucle]'"
            ) from exc

    @property
    def name(self) -> str:
        return f"raucle-detect-{self.mode}"

    @property
    def version(self) -> str:
        try:
            from raucle_detect import __version__

            return __version__
        except ImportError:
            return "unknown"

    def setup(self) -> None:
        from raucle_detect import Scanner

        self._scanner = Scanner(mode=self.mode)

    def teardown(self) -> None:
        self._scanner = None

    def predict(self, prompt: str) -> Prediction:
        if self._scanner is None:
            return Prediction(action="ALLOW", error="adapter not set up")
        try:
            result = self._scanner.scan(prompt)
        except Exception as exc:
            return Prediction(action="ALLOW", error=str(exc))
        detail = result.attack_technique or ",".join(result.categories) or ""
        return Prediction(
            action=result.action,
            confidence=result.confidence,
            detail=detail,
        )
