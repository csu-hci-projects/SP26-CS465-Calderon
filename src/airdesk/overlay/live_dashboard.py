"""Resizable live diagnostic dashboard rendering."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from airdesk.state.types import NormalizedHand

DrawHandOverlay = Callable[[Any, NormalizedHand, int, int, tuple[str, ...]], None]
DrawAlertBanner = Callable[[Any, str], None]


@dataclass(frozen=True)
class LiveDashboardRenderer:
    """Draw a resizable webcam-plus-diagnostics dashboard on an OpenCV canvas."""

    cv2: Any
    mirror: bool
    canvas_width: int
    canvas_height: int

    def render(
        self,
        *,
        image: Any,
        hands: tuple[NormalizedHand, ...],
        dashboard: dict[str, Any],
        status: str,
        candidates: dict[str, tuple[str, ...]],
        draw_hand_overlay: DrawHandOverlay,
        draw_alert_banner: DrawAlertBanner,
    ) -> Any:
        """Return a dashboard canvas with camera, evidence, timing, and event history."""
        import numpy as np

        canvas_width = max(960, int(self.canvas_width))
        canvas_height = max(620, int(self.canvas_height))
        canvas = np.full((canvas_height, canvas_width, 3), (18, 20, 24), dtype=image.dtype)

        self._draw_header(canvas, dashboard=dashboard, status=status, hands=hands)

        margin = 18
        gutter = 18
        header_height = 76
        footer_height = 138
        side_width = min(390, max(320, int(canvas_width * 0.32)))
        camera_width = canvas_width - side_width - margin * 2 - gutter
        camera_height = canvas_height - header_height - footer_height - margin * 2
        camera_x = margin
        camera_y = header_height + margin

        source = self.cv2.flip(image, 1) if self.mirror else image
        display_image, offset_x, offset_y = self._fit_image_to_box(
            source,
            width=camera_width,
            height=camera_height,
        )
        display_height, display_width = display_image.shape[:2]
        for hand in hands:
            draw_hand_overlay(
                display_image,
                hand,
                display_width,
                display_height,
                candidates.get(hand.hand_id, ()),
            )
        if status:
            draw_alert_banner(display_image, status)

        x0 = camera_x + offset_x
        y0 = camera_y + offset_y
        canvas[y0 : y0 + display_height, x0 : x0 + display_width] = display_image
        self.cv2.rectangle(
            canvas,
            (camera_x - 1, camera_y - 1),
            (camera_x + camera_width + 1, camera_y + camera_height + 1),
            (58, 64, 76),
            1,
        )

        side_x = camera_x + camera_width + gutter
        self._draw_side_panel(
            canvas,
            dashboard=dashboard,
            x=side_x,
            y=camera_y,
            width=side_width,
            height=camera_height,
        )
        self._draw_event_log(
            canvas,
            dashboard=dashboard,
            x=margin,
            y=camera_y + camera_height + gutter,
            width=canvas_width - margin * 2,
            height=footer_height - gutter,
        )
        return canvas

    def _fit_image_to_box(
        self,
        image: Any,
        *,
        width: int,
        height: int,
    ) -> tuple[Any, int, int]:
        source_height, source_width = image.shape[:2]
        scale = min(width / source_width, height / source_height)
        fitted_width = max(1, int(source_width * scale))
        fitted_height = max(1, int(source_height * scale))
        resized = self.cv2.resize(image, (fitted_width, fitted_height))
        return resized, (width - fitted_width) // 2, (height - fitted_height) // 2

    def _draw_header(
        self,
        image: Any,
        *,
        dashboard: dict[str, Any],
        status: str,
        hands: tuple[NormalizedHand, ...],
    ) -> None:
        width = image.shape[1]
        self.cv2.rectangle(image, (0, 0), (width, 76), (28, 31, 38), -1)
        title = str(dashboard.get("title", "AirDesk live dashboard"))
        mode = "mirror" if self.mirror else "camera"
        subtitle = str(
            dashboard.get(
                "subtitle",
                f"{mode} | hands={len(hands)} | q/esc quits",
            )
        )
        self._put_text_fit(
            image=image,
            text=title,
            x=18,
            y=31,
            max_width=max(160, width - 36),
            scale=0.72,
            color=(245, 247, 252),
            thickness=2,
        )
        self._put_text_fit(
            image=image,
            text=subtitle,
            x=18,
            y=60,
            max_width=max(160, width - 36),
            scale=0.52,
            color=(185, 195, 210),
            thickness=1,
        )
        alert = str(dashboard.get("alert", ""))
        if alert:
            self._draw_status_pill(image, text=alert, right=width - 18, y=26)
        elif "GESTURE " in status:
            self._draw_status_pill(
                image,
                text=status.split("GESTURE ", maxsplit=1)[1].strip(),
                right=width - 18,
                y=26,
            )

    def _draw_side_panel(
        self,
        image: Any,
        *,
        dashboard: dict[str, Any],
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> None:
        self.cv2.rectangle(image, (x, y), (x + width, y + height), (27, 30, 37), -1)
        self.cv2.rectangle(image, (x, y), (x + width, y + height), (58, 64, 76), 1)
        self._put_text_fit(
            image=image,
            text="TCN v2 evidence",
            x=x + 16,
            y=y + 30,
            max_width=width - 32,
            scale=0.55,
            color=(238, 242, 248),
            thickness=2,
        )
        line_y = y + 58
        for line in dashboard.get("summary_lines", [])[:4]:
            self._put_text_fit(
                image=image,
                text=str(line),
                x=x + 16,
                y=line_y,
                max_width=width - 32,
                scale=0.43,
                color=(178, 188, 204),
                thickness=1,
            )
            line_y += 24

        card_y = max(line_y + 8, y + 140)
        for hand in list(dashboard.get("hands", []))[:2]:
            if not isinstance(hand, dict):
                continue
            card_height = 190
            if card_y + card_height > y + height - 74:
                break
            self._draw_hand_card(
                image,
                hand=hand,
                x=x + 14,
                y=card_y,
                width=width - 28,
                height=card_height,
            )
            card_y += card_height + 12

        timing = dashboard.get("timing", {})
        if isinstance(timing, dict) and timing:
            self._put_text_fit(
                image=image,
                text=str(timing.get("line", "")),
                x=x + 16,
                y=y + height - 54,
                max_width=width - 32,
                scale=0.42,
                color=(160, 176, 198),
                thickness=1,
            )

    def _draw_hand_card(
        self,
        image: Any,
        *,
        hand: dict[str, Any],
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> None:
        self.cv2.rectangle(image, (x, y), (x + width, y + height), (36, 40, 49), -1)
        self.cv2.rectangle(image, (x, y), (x + width, y + height), (72, 80, 96), 1)
        label = str(hand.get("hand_id", "hand"))
        dx = hand.get("dx")
        dx_text = "" if dx is None else f" dx={float(dx):.2f}"
        self._put_text_fit(
            image=image,
            text=f"{label}{dx_text}",
            x=x + 12,
            y=y + 26,
            max_width=width - 24,
            scale=0.48,
            color=(235, 238, 245),
            thickness=2,
        )
        meters = (
            ("L", float(hand.get("left", 0.0)), (80, 170, 255)),
            ("R", float(hand.get("right", 0.0)), (75, 215, 120)),
            ("I", float(hand.get("intent", 0.0)), (245, 190, 80)),
            ("S", float(hand.get("start", 0.0)), (220, 120, 250)),
            ("E", float(hand.get("end", 0.0)), (120, 210, 235)),
        )
        meter_y = y + 48
        for name, value, color in meters:
            self._draw_meter(
                image,
                label=name,
                value=value,
                x=x + 12,
                y=meter_y,
                width=width - 24,
                color=color,
            )
            meter_y += 20
        evidence = hand.get("evidence", [])
        if isinstance(evidence, list) and evidence:
            top_line = self._evidence_summary_line(evidence)
            self._put_text_fit(
                image=image,
                text=top_line,
                x=x + 12,
                y=meter_y + 2,
                max_width=width - 24,
                scale=0.34,
                color=(200, 210, 228),
                thickness=1,
            )
            meter_y += 20
        features = hand.get("features", {})
        if isinstance(features, dict):
            motion_lines = self._motion_feature_lines(features)
            feature_y = meter_y + 6
            for line in motion_lines[:2]:
                self._put_text_fit(
                    image=image,
                    text=line,
                    x=x + 12,
                    y=feature_y,
                    max_width=width - 24,
                    scale=0.34,
                    color=(160, 174, 194),
                    thickness=1,
                )
                feature_y += 18

    def _evidence_summary_line(self, evidence: list[Any]) -> str:
        parts: list[str] = []
        for item in evidence[:4]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", ""))
            score = item.get("score")
            if not name or not isinstance(score, int | float):
                continue
            parts.append(f"{name} {float(score):.2f}")
        return "top " + " | ".join(parts) if parts else "top"

    def _motion_feature_lines(self, features: dict[str, Any]) -> tuple[str, ...]:
        def value(name: str) -> float:
            return float(features.get(name, 0.0))

        return (
            f"pos={value('palm_x'):.2f},{value('palm_y'):.2f} scale={value('hand_scale'):.2f}",
            (
                f"dx={value('dx_scale'):.2f} raw={value('dx_raw'):.2f} "
                f"vx={value('peak_vx'):.2f} c={value('consistency'):.2f}"
            ),
        )

    def _draw_meter(
        self,
        image: Any,
        *,
        label: str,
        value: float,
        x: int,
        y: int,
        width: int,
        color: tuple[int, int, int],
    ) -> None:
        value = min(1.0, max(0.0, value))
        label_width = 30
        bar_x = x + label_width
        bar_width = max(20, width - label_width - 48)
        self._put_text_fit(
            image=image,
            text=label,
            x=x,
            y=y + 12,
            max_width=label_width - 4,
            scale=0.36,
            color=(210, 216, 226),
            thickness=1,
        )
        self.cv2.rectangle(image, (bar_x, y), (bar_x + bar_width, y + 10), (58, 63, 75), -1)
        fill_width = int(bar_width * value)
        if fill_width > 0:
            self.cv2.rectangle(image, (bar_x, y), (bar_x + fill_width, y + 10), color, -1)
        self._put_text_fit(
            image=image,
            text=f"{value:.2f}",
            x=bar_x + bar_width + 8,
            y=y + 12,
            max_width=42,
            scale=0.34,
            color=(190, 200, 214),
            thickness=1,
        )

    def _draw_event_log(
        self,
        image: Any,
        *,
        dashboard: dict[str, Any],
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> None:
        self.cv2.rectangle(image, (x, y), (x + width, y + height), (27, 30, 37), -1)
        self.cv2.rectangle(image, (x, y), (x + width, y + height), (58, 64, 76), 1)
        self._put_text_fit(
            image=image,
            text="Recent decoded gestures",
            x=x + 14,
            y=y + 28,
            max_width=width - 28,
            scale=0.48,
            color=(238, 242, 248),
            thickness=2,
        )
        recent = list(dashboard.get("recent_candidates", []))
        if not recent:
            self._put_text_fit(
                image=image,
                text="none yet",
                x=x + 14,
                y=y + 58,
                max_width=width - 28,
                scale=0.44,
                color=(150, 160, 176),
                thickness=1,
            )
            return
        line_y = y + 58
        for item in recent[-4:]:
            if not isinstance(item, dict):
                continue
            line = (
                f"emit={float(item.get('emitted', 0.0)):6.2f}s "
                f"peak={float(item.get('peak', 0.0)):6.2f}s "
                f"delay={float(item.get('delay', 0.0)):4.2f}s "
                f"{item.get('hand_id', 'hand')} {item.get('name', '')} "
                f"{float(item.get('confidence', 0.0)):.2f}"
            )
            self._put_text_fit(
                image=image,
                text=line,
                x=x + 14,
                y=line_y,
                max_width=width - 28,
                scale=0.42,
                color=(190, 202, 218),
                thickness=1,
            )
            line_y += 24

    def _draw_status_pill(self, image: Any, *, text: str, right: int, y: int) -> None:
        scale = 0.5
        thickness = 2
        fitted = self._fit_text(text, max_width=310, scale=scale, thickness=thickness)
        text_width = self._text_width(fitted, scale=scale, thickness=thickness)
        x0 = max(18, right - text_width - 28)
        self.cv2.rectangle(image, (x0, y - 20), (right, y + 12), (0, 95, 220), -1)
        self.cv2.rectangle(image, (x0, y - 20), (right, y + 12), (90, 170, 255), 1)
        self.cv2.putText(
            image,
            fitted,
            (x0 + 14, y + 3),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            thickness,
            self.cv2.LINE_AA,
        )

    def _put_text_fit(
        self,
        *,
        image: Any,
        text: str,
        x: int,
        y: int,
        max_width: int,
        scale: float,
        color: tuple[int, int, int],
        thickness: int = 2,
    ) -> None:
        fitted = self._fit_text(text, max_width=max_width, scale=scale, thickness=thickness)
        self.cv2.putText(
            image,
            fitted,
            (x, y),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            self.cv2.LINE_AA,
        )

    def _fit_text(self, text: str, *, max_width: int, scale: float, thickness: int) -> str:
        if self._text_width(text, scale=scale, thickness=thickness) <= max_width:
            return text
        suffix = "..."
        low = 0
        high = max(0, len(text) - len(suffix))
        best = suffix
        while low <= high:
            middle = (low + high) // 2
            candidate = text[:middle].rstrip() + suffix
            if self._text_width(candidate, scale=scale, thickness=thickness) <= max_width:
                best = candidate
                low = middle + 1
            else:
                high = middle - 1
        return best

    def _text_width(self, text: str, *, scale: float, thickness: int) -> int:
        size, _baseline = self.cv2.getTextSize(
            text,
            self.cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            thickness,
        )
        return int(size[0])
