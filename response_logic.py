"""
response_logic.py
-------------------
Takes the model's live prediction for one GNSS epoch and decides what the
UAV's navigation system should DO about it. This is the "response" half of
the project — detection alone doesn't protect a UAV, the system also needs
to react sensibly when it sees jamming or spoofing.

Design, kept intentionally simple so you can explain every line:

  genuine -> trust it. Pass the reading through unchanged, update the
             "last known good" position.

  jammed  -> the signal is unreliable/degraded (not fake, just noisy/lost).
             Raise an alert. Since we still don't trust the reading, fall
             back to a DEAD-RECKONING estimate: assume the UAV kept moving
             in roughly the direction/speed it was last confidently moving,
             and extrapolate a short distance from the last good position.
             This is a standard, simple fallback used in real GNSS-denied
             navigation research (not our invention) — good enough to keep
             a UAV roughly on course for a few seconds while jamming clears.

  spoofed -> the signal LOOKS valid but is fake. Unlike jamming, extrapolating
             from it would be actively wrong (a good-looking fake reading
             has no useful information for dead reckoning). So instead:
             raise an alert AND discard the reading entirely, holding
             position at the last TRUSTED (genuine) fix until a genuine
             reading returns.

This module has no GNSS-specific dependencies beyond numbers in/out, so it
can be dropped straight into tomorrow's Streamlit GUI.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PositionEstimate:
    lat: float
    lon: float
    alt_m: float
    velocity_lat: float = 0.0   # degrees/epoch, estimated from recent genuine fixes
    velocity_lon: float = 0.0


@dataclass
class ResponseResult:
    label: str                  # "genuine" / "jammed" / "spoofed"
    alert: bool                 # True if the UAV's operator should be notified
    action_taken: str           # human-readable description of what happened
    position_used: PositionEstimate  # the position the navigation system should act on
    trusted: bool                # True only if this epoch's raw reading was actually used


class AntiJammingResponder:
    """
    Stateful responder: remembers the last trusted (genuine) position and
    a simple velocity estimate, so it can dead-reckon through jamming and
    hold position through spoofing.
    """

    def __init__(self, initial_position: Optional[PositionEstimate] = None):
        # Start with a placeholder if no real first fix is available yet.
        self.last_trusted_position = initial_position or PositionEstimate(lat=0.0, lon=0.0, alt_m=0.0)
        self.consecutive_jammed_epochs = 0  # tracks how long we've been dead-reckoning

    def handle(self, prediction_label: str, raw_lat: float = None, raw_lon: float = None,
               raw_alt_m: float = None) -> ResponseResult:
        """
        prediction_label: the model's output for this epoch — "genuine",
        "jammed", or "spoofed".
        raw_lat/raw_lon/raw_alt_m: the position reading that came with this
        epoch (only meaningful/trusted if prediction_label == "genuine").
        """
        if prediction_label == "genuine":
            return self._handle_genuine(raw_lat, raw_lon, raw_alt_m)
        elif prediction_label == "jammed":
            return self._handle_jammed()
        elif prediction_label == "spoofed":
            return self._handle_spoofed()
        else:
            raise ValueError(f"Unknown prediction label: {prediction_label!r}")

    # -----------------------------------------------------------------
    def _handle_genuine(self, lat, lon, alt_m) -> ResponseResult:
        """Trust it. Update velocity estimate from the last trusted fix so
        we have something to dead-reckon with if jamming hits next."""
        prev = self.last_trusted_position
        if lat is not None and lon is not None:
            velocity_lat = lat - prev.lat
            velocity_lon = lon - prev.lon
            new_position = PositionEstimate(
                lat=lat, lon=lon, alt_m=alt_m if alt_m is not None else prev.alt_m,
                velocity_lat=velocity_lat, velocity_lon=velocity_lon,
            )
        else:
            new_position = prev  # no position data available, keep as-is

        self.last_trusted_position = new_position
        self.consecutive_jammed_epochs = 0

        return ResponseResult(
            label="genuine",
            alert=False,
            action_taken="Reading trusted. Position and velocity estimate updated normally.",
            position_used=new_position,
            trusted=True,
        )

    def _handle_jammed(self) -> ResponseResult:
        """Don't trust the raw reading. Extrapolate forward from the last
        trusted position using the last known velocity (dead reckoning).
        The longer jamming persists, the more this estimate should be
        treated as increasingly uncertain — we track that with a counter
        so a GUI can show 'confidence decaying' rather than pretending the
        estimate stays perfectly accurate forever."""
        self.consecutive_jammed_epochs += 1
        prev = self.last_trusted_position

        extrapolated = PositionEstimate(
            lat=prev.lat + prev.velocity_lat,
            lon=prev.lon + prev.velocity_lon,
            alt_m=prev.alt_m,
            velocity_lat=prev.velocity_lat,
            velocity_lon=prev.velocity_lon,
        )
        # Note: we deliberately do NOT overwrite self.last_trusted_position
        # here — dead-reckoned guesses must never become the new "trusted"
        # baseline, or errors would compound epoch after epoch.

        return ResponseResult(
            label="jammed",
            alert=True,
            action_taken=(
                f"Signal jammed — reading discarded. Falling back to dead-reckoning "
                f"estimate (consecutive jammed epochs: {self.consecutive_jammed_epochs})."
            ),
            position_used=extrapolated,
            trusted=False,
        )

    def _handle_spoofed(self) -> ResponseResult:
        """A spoofed reading looks valid but carries no real information —
        extrapolating from it would just propagate a fake trajectory. So we
        hold entirely still at the last trusted position instead."""
        return ResponseResult(
            label="spoofed",
            alert=True,
            action_taken="Signal spoofed — reading rejected. Holding last trusted position.",
            position_used=self.last_trusted_position,
            trusted=False,
        )
