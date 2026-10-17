"""Telemetry, execution tracing, and introspection event logger."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from rich.console import Console

console = Console()


@dataclass
class IntrospectionEvent:
    timestamp: str
    phase: str
    agent_or_component: str
    action: str
    details: Dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None


class TelemetryTracer:
    """Tracks end-to-end execution path, meta-cognitive reflections, and error corrections."""

    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.events: List[IntrospectionEvent] = []
        self._logger = logging.getLogger("metacog_finance")

    def record_event(
        self,
        phase: str,
        agent_or_component: str,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        confidence: Optional[float] = None,
    ) -> None:
        event = IntrospectionEvent(
            timestamp=datetime.now().isoformat(),
            phase=phase,
            agent_or_component=agent_or_component,
            action=action,
            details=details or {},
            confidence=confidence,
        )
        self.events.append(event)
        
        # Format for terminal / rich output
        conf_str = f" [cyan](Conf: {confidence:.2f})[/cyan]" if confidence is not None else ""
        console.print(
            f"[dim]{event.timestamp[11:19]}[/dim] [bold blue][{phase}][/bold blue] "
            f"[magenta]{agent_or_component}[/magenta] -> {action}{conf_str}"
        )

    def get_summary(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "total_events": len(self.events),
            "events": [
                {
                    "timestamp": e.timestamp,
                    "phase": e.phase,
                    "agent": e.agent_or_component,
                    "action": e.action,
                    "confidence": e.confidence,
                    "details": e.details,
                }
                for e in self.events
            ],
        }

    def export_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.get_summary(), f, indent=2)
