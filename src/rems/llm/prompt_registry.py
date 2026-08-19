from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    """Versioned prompt pair for a single LLM task."""

    task_type: str
    version: str
    system: str
    user: str


class PromptRegistry:
    """Small registry for versioned prompt templates.

    The current code can keep importing constants from ``prompts.py``. This
    registry is the stable extension point for A/B tests, prompt version
    rollbacks, and task-specific prompt packs.
    """

    def __init__(self):
        self._templates: dict[tuple[str, str], PromptTemplate] = {}
        self._defaults: dict[str, str] = {}

    def register(self, template: PromptTemplate, *, make_default: bool = False) -> None:
        key = (template.task_type, template.version)
        self._templates[key] = template
        if make_default or template.task_type not in self._defaults:
            self._defaults[template.task_type] = template.version

    def get(self, task_type: str, version: str | None = None) -> PromptTemplate:
        selected_version = version or self._defaults.get(task_type)
        if selected_version is None:
            raise KeyError(f"No prompt registered for task type {task_type!r}")
        key = (task_type, selected_version)
        if key not in self._templates:
            raise KeyError(f"No prompt registered for task type {task_type!r} version {selected_version!r}")
        return self._templates[key]

    def default_version(self, task_type: str) -> str | None:
        return self._defaults.get(task_type)
