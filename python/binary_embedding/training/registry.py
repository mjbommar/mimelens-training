"""Tiny component registry used by losses, schedules, optimizers, callbacks, etc.

Pattern: each component declares its discriminator string and a builder. YAML
configs say `{type: foo, ...args}`; the registry returns the right object.
This stays out of the way for callers — pydantic validates the args, the
registry only handles dispatch.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class Registry:
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, key: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def deco(fn: Callable[..., T]) -> Callable[..., T]:
            if key in self._items:
                raise KeyError(f"{self.name}: {key!r} already registered")
            self._items[key] = fn
            return fn

        return deco

    def build(self, key: str, *args: Any, **kwargs: Any) -> Any:
        if key not in self._items:
            raise KeyError(
                f"{self.name}: unknown component {key!r}; known: {sorted(self._items)}"
            )
        return self._items[key](*args, **kwargs)

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def keys(self) -> list[str]:
        return sorted(self._items)


# Module-level registries, populated by the modules that own these components.
LOSSES = Registry("losses")
SCHEDULES = Registry("schedules")
OPTIMIZERS = Registry("optimizers")
CALLBACKS = Registry("callbacks")
HEADS = Registry("heads")
