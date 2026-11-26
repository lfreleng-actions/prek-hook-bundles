# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Pick the blobs a lint job needs from a bundle's index.

A job running two hooks needs two environments, their toolchains and
their clones, not the whole bundle. The choice is made by comparing
the requirement each hook in the job's configuration states with the
requirement each bundled environment was built for, using the same key
the planner used, so "the same environment" means one thing in both
places.

A hook the bundle has nothing for is not an error. prek builds that
environment on the runner, exactly as it does without a bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from hookbundles import plan, values
from hookbundles.archive import TOOL_BUCKETS

if TYPE_CHECKING:
    from collections.abc import Collection


@dataclass(frozen=True)
class Selection:
    """The blobs to install, and how far they cover what was asked.

    wanted counts the hooks that need an environment, missing names
    those the bundle has none for, and environments counts the distinct
    environments chosen, which several hooks may share.
    """

    blobs: tuple[str, ...]
    environments: int
    wanted: int
    missing: tuple[str, ...]


def _entries(index: dict[str, object], key: str) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    for item in values.sequence(index.get(key)) or []:
        entry = values.mapping(item)
        if entry is not None:
            found.append(entry)
    return found


def _blob(entry: dict[str, object]) -> str:
    return values.string(entry.get("blob")) or ""


def _serves(entry: dict[str, object], hook: str) -> bool:
    """Whether an environment is the one a hook of this id would use.

    Two hooks of one repository can take different languages or
    dependencies from its manifest while a configuration says the same
    of both, so an environment built for a remote unit lists the hook
    ids it serves. A null list is a local hook's environment, which the
    requirement alone identifies.
    """
    served = entry.get("hooks")
    return served is None or hook in (values.strings(served) or ())


def select(
    index: dict[str, object], config: object, hooks: Collection[str] | None = None
) -> Selection:
    """Choose the blobs that serve a configuration.

    hooks, when given, limits the choice to those hook ids, for a job
    that runs part of the configuration.
    """
    found, _ = plan.requirements(config)
    wants = {(key, name) for key, name in found if hooks is None or name in hooks}
    environments = _entries(index, "environments")
    tools = values.mapping(index.get("tools")) or {}

    chosen: dict[str, None] = {}
    buckets: set[str] = set()
    remote: set[tuple[object, object]] = set()
    missing: list[str] = []
    for key, name in sorted(wants):
        wanted = plan.key_json(key)
        matches = [
            entry
            for entry in environments
            if entry.get("requested") == wanted and _serves(entry, name)
        ]
        if not matches:
            label = key[0] if key[0] != "local" else f"local {key[2]}"
            missing.append(" ".join((label, *key[3], f"({name})")))
        elif key[0] != "local":
            # A clone is worth fetching for a hook with an environment
            # to go with it, and for no other.
            remote.add((key[0], key[1]))
        for entry in matches:
            chosen[_blob(entry)] = None
            language = values.string(entry.get("language")) or ""
            buckets.add(TOOL_BUCKETS.get(language, ""))
    clones = [
        _blob(entry)
        for entry in _entries(index, "clones")
        if (entry.get("repo"), entry.get("rev")) in remote
    ]
    toolchains = [
        _blob(tool)
        for bucket in sorted(buckets)
        if (tool := values.mapping(tools.get(bucket))) is not None
    ]
    blobs = [name for name in (*toolchains, *clones, *chosen) if name]
    return Selection(
        blobs=tuple(dict.fromkeys(blobs)),
        environments=len(chosen),
        wanted=len(wants),
        missing=tuple(missing),
    )


def everything(index: dict[str, object]) -> tuple[str, ...]:
    """Return every blob the index names."""
    tools = values.mapping(index.get("tools")) or {}
    entries = [
        *(tool for item in tools.values() if (tool := values.mapping(item))),
        *_entries(index, "clones"),
        *_entries(index, "environments"),
    ]
    return tuple(dict.fromkeys(name for entry in entries if (name := _blob(entry))))
