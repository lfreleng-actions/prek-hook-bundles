# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Decide whether an environment is the one its unit was asked to build.

prek chooses an environment by its marker: the hook repository and
revision, the language, the dependencies and the language version.
Whoever writes a marker therefore decides which hook an environment
serves, and a unit's marker is written inside the container that ran
the unit's install code. So a marker is evidence to be checked, never
a statement to be believed, and this module is the check.

What a marker is held to comes from places a unit could not reach:
the plan, which names the unit; the base stage's clones, whose
manifests say what a hook has when a configuration sets nothing; and
the base stage's own environments, which say which toolchain versions
the bundle carries.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from hookbundles import values
from hookbundles.archive import TOOL_BUCKETS, Rejected
from hookbundles.plan import key_json, load_yaml

if TYPE_CHECKING:
    from hookbundles.plan import Unit

# The markers prek writes: in an environment once its install succeeds,
# which it reads to decide which hook the environment serves, and in a
# clone once it is complete.
MARKERS = {"hooks": ".prek-hook.json", "repos": ".prek-repo.json"}

# A hook repository's manifest: what each of its hooks is, and the
# language and dependencies a hook has when a configuration sets none.
MANIFEST = ".pre-commit-hooks.yaml"

# The environment marker layout this code understands (prek's
# INSTALL_INFO_SCHEMA_VERSION). prek ignores an environment whose
# marker carries any other, so a bundle of them would be dead weight.
MARKER_SCHEMA = 1

# prek names an environment '<language>-' plus twenty random characters.
_ENV_DIR = re.compile(r"(python|pygrep|node|golang)-[A-Za-z0-9]{20}")

# What a hook environment is built from: its language, and its
# additional dependencies in order.
Shape = tuple[str, tuple[str, ...]]

# The shapes each hook repository's manifest gives its hooks, keyed by
# (URL, revision) and then by hook id.
Manifests = dict[tuple[str, str], dict[str, Shape]]

# The shapes a unit's environments may have, each with the hook ids it
# is the environment of, or None where any hook asking for it will do.
Shapes = dict[Shape, tuple[str, ...] | None]


@dataclass
class Group:
    """What one archive holds for one environment, clone or toolchain."""

    root: str
    name: str
    files: int = 0
    size: int = 0
    marker: bytes | None = None
    manifest: bytes | None = None
    problem: str | None = None
    children: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class Environment:
    """A hook environment accepted into the bundle."""

    group: Group
    language: str
    language_version: str
    dependencies: tuple[str, ...]
    requested: dict[str, object]
    # The hook ids this is the environment of, for a remote unit; None
    # where the requirement alone decides, as for a local hook.
    hooks: tuple[str, ...] | None
    users: int


def parse_marker(group: Group) -> dict[str, object]:
    """Parse a group's marker as a mapping, or raise Rejected."""
    try:
        marker = values.mapping(values.parse(group.marker or b""))
    except ValueError:
        raise Rejected("marker is not JSON") from None
    if marker is None:
        raise Rejected("marker is not a mapping")
    return marker


def _manifest_shapes(group: Group) -> dict[str, Shape]:
    """Return the shape a clone's manifest gives each of its hooks."""
    shapes: dict[str, Shape] = {}
    try:
        manifest = load_yaml((group.manifest or b"").decode())
    except Exception:  # noqa: BLE001 - whatever a YAML parser raises
        return shapes
    for item in values.sequence(manifest) or []:
        hook = values.mapping(item) or {}
        name = values.string(hook.get("id"))
        language = values.string(hook.get("language"))
        raw = hook.get("additional_dependencies")
        dependencies = () if raw is None else values.strings(raw)
        if name and language and dependencies is not None:
            shapes[name] = (language, dependencies)
    return shapes


def read_manifests(repos: dict[str, Group]) -> Manifests:
    """Read the manifest of every sound clone the base stage made.

    The base stage cloned these and ran nothing from them, so what a
    manifest says is the hook repository's word at that revision, and
    not something a unit's install code could have changed.
    """
    manifests: Manifests = {}
    for group in repos.values():
        if group.problem is not None or group.marker is None:
            continue
        try:
            marker = parse_marker(group)
        except Rejected:
            continue
        repo = values.string(marker.get("repo"))
        rev = values.string(marker.get("rev"))
        if repo is not None and rev is not None:
            manifests[repo, rev] = _manifest_shapes(group)
    return manifests


def expected_shapes(unit: Unit | None, manifests: Manifests) -> Shapes | None:
    """Return the shapes a unit's environments may have, and who each serves.

    None means any seeded language with no dependencies, which is a
    seed. A local unit states its language and dependencies itself, so
    its one shape serves whichever hook asks for exactly those. A
    remote unit's hooks each take the manifest's language and, where
    the unit sets no dependencies, the manifest's dependencies; two
    hooks of one repository can differ in either, so each shape lists
    the hook ids it is the environment of.
    """
    if unit is None:
        return None
    if unit.local:
        return {(unit.language, unit.dependencies): None}
    manifest = manifests.get((unit.repo, unit.rev), {})
    served: dict[Shape, list[str]] = {}
    for name in unit.hooks:
        if name in manifest:
            language, dependencies = manifest[name]
            shape = (unit.language or language, unit.dependencies or dependencies)
            served.setdefault(shape, []).append(name)
    return {shape: tuple(sorted(names)) for shape, names in served.items()}


@dataclass(frozen=True)
class Base:
    """What the trusted base stage established, which units are held to.

    tools is the toolchains it installed. versions is the version of
    each language's toolchain, as the seed environments record it: a
    bundle carries one toolchain a language, so an environment built
    with any other version is one the base stage did not provide for.
    """

    home: str
    tools: dict[str, Group]
    versions: dict[str, str]


def _check_toolchain(marker: dict[str, object], language: str, base: Base) -> None:
    """Require the toolchain to be one the base stage put in the store."""
    bucket = TOOL_BUCKETS[language]
    prefix = f"{base.home}/tools/{bucket}/"
    toolchain = values.string(marker.get("toolchain")) or ""
    if not toolchain.startswith(prefix) or posixpath.normpath(toolchain) != toolchain:
        raise Rejected("marker toolchain is not in the bundle's tools/")
    entry = toolchain.removeprefix(prefix).split("/", 1)[0]
    if bucket not in base.tools or entry not in base.tools[bucket].children:
        raise Rejected("marker toolchain is not one the base stage installed")


def _requested(unit: Unit | None, language: str) -> dict[str, object]:
    """Return the requirement an environment answers, as the plan keys it.

    A lint job computes the same key from its own configuration, and
    that equality is how it picks the blobs to fetch.
    """
    if unit is None:
        # A seed: what a local hook with no dependencies asks for.
        return key_json(("local", "", language, ()))
    return key_json((unit.repo, unit.rev, unit.language, unit.dependencies))


def check_marker(
    group: Group,
    base: Base,
    expected: tuple[Unit | None, Shapes | None],
) -> Environment:
    """Accept an environment for a unit, or raise Rejected.

    expected is the unit and the shapes its environments may have, from
    expected_shapes. The unit is None for a seed environment, which
    the base stage built from a local hook with no dependencies.
    """
    unit, shapes = expected
    if not _ENV_DIR.fullmatch(group.name):
        raise Rejected("directory is not named as prek names environments")
    marker = parse_marker(group)
    if marker.get("schema_version") != MARKER_SCHEMA:
        raise Rejected("marker schema is not one this builder knows")
    if marker.get("env_path") != f"{base.home}/hooks/{group.name}":
        raise Rejected("marker env_path is not the directory it is in")
    extra = values.mapping(marker.get("extra"))
    if extra is None or not all(isinstance(item, str) for item in extra.values()):
        raise Rejected("marker extra is not a mapping of strings")
    language = values.string(marker.get("language")) or ""
    if language not in TOOL_BUCKETS or not group.name.startswith(language + "-"):
        raise Rejected("marker language does not match the directory")
    _check_toolchain(marker, language, base)
    version = values.string(marker.get("language_version"))
    dependencies = values.strings(marker.get("dependencies"))
    if not version or dependencies is None:
        raise Rejected("marker language_version or dependencies is malformed")
    if unit is not None and version != base.versions.get(language):
        raise Rejected("marker language_version is not the base toolchain's")

    remote = unit is not None and not unit.local
    identity: object = {"url": unit.repo, "rev": unit.rev} if unit and remote else None
    if marker.get("repo") != identity:
        raise Rejected("marker names a repository other than the unit's")
    # prek selects an environment by its marker's language and
    # dependencies as well as its repository. Hold both to what the
    # unit's hooks ask for, so that a unit cannot supply the
    # environment of another unit of the same repository either.
    allowed: Shapes = {(language, ()): None} if shapes is None else shapes
    if (language, dependencies) not in allowed:
        raise Rejected("marker language or dependencies are not the unit's")
    return Environment(
        group=group,
        language=language,
        language_version=version,
        dependencies=dependencies,
        requested=_requested(unit, language),
        hooks=allowed[language, dependencies],
        users=0 if unit is None else len(unit.users),
    )
