# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Check what the sandbox produced, and write the bundle.

This is the one place that decides what a bundle contains, and it runs
nothing from the archives it reads: members are checked and copied
from tar to tar, never extracted. See docs/DESIGN.md, "Assembly".

What it takes from where:

  * toolchains, clones and the seed environments come from the base
    stage's export alone, whatever a unit's archive holds under the
    same names;
  * from a unit's archive it takes environment directories the base
    stage did not have, and only those whose marker names the unit's
    own repository and revision, and the language and dependencies
    its hooks ask for. Where the unit states no dependencies, those
    are the ones the hook repository's manifest gives, read from the
    base stage's clone. A unit's install code can put anything it
    likes in its own environment. It cannot supply an environment
    that prek would use for some other hook.

A bundle is an index and a directory of blobs. Each blob is one group
(a toolchain, a clone or an environment) as a tar of store-relative
paths, named by the SHA-256 of its bytes. The index records which blob
holds what, so a lint job fetches the few its hooks need, and one
signature over the index covers every blob it names.
"""

from __future__ import annotations

import hashlib
import subprocess
import tarfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hookbundles import SCHEMA, values
from hookbundles.archive import (
    Layout,
    Rejected,
    check_member,
    clean_name,
    group_of,
    reach,
    sanitised,
)
from hookbundles.markers import (
    MANIFEST,
    MARKERS,
    Base,
    Environment,
    Group,
    check_marker,
    expected_shapes,
    parse_marker,
    read_manifests,
)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import IO

    from hookbundles.markers import Shapes
    from hookbundles.plan import Plan, Unit
    from hookbundles.settings import Settings

_MAX_MARKER_BYTES = 64 * 1024
_MAX_MANIFEST_BYTES = 1024 * 1024
_CHUNK = 1024 * 1024


class AssemblyError(Exception):
    """The trusted base export is not what the builder itself made."""


def scan(path: Path, root: str, home: str) -> dict[str, Group]:
    """Read an archive of one store directory, group by group.

    A member that breaks a rule marks its group, and the rest of that
    group is then skipped. A member that cannot be placed in any group
    raises Rejected for the archive as a whole.
    """
    groups: dict[str, Group] = {}
    layout = Layout()
    with tarfile.open(path, "r|") as tar:
        for member in tar:
            name = clean_name(member.name)
            if name == root and member.isdir():
                continue
            placed = group_of(name)
            if placed is None or placed[0] != root:
                raise Rejected(f"{name}: member is outside {root}/")
            group = groups.setdefault(placed[1], Group(root, placed[1]))
            parts = name.split("/")
            if len(parts) == 3:
                group.children.add(parts[2])
            if group.problem is not None:
                continue
            try:
                _ = check_member(member, home, reach(root, group.name))
                layout.add(name, directory=member.isdir())
            except Rejected as error:
                group.problem = str(error)
                continue
            group.files += 1
            if not member.isreg():
                continue
            group.size += member.size
            base = f"{root}/{group.name}/"
            if name == base + str(MARKERS.get(root)):
                if member.size > _MAX_MARKER_BYTES:
                    group.problem = f"{name}: marker is implausibly large"
                    continue
                handle = tar.extractfile(member)
                group.marker = handle.read() if handle else None
            elif root == "repos" and name == base + MANIFEST:
                if member.size <= _MAX_MANIFEST_BYTES:
                    handle = tar.extractfile(member)
                    group.manifest = handle.read() if handle else None
    return groups


def _accept_unit(
    path: Path,
    base: Base,
    taken: set[str],
    expected: tuple[Unit | None, Shapes | None],
) -> list[Environment]:
    """Return the environments a unit's archive contributes."""
    groups = scan(path, "hooks", base.home)
    accepted: list[Environment] = []
    for name, group in sorted(groups.items()):
        if name in taken or group.marker is None and group.problem is None:
            # The base stage's own, or a directory prek never finished.
            continue
        if group.problem is not None:
            raise Rejected(group.problem)
        accepted.append(check_marker(group, base, expected))
    if not accepted:
        raise Rejected("the build produced no environment")
    return accepted


def _copy_group(stream: IO[bytes], path: Path, group: Group, home: str) -> None:
    """Write one group of an archive to a stream, as a tar of its own."""
    wanted = (group.root, group.name)
    with (
        tarfile.open(fileobj=stream, mode="w|", format=tarfile.PAX_FORMAT) as out,
        tarfile.open(path, "r|") as tar,
    ):
        for member in tar:
            name = clean_name(member.name)
            if group_of(name) != wanted:
                continue
            # Checked in the scan already; a failure here is a bug.
            _ = check_member(member, home, reach(*wanted))
            clean = sanitised(member, name)
            out.addfile(clean, tar.extractfile(member) if member.isreg() else None)


@dataclass(frozen=True)
class Writer:
    """Write groups as blobs under a directory.

    level is the zstd compression level. Zero writes plain tar, which
    the tests use so that they need no zstd.
    """

    blobs: Path
    home: str
    level: int

    def write(self, path: Path, group: Group) -> dict[str, object]:
        """Write one group as a blob, and return its index entry."""
        partial = self.blobs / f".partial-{group.root}-{group.name}"
        with partial.open("wb") as raw:
            if self.level == 0:
                _copy_group(raw, path, group, self.home)
            else:
                command = ["zstd", "--quiet", "-T0", f"-{self.level}", "--stdout"]
                with subprocess.Popen(  # noqa: S603
                    command, stdin=subprocess.PIPE, stdout=raw
                ) as process:
                    if process.stdin is None:
                        raise AssemblyError("zstd gave no input stream")
                    _copy_group(process.stdin, path, group, self.home)
                    process.stdin.close()
                    if process.wait() != 0:
                        raise AssemblyError("zstd could not compress a blob")
        digest = hashlib.sha256()
        with partial.open("rb") as handle:
            while chunk := handle.read(_CHUNK):
                digest.update(chunk)
        name = digest.hexdigest() + (".tar" if self.level == 0 else ".tar.zst")
        target = partial.replace(self.blobs / name)
        return {
            "blob": name,
            "size": target.stat().st_size,
            "files": group.files,
            "bytes": group.size,
        }


def _status(build: dict[str, object], uid: str) -> str:
    units = values.mapping(build.get("units")) or {}
    record = values.mapping(units.get(uid)) or {}
    return values.string(record.get("status")) or "not built"


def _clones(
    repos: dict[str, Group], path: Path, writer: Writer
) -> list[dict[str, object]]:
    """Write a blob for each sound clone, keyed by repository and rev."""
    clones: list[dict[str, object]] = []
    for name, group in sorted(repos.items()):
        if group.problem is not None or group.marker is None:
            continue
        try:
            marker = parse_marker(group)
        except Rejected:
            continue
        repo = values.string(marker.get("repo"))
        rev = values.string(marker.get("rev"))
        if repo is not None and rev is not None:
            entry = writer.write(path, group)
            clones.append({"repo": repo, "rev": rev, "directory": name, **entry})
    return clones


def assemble(
    settings: Settings,
    plan: Plan,
    build: dict[str, object],
    work: Path,
    writer: Writer,
) -> dict[str, object]:
    """Write the blobs through writer, and return the bundle's index."""
    home = settings.prek_home
    base = work / "base"
    writer.blobs.mkdir(parents=True, exist_ok=True)

    tools = scan(base / "tools.tar", "tools", home)
    repos = scan(base / "repos.tar", "repos", home)
    seeds = scan(base / "hooks.tar", "hooks", home)
    for group in [*tools.values(), *seeds.values()]:
        if group.problem is not None:
            raise AssemblyError(f"base export: {group.problem}")

    accepted: list[tuple[Path, str, Environment]] = []
    seeding = Base(home, tools, {})
    for group in seeds.values():
        if group.marker is None:
            continue
        try:
            environment = check_marker(group, seeding, (None, None))
        except Rejected as error:
            raise AssemblyError(f"seed {group.name}: {error}") from error
        accepted.append((base / "hooks.tar", "seed", environment))
    versions = {seed.language: seed.language_version for _, _, seed in accepted}
    established = Base(home, tools, versions)
    manifests = read_manifests(repos)
    taken = set(seeds)
    skipped: list[dict[str, object]] = []
    for unit in plan.units:
        path = work / "units" / f"{unit.uid}.tar"
        try:
            status = _status(build, unit.uid)
            if status != "built":
                raise Rejected(f"build {status}")
            expected = (unit, expected_shapes(unit, manifests))
            found = _accept_unit(path, established, taken, expected)
        except (Rejected, tarfile.TarError) as error:
            skipped.append(
                {
                    "unit": unit.uid,
                    "label": unit.label,
                    "users": len(unit.users),
                    "reason": str(error),
                }
            )
            continue
        accepted.extend((path, unit.uid, environment) for environment in found)
        taken.update(environment.group.name for environment in found)

    return {
        "schema": SCHEMA,
        "prek": {"version": build.get("prek"), "home": home},
        "platform": {"image_os": settings.image_os, "arch": build.get("arch")},
        "tools": {
            name: writer.write(base / "tools.tar", group)
            for name, group in sorted(tools.items())
        },
        "clones": _clones(repos, base / "repos.tar", writer),
        "environments": [
            {
                "directory": environment.group.name,
                "language": environment.language,
                "language_version": environment.language_version,
                "dependencies": list(environment.dependencies),
                "requested": environment.requested,
                "hooks": environment.hooks and list(environment.hooks),
                "unit": uid,
                "users": environment.users,
                **writer.write(path, environment.group),
            }
            for path, uid, environment in accepted
        ],
        "skipped": skipped,
    }
