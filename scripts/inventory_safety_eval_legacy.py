from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import stat
from typing import Any, Iterable


INVENTORY_VERSION = 1
_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
_SHA256_HEX_LENGTH = 64
_PROTECTED_SYSTEM_TREES = tuple(
    Path(path)
    for path in (
        "/boot",
        "/dev",
        "/etc",
        "/lib",
        "/lib64",
        "/media",
        "/mnt",
        "/opt",
        "/proc",
        "/root",
        "/run",
        "/sbin",
        "/srv",
        "/sys",
        "/usr",
        "/var",
    )
)


def _is_same_or_ancestor(candidate: Path, path: Path) -> bool:
    return candidate == path or candidate in path.parents


def _resolve_root(root: Path) -> Path:
    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"inventory root is not a directory: {resolved}")
    repository_root = Path(__file__).resolve().parents[1]
    working_directory = Path.cwd().resolve()
    is_broad = (
        len(resolved.parts) <= 3
        or _is_same_or_ancestor(resolved, repository_root)
        or _is_same_or_ancestor(resolved, working_directory)
        or any(_is_same_or_ancestor(system_root, resolved) for system_root in _PROTECTED_SYSTEM_TREES)
    )
    if is_broad or resolved == Path.home().resolve() or (resolved / ".git").exists():
        raise ValueError(f"unsafe inventory root: {resolved}")
    return resolved


def _validate_roots(roots: Iterable[Path]) -> tuple[Path, ...]:
    resolved_roots = tuple(sorted((_resolve_root(Path(root)) for root in roots), key=str))
    if not resolved_roots:
        raise ValueError("at least one inventory root is required")
    for index, root in enumerate(resolved_roots):
        for other in resolved_roots[index + 1 :]:
            if _is_same_or_ancestor(root, other) or _is_same_or_ancestor(other, root):
                raise ValueError(f"inventory roots overlap: {root} and {other}")
    return resolved_roots


def _absolute_lexical_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return Path(os.path.abspath(expanded))


def _open_directory_no_follow(path: Path, *, create: bool = False) -> int:
    path = _absolute_lexical_path(path)
    descriptor = os.open("/", _DIRECTORY_OPEN_FLAGS)
    try:
        for component in path.parts[1:]:
            try:
                child = os.open(component, _DIRECTORY_OPEN_FLAGS, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, mode=0o755, dir_fd=descriptor)
                child = os.open(component, _DIRECTORY_OPEN_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_relative_file_no_follow(root_descriptor: int, relative_path: Path) -> int:
    descriptor = os.dup(root_descriptor)
    try:
        parts = relative_path.parts
        if not parts or relative_path.is_absolute() or ".." in parts:
            raise ValueError(f"invalid relative legacy artifact path: {relative_path}")
        for component in parts[:-1]:
            child = os.open(component, _DIRECTORY_OPEN_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        file_descriptor = os.open(parts[-1], _FILE_OPEN_FLAGS, dir_fd=descriptor)
        os.close(descriptor)
        return file_descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _iter_relative_files(directory_descriptor: int, prefix: Path = Path()) -> Iterable[Path]:
    with os.scandir(os.dup(directory_descriptor)) as iterator:
        entries = sorted(iterator, key=lambda entry: entry.name)
        for entry in entries:
            relative_path = prefix / entry.name
            if entry.is_symlink():
                raise ValueError(f"symlinked legacy artifact is not supported: {relative_path}")
            if entry.is_file(follow_symlinks=False):
                yield relative_path
                continue
            if not entry.is_dir(follow_symlinks=False):
                raise ValueError(f"legacy artifact is not a regular file: {relative_path}")

            child_descriptor: int | None = None
            confirmation_descriptor: int | None = None
            try:
                child_descriptor = os.open(entry.name, _DIRECTORY_OPEN_FLAGS, dir_fd=directory_descriptor)
                before = os.fstat(child_descriptor)
                yield from _iter_relative_files(child_descriptor, relative_path)
                after = os.fstat(child_descriptor)
                confirmation_descriptor = os.open(
                    entry.name,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=directory_descriptor,
                )
                current = os.fstat(confirmation_descriptor)
            except OSError as error:
                raise ValueError(f"legacy directory changed while inventorying: {relative_path}") from error
            finally:
                if confirmation_descriptor is not None:
                    os.close(confirmation_descriptor)
                if child_descriptor is not None:
                    os.close(child_descriptor)
            if _stable_stat_identity(before) != _stable_stat_identity(after) or (
                before.st_dev,
                before.st_ino,
            ) != (current.st_dev, current.st_ino):
                raise ValueError(f"legacy directory changed while inventorying: {relative_path}")


def _hash_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    duplicate = os.dup(descriptor)
    os.lseek(duplicate, 0, os.SEEK_SET)
    with os.fdopen(duplicate, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _file_entry(root: Path, root_descriptor: int, relative_path: Path) -> dict[str, Any]:
    display_path = root / relative_path
    descriptor: int | None = None
    confirmation_descriptor: int | None = None
    try:
        descriptor = _open_relative_file_no_follow(root_descriptor, relative_path)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"legacy artifact is not a regular file: {display_path}")
        digest = _hash_descriptor(descriptor)
        after = os.fstat(descriptor)
        confirmation_digest = _hash_descriptor(descriptor)
        confirmation_descriptor = _open_relative_file_no_follow(root_descriptor, relative_path)
        current = os.fstat(confirmation_descriptor)
    except OSError as error:
        raise ValueError(f"symlinked or unstable legacy artifact is not supported: {display_path}") from error
    finally:
        if confirmation_descriptor is not None:
            os.close(confirmation_descriptor)
        if descriptor is not None:
            os.close(descriptor)

    if digest != confirmation_digest or _stable_stat_identity(before) != _stable_stat_identity(after) or (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
        current.st_ctime_ns,
    ):
        raise ValueError(f"legacy artifact changed while hashing: {display_path}")
    return {
        "path": relative_path.as_posix(),
        "size": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "sha256": digest,
    }


def build_inventory(roots: Iterable[Path]) -> dict[str, Any]:
    resolved_roots = _validate_roots(roots)
    return {
        "version": INVENTORY_VERSION,
        "roots": [_inventory_root(root) for root in resolved_roots],
    }


def _inventory_root(root: Path) -> dict[str, Any]:
    descriptor = _open_directory_no_follow(root)
    confirmation_descriptor: int | None = None
    try:
        before = os.fstat(descriptor)
        relative_paths = tuple(_iter_relative_files(descriptor))
        files = [_file_entry(root, descriptor, relative_path) for relative_path in relative_paths]
        confirmed_paths = tuple(_iter_relative_files(descriptor))
        after = os.fstat(descriptor)
        confirmation_descriptor = _open_directory_no_follow(root)
        current = os.fstat(confirmation_descriptor)
    except OSError as error:
        raise ValueError(f"legacy root changed while inventorying: {root}") from error
    finally:
        if confirmation_descriptor is not None:
            os.close(confirmation_descriptor)
        os.close(descriptor)

    if (
        relative_paths != confirmed_paths
        or _stable_stat_identity(before) != _stable_stat_identity(after)
        or (before.st_dev, before.st_ino) != (current.st_dev, current.st_ino)
    ):
        raise ValueError(f"legacy root changed while inventorying: {root}")
    return {"root": str(root), "files": files}


def _validate_inventory_file_entry(entry: object) -> dict[str, Any]:
    if not isinstance(entry, dict) or set(entry) != {"path", "size", "mtime_ns", "sha256"}:
        raise ValueError("invalid legacy inventory file entry")
    relative_path = entry["path"]
    parsed_path = PurePosixPath(relative_path) if isinstance(relative_path, str) else None
    if (
        parsed_path is None
        or not relative_path
        or parsed_path.is_absolute()
        or ".." in parsed_path.parts
        or parsed_path.as_posix() != relative_path
    ):
        raise ValueError("invalid legacy inventory file entry")
    size = entry["size"]
    mtime_ns = entry["mtime_ns"]
    sha256 = entry["sha256"]
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
        or not isinstance(mtime_ns, int)
        or isinstance(mtime_ns, bool)
        or not isinstance(sha256, str)
        or len(sha256) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise ValueError("invalid legacy inventory file entry")
    return entry


def verify_inventory(inventory: dict[str, Any]) -> None:
    if not isinstance(inventory, dict) or set(inventory) != {"version", "roots"}:
        raise ValueError("invalid legacy inventory document")
    if type(inventory["version"]) is not int or inventory["version"] != INVENTORY_VERSION:
        raise ValueError("unsupported legacy inventory version")
    roots = inventory["roots"]
    if not isinstance(roots, list) or not roots:
        raise ValueError("legacy inventory has no roots")

    root_paths: list[Path] = []
    for root_entry in roots:
        if (
            not isinstance(root_entry, dict)
            or set(root_entry) != {"root", "files"}
            or not isinstance(root_entry["root"], str)
        ):
            raise ValueError("invalid legacy inventory root")
        root_path = Path(root_entry["root"])
        if not root_path.is_absolute() or str(root_path) != root_entry["root"] or root_path.resolve() != root_path:
            raise ValueError("invalid legacy inventory root")
        root_paths.append(root_path)
        expected_files = root_entry["files"]
        if not isinstance(expected_files, list):
            raise ValueError("invalid legacy inventory files")
        validated_files = [_validate_inventory_file_entry(entry) for entry in expected_files]
        if len({entry["path"] for entry in validated_files}) != len(expected_files):
            raise ValueError("invalid legacy inventory file entry")

    resolved_roots = _validate_roots(root_paths)
    if tuple(root_paths) != resolved_roots:
        raise ValueError("legacy inventory roots must use stable ordering")
    for root_entry, root in zip(roots, resolved_roots, strict=True):
        current = _inventory_root(root)
        expected_files = root_entry["files"]
        if current["files"] != expected_files:
            expected_paths = {entry["path"] for entry in expected_files}
            current_paths = {entry["path"] for entry in current["files"]}
            missing = sorted(expected_paths - current_paths)
            if missing:
                raise ValueError(f"legacy artifact missing: {root / missing[0]}")
            added = sorted(current_paths - expected_paths)
            if added:
                raise ValueError(f"new legacy artifact: {root / added[0]}")
            raise ValueError(f"legacy artifact changed: {root}")


def _validate_destination(destination: Path, roots: Iterable[Path]) -> None:
    lexical_destination = _absolute_lexical_path(destination)
    resolved_destination = destination.expanduser().resolve()
    for root in _validate_roots(roots):
        if _is_same_or_ancestor(root, lexical_destination) or _is_same_or_ancestor(root, resolved_destination):
            raise ValueError(f"inventory destination is inside an inventory root: {destination}")


def write_inventory(destination: Path, inventory: dict[str, Any]) -> None:
    destination = _absolute_lexical_path(destination)
    parent_descriptor = _open_directory_no_follow(destination.parent, create=True)
    payload = json.dumps(inventory, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    temporary_name = f".{destination.name}.{secrets.token_hex(8)}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            temporary_name,
            destination.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        os.fsync(parent_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=parent_descriptor)
        except FileNotFoundError:
            pass
        os.close(parent_descriptor)


def _read_inventory(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("legacy inventory must be a JSON object")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inventory immutable legacy reviewer artifacts.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    write = subcommands.add_parser("write", help="write an inventory")
    write.add_argument("--root", action="append", required=True, type=Path)
    write.add_argument("--destination", required=True, type=Path)
    verify = subcommands.add_parser("verify", help="verify an existing inventory")
    verify.add_argument("--inventory", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "write":
            _validate_destination(args.destination, args.root)
            write_inventory(args.destination, build_inventory(args.root))
        else:
            verify_inventory(_read_inventory(args.inventory))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=__import__("sys").stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
