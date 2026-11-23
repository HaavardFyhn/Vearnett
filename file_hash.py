"""Streaming content hashes shared by runtime caches and transfer tooling."""
import hashlib
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_sha256(*sources: Path) -> str:
    """Hash source files, or their containing executable in a frozen app."""
    if getattr(sys, "frozen", False):
        return sha256(Path(sys.executable))
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.read_bytes())
    return digest.hexdigest()
