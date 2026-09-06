"""Create a SHA-256 manifest for the final E1/E2 evidence package."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
OUTPUT = HERE / "evidence_manifest_sha256.csv"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


files = [
    path
    for path in HERE.iterdir()
    if path.is_file()
    and path != OUTPUT
    and not path.name.endswith(".inspect.ndjson")
]
files.append(PROJECT / "论文修改与实验核对记录.docx")

with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["relative_or_absolute_path", "bytes", "sha256"])
    for path in sorted(files, key=lambda item: str(item).lower()):
        try:
            label = str(path.relative_to(PROJECT))
        except ValueError:
            label = str(path)
        writer.writerow([label, path.stat().st_size, digest(path)])

print(OUTPUT)
