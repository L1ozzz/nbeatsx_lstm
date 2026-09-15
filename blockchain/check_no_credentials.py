"""Confirm that evidence files do not contain the locally loaded credentials."""

from __future__ import annotations

import ast
import json
import os
import re
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
TEXT_SUFFIXES = {".py", ".sol", ".json", ".md", ".txt", ".csv"}


def load_secret_values(script_path: Path) -> tuple[str, str]:
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    private_key = None
    rpc_url = None
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id == "private_key":
            private_key = ast.literal_eval(node.value)
        elif target.id == "w3":
            for child in ast.walk(node.value):
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    if child.value.startswith(("http://", "https://")):
                        rpc_url = child.value
                        break
    if not private_key or not rpc_url:
        raise RuntimeError("Could not load credential values for comparison")
    return private_key, rpc_url


def main() -> None:
    source = Path(
        os.environ.get(
            "ETH_STORE_SCRIPT",
            r"D:\Research-Haozhi Liao\NBeatsx-LSTM-副本\NBeatsx-LSTM\eth-store.py",
        )
    )
    private_key, private_rpc = load_secret_values(source)
    scan_roots = [HERE, HERE.parent / "E3_GW_significance"]
    files = [
        path
        for root in scan_roots
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
    ]
    documents = [HERE.parent / "论文修改与实验核对记录.docx"]
    scan_items = [(path, str(path), path.read_text(encoding="utf-8", errors="ignore")) for path in files]
    for document in documents:
        with zipfile.ZipFile(document) as archive:
            for member in archive.namelist():
                if member.endswith(".xml"):
                    scan_items.append(
                        (document, f"{document}!{member}", archive.read(member).decode("utf-8", errors="ignore"))
                    )
    secret_matches = []
    assignment_matches = []
    assignment_pattern = re.compile(
        r"(?i)(private[_-]?key|mnemonic|seed[_-]?phrase|rpc[_-]?secret)\s*=\s*['\"][^'\"]+"
    )
    for path, label, text in scan_items:
        if private_key in text or private_rpc in text:
            secret_matches.append(label)
        # Environment-variable lookups and variable names without literal values
        # are allowed; only literal credential assignments are reportable.
        for match in assignment_pattern.finditer(text):
            snippet = match.group(0)
            if "os.environ" not in snippet:
                assignment_matches.append({"file": label, "kind": match.group(1)})

    result = {
        "roots_scanned": [str(path) for path in scan_roots],
        "files_scanned": len(files),
        "docx_files_scanned": len(documents),
        "docx_xml_parts_scanned": len(scan_items) - len(files),
        "exact_private_key_or_private_rpc_matches": secret_matches,
        "literal_credential_assignment_matches": assignment_matches,
        "credentials_absent": not secret_matches and not assignment_matches,
    }
    (HERE / "credential_scan.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if not result["credentials_absent"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
