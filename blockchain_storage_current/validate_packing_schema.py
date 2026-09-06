"""Validate the fixed-width field choices against the complete project CSV."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from packing import FIELD_LAYOUT, FIELD_NAMES, assert_round_trip, parse_text_record


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "imputed_data_KNN_1.csv"
SOURCE_RECORD = Path(__file__).with_name("source_record.txt")
OUTPUT = Path(__file__).with_name("packing_schema_validation.json")


def main() -> None:
    frame = pd.read_csv(DATASET)
    source_text = SOURCE_RECORD.read_text(encoding="utf-8").strip()
    source = parse_text_record(source_text)
    packed = assert_round_trip(source_text)

    dataset_row = frame.loc[frame["Day(Local_Date)"] == "2024-05-23"]
    if len(dataset_row) != 1:
        raise RuntimeError("Expected exactly one 2024-05-23 dataset row")
    row = dataset_row.iloc[0]
    row_scaled = tuple(int(round(float(row[name]) * 100)) for name in FIELD_NAMES)
    row_season = str(row["Season"])
    if row_scaled != source.readings or row_season != source.season:
        raise RuntimeError("source_record.txt does not match the 2024-05-23 CSV row")

    fields = []
    all_fit = True
    for name, size, signed in FIELD_LAYOUT:
        scaled = (frame[name].astype(float) * 100).round().astype("int64")
        minimum_allowed = -(1 << (size * 8 - 1)) if signed else 0
        maximum_allowed = (1 << (size * 8 - (1 if signed else 0))) - 1
        fits = int(scaled.min()) >= minimum_allowed and int(scaled.max()) <= maximum_allowed
        all_fit = all_fit and fits
        fields.append(
            {
                "name": name,
                "bytes": size,
                "signed": signed,
                "scale": 100,
                "dataset_min_scaled": int(scaled.min()),
                "dataset_max_scaled": int(scaled.max()),
                "allowed_min": minimum_allowed,
                "allowed_max": maximum_allowed,
                "all_rows_fit": fits,
            }
        )

    result = {
        "dataset": str(DATASET),
        "rows_checked": len(frame),
        "missing_cells_in_readings": int(frame[list(FIELD_NAMES)].isna().sum().sum()),
        "source_record_matches_csv_20240523": True,
        "source_text_bytes": len(source_text.encode("utf-8")),
        "packed_bytes": len(packed),
        "packed_hex": "0x" + packed.hex(),
        "round_trip_exact": True,
        "all_dataset_rows_fit_schema": all_fit,
        "fields": fields,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not all_fit:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
