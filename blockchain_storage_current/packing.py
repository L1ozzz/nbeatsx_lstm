"""Lossless fixed-width encoding for the 23 May 2024 weather-record schema.

The packed representation is exactly 32 bytes.  The first 31 bytes contain
the complete record (date, twelve readings, and season); the final byte is
reserved and must be zero.  All numeric readings use the same x100 scale as
the existing textual blockchain record.
"""

from __future__ import annotations

from dataclasses import dataclass


FIELD_NAMES = (
    "GustDir(Deg)",
    "GustSpd(m/s)",
    "WindRun(Km)",
    "Rain(mm)",
    "Tmean(C)",
    "Tmax(C)",
    "Tmin(C)",
    "Tgmin(C)",
    "VapPress(hPa)",
    "ET10(C)",
    "Rad(MJ/m2)",
    "SoilM(%)",
)

SEASON_TO_CODE = {"Spring": 0, "Summer": 1, "Autumn": 2, "Winter": 3}
CODE_TO_SEASON = {value: key for key, value in SEASON_TO_CODE.items()}

# WindRun needs four bytes because the full dataset reaches 1069.00 km after
# applying the x100 integer scale. Temperatures use signed two-byte fields.
FIELD_LAYOUT = (
    ("GustDir(Deg)", 2, False),
    ("GustSpd(m/s)", 2, False),
    ("WindRun(Km)", 4, False),
    ("Rain(mm)", 2, False),
    ("Tmean(C)", 2, True),
    ("Tmax(C)", 2, True),
    ("Tmin(C)", 2, True),
    ("Tgmin(C)", 2, True),
    ("VapPress(hPa)", 2, False),
    ("ET10(C)", 2, True),
    ("Rad(MJ/m2)", 2, False),
    ("SoilM(%)", 2, False),
)


@dataclass(frozen=True)
class WeatherRecord:
    date: int
    readings: tuple[int, ...]
    season: str


def parse_text_record(text: str) -> WeatherRecord:
    """Parse the existing ``YYYYMMDD,[...],Season`` record format."""

    stripped = text.strip()
    date_text, remainder = stripped.split(",[", 1)
    readings_text, season = remainder.rsplit("],", 1)
    readings = tuple(int(part.strip()) for part in readings_text.split(","))
    if len(readings) != len(FIELD_LAYOUT):
        raise ValueError(f"Expected {len(FIELD_LAYOUT)} readings, got {len(readings)}")
    if season not in SEASON_TO_CODE:
        raise ValueError(f"Unsupported season: {season}")
    return WeatherRecord(date=int(date_text), readings=readings, season=season)


def format_text_record(record: WeatherRecord) -> str:
    values = ", ".join(str(value) for value in record.readings)
    return f"{record.date},[{values}],{record.season}"


def pack_record(record: WeatherRecord) -> bytes:
    """Encode one record into the documented 32-byte big-endian layout."""

    if not 0 < record.date <= 0xFFFFFFFF:
        raise ValueError("Date does not fit uint32")
    if len(record.readings) != len(FIELD_LAYOUT):
        raise ValueError("Unexpected reading count")
    if record.season not in SEASON_TO_CODE:
        raise ValueError(f"Unsupported season: {record.season}")

    output = bytearray(record.date.to_bytes(4, "big", signed=False))
    for value, (name, size, signed) in zip(record.readings, FIELD_LAYOUT):
        minimum = -(1 << (size * 8 - 1)) if signed else 0
        maximum = (1 << (size * 8 - (1 if signed else 0))) - 1
        if not minimum <= value <= maximum:
            raise ValueError(f"{name}={value} is outside [{minimum}, {maximum}]")
        output.extend(value.to_bytes(size, "big", signed=signed))
    output.append(SEASON_TO_CODE[record.season])
    output.append(0)  # reserved/version-extension byte
    if len(output) != 32:
        raise AssertionError(f"Packed record must be 32 bytes, got {len(output)}")
    return bytes(output)


def unpack_record(packed: bytes) -> WeatherRecord:
    """Decode one 32-byte value and reject unknown reserved data."""

    if len(packed) != 32:
        raise ValueError(f"Packed record must be 32 bytes, got {len(packed)}")
    if packed[-1] != 0:
        raise ValueError("Unsupported packed-record version/reserved byte")

    date = int.from_bytes(packed[0:4], "big", signed=False)
    offset = 4
    readings = []
    for _, size, signed in FIELD_LAYOUT:
        readings.append(int.from_bytes(packed[offset : offset + size], "big", signed=signed))
        offset += size
    season_code = packed[offset]
    if season_code not in CODE_TO_SEASON:
        raise ValueError(f"Unknown season code: {season_code}")
    return WeatherRecord(date=date, readings=tuple(readings), season=CODE_TO_SEASON[season_code])


def assert_round_trip(text: str) -> bytes:
    record = parse_text_record(text)
    packed = pack_record(record)
    decoded = unpack_record(packed)
    if decoded != record or format_text_record(decoded) != text.strip():
        raise AssertionError("Packed record did not round-trip exactly")
    return packed
