"""Machine-checkable form of docs/schemas.md.

The prose file is the contract; this module is how the contract is enforced.
Change docs/schemas.md and this file in the same commit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

DATE_DTYPE = "datetime64[ns]"
ID_DTYPE = "object"
F32 = "float32"

UNIVERSE_SIZE = 500  # per CLAUDE.md; overridable in tests and fixtures


@dataclass(frozen=True)
class TableSpec:
    """Required columns of a stage output, plus the columns that key it."""

    name: str
    key: tuple[str, ...]
    required: dict[str, str]
    prefixes: tuple[str, ...] = ()
    optional: dict[str, str] = field(default_factory=dict)


RETURNS = TableSpec(
    name="returns",
    key=("date", "sec_id"),
    required={"date": DATE_DTYPE, "sec_id": ID_DTYPE, "ret": F32, "mktcap_lag": F32},
    optional={"country": ID_DTYPE, "currency": ID_DTYPE},
)

UNIVERSE = TableSpec(
    name="universe",
    key=("month", "sec_id"),
    required={"month": DATE_DTYPE, "sec_id": ID_DTYPE, "cap_rank": "int16"},
)

FEATURES = TableSpec(
    name="features",
    key=("date", "sec_id"),
    required={"date": DATE_DTYPE, "sec_id": ID_DTYPE, "rf": F32},
    prefixes=("char_", "med_"),
)

RESIDUALS = TableSpec(
    name="residuals",
    key=("date", "sec_id"),
    required={"date": DATE_DTYPE, "sec_id": ID_DTYPE, "resid": F32},
)

WEIGHTS = TableSpec(
    name="weights",
    key=("date", "sec_id"),
    required={"date": DATE_DTYPE, "sec_id": ID_DTYPE, "w": F32},
)


class SchemaError(AssertionError):
    """Raised when a table does not honour the contract."""


def validate(df: pd.DataFrame, spec: TableSpec) -> None:
    """Raise SchemaError unless df honours spec. Cheap enough to call at every seam."""
    missing = [c for c in spec.required if c not in df.columns]
    if missing:
        raise SchemaError(f"{spec.name}: missing required columns {missing}")

    for col, dtype in spec.required.items():
        actual = str(df[col].dtype)
        if actual != dtype:
            raise SchemaError(f"{spec.name}.{col}: dtype {actual}, contract says {dtype}")

    for col, dtype in spec.optional.items():
        if col in df.columns and str(df[col].dtype) != dtype:
            raise SchemaError(f"{spec.name}.{col}: dtype {df[col].dtype}, contract says {dtype}")

    for prefix in spec.prefixes:
        cols = [c for c in df.columns if c.startswith(prefix)]
        if not cols:
            raise SchemaError(f"{spec.name}: no columns with prefix {prefix!r}")
        bad = [c for c in cols if str(df[c].dtype) != F32]
        if bad:
            raise SchemaError(f"{spec.name}: {prefix}* columns must be {F32}, offenders {bad[:5]}")

    nulls = [c for c in spec.required if df[c].isna().any()]
    if nulls:
        raise SchemaError(f"{spec.name}: nulls in required columns {nulls}")

    key = list(spec.key)
    if df.duplicated(subset=key).any():
        n = int(df.duplicated(subset=key).sum())
        raise SchemaError(f"{spec.name}: {n} duplicate rows on key {key}")

    if not pd.MultiIndex.from_frame(df[key]).is_monotonic_increasing:
        raise SchemaError(f"{spec.name}: rows are not sorted by {key}")


def validate_universe(df: pd.DataFrame, size: int = UNIVERSE_SIZE) -> None:
    """Universe-specific checks on top of the schema: fixed size, dense ranks."""
    validate(df, UNIVERSE)
    counts = df.groupby("month").size()
    off = counts[counts != size]
    if len(off):
        raise SchemaError(
            f"universe: {len(off)} months do not have exactly {size} names, "
            f"e.g. {off.head(3).to_dict()}"
        )
    for month, grp in df.groupby("month"):
        ranks = sorted(grp["cap_rank"].tolist())
        if ranks != list(range(1, size + 1)):
            raise SchemaError(f"universe: cap_rank is not 1..{size} in {month.date()}")


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The 79 feature columns, in a stable order. Never rely on DataFrame order."""
    chars = sorted(c for c in df.columns if c.startswith("char_"))
    meds = sorted(c for c in df.columns if c.startswith("med_"))
    return chars + meds + ["rf"]
