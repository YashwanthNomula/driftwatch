"""Per-feature drift thresholds loaded from a JSON config file.

The project ships with global PSI/KS warning/drift cutoffs (see
``driftwatch.stats``). A config file lets you override the defaults
globally and/or per feature — e.g. a stable-but-important feature that
should alarm earlier, or a noisy feature that should alarm later.

Example ``driftwatch.json``::

    {
      "defaults": {"psi_warning": 0.1, "psi_drift": 0.25,
                   "ks_warning": 0.1, "ks_drift": 0.3},
      "features": {
        "age":    {"psi_warning": 0.05, "psi_drift": 0.15},
        "income": {"ks_drift": 0.4}
      }
    }

Any key left out falls back to the global default; any feature not listed
falls back to the defaults entirely. Invalid configs raise
``ConfigError`` (a ``ValueError``) with a message that says what's wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .stats import KS_MODERATE, KS_NO_CHANGE, PSI_MODERATE, PSI_NO_CHANGE

#: Threshold keys a config may set.
THRESHOLD_KEYS = ("psi_warning", "psi_drift", "ks_warning", "ks_drift")


class ConfigError(ValueError):
    """Raised when a driftwatch config file is missing or invalid."""


@dataclass(frozen=True)
class ThresholdSet:
    """Warning/drift cutoffs for PSI and KS for one feature."""
    psi_warning: float = PSI_NO_CHANGE
    psi_drift: float = PSI_MODERATE
    ks_warning: float = KS_NO_CHANGE
    ks_drift: float = KS_MODERATE


@dataclass
class ThresholdConfig:
    """Parsed config: global defaults plus per-feature overrides."""
    defaults: ThresholdSet = field(default_factory=ThresholdSet)
    features: dict[str, ThresholdSet] = field(default_factory=dict)

    def thresholds_for(self, name: str) -> ThresholdSet:
        """Thresholds for ``name``, falling back to the global defaults."""
        return self.features.get(name, self.defaults)


def _check_number(where: str, key: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where}: '{key}' must be a number, got {value!r}")
    if value < 0:
        raise ConfigError(f"{where}: '{key}' must be >= 0, got {value!r}")
    return float(value)


def _parse_threshold_set(data: object, where: str,
                         base: ThresholdSet) -> ThresholdSet:
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: must be an object of thresholds, "
                          f"got {data!r}")
    unknown = [k for k in data if k not in THRESHOLD_KEYS]
    if unknown:
        raise ConfigError(f"{where}: unknown threshold key(s) "
                          f"{unknown}; expected one of {list(THRESHOLD_KEYS)}")
    vals: dict[str, float] = {}
    for key in THRESHOLD_KEYS:
        if key in data:
            vals[key] = _check_number(where, key, data[key])
    merged = ThresholdSet(
        psi_warning=vals.get("psi_warning", base.psi_warning),
        psi_drift=vals.get("psi_drift", base.psi_drift),
        ks_warning=vals.get("ks_warning", base.ks_warning),
        ks_drift=vals.get("ks_drift", base.ks_drift),
    )
    if merged.psi_warning >= merged.psi_drift:
        raise ConfigError(
            f"{where}: psi_warning ({merged.psi_warning}) must be below "
            f"psi_drift ({merged.psi_drift})")
    if merged.ks_warning >= merged.ks_drift:
        raise ConfigError(
            f"{where}: ks_warning ({merged.ks_warning}) must be below "
            f"ks_drift ({merged.ks_drift})")
    return merged


def load_config(path: str) -> ThresholdConfig:
    """Read and validate a JSON driftwatch config file."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise ConfigError(f"{path}: config file not found")
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read config file: {exc}")
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: invalid JSON: {exc}")

    if not isinstance(data, dict):
        raise ConfigError(f"{path}: config must be a JSON object")
    unknown = [k for k in data if k not in ("defaults", "features")]
    if unknown:
        raise ConfigError(f"{path}: unknown top-level key(s) {unknown}; "
                          "expected 'defaults' and/or 'features'")

    builtin = ThresholdSet()
    defaults = builtin
    if "defaults" in data:
        defaults = _parse_threshold_set(data["defaults"],
                                        f"{path}: 'defaults'", builtin)

    features: dict[str, ThresholdSet] = {}
    raw_features = data.get("features", {})
    if not isinstance(raw_features, dict):
        raise ConfigError(f"{path}: 'features' must be an object mapping "
                          f"feature names to thresholds")
    for name, spec in raw_features.items():
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"{path}: feature name must be a non-empty "
                              f"string, got {name!r}")
        features[name] = _parse_threshold_set(
            spec, f"{path}: feature '{name}'", defaults)

    return ThresholdConfig(defaults=defaults, features=features)
