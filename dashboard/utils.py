from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

try:
    from tensorflow.keras.models import load_model
except Exception:  # pragma: no cover
    load_model = None


BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "ml models"
DATA_PATH = ARTIFACTS_DIR / "cleaned_data.csv"

TARGET = "Active power(+)(kW)"
TIME_COLUMN = "time_corrected"
LOOKBACK = 24
EXOG_CANDIDATES = [
    "Power factor(%)",
    "A phase voltage(V)",
    "B phase voltage(V)",
    "C phase voltage(V)",
    "A phase current(A)",
    "B phase current(A)",
    "C phase current(A)",
]

_ARTIFACT_CACHE: dict[str, Any] = {
    "loaded": False,
    "model": None,
    "scaler": None,
    "error": None,
    "model_path": None,
    "scaler_path": None,
}


def _pick_first_existing(patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(ARTIFACTS_DIR.glob(pattern))
        if matches:
            return matches[0]
    return None


def _get_model_path() -> Path | None:
    return _pick_first_existing(["*.keras", "*.h5", "*.hdf5"])


def _get_scaler_path() -> Path | None:
    return _pick_first_existing(["scaler.save", "*.save", "*.joblib", "*.pkl"])


def get_artifact_status() -> dict[str, Any]:
    model_path = _get_model_path()
    scaler_path = _get_scaler_path()
    return {
        "data_available": DATA_PATH.exists(),
        "model_available": model_path is not None,
        "scaler_available": scaler_path is not None,
        "model_name": model_path.name if model_path else None,
        "scaler_name": scaler_path.name if scaler_path else None,
        "runtime_error": _ARTIFACT_CACHE["error"],
    }


def load_forecasting_artifacts() -> tuple[Any, Any]:
    if _ARTIFACT_CACHE["loaded"]:
        return _ARTIFACT_CACHE["model"], _ARTIFACT_CACHE["scaler"]

    _ARTIFACT_CACHE["loaded"] = True
    model_path = _get_model_path()
    scaler_path = _get_scaler_path()
    _ARTIFACT_CACHE["model_path"] = model_path
    _ARTIFACT_CACHE["scaler_path"] = scaler_path

    if model_path is None or scaler_path is None:
        _ARTIFACT_CACHE["error"] = (
            "Saved LSTM model/scaler not found in 'ml models'. "
            "Dashboard will use the fallback forecast."
        )
        return None, None

    if load_model is None or joblib is None:
        _ARTIFACT_CACHE["error"] = (
            "TensorFlow or joblib is not installed, so the saved model could not be loaded."
        )
        return None, None

    try:
        _ARTIFACT_CACHE["model"] = load_model(model_path)
        _ARTIFACT_CACHE["scaler"] = joblib.load(scaler_path)
    except Exception as exc:  # pragma: no cover
        _ARTIFACT_CACHE["error"] = f"Could not load forecasting artifacts: {exc}"
        _ARTIFACT_CACHE["model"] = None
        _ARTIFACT_CACHE["scaler"] = None

    return _ARTIFACT_CACHE["model"], _ARTIFACT_CACHE["scaler"]


def correct_meter_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Recreate the notebook's timestamp correction and outage expansion."""
    if "Time" not in df.columns:
        raise ValueError("Input data must contain a 'Time' column.")

    corrected = df.copy()
    corrected["time_original"] = pd.to_datetime(
        corrected["Time"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    corrected = corrected.dropna(subset=["time_original"]).copy()
    corrected["date"] = corrected["time_original"].dt.date
    corrected["hour_12"] = corrected["time_original"].dt.hour
    corrected["minute"] = corrected["time_original"].dt.minute

    def _determine_ampm(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("time_original", ascending=False).copy()
        group["hour_occurrence"] = group.groupby("hour_12").cumcount()
        group["period"] = group["hour_occurrence"].apply(
            lambda count: "AM" if count == 1 else "PM"
        )
        mask_12 = group["hour_12"] == 12
        if mask_12.any():
            group.loc[mask_12, "period"] = group.loc[mask_12, "hour_occurrence"].apply(
                lambda count: "PM" if count == 0 else "AM"
            )
        return group

    corrected = pd.concat(
        [_determine_ampm(group) for _, group in corrected.groupby("date", sort=False)],
        axis=0,
    )

    def _to_24_hour(row: pd.Series) -> int:
        hour = int(row["hour_12"])
        period = row["period"]
        if period == "AM":
            return 0 if hour == 12 else hour
        return 12 if hour == 12 else hour + 12

    corrected["hour_24"] = corrected.apply(_to_24_hour, axis=1)
    corrected[TIME_COLUMN] = pd.to_datetime(
        corrected["date"].astype(str)
        + " "
        + corrected["hour_24"].astype(str).str.zfill(2)
        + ":"
        + corrected["minute"].astype(str).str.zfill(2)
        + ":00"
    )

    start = corrected[TIME_COLUMN].min().floor("h")
    end = corrected[TIME_COLUMN].max().ceil("h")
    complete_timeline = pd.date_range(start=start, end=end, freq="h")
    df_complete = pd.DataFrame({TIME_COLUMN: complete_timeline})

    drop_intermediates = {
        "time_original",
        "date",
        "hour_12",
        "minute",
        "hour_occurrence",
        "period",
        "hour_24",
    }
    data_cols = [
        column
        for column in corrected.columns
        if column not in drop_intermediates and column != TIME_COLUMN
    ]
    merged = (
        corrected[[TIME_COLUMN] + data_cols]
        .drop_duplicates(subset=TIME_COLUMN, keep="first")
        .sort_values(TIME_COLUMN)
    )
    full = df_complete.merge(merged, on=TIME_COLUMN, how="left")
    primary_column = data_cols[0]
    full["is_outage"] = full[primary_column].isna().map({True: "Yes", False: "No"})

    return full[[TIME_COLUMN, "is_outage"] + data_cols]


def _normalise_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    if TIME_COLUMN in df.columns:
        prepared = df.copy()
        prepared[TIME_COLUMN] = pd.to_datetime(prepared[TIME_COLUMN], errors="coerce")
        if "is_outage" not in prepared.columns:
            prepared["is_outage"] = "No"
        return prepared.dropna(subset=[TIME_COLUMN]).copy()

    if isinstance(df.index, pd.DatetimeIndex):
        prepared = df.reset_index().rename(columns={df.index.name or "index": TIME_COLUMN})
        if "is_outage" not in prepared.columns:
            prepared["is_outage"] = "No"
        prepared[TIME_COLUMN] = pd.to_datetime(prepared[TIME_COLUMN], errors="coerce")
        return prepared.dropna(subset=[TIME_COLUMN]).copy()

    if "Time" in df.columns:
        return correct_meter_timestamps(df)

    if "time_original" in df.columns:
        prepared = df.copy()
        prepared[TIME_COLUMN] = pd.to_datetime(prepared["time_original"], errors="coerce")
        if "is_outage" not in prepared.columns:
            prepared["is_outage"] = "No"
        return prepared.dropna(subset=[TIME_COLUMN]).copy()

    raise ValueError(
        "Input data must include either 'Time', 'time_corrected', or a DatetimeIndex."
    )


def clean_meter_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook cleaning pipeline and return a time-indexed dataframe."""
    cleaned = _normalise_input_frame(df)
    cleaned = cleaned.sort_values(TIME_COLUMN).reset_index(drop=True)

    total_energy = "Total cumulative energy(T1+T2)(kWh)"
    apparent_power = "Apparent power(+)(kVA)"
    power_factor = "Power factor(%)"
    voltage_cols = [
        "A phase voltage(V)",
        "B phase voltage(V)",
        "C phase voltage(V)",
    ]
    current_cols = [
        "A phase current(A)",
        "B phase current(A)",
        "C phase current(A)",
    ]

    if total_energy in cleaned.columns:
        cleaned["energy_diff"] = cleaned[total_energy].diff()
        if TARGET in cleaned.columns:
            cleaned[TARGET] = cleaned[TARGET].fillna(cleaned["energy_diff"])

    if all(column in cleaned.columns for column in [TARGET, apparent_power, power_factor]):
        mask = (
            cleaned[TARGET].isna()
            & cleaned[apparent_power].notna()
            & cleaned[power_factor].notna()
        )
        pf_values = cleaned.loc[mask, power_factor].astype(float)
        pf_ratio = np.where(pf_values.abs() > 1.5, pf_values / 100.0, pf_values)
        cleaned.loc[mask, TARGET] = cleaned.loc[mask, apparent_power].astype(float) * pf_ratio

    time_indexed = cleaned.set_index(TIME_COLUMN)
    numeric_cols = time_indexed.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        time_indexed[numeric_cols] = time_indexed[numeric_cols].interpolate(
            method="time", limit=3
        )
    cleaned = time_indexed.reset_index()

    cleaned["hour"] = cleaned[TIME_COLUMN].dt.hour
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_cols:
        if column == "hour":
            continue
        cleaned[column] = cleaned[column].fillna(
            cleaned.groupby("hour")[column].transform("median")
        )

    available_voltage_cols = [column for column in voltage_cols if column in cleaned.columns]
    if available_voltage_cols:
        cleaned[available_voltage_cols] = cleaned[available_voltage_cols].apply(
            lambda row: row.fillna(row.mean()),
            axis=1,
        )

    available_current_cols = [column for column in current_cols if column in cleaned.columns]
    if available_current_cols:
        cleaned[available_current_cols] = cleaned[available_current_cols].apply(
            lambda row: row.fillna(row.mean()),
            axis=1,
        )

    cleaned = cleaned.ffill().bfill()
    cleaned = cleaned.drop(columns=["energy_diff", "hour"], errors="ignore")
    cleaned = cleaned.sort_values(TIME_COLUMN).set_index(TIME_COLUMN)
    cleaned.index.name = TIME_COLUMN
    return cleaned


def load_and_prepare_data(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load a CSV/Excel file or dataframe and return cleaned time-indexed data."""
    if isinstance(source, pd.DataFrame):
        raw = source.copy()
    else:
        file_path = Path(source)
        suffix = file_path.suffix.lower()
        if suffix in {".xls", ".xlsx"}:
            raw = pd.read_excel(file_path)
        else:
            raw = pd.read_csv(file_path)

    return clean_meter_data(raw)


def load_default_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "No default cleaned dataset was found in 'ml models/cleaned_data.csv'."
        )
    return load_and_prepare_data(DATA_PATH)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [TARGET] + [column for column in EXOG_CANDIDATES if column in df.columns]


def _hourly_reference_values(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    history = df[columns].copy().ffill().bfill()
    reference = history.groupby(history.index.hour).median()
    reference = reference.fillna(history.tail(LOOKBACK * 7).median())
    return reference


def _fallback_forecast(history: pd.DataFrame, steps: int) -> list[dict[str, Any]]:
    series = history[TARGET].dropna().astype(float)
    if series.empty:
        raise ValueError(f"Column '{TARGET}' is missing or empty after preprocessing.")

    recent_mean = float(series.tail(24).mean())
    hourly_profile = series.groupby(series.index.hour).mean()
    last_timestamp = history.index.max()
    forecast: list[dict[str, Any]] = []

    for offset in range(1, steps + 1):
        timestamp = last_timestamp + pd.Timedelta(hours=offset)
        hourly_estimate = float(hourly_profile.get(timestamp.hour, recent_mean))
        value = (0.7 * hourly_estimate) + (0.3 * recent_mean)
        forecast.append(
            {
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "forecast_kw": round(float(value), 2),
                "method": "hourly-baseline",
            }
        )

    return forecast


def _lstm_forecast(history: pd.DataFrame, steps: int) -> list[dict[str, Any]] | None:
    model, scaler = load_forecasting_artifacts()
    if model is None or scaler is None:
        return None

    feature_cols = _feature_columns(history)
    if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != len(feature_cols):
        _ARTIFACT_CACHE["error"] = (
            "Saved scaler feature count does not match the prepared dashboard data. "
            "Fallback forecast is being used."
        )
        return None

    history_window = history[feature_cols].copy().ffill().bfill()
    if len(history_window) < LOOKBACK:
        _ARTIFACT_CACHE["error"] = (
            f"At least {LOOKBACK} hourly observations are required for LSTM forecasting."
        )
        return None

    hourly_reference = _hourly_reference_values(history_window, feature_cols)
    recent_window = history_window.tail(LOOKBACK).to_numpy(dtype=float)
    last_timestamp = history_window.index.max()
    forecast: list[dict[str, Any]] = []

    for offset in range(1, steps + 1):
        scaled_window = scaler.transform(recent_window)
        model_input = scaled_window.reshape(1, LOOKBACK, len(feature_cols))
        prediction = np.asarray(model.predict(model_input, verbose=0)).reshape(-1)
        next_target_scaled = float(prediction[0])

        dummy = np.zeros((1, len(feature_cols)))
        dummy[0, 0] = next_target_scaled
        next_target = float(scaler.inverse_transform(dummy)[0, 0])

        next_timestamp = last_timestamp + pd.Timedelta(hours=offset)
        next_features = [next_target]
        for column in feature_cols[1:]:
            if next_timestamp.hour in hourly_reference.index:
                exog_value = float(hourly_reference.loc[next_timestamp.hour, column])
            else:
                exog_value = float(history_window[column].tail(LOOKBACK).median())
            next_features.append(exog_value)

        recent_window = np.vstack([recent_window[1:], np.array(next_features, dtype=float)])
        forecast.append(
            {
                "timestamp": next_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "forecast_kw": round(next_target, 2),
                "method": "lstm",
            }
        )

    return forecast


def make_forecast(input_df: pd.DataFrame, steps: int = 24) -> list[dict[str, Any]]:
    """Generate a forecast from cleaned history using LSTM or a fallback profile."""
    if steps < 1:
        raise ValueError("Forecast horizon must be at least 1 hour.")

    if not isinstance(input_df.index, pd.DatetimeIndex):
        history = clean_meter_data(input_df)
    else:
        history = input_df.copy()

    lstm_forecast = _lstm_forecast(history, steps)
    if lstm_forecast is not None:
        return lstm_forecast

    return _fallback_forecast(history, steps)