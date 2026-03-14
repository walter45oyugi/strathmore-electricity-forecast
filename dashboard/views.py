
import pandas as pd
from django.shortcuts import render
from django.utils.safestring import mark_safe
import plotly.graph_objects as go
from plotly.io import to_html
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .serializers import ForecastRequestSerializer
from .utils import (
    TARGET,
    get_artifact_status,
    load_and_prepare_data,
    load_default_data,
    make_forecast,
)

CURRENT_UPPER_CAP = 120
VOLTAGE_LOWER_CAP = 210
VOLTAGE_UPPER_CAP = 260


def _safe_mean(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def _average_across_columns(
    df: pd.DataFrame,
    columns: list[str],
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> float | None:
    available = [column for column in columns if column in df.columns]
    if not available:
        return None

    numeric = df[available].apply(pd.to_numeric, errors="coerce")
    if lower_bound is not None:
        numeric = numeric.where(numeric >= lower_bound)
    if upper_bound is not None:
        numeric = numeric.where(numeric <= upper_bound)

    row_mean = numeric.mean(axis=1).dropna()
    return _safe_mean(row_mean)


def _power_factor_average(df: pd.DataFrame) -> float | None:
    if "Power factor(%)" in df.columns:
        value = _safe_mean(df["Power factor(%)"])
    else:
        phase_columns = [
            "A phase power factor(%)",
            "B phase power factor(%)",
            "C phase power factor(%)",
        ]
        value = _average_across_columns(df, phase_columns)

    if value is None:
        return None

    return float(value * 100.0) if abs(value) <= 1.5 else float(value)


def _filter_series_bounds(
    series: pd.Series,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> pd.Series:
    filtered = pd.to_numeric(series, errors="coerce").dropna()
    if lower_bound is not None:
        filtered = filtered[filtered >= lower_bound]
    if upper_bound is not None:
        filtered = filtered[filtered <= upper_bound]
    return filtered


def _build_active_power_chart(df: pd.DataFrame) -> str:
    series = pd.to_numeric(df[TARGET], errors="coerce").dropna()
    chart_modes = {
        "Hourly": series.groupby(series.index.hour).mean(),
        "Daily": series.resample("D").mean(),
        "Weekly": series.resample("W-MON").mean(),
        "Monthly": series.resample("MS").mean(),
    }

    figure = go.Figure()
    mode_names = list(chart_modes.keys())

    for index, (label, aggregated) in enumerate(chart_modes.items()):
        cleaned = aggregated.dropna()
        is_hourly = label == "Hourly"
        x_values = (
            [f"{int(hour):02d}:00" for hour in cleaned.index]
            if is_hourly
            else [timestamp.strftime("%Y-%m-%d") for timestamp in cleaned.index]
        )
        common = {
            "name": label,
            "x": x_values,
            "y": cleaned.tolist(),
            "visible": label == "Daily",
            "hovertemplate": "%{x}<br>%{y:.2f} kW<extra></extra>",
        }
        if is_hourly:
            figure.add_trace(
                go.Bar(
                    **common,
                    marker={
                        "color": "#b91c1c",
                        "line": {"color": "#991b1b", "width": 1},
                    },
                )
            )
        else:
            figure.add_trace(
                go.Scatter(
                    **common,
                    mode="lines",
                    line={"color": "#b91c1c", "width": 3},
                    fill="tozeroy",
                    fillcolor="rgba(185, 28, 28, 0.10)",
                )
            )

    buttons = []
    for index, label in enumerate(mode_names):
        visible = [i == index for i in range(len(mode_names))]
        xaxis_title = "Hour of Day" if label == "Hourly" else "Period"
        buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"xaxis": {"title": xaxis_title}},
                ],
            }
        )

    figure.update_layout(
        margin={"t": 60, "r": 24, "b": 56, "l": 64},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": "Arial, sans-serif", "color": "#1f2937"},
        showlegend=False,
        hovermode="x unified",
        xaxis={"title": "Period", "showgrid": False},
        yaxis={
            "title": "Active Power (kW)",
            "gridcolor": "#e5e7eb",
            "zerolinecolor": "#d1d5db",
        },
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 1,
                "xanchor": "right",
                "y": 1.18,
                "yanchor": "top",
            }
        ],
    )

    return mark_safe(
        to_html(
            figure,
            full_html=False,
            include_plotlyjs=True,
            config={"responsive": True, "displayModeBar": False},
        )
    )


def _build_phase_chart(
    df: pd.DataFrame,
    columns: dict[str, str],
    title: str,
    y_axis_title: str,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    y_range: list[float] | None = None,
) -> str:
    colors = {
        "Phase A": "#2563eb",
        "Phase B": "#dc2626",
        "Phase C": "#059669",
    }
    figure = go.Figure()

    for phase_name, column in columns.items():
        if column not in df.columns:
            continue

        series = _filter_series_bounds(df[column], lower_bound, upper_bound)
        if series.empty:
            continue

        aggregated = series.resample("D").mean().dropna()
        if aggregated.empty:
            continue

        figure.add_trace(
            go.Scatter(
                x=[timestamp.strftime("%Y-%m-%d") for timestamp in aggregated.index],
                y=aggregated.tolist(),
                mode="lines",
                name=phase_name,
                line={"color": colors[phase_name], "width": 3},
                hovertemplate=f"{phase_name}<br>%{{x}}<br>%{{y:.2f}}<extra></extra>",
            )
        )

    figure.update_layout(
        title={"text": title, "x": 0},
        margin={"t": 56, "r": 24, "b": 56, "l": 64},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": "Arial, sans-serif", "color": "#1f2937"},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        xaxis={"title": "Period", "showgrid": False},
        yaxis={
            "title": y_axis_title,
            "gridcolor": "#e5e7eb",
            "zerolinecolor": "#d1d5db",
            "range": y_range,
        },
    )

    return mark_safe(
        to_html(
            figure,
            full_html=False,
            include_plotlyjs=False,
            config={"responsive": True, "displayModeBar": False},
        )
    )


def _build_forecast_chart(df: pd.DataFrame, forecast: list[dict]) -> str:
    """Actual last 48 h + LSTM 24 h forecast on one Plotly chart."""
    actual_series = pd.to_numeric(df[TARGET], errors="coerce").dropna()
    cutoff_start = actual_series.index.max() - pd.Timedelta(hours=48)
    last_48h = actual_series[actual_series.index >= cutoff_start]

    forecast_df = pd.DataFrame(forecast)
    forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])

    # Convert timestamps to ISO strings to avoid Plotly integer+Timestamp arithmetic issues
    actual_x = [ts.isoformat() for ts in last_48h.index]
    forecast_x = [ts.isoformat() for ts in forecast_df["timestamp"]]
    cutoff_str = last_48h.index.max().isoformat()

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=actual_x,
            y=last_48h.tolist(),
            mode="lines",
            name="Actual (last 48 h)",
            line={"color": "#6b7280", "width": 3},
            hovertemplate="Actual<br>%{x}<br>%{y:.2f} kW<extra></extra>",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=forecast_x,
            y=forecast_df["forecast_kw"].tolist(),
            mode="lines+markers",
            name="LSTM Forecast (next 24 h)",
            line={"color": "#b91c1c", "width": 3, "dash": "dash"},
            marker={"size": 6, "color": "#b91c1c"},
            hovertemplate="Forecast<br>%{x}<br>%{y:.2f} kW<extra></extra>",
        )
    )

    figure.update_layout(
        height=480,
        margin={"t": 48, "r": 24, "b": 56, "l": 64},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": "Arial, sans-serif", "color": "#1f2937"},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.1, "x": 0},
        xaxis={"title": "Time", "showgrid": False},
        yaxis={
            "title": "Active Power (kW)",
            "gridcolor": "#e5e7eb",
            "zerolinecolor": "#d1d5db",
        },
        shapes=[
            {
                "type": "line",
                "x0": cutoff_str,
                "x1": cutoff_str,
                "y0": 0,
                "y1": 1,
                "xref": "x",
                "yref": "paper",
                "line": {"color": "#9ca3af", "width": 2, "dash": "dot"},
            }
        ],
        annotations=[
            {
                "x": cutoff_str,
                "y": 1,
                "xref": "x",
                "yref": "paper",
                "text": "Now",
                "showarrow": False,
                "xanchor": "left",
                "yanchor": "top",
                "font": {"color": "#6b7280", "size": 13},
                "xshift": 6,
            }
        ],
    )

    return mark_safe(
        to_html(
            figure,
            full_html=False,
            include_plotlyjs="cdn",
            config={"responsive": True, "displayModeBar": False},
        )
    )


def overview(request):
    try:
        df = load_default_data()
        stats = df[TARGET].describe().to_dict()
        avg_current = _average_across_columns(
            df,
            ["A phase current(A)", "B phase current(A)", "C phase current(A)"],
            lower_bound=0,
            upper_bound=CURRENT_UPPER_CAP,
        )
        avg_voltage = _average_across_columns(
            df,
            ["A phase voltage(V)", "B phase voltage(V)", "C phase voltage(V)"],
            lower_bound=VOLTAGE_LOWER_CAP,
            upper_bound=VOLTAGE_UPPER_CAP,
        )
        avg_power_factor = _power_factor_average(df)

        context = {
            "stats": stats,
            "avg_current": avg_current,
            "avg_voltage": avg_voltage,
            "avg_power_factor": avg_power_factor,
            "date_start": df.index.min(),
            "date_end": df.index.max(),
            "active_power_chart": _build_active_power_chart(df),
            "voltage_chart": _build_phase_chart(
                df,
                {
                    "Phase A": "A phase voltage(V)",
                    "Phase B": "B phase voltage(V)",
                    "Phase C": "C phase voltage(V)",
                },
                title="Voltage Trend by Phase",
                y_axis_title="Voltage (V)",
                lower_bound=VOLTAGE_LOWER_CAP,
                upper_bound=VOLTAGE_UPPER_CAP,
                y_range=[VOLTAGE_LOWER_CAP, VOLTAGE_UPPER_CAP],
            ),
            "current_chart": _build_phase_chart(
                df,
                {
                    "Phase A": "A phase current(A)",
                    "Phase B": "B phase current(A)",
                    "Phase C": "C phase current(A)",
                },
                title="Current Trend by Phase",
                y_axis_title="Current (A)",
                lower_bound=0,
                upper_bound=CURRENT_UPPER_CAP,
                y_range=[0, CURRENT_UPPER_CAP],
            ),
        }
        return render(request, "dashboard/overview.html", context)
    except Exception as exc:
        return render(request, "dashboard/error.html", {"error": str(exc)})


def forecast_view(request):
    try:
        df = load_default_data()
        forecast = make_forecast(df, steps=24)
        forecast_df = pd.DataFrame(forecast)
        forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])

        peak_kw = float(forecast_df["forecast_kw"].max())
        mean_kw = float(forecast_df["forecast_kw"].mean())
        forecast_start = forecast_df["timestamp"].iloc[0]
        forecast_end = forecast_df["timestamp"].iloc[-1]
        method = forecast_df["method"].iloc[0]
        data_end = df.index.max()

        context = {
            "forecast_chart": _build_forecast_chart(df, forecast),
            "forecast": forecast,
            "peak_kw": peak_kw,
            "mean_kw": mean_kw,
            "forecast_start": forecast_start,
            "forecast_end": forecast_end,
            "method": method,
            "data_end": data_end,
            "model_status": get_artifact_status(),
        }
        return render(request, "dashboard/forecast.html", context)
    except Exception as exc:
        return render(request, "dashboard/error.html", {"error": str(exc)})


class ForecastAPI(APIView):
    def post(self, request):
        serializer = ForecastRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        steps = serializer.validated_data.get("steps", 24)
        uploaded_file = serializer.validated_data.get("file")
        data_records = serializer.validated_data.get("data")

        import os
        import tempfile
        from pathlib import Path

        try:
            if uploaded_file is not None:
                suffix = Path(uploaded_file.name).suffix or ".csv"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    for chunk in uploaded_file.chunks():
                        temp_file.write(chunk)
                    temp_path = temp_file.name
                try:
                    prepared = load_and_prepare_data(temp_path)
                finally:
                    os.unlink(temp_path)
            elif data_records:
                prepared = load_and_prepare_data(pd.DataFrame(data_records))
            else:
                return Response(
                    {"error": "Provide either a file or a JSON data list."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            forecast = make_forecast(prepared, steps)
            return Response(
                {
                    "forecast": forecast,
                    "model_status": get_artifact_status(),
                },
                status=status.HTTP_200_OK,
            )
        except Exception as exc:
            return Response(
                {"error": str(exc)},
                status=status.HTTP_400_BAD_REQUEST,
            )
