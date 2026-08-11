from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gamma


ROOT = Path(__file__).resolve().parent
STATIONS = ROOT / "station_features.csv"
IFS_DIR = ROOT / "ifs"
ERA5_DIR = ROOT / "era5"
CONTINUOUS_OUTPUT = ROOT / "continuous_error_parameters.npz"
PRECIPITATION_OUTPUT = ROOT / "precipitation_error_parameters.npz"

EXPECTED_RUNS = 183
EXPECTED_LEADS = 48
FIRST_TIME = "2024-03-14T09:00"
LAST_TIME = "2025-03-15T08:00"
TIMEZONE = "Asia/Shanghai"
WET_THRESHOLD = 0.0

VARIABLES = np.asarray([
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_speed_100m",
    "wind_direction_100m",
    "pressure_msl",
    "cloud_cover",
])
CONTINUOUS_VARIABLES = np.asarray([
    "temperature_2m",
    "relative_humidity_2m",
    "wind_u_10m",
    "wind_v_10m",
    "wind_u_100m",
    "wind_v_100m",
    "pressure_msl",
    "cloud_cover",
])
CONTINUOUS_PARAMETER_NAMES = np.asarray([
    "ar_coefficient",
    "additive_bias",
    "stochastic_mean",
    "stochastic_std",
])
PRECIPITATION_PARAMETER_NAMES = np.asarray([
    "wet_probability_prev_dry_truth_dry",
    "wet_probability_prev_dry_truth_wet",
    "wet_probability_prev_wet_truth_dry",
    "wet_probability_prev_wet_truth_wet",
    "hit_probability",
    "miss_probability",
    "false_alarm_probability",
    "correct_negative_probability",
    "hit_log_ratio_mean",
    "hit_log_ratio_std",
    "false_alarm_gamma_shape",
    "false_alarm_gamma_scale",
])


def main():
    """Estimate error-generator parameters from exactly aligned station and valid-time samples.

    Wind speed and meteorological direction are converted to u and v before forecast
    errors are calculated, so the continuous wind parameters describe vector-component
    errors rather than separate speed and direction errors. The continuous model uses
    one station-channel AR coefficient and additive bias across all leads; lead 1 stores
    the initial-error distribution, while leads 2-48 store lead-specific innovation
    distributions. The precipitation model separately fits occurrence transitions,
    categorical errors, hit ratios, and false-alarm amounts by station, then broadcasts
    those pooled estimates across the lead axis.
    """
    stations = pd.read_csv(STATIONS, dtype={"站点编号": str})
    station_ids = stations["站点编号"].to_numpy(dtype=str)
    ifs_files = sorted(IFS_DIR.glob("*.npz"))
    era5_files = sorted(ERA5_DIR.glob("*.csv"))

    if len(ifs_files) != EXPECTED_RUNS:
        raise ValueError(f"Expected {EXPECTED_RUNS} IFS files, found {len(ifs_files)}.")
    if {path.stem for path in era5_files} != set(station_ids):
        raise ValueError("ERA5 station files do not exactly match station_features.csv.")

    station_count = len(station_ids)
    variable_count = len(VARIABLES)
    forecasts = np.empty(
        (EXPECTED_RUNS, station_count, EXPECTED_LEADS, variable_count),
        dtype=np.float32,
    )
    forecast_times = np.empty(
        (EXPECTED_RUNS, EXPECTED_LEADS),
        dtype="U16",
    )

    for k, path in enumerate(ifs_files):
        with np.load(path, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["station_id"], station_ids)
            np.testing.assert_array_equal(archive["variables"], VARIABLES)
            if archive["data"].shape != (
                station_count,
                EXPECTED_LEADS,
                variable_count,
            ):
                raise ValueError(f"Unexpected data shape in {path.name}: {archive['data'].shape}")
            if str(archive["timezone"]) != TIMEZONE:
                raise ValueError(f"Unexpected timezone in {path.name}: {archive['timezone']}")
            forecasts[k] = archive["data"]
            forecast_times[k] = archive["time"]

    valid_times = forecast_times.reshape(-1)
    expected_times = pd.date_range(
        FIRST_TIME,
        LAST_TIME,
        freq="h",
    ).strftime("%Y-%m-%dT%H:%M").to_numpy()
    np.testing.assert_array_equal(valid_times, expected_times)

    precipitation_index = int(np.flatnonzero(VARIABLES == "precipitation")[0])
    ifs_precipitation = forecasts[:, :, :, precipitation_index].copy()
    era5_precipitation = np.empty_like(ifs_precipitation)
    continuous_errors = np.empty(
        (
            EXPECTED_RUNS,
            station_count,
            EXPECTED_LEADS,
            len(CONTINUOUS_VARIABLES),
        ),
        dtype=np.float32,
    )

    for s, station_id in enumerate(station_ids):
        era5 = pd.read_csv(ERA5_DIR / f"{station_id}.csv")
        np.testing.assert_array_equal(era5["time"].to_numpy(), valid_times)
        era5_values = era5[VARIABLES].to_numpy(dtype=np.float32).reshape(
            EXPECTED_RUNS,
            EXPECTED_LEADS,
            variable_count,
        )
        era5_precipitation[:, s] = era5_values[:, :, precipitation_index]

        continuous_errors[:, s, :, 0] = (
            forecasts[:, s, :, 0] - era5_values[:, :, 0]
        )
        continuous_errors[:, s, :, 1] = (
            forecasts[:, s, :, 1] - era5_values[:, :, 1]
        )

        ifs_direction_10m = np.deg2rad(forecasts[:, s, :, 4])
        era5_direction_10m = np.deg2rad(era5_values[:, :, 4])
        continuous_errors[:, s, :, 2] = (
            -forecasts[:, s, :, 3] * np.sin(ifs_direction_10m)
            + era5_values[:, :, 3] * np.sin(era5_direction_10m)
        )
        continuous_errors[:, s, :, 3] = (
            -forecasts[:, s, :, 3] * np.cos(ifs_direction_10m)
            + era5_values[:, :, 3] * np.cos(era5_direction_10m)
        )

        ifs_direction_100m = np.deg2rad(forecasts[:, s, :, 6])
        era5_direction_100m = np.deg2rad(era5_values[:, :, 6])
        continuous_errors[:, s, :, 4] = (
            -forecasts[:, s, :, 5] * np.sin(ifs_direction_100m)
            + era5_values[:, :, 5] * np.sin(era5_direction_100m)
        )
        continuous_errors[:, s, :, 5] = (
            -forecasts[:, s, :, 5] * np.cos(ifs_direction_100m)
            + era5_values[:, :, 5] * np.cos(era5_direction_100m)
        )

        continuous_errors[:, s, :, 6] = (
            forecasts[:, s, :, 7] - era5_values[:, :, 7]
        )
        continuous_errors[:, s, :, 7] = (
            forecasts[:, s, :, 8] - era5_values[:, :, 8]
        )

    del forecasts

    if not np.isfinite(continuous_errors).all():
        raise ValueError("IFS-minus-ERA5 errors contain non-finite values.")

    previous_error = continuous_errors[:, :, :-1, :]
    next_error = continuous_errors[:, :, 1:, :]
    sample_count = EXPECTED_RUNS * (EXPECTED_LEADS - 1)

    sum_previous = previous_error.sum(axis=(0, 2), dtype=np.float64)
    sum_next = next_error.sum(axis=(0, 2), dtype=np.float64)
    sum_previous_squared = np.square(previous_error).sum(
        axis=(0, 2),
        dtype=np.float64,
    )
    sum_previous_next = np.multiply(previous_error, next_error).sum(
        axis=(0, 2),
        dtype=np.float64,
    )

    denominator = (
        sample_count * sum_previous_squared
        - np.square(sum_previous)
    )
    ar_coefficient = (
        sample_count * sum_previous_next
        - sum_previous * sum_next
    ) / denominator
    additive_bias = (
        sum_next - ar_coefficient * sum_previous
    ) / sample_count

    residual = (
        next_error
        - previous_error * ar_coefficient.astype(np.float32)[None, :, None, :]
        - additive_bias.astype(np.float32)[None, :, None, :]
    )
    innovation_mean = residual.mean(axis=(0, 2), dtype=np.float64)
    innovation_std = residual.std(axis=0, ddof=1, dtype=np.float64).transpose(0, 2, 1)
    initial_error_mean = continuous_errors[:, :, 0, :].mean(axis=0, dtype=np.float64)
    initial_error_std = continuous_errors[:, :, 0, :].std(
        axis=0,
        ddof=1,
        dtype=np.float64,
    )

    continuous_parameters = np.empty(
        (
            station_count,
            len(CONTINUOUS_VARIABLES),
            EXPECTED_LEADS,
            len(CONTINUOUS_PARAMETER_NAMES),
        ),
        dtype=np.float64,
    )
    continuous_parameters[:, :, :, 0] = ar_coefficient[:, :, None]
    continuous_parameters[:, :, :, 1] = additive_bias[:, :, None]
    continuous_parameters[:, :, 0, 2] = initial_error_mean
    continuous_parameters[:, :, 1:, 2] = innovation_mean[:, :, None]
    continuous_parameters[:, :, 0, 3] = initial_error_std
    continuous_parameters[:, :, 1:, 3] = innovation_std

    if not np.isfinite(continuous_parameters).all():
        raise ValueError("Continuous-variable parameters contain non-finite values.")

    np.savez_compressed(
        CONTINUOUS_OUTPUT,
        parameters=continuous_parameters,
        station_id=station_ids,
        variables=CONTINUOUS_VARIABLES,
        lead_hours=np.arange(1, EXPECTED_LEADS + 1),
        parameter_names=CONTINUOUS_PARAMETER_NAMES,
        axis_order=np.asarray(["station", "variable", "lead", "parameter"]),
        stochastic_term_kind=np.asarray(
            ["initial_error"] + ["ar_innovation"] * (EXPECTED_LEADS - 1)
        ),
        timezone=TIMEZONE,
        error_definition="IFS_minus_ERA5_in_parameter_variable_space",
        wind_component_convention="u=-speed*sin(direction), v=-speed*cos(direction)",
    )

    if not np.isfinite(ifs_precipitation).all():
        raise ValueError("IFS precipitation contains non-finite values.")
    if not np.isfinite(era5_precipitation).all():
        raise ValueError("ERA5 precipitation contains non-finite values.")

    ifs_wet = ifs_precipitation > WET_THRESHOLD
    era5_wet = era5_precipitation > WET_THRESHOLD
    hit = ifs_wet & era5_wet
    miss = ~ifs_wet & era5_wet
    false_alarm = ifs_wet & ~era5_wet
    correct_negative = ~ifs_wet & ~era5_wet

    hit_count = hit.sum(axis=(0, 2))
    miss_count = miss.sum(axis=(0, 2))
    false_alarm_count = false_alarm.sum(axis=(0, 2))
    correct_negative_count = correct_negative.sum(axis=(0, 2))

    hit_probability = hit_count / (hit_count + miss_count)
    miss_probability = miss_count / (hit_count + miss_count)
    false_alarm_probability = false_alarm_count / (
        false_alarm_count + correct_negative_count
    )
    correct_negative_probability = correct_negative_count / (
        false_alarm_count + correct_negative_count
    )

    previous_wet = ifs_wet[:, :, :-1]
    current_wet = ifs_wet[:, :, 1:]
    current_truth_wet = era5_wet[:, :, 1:]
    wet_transition_probability = np.empty((station_count, 2, 2), dtype=np.float64)

    for previous_state in (0, 1):
        for truth_state in (0, 1):
            condition = (
                (previous_wet == previous_state)
                & (current_truth_wet == truth_state)
            )
            wet_transition_probability[:, previous_state, truth_state] = (
                (condition & current_wet).sum(axis=(0, 2))
                / condition.sum(axis=(0, 2))
            )

    hit_log_ratio_mean = np.empty(station_count, dtype=np.float64)
    hit_log_ratio_std = np.empty(station_count, dtype=np.float64)
    false_alarm_gamma_shape = np.empty(station_count, dtype=np.float64)
    false_alarm_gamma_scale = np.empty(station_count, dtype=np.float64)

    for s in range(station_count):
        hit_log_ratio = np.log(
            ifs_precipitation[:, s][hit[:, s]]
            / era5_precipitation[:, s][hit[:, s]]
        )
        hit_log_ratio_mean[s] = hit_log_ratio.mean()
        hit_log_ratio_std[s] = hit_log_ratio.std(ddof=1)

        false_alarm_amount = ifs_precipitation[:, s][false_alarm[:, s]]
        shape, location, scale = gamma.fit(false_alarm_amount, floc=0.0)
        if location != 0.0:
            raise ValueError(f"Nonzero fitted Gamma location for station {station_ids[s]}.")
        false_alarm_gamma_shape[s] = shape
        false_alarm_gamma_scale[s] = scale

    precipitation_by_station = np.column_stack([
        wet_transition_probability[:, 0, 0],
        wet_transition_probability[:, 0, 1],
        wet_transition_probability[:, 1, 0],
        wet_transition_probability[:, 1, 1],
        hit_probability,
        miss_probability,
        false_alarm_probability,
        correct_negative_probability,
        hit_log_ratio_mean,
        hit_log_ratio_std,
        false_alarm_gamma_shape,
        false_alarm_gamma_scale,
    ])
    precipitation_parameters = np.broadcast_to(
        precipitation_by_station[:, None, :],
        (
            station_count,
            EXPECTED_LEADS,
            len(PRECIPITATION_PARAMETER_NAMES),
        ),
    ).copy()

    if not np.isfinite(precipitation_parameters).all():
        raise ValueError("Precipitation parameters contain non-finite values.")

    np.savez_compressed(
        PRECIPITATION_OUTPUT,
        parameters=precipitation_parameters,
        station_id=station_ids,
        variable="precipitation",
        lead_hours=np.arange(1, EXPECTED_LEADS + 1),
        parameter_names=PRECIPITATION_PARAMETER_NAMES,
        axis_order=np.asarray(["station", "lead", "parameter"]),
        wet_threshold=WET_THRESHOLD,
        timezone=TIMEZONE,
        occurrence_initialization=np.asarray([
            "false_alarm_probability_if_truth_dry",
            "hit_probability_if_truth_wet",
        ]),
        hit_amount_model="ERA5_times_lognormal_ratio",
        false_alarm_amount_model="Gamma_shape_scale_location_zero",
    )

    print(CONTINUOUS_OUTPUT.name, continuous_parameters.shape)
    print(PRECIPITATION_OUTPUT.name, precipitation_parameters.shape)


if __name__ == "__main__":
    main()
