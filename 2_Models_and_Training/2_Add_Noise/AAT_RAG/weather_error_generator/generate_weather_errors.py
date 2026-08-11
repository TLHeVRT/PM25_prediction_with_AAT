import numpy as np


LEAD_COUNT = 48
VARIABLE_COUNT = 9
PRECIPITATION_CHANNEL = 2
BOUNDED_PARAMETER_CHANNELS = np.asarray([1, 7])


def generate_noisy_weather(
    continuous_parameters,
    precipitation_parameters,
    station_era5_pairs,
    rng,
):
    """Generate IFS-like 48-hour weather sequences from station-aligned ERA5 sequences.

    The input channel order is temperature_2m, relative_humidity_2m, precipitation,
    wind_speed_10m, wind_direction_10m, wind_speed_100m, wind_direction_100m,
    pressure_msl, and cloud_cover. The parameter-variable order is temperature_2m,
    relative_humidity_2m, wind_u_10m, wind_v_10m, wind_u_100m, wind_v_100m,
    pressure_msl, and cloud_cover; each parameter tuple is (phi, xi, mu, sigma).
    Meteorological wind is represented as u = -speed sin(direction) and
    v = -speed cos(direction). Lead 1 draws the initial error as
    e[s,v,1] = mu[s,v,1] + sigma[s,v,1] z[s,v,1]. Later leads use
    e[s,v,l] = phi[s,v,l] e[s,v,l-1] + xi[s,v,l]
               + mu[s,v,l] + sigma[s,v,l] z[s,v,l],
    where all z are independent standard-normal draws. The generated value is ERA5
    plus e in parameter-variable space. Humidity and cloud cover are clipped to
    [0, 100] at every lead, and their clipped errors are used by the next AR step.
    Generated wind components are converted back to speed and direction.

    Precipitation first draws a wet/dry occurrence. At lead 1, its wet probability is
    the fitted hit probability when ERA5 is wet and the fitted false-alarm probability
    otherwise. At later leads it uses P(B[s,l]=1 | B[s,l-1], T[s,l]), where B is the
    generated wet indicator and T is the ERA5 wet indicator. A wet hit is generated as
    ERA5 * exp(mu_ratio + sigma_ratio z); a wet false alarm is drawn from the fitted
    Gamma(shape, scale) distribution; a generated dry hour is zero.

    station_era5_pairs contains (station_index, array) pairs, where station_index is the
    shared station-axis index of the parameter tensors and training data, and each array
    has shape (48, 9). The return value preserves that pair order and contains newly
    allocated (station_index, generated_array) pairs.
    """
    continuous_parameters = np.asarray(continuous_parameters, dtype=np.float64)
    precipitation_parameters = np.asarray(
        precipitation_parameters,
        dtype=np.float64,
    )
    station_count = continuous_parameters.shape[0]

    if continuous_parameters.shape[1:] != (8, 48, 4):
        raise ValueError(
            "continuous_parameters must have shape (station, 8, 48, 4)."
        )
    if precipitation_parameters.shape != (station_count, 48, 12):
        raise ValueError(
            "precipitation_parameters must have shape (station, 48, 12)."
        )
    if not np.isfinite(continuous_parameters).all():
        raise ValueError("continuous_parameters contains non-finite values.")
    if not np.isfinite(precipitation_parameters).all():
        raise ValueError("precipitation_parameters contains non-finite values.")
    if np.any(continuous_parameters[:, :, :, 3] < 0.0):
        raise ValueError("A continuous-variable standard deviation is negative.")
    if np.any(
        (precipitation_parameters[:, :, :8] < 0.0)
        | (precipitation_parameters[:, :, :8] > 1.0)
    ):
        raise ValueError("A precipitation occurrence probability is outside [0, 1].")
    if np.any(precipitation_parameters[:, :, 9] < 0.0):
        raise ValueError("A precipitation log-ratio standard deviation is negative.")
    if np.any(precipitation_parameters[:, :, 10:12] <= 0.0):
        raise ValueError("A precipitation Gamma parameter is not positive.")

    generated_pairs = []

    for station_index, era5_sequence in station_era5_pairs:
        if station_index < 0 or station_index >= station_count:
            raise IndexError(f"Station index {station_index} is outside the parameter tensor.")

        row = station_index
        era5_sequence = np.asarray(era5_sequence, dtype=np.float64)

        if era5_sequence.shape != (LEAD_COUNT, VARIABLE_COUNT):
            raise ValueError(
                f"ERA5 sequence for station index {station_index} must have shape (48, 9)."
            )
        if not np.isfinite(era5_sequence).all():
            raise ValueError(
                f"ERA5 sequence for station index {station_index} contains non-finite values."
            )
        if np.any(era5_sequence[:, PRECIPITATION_CHANNEL] < 0.0):
            raise ValueError(
                f"ERA5 precipitation for station index {station_index} contains negative values."
            )

        era5_direction_10m = np.deg2rad(era5_sequence[:, 4])
        era5_direction_100m = np.deg2rad(era5_sequence[:, 6])
        era5_continuous = np.column_stack([
            era5_sequence[:, 0],
            era5_sequence[:, 1],
            -era5_sequence[:, 3] * np.sin(era5_direction_10m),
            -era5_sequence[:, 3] * np.cos(era5_direction_10m),
            -era5_sequence[:, 5] * np.sin(era5_direction_100m),
            -era5_sequence[:, 5] * np.cos(era5_direction_100m),
            era5_sequence[:, 7],
            era5_sequence[:, 8],
        ])

        station_continuous = continuous_parameters[row]
        generated_continuous = np.empty_like(era5_continuous)
        previous_error = None

        for lead in range(LEAD_COUNT):
            if lead == 0:
                error = rng.normal(
                    station_continuous[:, lead, 2],
                    station_continuous[:, lead, 3],
                )
            else:
                error = (
                    station_continuous[:, lead, 0] * previous_error
                    + station_continuous[:, lead, 1]
                    + rng.normal(
                        station_continuous[:, lead, 2],
                        station_continuous[:, lead, 3],
                    )
                )

            generated_continuous[lead] = era5_continuous[lead] + error
            generated_continuous[lead, BOUNDED_PARAMETER_CHANNELS] = np.clip(
                generated_continuous[lead, BOUNDED_PARAMETER_CHANNELS],
                0.0,
                100.0,
            )
            previous_error = generated_continuous[lead] - era5_continuous[lead]

        generated = era5_sequence.copy()
        generated[:, 0] = generated_continuous[:, 0]
        generated[:, 1] = generated_continuous[:, 1]
        generated[:, 3] = np.hypot(
            generated_continuous[:, 2],
            generated_continuous[:, 3],
        )
        generated[:, 4] = np.degrees(np.arctan2(
            -generated_continuous[:, 2],
            -generated_continuous[:, 3],
        )) % 360.0
        generated[:, 5] = np.hypot(
            generated_continuous[:, 4],
            generated_continuous[:, 5],
        )
        generated[:, 6] = np.degrees(np.arctan2(
            -generated_continuous[:, 4],
            -generated_continuous[:, 5],
        )) % 360.0
        generated[:, 7] = generated_continuous[:, 6]
        generated[:, 8] = generated_continuous[:, 7]

        station_precipitation = precipitation_parameters[row]
        truth_precipitation = era5_sequence[:, PRECIPITATION_CHANNEL]
        generated_precipitation = np.zeros(LEAD_COUNT, dtype=np.float64)
        previous_wet = False

        for lead in range(LEAD_COUNT):
            truth_wet = truth_precipitation[lead] > 0.0

            if lead == 0:
                wet_probability = station_precipitation[
                    lead,
                    4 if truth_wet else 6,
                ]
            else:
                transition_index = 2 * int(previous_wet) + int(truth_wet)
                wet_probability = station_precipitation[lead, transition_index]

            generated_wet = rng.random() < wet_probability

            if generated_wet and truth_wet:
                generated_precipitation[lead] = truth_precipitation[lead] * np.exp(
                    rng.normal(
                        station_precipitation[lead, 8],
                        station_precipitation[lead, 9],
                    )
                )
            elif generated_wet:
                generated_precipitation[lead] = rng.gamma(
                    station_precipitation[lead, 10],
                    station_precipitation[lead, 11],
                )

            previous_wet = generated_wet

        generated[:, PRECIPITATION_CHANNEL] = generated_precipitation
        generated_pairs.append((station_index, generated))

    return generated_pairs
