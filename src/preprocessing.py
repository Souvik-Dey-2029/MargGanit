import numpy as np


FEATURE_NAMES = [
    "Year",
    "Month",
    "Day",
    "Hour",
    "Minute",
    "Day_of_Week",
    "Location",
    "Chainage_km",
    "RoadSide",
    "Severity",
    "Cause",
    "RoadFeature",
    "RoadCondition",
    "Weather",
    "Vehicle_V1",
    "Vehicle_V2",
]


def remove_duplicates(data):
    """
    Remove exact duplicate accident records.

    Parameters
    ----------
    data : numpy.ndarray
        Raw structured accident data.

    Returns
    -------
    numpy.ndarray
        Deduplicated accident records.
    """
    return np.unique(data)


def extract_datetime_features(clean_data):
    """
    Extract numerical date and time features.

    Returns
    -------
    tuple
        years, months, days, hours, minutes
    """

    dates = clean_data["Date"]

    years = np.array([
        int(date.split("/")[2])
        for date in dates
    ])

    months = np.array([
        int(date.split("/")[1])
        for date in dates
    ])

    days = np.array([
        int(date.split("/")[0])
        for date in dates
    ])

    times = clean_data["Time_of_Accident"]

    hours = np.array([
        int(time.split(":")[0])
        for time in times
    ])

    minutes = np.array([
        int(time.split(":")[1])
        for time in times
    ])

    return years, months, days, hours, minutes


def clean_vehicle_data(clean_data):
    """
    Handle missing second-vehicle values.

    Raw value -1 means no second vehicle / N/A.
    Internally, this is represented as 0.

    Returns
    -------
    tuple
        Cleaned Vehicle_V2 array and original missing-value mask.
    """

    vehicle_v2 = clean_data["Vehicle_Type_Involved_J_V2"]

    v2_missing_mask = vehicle_v2 == -1

    vehicle_v2_clean = vehicle_v2.copy()

    vehicle_v2_clean[v2_missing_mask] = 0

    return vehicle_v2_clean, v2_missing_mask


def create_feature_matrix(clean_data):
    """
    Create the numerical feature matrix used by MargGanit.

    Returns
    -------
    tuple
        clean_features, feature_names
    """

    (
        years,
        months,
        days,
        hours,
        minutes
    ) = extract_datetime_features(clean_data)

    vehicle_v2_clean, _ = clean_vehicle_data(clean_data)

    clean_features = np.column_stack([
        years,
        months,
        days,
        hours,
        minutes,
        clean_data["Day_of_Week"],
        clean_data["Accident_Location_A"],
        clean_data["Accident_Location_A_Chainage_km"],
        clean_data["Accident_Location_A_Chainage_km_RoadSide"],
        clean_data["Accident_Severity_C"],
        clean_data["Causes_D"],
        clean_data["Road_Feature_E"],
        clean_data["Road_Condition_F"],
        clean_data["Weather_Conditions_H"],
        clean_data["Vehicle_Type_Involved_J_V1"],
        vehicle_v2_clean,
    ])

    return clean_features, FEATURE_NAMES


def preprocess_data(data):
    """
    Run the complete MargGanit preprocessing pipeline.

    Parameters
    ----------
    data : numpy.ndarray
        Raw accident records.

    Returns
    -------
    tuple
        clean_data, clean_features, feature_names
    """

    clean_data = remove_duplicates(data)

    clean_features, feature_names = create_feature_matrix(
        clean_data
    )

    return (
        clean_data,
        clean_features,
        feature_names
    )

# VALIDATION

def validate_feature_matrix(clean_features):
    """
    Validate the final numerical feature matrix.

    Returns
    -------
    dict
        Validation results.
    """

    validation = {
        "shape": clean_features.shape,
        "nan_count": int(np.isnan(clean_features).sum()),
        "inf_count": int(np.isinf(clean_features).sum()),
        "valid": True,
    }

    # Basic validity
    if validation["nan_count"] > 0:
        validation["valid"] = False

    if validation["inf_count"] > 0:
        validation["valid"] = False

    # Expected feature ranges
    ranges = {
        "Year": (2003, 2023),
        "Month": (1, 12),
        "Day": (1, 31),
        "Hour": (0, 23),
        "Minute": (0, 59),
        "Day_of_Week": (1, 7),
        "Location": (1, 2),
        "Chainage_km": (0.0, 741.395),
        "RoadSide": (1, 2),
        "Severity": (1, 4),
        "Cause": (1, 8),
        "RoadFeature": (1, 4),
        "RoadCondition": (1, 8),
        "Weather": (1, 12),
        "Vehicle_V1": (1, 15),
        "Vehicle_V2": (0, 15),
    }

    for index, feature_name in enumerate(FEATURE_NAMES):

        minimum, maximum = ranges[feature_name]

        column = clean_features[:, index]

        if column.min() < minimum or column.max() > maximum:
            validation["valid"] = False

    return validation