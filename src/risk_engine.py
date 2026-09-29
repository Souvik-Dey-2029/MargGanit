import numpy as np


def calculate_risk(clean_data, k=10):
    """
    Calculate the MargGanit risk indicator for each 10-km road segment.

    Parameters
    ----------
    clean_data : numpy.ndarray
        Deduplicated accident records.

    k : int, default=10
        Reliability regularization parameter used to reduce
        small-sample instability.

    Returns
    -------
    dict
        Segment-level risk analysis results.
    """

    chainage = clean_data[
        "Accident_Location_A_Chainage_km"
    ]

    severity = clean_data[
        "Accident_Severity_C"
    ]

    hours = np.array([
        int(time.split(":")[0])
        for time in clean_data["Time_of_Accident"]
    ])

    # ---------------------------------------------------------
    # 1. Create 10-km road segments
    # ---------------------------------------------------------

    segments = np.floor(chainage / 10).astype(int)

    segment_values, segment_counts = np.unique(
        segments,
        return_counts=True
    )

    # ---------------------------------------------------------
    # 2. Spatial concentration score
    # ---------------------------------------------------------

    min_count = segment_counts.min()
    max_count = segment_counts.max()

    concentration_score = (
        (segment_counts - min_count)
        / (max_count - min_count)
    )

    # ---------------------------------------------------------
    # 3. Mean severity per segment
    # ---------------------------------------------------------

    segment_mean_severity = np.array([
        severity[segments == segment].mean()
        for segment in segment_values
    ])

    min_severity = segment_mean_severity.min()
    max_severity = segment_mean_severity.max()

    severity_score = (
        (segment_mean_severity - min_severity)
        / (max_severity - min_severity)
    )

    # Reliability adjustment
    severity_reliability = (
        segment_counts
        / (segment_counts + k)
    )

    adjusted_severity_score = (
        severity_score
        * severity_reliability
    )

    # ---------------------------------------------------------
    # 4. Temporal concentration using entropy
    # ---------------------------------------------------------

    hour_distribution = np.zeros(
        (len(segment_values), 24)
    )

    for i, segment in enumerate(segment_values):

        segment_hours = hours[
            segments == segment
        ]

        values, counts = np.unique(
            segment_hours,
            return_counts=True
        )

        hour_distribution[i, values] = counts

    hour_probabilities = (
        hour_distribution
        / hour_distribution.sum(
            axis=1,
            keepdims=True
        )
    )

    safe_probabilities = np.where(
        hour_probabilities > 0,
        hour_probabilities,
        1
    )

    hour_entropy = -np.sum(
        hour_probabilities
        * np.log2(safe_probabilities),
        axis=1
    )

    min_entropy = hour_entropy.min()
    max_entropy = hour_entropy.max()

    temporal_concentration_score = (
        1
        - (
            (hour_entropy - min_entropy)
            / (max_entropy - min_entropy)
        )
    )

    temporal_reliability = (
        segment_counts
        / (segment_counts + k)
    )

    adjusted_temporal_score = (
        temporal_concentration_score
        * temporal_reliability
    )

    # ---------------------------------------------------------
    # 5. Final MargGanit risk indicator
    # ---------------------------------------------------------

    risk_indicator = (
        concentration_score
        + adjusted_severity_score
        + adjusted_temporal_score
    ) / 3

    # ---------------------------------------------------------
    # 6. Ranking
    # ---------------------------------------------------------

    risk_ranking = np.argsort(
        risk_indicator
    )[::-1]

    return {
        "segment_values": segment_values,
        "segment_counts": segment_counts,
        "segment_mean_severity": segment_mean_severity,
        "concentration_score": concentration_score,
        "severity_score": severity_score,
        "adjusted_severity_score": adjusted_severity_score,
        "temporal_concentration_score": temporal_concentration_score,
        "adjusted_temporal_score": adjusted_temporal_score,
        "risk_indicator": risk_indicator,
        "risk_ranking": risk_ranking,
    }