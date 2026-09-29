from src.data_loader import load_accident_data
from src.preprocessing import (
    preprocess_data,
    validate_feature_matrix,
)
from src.risk_engine import calculate_risk


DATA_PATH = "data/ETP_4_New_Data_Accidents.csv"


def main():
    print("=" * 60)
    print("MARGGANIT — ROAD SAFETY ANALYTICS ENGINE")
    print("=" * 60)

    # 1. Load raw data
    data = load_accident_data(DATA_PATH)

    print(f"\nRaw records: {len(data)}")

    # 2. Preprocess
    clean_data, clean_features, feature_names = preprocess_data(
        data
    )
    validation = validate_feature_matrix(
        clean_features
    )

    if not validation["valid"]:
        raise ValueError(
            "Feature matrix validation failed."
        )

    print(f"Unique records: {len(clean_data)}")
    print(f"Feature matrix: {clean_features.shape}")
    print(f"Data validation: {'PASSED' if validation['valid'] else 'FAILED'}")

    # 3. Calculate risk
    results = calculate_risk(clean_data)

    segment_values = results["segment_values"]
    segment_counts = results["segment_counts"]
    segment_mean_severity = results["segment_mean_severity"]
    risk_indicator = results["risk_indicator"]
    risk_ranking = results["risk_ranking"]

    print(f"Road segments analyzed: {len(segment_values)}")

    # 4. Display top risk segments
    print("\nTOP 10 MARGGANIT RISK SEGMENTS")
    print("-" * 60)

    for rank, index in enumerate(
        risk_ranking[:10],
        start=1
    ):
        start_km = segment_values[index] * 10
        end_km = start_km + 10

        print(
            f"{rank:2}. "
            f"{start_km:.0f}-{end_km:.0f} km"
            f" | Risk: {risk_indicator[index]:.3f}"
            f" | Accidents: {segment_counts[index]}"
            f" | Mean Severity: "
            f"{segment_mean_severity[index]:.3f}"
        )

    print("\n" + "=" * 60)
    print("Analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()