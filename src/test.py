from src.data_loader import load_accident_data
from src.preprocessing import preprocess_data
from src.risk_engine import calculate_risk
from src.preprocessing import validate_feature_matrix

# Load
data = load_accident_data(
    "data/ETP_4_New_Data_Accidents.csv"
)

# Preprocess
clean_data, clean_features, feature_names = preprocess_data(
    data
)
validation = validate_feature_matrix(
    clean_features
)

print("\nFeature Validation")
print("-" * 40)
print("Shape:", validation["shape"])
print("NaN values:", validation["nan_count"])
print("Infinite values:", validation["inf_count"])
print("Valid:", validation["valid"])

# Risk engine
results = calculate_risk(clean_data)

risk_ranking = results["risk_ranking"]
segment_values = results["segment_values"]
segment_counts = results["segment_counts"]
segment_mean_severity = results["segment_mean_severity"]
risk_indicator = results["risk_indicator"]


print("MARGGANIT RISK ENGINE")
print("=" * 40)

print("Raw records:", len(data))
print("Clean records:", len(clean_data))
print("Feature matrix:", clean_features.shape)
print("Road segments:", len(segment_values))

print("\nTop 10 Risk Segments:\n")

for rank, index in enumerate(
    risk_ranking[:10],
    start=1
):
    start_km = segment_values[index] * 10
    end_km = start_km + 10

    print(
        f"{rank}. "
        f"{start_km:.0f}-{end_km:.0f} km"
        f" | Risk: {risk_indicator[index]:.3f}"
        f" | Accidents: {segment_counts[index]}"
        f" | Mean Severity: "
        f"{segment_mean_severity[index]:.3f}"
    )