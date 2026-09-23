import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- Settings --------------------

# Load data

with open(r"C:\Users\bunsr\OneDrive\Bunsree\OneDrive\Desktop\ML Practice\Bispecific_mAb\ML Practice_Structure_Based_HAC_Module_Bispecific_scFv.csv") as file:
    print(file.read())

df = pd.read_csv(
    r"C:\Users\bunsr\OneDrive\Bunsree\OneDrive\Desktop\ML Practice\Bispecific_scFv\ML Practice_Structure_Based_HAC_Module_Bispecific_scFv.csv"
)

print("Dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# -------------------- Feature groups --------------------

hydrophobicity_features = [
    "Fab_Total_SASA",
    "Fab_Total_SAP",
    "Num_Hydrophobic_Patches",
    "Largest_Hydrophobic_Patch_SASA",
    "Max_Hydrophobic_Patch_Intensity",
    "Mean_Hydrophobic_Patch_Intensity",
    "Sum_Total_Hydrophobic_Patch_SASA",
    "Top_Hydrophobic_Patch_Burden",
    "Num_Aromatic_Patches",
    "Largest_Aromatic_Patch_SASA",
]

aggregation_features = [
    "Fab_Total_SASA",
    "Fab_Total_SAP",
    "Num_Hydrophobic_Patches",
    "Largest_Hydrophobic_Patch_SASA",
    "Max_Hydrophobic_Patch_Intensity",
    "Mean_Hydrophobic_Patch_Intensity",
    "Top_Hydrophobic_Patch_Burden",
    "Largest_Aromatic_Patch_SASA",
]

charge_features = [
    "Net_Charge",
    "Dipole_Moment_X",
    "Dipole_Moment_Y",
    "Dipole_Moment_Z",
    "Complementary_Charge_Patches"
]


# -------------------- Validation function --------------------

def validate_feature_group(df, features, group_name):
    print("\n" + "=" * 80)
    print(group_name.upper())
    print("=" * 80)

    X = df[features].copy()

    # Confirm that all selected features are numeric
    non_numeric = X.select_dtypes(exclude=np.number).columns.tolist()

    if non_numeric:
        raise TypeError(
            f"Non-numeric columns in {group_name}: {non_numeric}"
        )

    # Confirm that there are no missing values
    if X.isnull().any().any():
        raise ValueError(f"Missing values detected in {group_name}")

    # Check for zero-variance features
    variances = X.var()
    zero_variance_features = variances[variances == 0].index.tolist()

    print("\nZero-variance features:")
    print(zero_variance_features if zero_variance_features else "None")

    usable_features = [
        feature for feature in features
        if feature not in zero_variance_features
    ]

    print("\nFeatures ranked by variance:")
    print(variances.to_string())

    print("\nCorrelation matrix:")
    print(X[usable_features].corr().to_string())

    # Visualize feature distributions
    X[usable_features].hist(
        figsize=(16, 12),
        bins=20
    )

    plt.suptitle(
        f"{group_name} Feature Distributions",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()

    # Visualize feature correlations
    correlation_matrix = X[usable_features].corr()

    plt.figure(figsize=(12, 10))

    plt.imshow(
        correlation_matrix,
        cmap="coolwarm",
        interpolation="nearest",
        aspect="auto",
        vmin=-1,
        vmax=1
    )

    plt.colorbar(label="Correlation")

    plt.xticks(
        range(len(usable_features)),
        usable_features,
        rotation=90
    )

    plt.yticks(
        range(len(usable_features)),
        usable_features
    )

    plt.title(f"{group_name} Feature Correlations")
    plt.tight_layout()
    plt.show()

    return {
        "usable_features": usable_features,
        "variances": variances,
        "correlations": correlation_matrix
    }


# -------------------- Run validation --------------------

hydrophobicity_results = validate_feature_group(
    df,
    hydrophobicity_features,
    "Hydrophobicity"
)

aggregation_results = validate_feature_group(
    df,
    aggregation_features,
    "Aggregation"
)

charge_results = validate_feature_group(
    df,
    charge_features,
    "Charge"
)

# -------------------- Manual review and drop --------------------

# Review the printed correlation matrices and heatmaps above, then list which features to drop per group and filter.

hydrophobicity_to_drop = []  # fill in after reviewing output
aggregation_to_drop = []     # fill in after reviewing output
charge_to_drop = []          # fill in after reviewing output

hydrophobicity_final = [
    f for f in hydrophobicity_results["usable_features"]
    if f not in hydrophobicity_to_drop
]

aggregation_final = [
    f for f in aggregation_results["usable_features"]
    if f not in aggregation_to_drop
]

charge_final = [
    f for f in charge_results["usable_features"]
    if f not in charge_to_drop
]

print("\nFinal Hydrophobicity features:", hydrophobicity_final)
print("Final Aggregation features:", aggregation_final)
print("Final Charge features:", charge_final)