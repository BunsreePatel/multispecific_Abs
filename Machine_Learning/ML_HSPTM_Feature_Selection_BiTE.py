import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- Settings --------------------

# Load data
df = pd.read_csv(
    r"C:\Users\meeko\OneDrive\Desktop\ML Practice\BiTE\ML Practice_Structure_Based_HSPTM_Module_BiTE.csv"
)

print("Dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

PLOT_SIZE = (16, 12)
PLOT_DPI = 100

# -------------------- Feature groups --------------------

general_structure_features = [
    "Fab_Total_SASA",
    "N_Coil",
    "N_Helix",
    "N_Sheet",
    "Pct_Coil",
    "Pct_Helix",
    "Pct_Sheet",
]

ASN_deamidation_features = [
    "ASN1_AbsSASA",
    "ASN1_BFactor",
    "ASN1_RelSASA",
    "ASN2_AbsSASA",
    "ASN2_BFactor",
    "ASN2_RelSASA",
    "ASN3_AbsSASA",
    "ASN3_BFactor",
    "ASN3_RelSASA",
    "ASN_Deamidation_Count",
    "CDR_ASN_Deamidation",
]

ASP_isomerization_features = [
    "ASP1_AbsSASA",
    "ASP1_BFactor",
    "ASP1_RelSASA",
    "ASP2_AbsSASA",
    "ASP2_BFactor",
    "ASP2_RelSASA",
    "ASP3_AbsSASA",
    "ASP3_BFactor",
    "ASP3_RelSASA",
    "ASP4_AbsSASA",
    "ASP4_BFactor",
    "ASP4_RelSASA",
    "ASP_Isomerization_Count",
    "CDR_ASP_Isomerization",
]

CYS_oxidation_features = [
    "CYS_OX1_AbsSASA",
    "CYS_OX1_BFactor",
    "CYS_OX1_RelSASA",
    "CYS_OX2_AbsSASA",
    "CYS_OX2_BFactor",
    "CYS_OX2_RelSASA",
    "CYS_OX3_AbsSASA",
    "CYS_OX3_BFactor",
    "CYS_OX3_RelSASA",
    "CYS_OX4_AbsSASA",
    "CYS_OX4_BFactor",
    "CYS_OX4_RelSASA",
    "CYS_OX5_AbsSASA",
    "CYS_OX5_BFactor",
    "CYS_OX5_RelSASA", 
    "CYS_OX6_AbsSASA",
    "CYS_OX6_BFactor",
    "CYS_OX6_RelSASA",
    "CYS_Oxidation_Count",
    "CDR_CYS_Oxidation",
]

Free_CYS_features = [
    "Free_CYS1_AbsSASA",
    "Free_CYS1_BFactor",
    "Free_CYS1_RelSASA",
    "Free_CYS2_AbsSASA",
    "Free_CYS2_BFactor",
    "Free_CYS2_RelSASA",
    "Free_CYS3_AbsSASA",
    "Free_CYS3_BFactor",
    "Free_CYS3_RelSASA",
    "Free_CYS4_AbsSASA",
    "Free_CYS4_BFactor",
    "Free_CYS4_RelSASA",
    "Free_CYS5_AbsSASA",
    "Free_CYS5_BFactor",
    "Free_CYS5_RelSASA", 
    "Free_CYS6_AbsSASA",
    "Free_CYS6_BFactor",
    "Free_CYS6_RelSASA",
    "Free_CYS_Count",
    "CDR_CYS_Oxidation",
]

GLN_deamidation_features = [
    "GLN1_AbsSASA",
    "GLN1_BFactor",
    "GLN1_RelSASA",
    "GLN2_AbsSASA",
    "GLN2_BFactor",
    "GLN2_RelSASA",
    "GLN3_AbsSASA",
    "GLN3_BFactor",
    "GLN3_RelSASA",
    "GLN4_AbsSASA",
    "GLN4_BFactor",
    "GLN4_RelSASA",
    "GLN5_AbsSASA",
    "GLN5_BFactor",
    "GLN5_RelSASA", 
    "GLN6_AbsSASA",
    "GLN6_BFactor",
    "GLN6_RelSASA",
    "GLN7_AbsSASA",
    "GLN7_BFactor",
    "GLN7_RelSASA",
    "GLN_Deamidation_Count",
    "CDR_GLN_Deamidation",
]

HIS_oxidation_features = [
    "HIS1_AbsSASA",
    "HIS1_BFactor",
    "HIS1_RelSASA",
    "HIS2_AbsSASA",
    "HIS2_BFactor",
    "HIS2_RelSASA",
    "HIS3_AbsSASA",
    "HIS3_BFactor",
    "HIS3_RelSASA",
    "HIS4_AbsSASA",
    "HIS4_BFactor",
    "HIS4_RelSASA",
    "HIS_Oxidation_Count",
    "CDR_HIS_Oxidation",
]

MET_oxidation_features = [
    "MET1_AbsSASA",
    "MET1_BFactor",
    "MET1_RelSASA",
    "MET2_AbsSASA",
    "MET2_BFactor",
    "MET2_RelSASA",
    "MET3_AbsSASA",
    "MET3_BFactor",
    "MET3_RelSASA",
    "MET_Oxidation_Count",
    "CDR_MET_Oxidation",
]

N_glycosylation_features = [
    "NGLY1_AbsSASA",
    "NGLY1_BFactor",
    "NGLY1_RelSASA",
    "NGLY2_AbsSASA",
    "NGLY2_BFactor",
    "NGLY2_RelSASA",
    "NGLY3_AbsSASA",
    "NGLY3_BFactor",
    "NGLY3_RelSASA",
    "N_Glycosylation_Count",
    "CDR_N_Glycosylation",
]

TRP_oxidation_features = [
    "TRP1_AbsSASA",
    "TRP1_BFactor",
    "TRP1_RelSASA",
    "TRP2_AbsSASA",
    "TRP2_BFactor",
    "TRP2_RelSASA",
    "TRP3_AbsSASA",
    "TRP3_BFactor",
    "TRP3_RelSASA",
    "TRP4_AbsSASA",
    "TRP4_BFactor",
    "TRP4_RelSASA",
    "TRP5_AbsSASA",
    "TRP5_BFactor",
    "TRP5_RelSASA", 
    "TRP6_AbsSASA",
    "TRP6_BFactor",
    "TRP6_RelSASA",
    "TRP7_AbsSASA",
    "TRP7_BFactor",
    "TRP7_RelSASA",
    "TRP8_AbsSASA",
    "TRP8_BFactor",
    "TRP8_RelSASA",
    "TRP9_AbsSASA",
    "TRP9_BFactor",
    "TRP9_RelSASA",
    "TRP10_AbsSASA",
    "TRP10_BFactor",
    "TRP_Oxidation_Count",
    "CDR_TRP_Oxidation",
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
        figsize=(16,16),
        bins=20,
        color="#040454"
    )

    plt.suptitle(
        f"{group_name} Feature Distributions",
        fontsize=16
    )

    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.90,
                        wspace=0.25, hspace=0.5)
    plt.show()

    #plt.tight_layout()
    #plt.show()

    # Visualize feature correlations
    correlation_matrix = X[usable_features].corr()

    plt.figure(figsize=PLOT_SIZE, dpi=PLOT_DPI)
    #plt.figure(figsize=(12, 10))

    plt.imshow(
        correlation_matrix,
        cmap="RdBu_r",
        interpolation="nearest",
        aspect="auto",
        vmin=-1,
        vmax=1
    )

    plt.colorbar(label="Correlation")

    plt.xticks(
        range(len(usable_features)),
        usable_features,
        rotation=90,
        fontsize=9,
    )

    plt.yticks(
        range(len(usable_features)),
        usable_features,
        fontsize=9,
    )

    plt.title(f"{group_name} Feature Correlations")
    plt.subplots_adjust(left=0.30, bottom=0.40, right=0.94, top=0.90)
    plt.show()

    return {
        "usable_features": usable_features,
        "variances": variances,
        "correlations": correlation_matrix
    }


# -------------------- Run validation --------------------

general_structure_results = validate_feature_group(
    df,
    general_structure_features,
    "General Structure"
)

ASN_deamidation_results = validate_feature_group(
    df,
    ASN_deamidation_features,
    "ASN Deamidation"
)

ASP_isomerization_results = validate_feature_group(
    df,
    ASP_isomerization_features,
    "ASP Isomerization"
)

CYS_oxidation_results = validate_feature_group(
    df,
    CYS_oxidation_features,
    "CYS Oxidation"
)

Free_CYS_results = validate_feature_group(
    df,
    Free_CYS_features,
    "Free CYS"
)

GLN_deamidation_results = validate_feature_group(
    df,
    GLN_deamidation_features,
    "GLN Deamidation"
)

HIS_oxidation_results = validate_feature_group(
    df,
    HIS_oxidation_features,
    "HIS Oxidation"
)

MET_oxidation_results = validate_feature_group(
    df,
    MET_oxidation_features,
    "MET Oxidation"
)

N_glycosylation_results = validate_feature_group(
    df,
    N_glycosylation_features,
    "N-Glycosylation"
)

TRP_oxidation_results = validate_feature_group(
    df,
    TRP_oxidation_features,
    "TRP Oxidation"
)


# -------------------- Optional manual feature drops --------------------

ASN_deamidation_to_drop = []
ASP_isomerization_to_drop = []
CYS_oxidation_to_drop = []
Free_CYS_to_drop = []
GLN_deamidation_to_drop = []
HIS_oxidation_to_drop = []
MET_oxidation_to_drop = []
N_glycosylation_to_drop = []
TRP_oxidation_to_drop = []
general_structure_to_drop = []

ASN_deamidation_final = [
    f for f in ASN_deamidation_results["usable_features"]
    if f not in ASN_deamidation_to_drop
]

ASP_isomerization_final = [
    f for f in ASP_isomerization_results["usable_features"]
    if f not in ASP_isomerization_to_drop
]

CYS_oxidation_final = [
    f for f in CYS_oxidation_results["usable_features"]
    if f not in CYS_oxidation_to_drop
]

Free_CYS_final = [
    f for f in Free_CYS_results["usable_features"]
    if f not in Free_CYS_to_drop
]

GLN_deamidation_final = [
    f for f in GLN_deamidation_results["usable_features"]
    if f not in GLN_deamidation_to_drop
]

HIS_oxidation_final = [
    f for f in HIS_oxidation_results["usable_features"]
    if f not in HIS_oxidation_to_drop
]

MET_oxidation_final = [
    f for f in MET_oxidation_results["usable_features"]
    if f not in MET_oxidation_to_drop
]

N_glycosylation_final = [
    f for f in N_glycosylation_results["usable_features"]
    if f not in N_glycosylation_to_drop
]

TRP_oxidation_final = [
    f for f in TRP_oxidation_results["usable_features"]
    if f not in TRP_oxidation_to_drop
]

general_structure_final = [
    f for f in general_structure_results["usable_features"]
    if f not in general_structure_to_drop
]

print("\nFinal General Structure features:", general_structure_final)
print("\nFinal ASN Deamidation features:", ASN_deamidation_final)
print("\nFinal ASP Isomerization features:", ASP_isomerization_final)
print("\nFinal CYS Oxidation features:", CYS_oxidation_final)
print("\nFinal Free CYS features:", Free_CYS_final)
print("\nFinal GLN Deamidation features:", GLN_deamidation_final)
print("\nFinal HIS Oxidation features:", HIS_oxidation_final)
print("\nFinal MET Oxidation features:", MET_oxidation_final)
print("\nFinal N-Glycosylation features:", N_glycosylation_final)
print("\nFinal TRP Oxidation features:", TRP_oxidation_final)

# Plot again using features remaining after manual drops
final_feature_groups = [
    ("General Structure", general_structure_final),
    ("ASN Deamidation", ASN_deamidation_final),
    ("ASP Isomerization", ASP_isomerization_final),
    ("CYS Oxidation", CYS_oxidation_final),
    ("Free CYS", Free_CYS_final),
    ("GLN Deamidation", GLN_deamidation_final),
    ("HIS Oxidation", HIS_oxidation_final),
    ("MET Oxidation", MET_oxidation_final),
    ("N-Glycosylation", N_glycosylation_final),
    ("TRP Oxidation", TRP_oxidation_final),
]


for group_name, features in final_feature_groups:
    print(f"\n{group_name} features after manual drop:", features)

    df[features].hist(figsize=(16, 16), bins=20, color="#040454")
    plt.suptitle(f"{group_name} Feature Distributions — After Manual Drop", fontsize=16)
    plt.subplots_adjust(
        left=0.08, bottom=0.08, right=0.98, top=0.90,
        wspace=0.25, hspace=0.5
    )
    plt.show()

    correlation_matrix = df[features].corr()

    plt.figure(figsize=PLOT_SIZE, dpi=PLOT_DPI)
    plt.imshow(
        correlation_matrix,
        cmap="RdBu_r",
        interpolation="nearest",
        aspect="auto",
        vmin=-1,
        vmax=1
    )
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(features)), features, rotation=90, fontsize=9)
    plt.yticks(range(len(features)), features, fontsize=9)
    plt.title(f"{group_name} Feature Correlations — After Manual Drop")
    plt.subplots_adjust(left=0.30, bottom=0.40, right=0.94, top=0.90)
    plt.show()