import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Load data
df = pd.read_csv(
    r"C:\Users\bunsr\OneDrive\Bunsree\OneDrive\Desktop\ML Practice\BiTE\ML Practice_Structure_Based_HSPTM_Module_BiTE.csv"
)

# Verify successful load with some randomly selected records
print(df.sample(min(5, len(df))))
print(df.describe())

print("First 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

# Feature groups
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
    "CDR_Free_CYS",
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


# Apply the manual drops from the validation script
manual_drops = {
}

def get_final_features(features, group_name):
    X = df[features].copy()

    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise TypeError(
            f"Non-numeric columns in {group_name}: {non_numeric}"
        )

    if X.isnull().any().any():
        missing_columns = X.columns[X.isnull().any()].tolist()
        print(
            f"Missing values detected in {group_name}: {missing_columns}"
        )

    variances = X.var()
    zero_variance_features = set(variances[variances == 0].index)
    #dropped_features = manual_drops[group_name] | zero_variance_features

    final_features = [
        feature for feature in features
        #if feature not in dropped_features
    ]

    if not final_features:
        raise ValueError(f"No usable features remain in {group_name}.")

    print(f"\n{group_name} zero-variance features:")
    print(sorted(zero_variance_features) if zero_variance_features else "None")

    #print(f"{group_name} manually dropped features:")
    #print(sorted(manual_drops[group_name]) if manual_drops[group_name] else "None")

    print(f"{group_name} final features:")
    print(final_features)

    return final_features

# Filter features before creating the input datasets
general_structure_features = get_final_features(
    general_structure_features, "General Structure"
)
ASN_deamidation_features = get_final_features(
    ASN_deamidation_features, "ASN deamidation"
)
ASP_isomerization_features = get_final_features(
    ASP_isomerization_features, "ASP isomerization"
)
CYS_oxidation_features = get_final_features(
    CYS_oxidation_features, "CYS oxidation"
)
Free_CYS_features = get_final_features(
    Free_CYS_features, "Free CYS"
)
GLN_deamidation_features= get_final_features(
    GLN_deamidation_features, "GLN Deamidation"
)
HIS_oxidation_features = get_final_features(
    HIS_oxidation_features, "HIS oxidation"
)
MET_oxidation_features = get_final_features(
    MET_oxidation_features, "MET oxidation"
)
N_glycosylation_features = get_final_features(
    N_glycosylation_features, "N Glycosylation"
)
TRP_oxidation_features = get_final_features(
    TRP_oxidation_features, "TRP oxidation"
)

# Create multiple input datasets using the final feature lists
X_general_structure = df[general_structure_features]
X_ASN_deamidation = df[ASN_deamidation_features]
X_ASP_isomerization = df[ASP_isomerization_features]
X_CYS_oxidation = df[CYS_oxidation_features]
X_Free_CYS = df [Free_CYS_features]
X_GLN_deamidation = df[GLN_deamidation_features]
X_HIS_oxidation = df[HIS_oxidation_features]
X_MET_oxidation = df[MET_oxidation_features]
X_N_glycosylation = df[N_glycosylation_features]
X_TRP_oxidation = df[TRP_oxidation_features]

print("\nGeneral Structure input shape:")
print(X_general_structure.shape)

print("\nASN_deamidation input shape:")
print(X_ASN_deamidation.shape)

print("\nASP_isomerization input shape:")
print(X_ASP_isomerization.shape)

print("\nCYS_oxidation input shape:")
print(X_CYS_oxidation.shape)

print("\nFree_CYS input shape:")
print(X_Free_CYS.shape)

print("\nGLN_deamidation input shape:")
print(X_GLN_deamidation.shape)

print("\nHIS_oxidation input shape:")
print(X_HIS_oxidation.shape)

print("\nMET_oxidation input shape:")
print(X_MET_oxidation.shape)

print("\nN_glycosylation input shape:")
print(X_N_glycosylation.shape)

print("\nTRP_oxidation input shape:")
print(X_TRP_oxidation.shape)

# ---------- Visualize Feature Distributions ----------

# General Structure
X_general_structure.hist(figsize=(15, 12), bins=20)
plt.suptitle("General Structure Feature Distributions")
plt.tight_layout()
plt.show()

# ASN deamidation
X_ASN_deamidation.hist(figsize=(15, 12), bins=20)
plt.suptitle("ASN Deamidation Feature Distributions")
plt.tight_layout()
plt.show()

# ASP isomerization
X_ASP_isomerization.hist(figsize=(15, 12), bins=20)
plt.suptitle("ASP Isomerization Feature Distributions")
plt.tight_layout()
plt.show()

# CYS oxidation
X_CYS_oxidation.hist(figsize=(15, 12), bins=20)
plt.suptitle("CYS Oxidation Feature Distributions")
plt.tight_layout()
plt.show()

# Free CYS
X_Free_CYS.hist(figsize=(15, 12), bins=20)
plt.suptitle("Free CYS Feature Distributions")
plt.tight_layout()
plt.show()

# GLN deamidation
X_GLN_deamidation.hist(figsize=(15, 12), bins=20)
plt.suptitle("GLN Deamidation Feature Distributions")
plt.tight_layout()
plt.show()

# HIS oxidation
X_HIS_oxidation.hist(figsize=(15, 12), bins=20)
plt.suptitle("HIS Oxidation Feature Distributions")
plt.tight_layout()
plt.show()

# MET oxidation
X_MET_oxidation.hist(figsize=(15, 12), bins=20)
plt.suptitle("MET Oxidation Feature Distributions")
plt.tight_layout()
plt.show()

# N-glycosylation
X_N_glycosylation.hist(figsize=(15, 12), bins=20)
plt.suptitle("N-glycosylation Feature Distributions")
plt.tight_layout()
plt.show()

# TRP oxidation
X_TRP_oxidation.hist(figsize=(15, 12), bins=20)
plt.suptitle("TRP Oxidation Feature Distributions")
plt.tight_layout()
plt.show()

# ---------- UNSUPERVISED LEARNING ----------

# ---------- General Structure ----------
# Impute missing values in General Structure input features
general_structure_imputer = SimpleImputer(strategy="mean")
X_general_structure_imputed = general_structure_imputer.fit_transform(X_general_structure)

# Standardize all general structure input features
general_structure_scaler = StandardScaler()
X_general_structure_scaled = general_structure_scaler.fit_transform(X_general_structure_imputed)

# Create PCA model
general_structure_pca = PCA(n_components=1)

# Combine all general structure features into one derived component
general_structure_score = general_structure_pca.fit_transform(X_general_structure_scaled).flatten()

# Add the derived output to the dataframe
df["General_Structure_Score"] = general_structure_score

# Print PCA information
print("\nGeneral Structure PCA")
print("Number of input features:", X_general_structure.shape[1])
print("Number of derived outputs:", 1)
print("Variance explained by General Structure Score:", general_structure_pca.explained_variance_ratio_[0])
print("\nGeneral Structure PCA coefficients:")

general_structure_coefficients = pd.DataFrame(general_structure_pca.components_[0], index=general_structure_features, columns=["Coefficient"])
print(general_structure_coefficients)

# Scatterplot of observations using the derived score
plt.figure(figsize=(8, 6))
plt.scatter(range(len(general_structure_score)), general_structure_score, color="green")
plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Observation")
plt.ylabel("General Structure Score")
plt.title("Derived General Structure Score")
plt.show()

# ---------- ASN deamidation ----------
# Impute missing values in ASN deamidation input features
ASN_deamidation_imputer = SimpleImputer(strategy="mean")
X_ASN_deamidation_imputed = ASN_deamidation_imputer.fit_transform(X_ASN_deamidation)

# Standardize all ASN deamidation input features
ASN_deamidation_scaler = StandardScaler()
X_ASN_deamidation_scaled = ASN_deamidation_scaler.fit_transform(X_ASN_deamidation_imputed)

# Create PCA model
ASN_deamidation_pca = PCA(n_components=1)

# Combine all ASN deamidation features into one derived component
ASN_deamidation_score = ASN_deamidation_pca.fit_transform(X_ASN_deamidation_scaled).flatten()

# Add the derived output to the dataframe
df["ASN_deamidation_Score"] = ASN_deamidation_score

# Print PCA information
print("\nASN deamidation PCA")
print("Number of input features:", X_ASN_deamidation.shape[1])
print("Number of derived outputs:", 1)
print("Variance explained by ASN deamidation Score:", ASN_deamidation_pca.explained_variance_ratio_[0])
print("\nASN deamidation PCA coefficients:")

ASN_deamidation_coefficients = pd.DataFrame(ASN_deamidation_pca.components_[0], index=ASN_deamidation_features, columns=["Coefficient"])
print(ASN_deamidation_coefficients)

# Scatterplot of observations using the derived score
plt.figure(figsize=(8, 6))
plt.scatter(range(len(ASN_deamidation_score)), ASN_deamidation_score, color="blue")
plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Observation")
plt.ylabel("ASN deamidation Score")
plt.title("Derived ASN deamidation Score")
plt.show()

# ---------- ASP isomerization ----------
# Impute missing values in ASP isomerization input features
ASP_isomerization_imputer = SimpleImputer(strategy="mean")
X_ASP_isomerization_imputed = ASP_isomerization_imputer.fit_transform(X_ASP_isomerization)

# Standardize all ASP isomerization input features
ASP_isomerization_scaler = StandardScaler()
X_ASP_isomerization_scaled = ASP_isomerization_scaler.fit_transform(X_ASP_isomerization_imputed)

# Create PCA model
ASP_isomerization_pca = PCA(n_components=1)

# Combine all ASP isomerization features into one derived component
ASP_isomerization_score = ASP_isomerization_pca.fit_transform(X_ASP_isomerization_scaled).flatten()

# Add the derived output to the dataframe
df["ASP_isomerization_Score"] = ASP_isomerization_score

# Print PCA information
print("\nASP isomerization PCA")
print("Number of input features:", X_ASP_isomerization.shape[1])
print("Number of derived outputs:", 1)
print("Variance explained by ASP isomerization Score:", ASP_isomerization_pca.explained_variance_ratio_[0])
print("\nASP isomerization PCA coefficients:")

ASP_isomerization_coefficients = pd.DataFrame(ASP_isomerization_pca.components_[0], index=ASP_isomerization_features, columns=["Coefficient"])
print(ASP_isomerization_coefficients)

# Scatterplot of observations using the derived score
plt.figure(figsize=(8, 6))
plt.scatter(range(len(ASP_isomerization_score)), ASP_isomerization_score, color="purple")
plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Observation")
plt.ylabel("ASP isomerization Score")
plt.title("Derived ASP isomerization Score")
plt.show()

# ---------- CYS oxidation ----------
# Impute missing values in CYS oxidation input features
CYS_oxidation_imputer = SimpleImputer(strategy="mean")
X_CYS_oxidation_imputed = CYS_oxidation_imputer.fit_transform(X_CYS_oxidation)

# Standardize all CYS oxidation input features
CYS_oxidation_scaler = StandardScaler()
X_CYS_oxidation_scaled = CYS_oxidation_scaler.fit_transform(X_CYS_oxidation_imputed)

# Create PCA model
CYS_oxidation_pca = PCA(n_components=1)

# Combine all CYS oxidation features into one derived component
CYS_oxidation_score = CYS_oxidation_pca.fit_transform(X_CYS_oxidation_scaled).flatten()

# Add the derived output to the dataframe
df["CYS_oxidation_Score"] = CYS_oxidation_score

# Print PCA information
print("\nCYS oxidation PCA")
print("Number of input features:", X_CYS_oxidation.shape[1])
print("Number of derived outputs:", 1)
print("Variance explained by CYS oxidation Score:", CYS_oxidation_pca.explained_variance_ratio_[0])
print("\nCYS oxidation PCA coefficients:")

CYS_oxidation_coefficients = pd.DataFrame(CYS_oxidation_pca.components_[0], index=CYS_oxidation_features, columns=["Coefficient"])
print(CYS_oxidation_coefficients)

# Scatterplot of observations using the derived score
plt.figure(figsize=(8, 6))
plt.scatter(range(len(CYS_oxidation_score)), CYS_oxidation_score, color="orange")
plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Observation")
plt.ylabel("CYS oxidation Score")
plt.title("Derived CYS oxidation Score")
plt.show()

# ---------- Free CYS ----------
# Impute missing values in Free CYS input features
Free_CYS_imputer = SimpleImputer(strategy="mean")
X_Free_CYS_imputed = Free_CYS_imputer.fit_transform(X_Free_CYS)

# Standardize all Free CYS input features
Free_CYS_scaler = StandardScaler()
X_Free_CYS_scaled = Free_CYS_scaler.fit_transform(X_Free_CYS_imputed)

# Create PCA model
Free_CYS_pca = PCA(n_components=1)

# Combine all Free CYS features into one derived component
Free_CYS_score = Free_CYS_pca.fit_transform(X_Free_CYS_scaled).flatten()

# Add the derived output to the dataframe
df["Free_CYS_Score"] = Free_CYS_score

# Print PCA information
print("\nFree CYS PCA")
print("Number of input features:", X_Free_CYS.shape[1])
print("Number of derived outputs:", 1)
print("Variance explained by Free CYS Score:", Free_CYS_pca.explained_variance_ratio_[0])
print("\nFree CYS PCA coefficients:")

Free_CYS_coefficients = pd.DataFrame(Free_CYS_pca.components_[0], index=Free_CYS_features, columns=["Coefficient"])
print(Free_CYS_coefficients)

# Scatterplot of observations using the derived score
plt.figure(figsize=(8, 6))
plt.scatter(range(len(Free_CYS_score)), Free_CYS_score, color="red")
plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Observation")
plt.ylabel("Free CYS Score")
plt.title("Derived Free CYS Score")
plt.show()

# ---------- GLN deamidation ----------
# Impute missing values in GLN deamidation input features
GLN_deamidation_imputer = SimpleImputer(strategy="mean")
X_GLN_deamidation_imputed = GLN_deamidation_imputer.fit_transform(X_GLN_deamidation)

# Standardize all GLN deamidation input features
GLN_deamidation_scaler = StandardScaler()
X_GLN_deamidation_scaled = GLN_deamidation_scaler.fit_transform(X_GLN_deamidation_imputed)

# Create PCA model
GLN_deamidation_pca = PCA(n_components=1)

# Combine all GLN deamidation features into one derived component
GLN_deamidation_score = GLN_deamidation_pca.fit_transform(X_GLN_deamidation_scaled).flatten()

# Add the derived output to the dataframe
df["GLN_deamidation_Score"] = GLN_deamidation_score

# Print PCA information
print("\nGLN deamidation PCA")
print("Number of input features:", X_GLN_deamidation.shape[1])
print("Number of derived outputs:", 1)
print("Variance explained by GLN deamidation Score:", GLN_deamidation_pca.explained_variance_ratio_[0])
print("\nGLN deamidation PCA coefficients:")

GLN_deamidation_coefficients = pd.DataFrame(GLN_deamidation_pca.components_[0], index=GLN_deamidation_features, columns=["Coefficient"])
print(GLN_deamidation_coefficients)

# Scatterplot of observations using the derived score
plt.figure(figsize=(8, 6))
plt.scatter(range(len(GLN_deamidation_score)), GLN_deamidation_score, color="brown")
plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Observation")
plt.ylabel("GLN deamidation Score")
plt.title("Derived GLN deamidation Score")
plt.show()

# ---------- HIS oxidation ----------
# Impute missing values in HIS oxidation input features
HIS_oxidation_imputer = SimpleImputer(strategy="mean")
X_HIS_oxidation_imputed = HIS_oxidation_imputer.fit_transform(X_HIS_oxidation)

# Standardize all HIS oxidation input features
HIS_oxidation_scaler = StandardScaler()
X_HIS_oxidation_scaled = HIS_oxidation_scaler.fit_transform(X_HIS_oxidation_imputed)

# Create PCA model
HIS_oxidation_pca = PCA(n_components=1)

# Combine all HIS oxidation features into one derived component
HIS_oxidation_score = HIS_oxidation_pca.fit_transform(X_HIS_oxidation_scaled).flatten()

# Add the derived output to the dataframe
df["HIS_oxidation_Score"] = HIS_oxidation_score

# Print PCA information
print("\nHIS oxidation PCA")
print("Number of input features:", X_HIS_oxidation.shape[1])
print("Number of derived outputs:", 1)
print("Variance explained by HIS oxidation Score:", HIS_oxidation_pca.explained_variance_ratio_[0])
print("\nHIS oxidation PCA coefficients:")

HIS_oxidation_coefficients = pd.DataFrame(HIS_oxidation_pca.components_[0], index=HIS_oxidation_features, columns=["Coefficient"])
print(HIS_oxidation_coefficients)

# Scatterplot of observations using the derived score
plt.figure(figsize=(8, 6))
plt.scatter(range(len(HIS_oxidation_score)), HIS_oxidation_score, color="cyan")
plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Observation")
plt.ylabel("HIS oxidation Score")
plt.title("Derived HIS oxidation Score")
plt.show()

# ---------- MET oxidation ----------
# Impute missing values in MET oxidation input features
MET_oxidation_imputer = SimpleImputer(strategy="mean")
X_MET_oxidation_imputed = MET_oxidation_imputer.fit_transform(X_MET_oxidation)

# Standardize all MET oxidation input features
MET_oxidation_scaler = StandardScaler()
X_MET_oxidation_scaled = MET_oxidation_scaler.fit_transform(X_MET_oxidation_imputed)

# Create PCA model
MET_oxidation_pca = PCA(n_components=1)

# Combine all MET oxidation features into one derived component
MET_oxidation_score = MET_oxidation_pca.fit_transform(X_MET_oxidation_scaled).flatten()

# Add the derived output to the dataframe
df["MET_oxidation_Score"] = MET_oxidation_score

# Print PCA information
print("\nMET oxidation PCA")
print("Number of input features:", X_MET_oxidation.shape[1])
print("Number of derived outputs:", 1)
print("Variance explained by MET oxidation Score:", MET_oxidation_pca.explained_variance_ratio_[0])
print("\nMET oxidation PCA coefficients:")

MET_oxidation_coefficients = pd.DataFrame(MET_oxidation_pca.components_[0], index=MET_oxidation_features, columns=["Coefficient"])
print(MET_oxidation_coefficients)

# Scatterplot of observations using the derived score
plt.figure(figsize=(8, 6))
plt.scatter(range(len(MET_oxidation_score)), MET_oxidation_score, color="magenta")
plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Observation")
plt.ylabel("MET oxidation Score")
plt.title("Derived MET oxidation Score")
plt.show()

# ---------- N-glycosylation ----------
# Impute missing values in N-glycosylation input features
N_glycosylation_imputer = SimpleImputer(strategy="mean")
X_N_glycosylation_imputed = N_glycosylation_imputer.fit_transform(X_N_glycosylation)

# Standardize all N-glycosylation input features
N_glycosylation_scaler = StandardScaler()
X_N_glycosylation_scaled = N_glycosylation_scaler.fit_transform(X_N_glycosylation_imputed)

# Create PCA model
N_glycosylation_pca = PCA(n_components=1)

# Combine all N-glycosylation features into one derived component
N_glycosylation_score = N_glycosylation_pca.fit_transform(X_N_glycosylation_scaled).flatten()

# Add the derived output to the dataframe
df["N_glycosylation_Score"] = N_glycosylation_score

# Print PCA information
print("\nN-glycosylation PCA")
print("Number of input features:", X_N_glycosylation.shape[1])
print("Number of derived outputs:", 1)
print("Variance explained by N-glycosylation Score:", N_glycosylation_pca.explained_variance_ratio_[0])
print("\nN-glycosylation PCA coefficients:")

N_glycosylation_coefficients = pd.DataFrame(N_glycosylation_pca.components_[0], index=N_glycosylation_features, columns=["Coefficient"])
print(N_glycosylation_coefficients)

# Scatterplot of observations using the derived score
plt.figure(figsize=(8, 6))
plt.scatter(range(len(N_glycosylation_score)), N_glycosylation_score, color="olive")
plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Observation")
plt.ylabel("N-glycosylation Score")
plt.title("Derived N-glycosylation Score")
plt.show()

# ---------- TRP oxidation ----------
# Impute missing values in TRP oxidation input features
TRP_oxidation_imputer = SimpleImputer(strategy="mean")
X_TRP_oxidation_imputed = TRP_oxidation_imputer.fit_transform(X_TRP_oxidation)

# Standardize all TRP oxidation input features
TRP_oxidation_scaler = StandardScaler()
X_TRP_oxidation_scaled = TRP_oxidation_scaler.fit_transform(X_TRP_oxidation_imputed)

# Create PCA model
TRP_oxidation_pca = PCA(n_components=1)

# Combine all TRP oxidation features into one derived component
TRP_oxidation_score = TRP_oxidation_pca.fit_transform(X_TRP_oxidation_scaled).flatten()

# Add the derived output to the dataframe
df["TRP_oxidation_Score"] = TRP_oxidation_score

# Print PCA information
print("\nTRP oxidation PCA")
print("Number of input features:", X_TRP_oxidation.shape[1])
print("Number of derived outputs:", 1)
print("Variance explained by TRP oxidation Score:", TRP_oxidation_pca.explained_variance_ratio_[0])
print("\nTRP oxidation PCA coefficients:")

TRP_oxidation_coefficients = pd.DataFrame(TRP_oxidation_pca.components_[0], index=TRP_oxidation_features, columns=["Coefficient"])
print(TRP_oxidation_coefficients)

# Scatterplot of observations using the derived score
plt.figure(figsize=(8, 6))
plt.scatter(range(len(TRP_oxidation_score)), TRP_oxidation_score, color="black")
plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Observation")
plt.ylabel("TRP oxidation Score")
plt.title("Derived TRP oxidation Score")
plt.show()