import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Load data
df = pd.read_csv(
    r"C:\Users\bunsr\OneDrive\Bunsree\OneDrive\Desktop\ML Practice\Bispecific_mAb\ML Practice_Structure_Based_HAC_Module_Bispecific_mAb.csv"
)

# Verify successful load with some randomly selected records
print(df.sample(5))
print(df.describe())

print("First 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

# Feature groups
hydrophobicity_features = [
    "Fab_Total_SASA",
    "Fab_Total_SAP",
    "Num_Hydrophobic_Patches",
    "Largest_Hydrophobic_Patch_Size",
    "Largest_Hydrophobic_Patch_SASA",
    "Max_Hydrophobic_Patch_Intensity",
    "Mean_Hydrophobic_Patch_Intensity",
    "Sum_Total_Hydrophobic_Patch_SASA",
    "Top_Hydrophobic_Patch_Burden",
    "Hydro_Dipole_Moment_X",
    "Hydro_Dipole_Moment_Y",
    "Hydro_Dipole_Moment_Z",
    "Num_Aromatic_Patches",
    "Largest_Aromatic_Patch_Size",
    "Largest_Aromatic_Patch_SASA",
    "Max_Aromatic_Patch_Intensity",
    "Mean_Aromatic_Patch_Intensity",
    "Sum_Total_Aromatic_Patch_SASA",
    "Top_Aromatic_Patch_Burden"
]

aggregation_features = [
    "Fab_Total_SASA",
    "Fab_Total_SAP",
    "Num_Hydrophobic_Patches",
    "Largest_Hydrophobic_Patch_Size",
    "Largest_Hydrophobic_Patch_SASA",
    "Max_Hydrophobic_Patch_Intensity",
    "Mean_Hydrophobic_Patch_Intensity",
    "Sum_Total_Hydrophobic_Patch_SASA",
    "Top_Hydrophobic_Patch_Burden",
    "Num_Aromatic_Patches",
    "Largest_Aromatic_Patch_Size",
    "Largest_Aromatic_Patch_SASA",
    "Max_Aromatic_Patch_Intensity",
    "Mean_Aromatic_Patch_Intensity",
    "Sum_Total_Aromatic_Patch_SASA",
    "Top_Aromatic_Patch_Burden"
]

charge_features = [
    "Net_Charge",
    "Dipole_Moment_X",
    "Dipole_Moment_Y",
    "Dipole_Moment_Z",
    "Complementary_Charge_Patches"
]

# Create three input datasets
X_hydrophobicity = df[hydrophobicity_features]
X_aggregation = df[aggregation_features]
X_charge = df[charge_features]

print("\nHydrophobicity input shape:")
print(X_hydrophobicity.shape)

print("\nAggregation input shape:")
print(X_aggregation.shape)

print("\nCharge input shape:")
print(X_charge.shape)

# ---------- Visualize Feature Distributions ----------

# Hydrophobicity
X_hydrophobicity.hist(figsize=(15, 12), bins=20)
plt.suptitle("Hydrophobicity Feature Distributions")
plt.tight_layout()
plt.show()

# Aggregation
X_aggregation.hist(figsize=(15, 12), bins=20)
plt.suptitle("Aggregation Feature Distributions")
plt.tight_layout()
plt.show()

# Charge
X_charge.hist(figsize=(15, 8), bins=20)
plt.suptitle("Charge Feature Distributions")
plt.tight_layout()
plt.show()

# ---------- UNSUPERVISED LEARNING ----------

# ---------- Hydrophobicity ----------
# Standardize all hydrophobicity input features
hydrophobicity_scaler = StandardScaler()
X_hydrophobicity_scaled = hydrophobicity_scaler.fit_transform(X_hydrophobicity)

# Create PCA model
hydrophobicity_pca = PCA(n_components=1)

# Combine all hydrophobicity features into one derived component
hydrophobicity_score = hydrophobicity_pca.fit_transform(X_hydrophobicity_scaled).flatten()

# Add the derived output to the dataframe
df["Hydrophobicity_Score"] = hydrophobicity_score

# Print PCA information
print("\nHydrophobicity PCA")
print("Number of input features:", X_hydrophobicity.shape[1])
print("Number of derived outputs:", 1)
print("Variance explained by Hydrophobicity Score:", hydrophobicity_pca.explained_variance_ratio_[0])
print("\nHydrophobicity PCA coefficients:")

hydrophobicity_coefficients = pd.DataFrame(hydrophobicity_pca.components_[0], index=hydrophobicity_features, columns=["Coefficient"])
print(hydrophobicity_coefficients)

# Scatterplot of observations using the derived score
plt.figure(figsize=(8, 6))
plt.scatter(range(len(hydrophobicity_score)), hydrophobicity_score, color="green")
plt.xlabel("Observation")
plt.ylabel("Hydrophobicity Score")
plt.title("Derived Hydrophobicity Score")
plt.show()


# ---------- Aggregation ----------
# Standardize all aggregation input features
aggregation_scaler = StandardScaler()
X_aggregation_scaled = aggregation_scaler.fit_transform(X_aggregation)

# Create PCA model
aggregation_pca = PCA(n_components=1)

# Combine all aggregation features into one derived component
aggregation_score = aggregation_pca.fit_transform(X_aggregation_scaled).flatten()

# Add the derived output to the dataframe
df["Aggregation_Score"] = aggregation_score

# Print PCA information
print("\nAggregation PCA")
print("Number of input features:", X_aggregation.shape[1])
print("Number of derived outputs:", 1)
print("Variance explained by Aggregation Score:", aggregation_pca.explained_variance_ratio_[0])
print("\nAggregation PCA coefficients:")

aggregation_coefficients = pd.DataFrame(aggregation_pca.components_[0], index=aggregation_features, columns=["Coefficient"])
print(aggregation_coefficients)

# Scatterplot of observations using the derived score
plt.figure(figsize=(8, 6))
plt.scatter(range(len(aggregation_score)), aggregation_score, color="blue")
plt.xlabel("Observation")
plt.ylabel("Aggregation Score")
plt.title("Derived Aggregation Score")
plt.show()


# ---------- Charge ----------
# Standardize all charge input features
charge_scaler = StandardScaler()
X_charge_scaled = charge_scaler.fit_transform(X_charge)

# Create PCA model
charge_pca = PCA(n_components=1)

# Combine all charge features into one derived component
charge_score = charge_pca.fit_transform(X_charge_scaled).flatten()

# Add the derived output to the dataframe
df["Charge_Score"] = charge_score

# Print PCA information
print("\nCharge PCA")
print("Number of input features:", X_charge.shape[1])
print("Number of derived outputs:", 1)
print("Variance explained by Charge Score:", charge_pca.explained_variance_ratio_[0])
print("\nCharge PCA coefficients:")

charge_coefficients = pd.DataFrame(charge_pca.components_[0], index=charge_features, columns=["Coefficient"])
print(charge_coefficients)

# Scatterplot of observations using the derived score
plt.figure(figsize=(8, 6))
plt.scatter(range(len(charge_score)), charge_score, color="purple")
plt.xlabel("Observation")
plt.ylabel("Charge Score")
plt.title("Derived Charge Score")
plt.show()