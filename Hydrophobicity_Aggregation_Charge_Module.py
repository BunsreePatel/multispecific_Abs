import os
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.SASA import ShrakeRupley
from Bio.Data.PDBData import residue_sasa_scales

warnings.filterwarnings("ignore")

# --- PATHS ---
BASE_DIRS = [
    Path("/home/bunsree/projects/multispecific_Abs/scFv"),
    Path("/home/bunsree/projects/multispecific_Abs/Bispecific_scFv"),
    Path("/home/bunsree/projects/multispecific_Abs/BiTE (Bispecific T-Cell Engager)"),
    Path("/home/bunsree/projects/multispecific_Abs/Bispecific_mAb"),
    Path("/home/bunsree/projects/multispecific_Abs/Whole_mAb")
]

MASTER_CSV = Path("/home/bunsree/projects/multispecific_Abs/TheraSAbDab_SeqStruc_07Dec2025.csv")

"""
THEORY: STRUCTURE-BASED HYDROPHOBICITY AND AGGREGATION MODULE
=============================================================
 
HYDROPHOBICITY FEATURES (3D-Dependent):
 
1. WIMLEY-WHITE SCALE
    - Experimentally derived whole-residue hydrophobicity scale used to score surface-exposed residues for aggregation and hydrophobic patch analysis.
    - Values used are at pH 8; ARG and LYS excluded. Scale provides separate pH 2 and pH 8 values — pH 8 selected as physiologically relevant for antibody developability.
    - Literature: Wimley, W. C., & White, S. H. (1996). Experimentally determined hydrophobicity scale for proteins at membrane interfaces. Nature Structural Biology, 3(10), 842-848. 
    
2. WILKE MAX-ASA
    - A dictionary of maximum solvent accessible surface area (ASA) values for the 20 standard amino acids.
    - In Biopython, these values are used as normalization factors to calculate Relative Solvent Accessibility (RSA) from absolute SASA values.
    - This scale is often preferred over older scales because it provides a tighter upper bound.
    - Literature: Tien, M. Z., Meyer, A. G., Sydykova, D. K., et al. (2013). Maximum allowed solvent accessibilities of residues in proteins. PLoS One, 8(11), e80635.

3. N_MASK
    - Fab-only structures (VH + CH1 / VL + CL) have artificially exposed C-termini because CH1 and CL C-terminal regions are interface residues designed to be buried in full IgG.
      When Fc is removed, these hydrophobic residues become artificially exposed, causing computational tools to incorrectly flag them as aggregation-prone patches.
    - Masking these C-terminal residues prevents false positive hydrophobic/charge patch calls at truncation boundaries, restoring biological relevance to developability predictions by focusing analysis on the actual Fab surface exposed in solution.
      Zeroing out the last N residues of each chain prevents false positive hydrophobic/charge patch calls at the CH1/CL truncation interface before any feature extraction.
    - N_MASK values are conservative engineering estimates based on domain architecture
    - Literature: Röthisberger, D., Honegger, A., & Plückthun, A. (2005). Domain interactions in the Fab fragment: A comparative evaluation of the single-chain Fv and Fab format engineered with variable domains of different stability. Journal of Molecular Biology, 347(4), 773-789.

4. SOLVENT-ACCESSIBLE SURFACE AREA (SASA)
    - Folded (Hydrophobic and Polar) calculated using Shrake-Rupley rolling-probe algorithm (Biopython)
    - Unfolded (Hydrophobic and Polar) approximated from MAX_ASA (Wilke scale)
    - Literature: Shrake, A., & Rupley, J. A. (1973). Environment and exposure to solvent of protein atoms. Lysozyme and insulin. Journal of Molecular Biology, 79(2), 351-371.

5. SPATIAL AGGREGATION PROPENSITY (SAP)
    - SAP quantifies solvent-exposed hydrophobic patches by combining relative SASA with Wimley-White hydrophobicity summed over sidechain neighbors within 5.0 Å.
    - Spatial Cutoffs:
        R = 5.0 Å (SAP sidechain neighbor search radius)
    - Literature: Chennamsetty, N., Voynova, V., Kayser, et al. (2009). Design of therapeutic proteins with enhanced stability. Proceedings of the National Academy of Sciences, 106(29), 11937-11942.
    - Literature: Lauer, T. M., Agrawal, N. J., Chennamsetty, N., et al. (2012). Developability index: A rapid in silico tool for the screening of antibody aggregation propensity. Journal of Pharmaceutical Sciences, 101(1), 102-115.

6. HYDROPHOBIC PATCHES
    - Full characterization of all patches, not just the largest
    - Outputs: number of patches, largest patch size, largest patch SASA, max/mean patch intensity, total patch SASA, top patch burden
    - Distinguishes between a Fab with one dominant patch vs many diffuse patches
    - Spatial Cutoffs:
        PATCH_RADIUS = 6.0 Å  (hydrophobic patch clustering radius)
    - Literature: Waibl, F., Fernández-Quintero, M. L., Wedl, F. S., et al. (2022). Comparison of hydrophobicity scales for predicting biophysical properties of antibodies. Frontiers in Molecular Biosciences, 9, 960194.

7. HYDROPHOBIC DIPOLE MOMENT
    - Captures directional asymmetry of surface hydrophobicity; high magnitude reflects uneven hydrophobic distribution and elevated aggregation/viscosity risk.
    - Literature: Eisenberg, D., Weiss, R. M., Terwilliger, T. C., & Wilcox, W. (1982). Hydrophobic Moments and Protein Structure. Faraday Symp. Chem. Soc., 17, 109-120.

8. AROMATIC PATCHES
    - Identifies spatially clustered aromatic residues (PHE, TYR, TRP) on the protein surface via proximity-based clustering; dense aromatic surface exposure is associated with aggregation risk.
    - Outputs: number of patches, largest patch size, largest patch SASA, max/mean patch intensity, total patch SASA, top patch burden
    - Literature: Burley, S. K., & Petsko, G. A. (1985). Aromatic-aromatic interaction: A mechanism of protein structure stabilization. Science, 229(4708), 23-28.
    
9. NET CHARGE, DIPOLE MOMENT, COMPLEMENTARY CHARGE PATCHES

    - Literature: 

10. VISCOSITY

    - Literature: 

"""

# --- BIOPHYSICAL CONSTANTS AND SCALES ---
HYDROPHOBIC_RESIDUES        = ['ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'PRO', 'TYR', 'TRP']     # TYR included here as hydrophobic: WW_SCALE value = +0.94 kcal/mol at pH 8
AROMATIC_RESIDUES           = ['PHE', 'TYR', 'TRP']
POSITIVE_RESIDUES           = ['ARG', 'LYS', 'HIS']
NEGATIVE_RESIDUES           = ['ASP', 'GLU']

# --- MODULE-SPECIFIC SCALES/THRESHOLDS (from literature) ---
PATCH_SASA_THRESHOLD        = 200.0 # Å²
PATCH_RADIUS                = 6.0   # Å (standard for spatial clustering, smoother patches)
MIN_PATCH_SIZE              = 4     # residues
SURFACE_EXPOSURE_THRESHOLD  = 0.1   # balanced, higher threshold would be 0.2; also possible to make it 0 an remove as a factor

MAX_ASA = residue_sasa_scales["Wilke"]

# WIMLEY-WHITE HYDROPHOBICITY SCALE (kcal/mol)
# Positive = Hydrophobic; scale values are at pH 8, ARG and LYS excluded
WW_SCALE = {
    "ALA": -0.17,
    "ASN": -0.42,
    "ASP": -1.23,
    "CYS": 0.24,
    "GLN": -0.58,
    "GLU": -2.02,
    "GLY": -0.01,
    "HIS": -0.17,
    "ILE": 0.31,
    "LEU": 0.56,
    "MET": 0.23,
    "PHE": 1.13,
    "PRO": -0.45,
    "SER": -0.13,
    "THR": -0.14,
    "TRP": 1.85,
    "TYR": 0.94,
    "VAL": -0.07,
}

# --- TRUNCATION MASKING PARAMETERS ---
"""
Since pdb structures are Fab or Fv structures, the C-termini for the Heavy and Light Chains will be masked.
"""

N_MASK_HEAVY = 10   # Chain A (heavy, CH1 C-terminus)
N_MASK_LIGHT = 5    # Chain B (light, CL C-terminus)

# --- HYDROPHOBICITY, AGGREGATION, AND CHARGE HELPER FUNCTIONS ---
def get_max_asa(resname):
    """Get max ASA for 3-letter residue code."""

    return MAX_ASA.get(resname.upper(), 1.0)


def mask_truncated_termini(chain, chain_id):
    """
    Mask the C-terminal residues of a chain (Fab/Fv artifact) for artifact-free feature extraction.
    Change the field and value as appropriate for each specific module.
    """
    
    residues = list(chain.get_residues())
    if chain_id == 'A':
        n_mask = N_MASK_HEAVY  
    elif chain_id == 'B':
        n_mask = N_MASK_LIGHT
    else:
        print(f"WARNING: Unexpected chain ID '{chain_id}' — skipping truncation masking.")
        return
    for res in residues[-n_mask:]:
        res.abs_sasa = 0.0  
        # Add module-specific helper functions here.


def compute_abs_sasa(chain, chain_id):
    """
    Run Shrake-Rupley SASA on a chain and store residue-level absolute SASA
    as res.abs_sasa, then applies the termini truncation masking.
    """

    sr = ShrakeRupley()
    sr.compute(chain, level="R")

    for res in chain.get_residues():
        res.abs_sasa = getattr(res, "sasa", 0.0)

    mask_truncated_termini(chain, chain_id)


def calculate_spatial_agg_propensity(struct, WW_SCALE, R=5.0):
    """Spatial Aggregation Propensity (SAP) = relative sasa * WW_SCALE hydrophobicity, summed over side-chain neighbors within "R" Å."""
    
    residues = []
    for chain in struct[0]:
        for res in chain.get_residues():
            if res.id[0] != " ":
                continue
            residues.append(res)

    # Calculate and store ww_rel_sap for each residue
    for res in residues:
        abs_sasa = getattr(res, "abs_sasa", 0.0)
        max_asa = get_max_asa(res.get_resname())
        rel_sasa = (abs_sasa / max_asa) if max_asa > 0 else 0.0
        ww = WW_SCALE.get(res.get_resname(), 0.0)
        res.ww_rel_sap = rel_sasa * ww

    # Collect all sidechain atoms from all residues
    sidechain_atoms = [
        atom
        for residue in residues
        for atom in residue.get_atoms()
        if atom.name not in ['N', 'CA', 'C', 'O']
    ]

    if not sidechain_atoms:
       return 0.0
    
    ns = NeighborSearch(sidechain_atoms)    
    residue_saps = {}
    
    for atom_i in sidechain_atoms:
        # Find neighbors within R=5.0Å
        neighbors = ns.search(atom_i.get_coord(), R)
        sap_i = 0.0

        for atom_j in neighbors:
            if atom_i == atom_j:  # Skip self
                continue
            residue_j = atom_j.get_parent()
            sap_i += getattr(residue_j, 'ww_rel_sap', 0.0)
        
        residue_i = atom_i.get_parent()
        residue_saps.setdefault(residue_i, []).append(sap_i)
    
    # Residue SAP = avg atom SAP; chain SAP = sum residue SAP
    total_sap = sum(np.mean(vals) for vals in residue_saps.values())
    return round(total_sap, 3)


def calculate_chain_spatial_agg_propensity(chain, WW_SCALE, R=5.0):
    """Spatial Aggregation Propensity (SAP) = relative sasa * WW_SCALE hydrophobicity, summed over side-chain neighbors within "R" Å."""
    
    residues = []
    for res in chain.get_residues():
        if res.id[0] != " ":
            continue
        residues.append(res)

    # Calculate and store ww_rel_sap for each residue    
    for res in residues:    
        abs_sasa = getattr(res, "abs_sasa", 0.0)
        max_asa = get_max_asa(res.get_resname())
        rel_sasa = (abs_sasa / max_asa) if max_asa > 0 else 0.0
        ww = WW_SCALE.get(res.get_resname(), 0.0)
        res.ww_rel_sap = rel_sasa * ww

    # Collect all sidechain atoms from all residues
    sidechain_atoms = [
        atom
        for residue in residues
        for atom in residue.get_atoms()
        if atom.name not in ['N', 'CA', 'C', 'O']
    ]

    if not sidechain_atoms:
       return 0.0
    
    ns = NeighborSearch(sidechain_atoms)    
    residue_saps = {}
    
    for atom_i in sidechain_atoms:
        # Find neighbors within R=5.0Å
        neighbors = ns.search(atom_i.get_coord(), R)
        sap_i = 0.0

        for atom_j in neighbors:
            if atom_i == atom_j:  # Skip self
                continue
            residue_j = atom_j.get_parent()
            sap_i += getattr(residue_j, 'ww_rel_sap', 0.0)
        
        residue_i = atom_i.get_parent()
        residue_saps.setdefault(residue_i, []).append(sap_i)
    
    # Residue SAP = avg atom SAP; chain SAP = sum residue SAP
    total_sap = sum(np.mean(vals) for vals in residue_saps.values())
    return round(total_sap, 3)


def get_patch_anchor_coord(res):
    """Use beta carbon (CB) atom as anchor; fallback to alpha carbon (CA)."""

    if res.has_id("CB"):
        return res["CB"].get_coord()
    if res.has_id("CA"):
        return res["CA"].get_coord()
    return None


def build_hydrophobic_patch_candidates(struct):
    """
    Build candidate residues from model 0:
    - standard amino acid residue record
    - hydrophobic residue
    - rel_sasa >= SURFACE_EXPOSURE_THRESHOLD
    - has anchor coordinate (CB or CA)
    """

    hydrophobic_patch_candidates = []
    
    # Process all standard residues from all chains
    for chain in struct[0]:
        for res in chain.get_residues():
            if res.id[0] != " ":
                continue
    
            resname = res.get_resname().upper()
            rel_sasa = getattr(res, "rel_sasa", 0.0)      

            if resname not in HYDROPHOBIC_RESIDUES:
                continue
            if rel_sasa < SURFACE_EXPOSURE_THRESHOLD:
                continue

            anchor = get_patch_anchor_coord(res)
            if anchor is None:
                continue

            abs_sasa = getattr(res, "abs_sasa", 0.0)
            ww = WW_SCALE.get(resname, 0.0)

            hydrophobic_patch_candidates.append({
                "chain_id": chain.get_id(),
                "resname": resname,
                "resseq": res.id[1],
                "icode": str(res.id[2]).strip() if res.id[2] is not None else "",
                "coord": anchor,
                "abs_sasa": abs_sasa,
                "ww": ww,
                })

    return hydrophobic_patch_candidates


def build_chain_hydrophobic_patch_candidates(chain):
    """
    Build candidate residues from a single chain:
    - standard amino acid residue record
    - hydrophobic residue
    - rel_sasa >= SURFACE_EXPOSURE_THRESHOLD
    - has anchor coordinate (CB or CA)
    """

    chain_hydrophobic_patch_candidates = []

    # Process all standard residues from a single chain
    for res in chain.get_residues():
        if res.id[0] != " ":
            continue

        resname = res.get_resname().upper()
        rel_sasa = getattr(res, "rel_sasa", 0.0)
    
        if resname not in HYDROPHOBIC_RESIDUES:
            continue
        if rel_sasa < SURFACE_EXPOSURE_THRESHOLD:
            continue

        anchor = get_patch_anchor_coord(res)
        if anchor is None:
            continue

        abs_sasa = getattr(res, "abs_sasa", 0.0)
        ww = WW_SCALE.get(resname, 0.0)

        chain_hydrophobic_patch_candidates.append({
            "chain_id": chain.get_id(),
            "resname": resname,
            "resseq": res.id[1],
            "icode": str(res.id[2]).strip() if res.id[2] is not None else "",
            "coord": anchor,
            "abs_sasa": abs_sasa,
            "ww": ww,
            })

    return chain_hydrophobic_patch_candidates


def build_hydrophobic_patch_graph(hydrophobic_patch_candidates):
    """Connected components over residue anchors with patch cutoff radius."""

    n = len(hydrophobic_patch_candidates)
    hydrophobic_patch_graph = [[] for _ in range(n)]

    for i in range(n):
        ci = hydrophobic_patch_candidates[i]["coord"]
        for j in range(i + 1, n):
            cj = hydrophobic_patch_candidates[j]["coord"]
            if np.linalg.norm(ci - cj) <= PATCH_RADIUS:
                hydrophobic_patch_graph[i].append(j)
                hydrophobic_patch_graph[j].append(i)

    return hydrophobic_patch_graph


def build_chain_hydrophobic_patch_graph(chain_hydrophobic_patch_candidates):
    """Connected components over residue anchors with patch cutoff radius."""
    
    n = len(chain_hydrophobic_patch_candidates)
    chain_hydrophobic_patch_graph = [[] for _ in range(n)]

    for i in range(n):
        ci = chain_hydrophobic_patch_candidates[i]["coord"]
        for j in range(i + 1, n):
            cj = chain_hydrophobic_patch_candidates[j]["coord"]
            if np.linalg.norm(ci - cj) <= PATCH_RADIUS:
                chain_hydrophobic_patch_graph[i].append(j)
                chain_hydrophobic_patch_graph[j].append(i)

    return chain_hydrophobic_patch_graph


def enumerate_hydrophobic_patches(hydrophobic_patch_graph):
    n = len(hydrophobic_patch_graph)
    visited = [False] * n
    hydrophobic_patches = []

    for i in range(n):
        if visited[i]:
            continue

        stack = [i]
        visited[i] = True
        patch = []

        while stack:
            u = stack.pop()
            patch.append(u)
            for v in hydrophobic_patch_graph[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)

        hydrophobic_patches.append(patch)

    return hydrophobic_patches


def enumerate_chain_hydrophobic_patches(chain_hydrophobic_patch_graph):
    n = len(chain_hydrophobic_patch_graph)
    visited = [False] * n
    chain_hydrophobic_patches = []

    for i in range(n):
        if visited[i]:
            continue

        stack = [i]
        visited[i] = True
        patch = []

        while stack:
            u = stack.pop()
            patch.append(u)
            for v in chain_hydrophobic_patch_graph[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)

        chain_hydrophobic_patches.append(patch)

    return chain_hydrophobic_patches


def compute_hydrophobic_patch_metrics(hydrophobic_patches, hydrophobic_patch_candidates):
    hydrophobic_patch_metrics = []

    for patch in hydrophobic_patches:
        patch_size = len(patch)
        patch_sasa = sum(hydrophobic_patch_candidates[i]["abs_sasa"] for i in patch)

        if patch_sasa > 0:
            weighted = sum(
                hydrophobic_patch_candidates[i]["ww"] * hydrophobic_patch_candidates[i]["abs_sasa"]
                for i in patch
            )
            patch_intensity = weighted / patch_sasa
        else:
            patch_intensity = 0.0

        patch_burden = patch_intensity * patch_sasa

        hydrophobic_patch_metrics.append({
            "patch_size": patch_size,
            "patch_sasa": patch_sasa,
            "patch_intensity": patch_intensity,
            "patch_burden": patch_burden,
        })

    return hydrophobic_patch_metrics


def compute_chain_hydrophobic_patch_metrics(chain_hydrophobic_patches, chain_hydrophobic_patch_candidates):
    chain_hydrophobic_patch_metrics = []

    for patch in chain_hydrophobic_patches:
        patch_size = len(patch)
        patch_sasa = sum(chain_hydrophobic_patch_candidates[i]["abs_sasa"] for i in patch)

        if patch_sasa > 0:
            weighted = sum(
                chain_hydrophobic_patch_candidates[i]["ww"] * chain_hydrophobic_patch_candidates[i]["abs_sasa"]
                for i in patch
            )
            patch_intensity = weighted / patch_sasa
        else:
            patch_intensity = 0.0

        patch_burden = patch_intensity * patch_sasa

        chain_hydrophobic_patch_metrics.append({
            "patch_size": patch_size,
            "patch_sasa": patch_sasa,
            "patch_intensity": patch_intensity,
            "patch_burden": patch_burden,
        })

    return chain_hydrophobic_patch_metrics


def filter_valid_hydrophobic_patches(hydrophobic_patch_metrics):
    valid_hydrophobic_patches = [
        p for p in hydrophobic_patch_metrics
        if p["patch_size"] >= MIN_PATCH_SIZE and p["patch_sasa"] >= PATCH_SASA_THRESHOLD
    ]
    return valid_hydrophobic_patches


def filter_valid_chain_hydrophobic_patches(chain_hydrophobic_patch_metrics):
    valid_chain_hydrophobic_patches = [
        p for p in chain_hydrophobic_patch_metrics
        if p["patch_size"] >= MIN_PATCH_SIZE and p["patch_sasa"] >= PATCH_SASA_THRESHOLD
    ]
    return valid_chain_hydrophobic_patches


def summarize_hydrophobic_patches(valid_hydrophobic_patches):
    if not valid_hydrophobic_patches:
        return {
            "Num_Hydrophobic_Patches": 0,
            "Largest_Hydrophobic_Patch_Size": 0,
            "Largest_Hydrophobic_Patch_SASA": 0.0,
            "Max_Hydrophobic_Patch_Intensity": 0.0,
            "Mean_Hydrophobic_Patch_Intensity": 0.0,
            "Sum_Total_Hydrophobic_Patch_SASA": 0.0,
            "Top_Hydrophobic_Patch_Burden": 0.0,
        }

    sizes = [p["patch_size"] for p in valid_hydrophobic_patches]
    sasas = [p["patch_sasa"] for p in valid_hydrophobic_patches]
    intensities = [p["patch_intensity"] for p in valid_hydrophobic_patches]
    burdens = [p["patch_burden"] for p in valid_hydrophobic_patches]

    return {
        "Num_Hydrophobic_Patches": len(valid_hydrophobic_patches),
        "Largest_Hydrophobic_Patch_Size": int(max(sizes)),
        "Largest_Hydrophobic_Patch_SASA": round(max(sasas), 2),
        "Max_Hydrophobic_Patch_Intensity": round(max(intensities), 4),
        "Mean_Hydrophobic_Patch_Intensity": round(float(np.mean(intensities)), 4),
        "Sum_Total_Hydrophobic_Patch_SASA": round(sum(sasas), 2),
        "Top_Hydrophobic_Patch_Burden": round(max(burdens), 4),
    }


def summarize_chain_hydrophobic_patches(valid_chain_hydrophobic_patches):
    if not valid_chain_hydrophobic_patches:
        return {
            "Chain_Num_Hydrophobic_Patches": 0,
            "Chain_Largest_Hydrophobic_Patch_Size": 0,
            "Chain_Largest_Hydrophobic_Patch_SASA": 0.0,
            "Chain_Max_Hydrophobic_Patch_Intensity": 0.0,
            "Chain_Mean_Hydrophobic_Patch_Intensity": 0.0,
            "Chain_Sum_Total_Hydrophobic_Patch_SASA": 0.0,
            "Chain_Top_Hydrophobic_Patch_Burden": 0.0,
        }

    sizes = [p["patch_size"] for p in valid_chain_hydrophobic_patches]
    sasas = [p["patch_sasa"] for p in valid_chain_hydrophobic_patches]
    intensities = [p["patch_intensity"] for p in valid_chain_hydrophobic_patches]
    burdens = [p["patch_burden"] for p in valid_chain_hydrophobic_patches]

    return {
        "Chain_Num_Hydrophobic_Patches": len(valid_chain_hydrophobic_patches),
        "Chain_Largest_Hydrophobic_Patch_Size": int(max(sizes)),
        "Chain_Largest_Hydrophobic_Patch_SASA": round(max(sasas), 2),
        "Chain_Max_Hydrophobic_Patch_Intensity": round(max(intensities), 4),
        "Chain_Mean_Hydrophobic_Patch_Intensity": round(float(np.mean(intensities)), 4),
        "Chain_Sum_Total_Hydrophobic_Patch_SASA": round(sum(sasas), 2),
        "Chain_Top_Hydrophobic_Patch_Burden": round(max(burdens), 4),
    }


def calculate_hydrophobic_dipole_moment(struct, WW_SCALE):
    """Hydrophobic dipole moment(μs1) = sum of WW_SCALE hydrophobicity * unit vector (CA→anchor) per residue,
    calculated for the entire structure (all chains).
    """

    hydro_dipole_moment = np.zeros(3)

    # Process all standard residues for all chains
    for chain in struct[0]:
        for res in chain.get_residues():
            if res.id[0] != " ":
                continue

            resname = res.get_resname().upper()

            # Returns the coordinate of the side-chain anchor atom (CB if present, else CA)
            ca = res["CA"].get_coord() if res.has_id("CA") else None
            anchor = get_patch_anchor_coord(res)
            if ca is None or anchor is None:
                continue
                
            # Vector is computed from the CA atom (backbone) to this anchor (side chain)
            vector = anchor - ca
            norm = np.linalg.norm(vector)
            s_i = vector / norm if norm != 0 else np.zeros(3)
            ww = WW_SCALE.get(resname, 0.0)
            hydro_dipole_moment += ww * s_i

    return hydro_dipole_moment


def calculate_chain_hydrophobic_dipole_moment(chain, WW_SCALE):
    """Hydrophobic dipole moment(μs1) = sum of WW_SCALE hydrophobicity * unit vector (CA→anchor) per residue,
    calculated for a single chain.
    """

    chain_hydro_dipole_moment = np.zeros(3)

    # Process all standard residues for a single chain
    for res in chain.get_residues():
        if res.id[0] != " ":
            continue
    
        resname = res.get_resname().upper()

        # Returns the coordinate of the side-chain anchor atom (CB if present, else CA)
        ca = res["CA"].get_coord() if res.has_id("CA") else None
        anchor = get_patch_anchor_coord(res)
        if ca is None or anchor is None:
            continue
            
        # Vector is computed from the CA atom (backbone) to this anchor (side chain)
        vector = anchor - ca
        norm = np.linalg.norm(vector)
        s_i = vector / norm if norm != 0 else np.zeros(3)
        ww = WW_SCALE.get(resname, 0.0)
        chain_hydro_dipole_moment += ww * s_i

    return chain_hydro_dipole_moment


def build_aromatic_patch_candidates(struct):
    """
    Build candidate residues from model 0:
    - standard amino acid residue record
    - aromatic residue
    - rel_sasa >= SURFACE_EXPOSURE_THRESHOLD
    - has anchor coordinate (CB or CA)
    """
    
    aromatic_patch_candidates = []

    # Process all standard residues from all chains
    for chain in struct[0]:
        for res in chain.get_residues():
            if res.id[0] != " ":
                continue

            resname = res.get_resname().upper()
            rel_sasa = getattr(res, "rel_sasa", 0.0)        # CHECK ON THIS???

            if resname not in AROMATIC_RESIDUES:
                continue
            if rel_sasa < SURFACE_EXPOSURE_THRESHOLD:       # CHECK ON THIS??
                continue

            anchor = get_patch_anchor_coord(res)
            if anchor is None:
                continue

            abs_sasa = getattr(res, "abs_sasa", 0.0)        # CHECK ON THIS??

            aromatic_patch_candidates.append({
                "chain_id": chain.get_id(),
                "resname": resname,
                "resseq": res.id[1],
                "icode": str(res.id[2]).strip() if res.id[2] is not None else "",
                "coord": anchor,
                "abs_sasa": abs_sasa,
                })

    return aromatic_patch_candidates


def build_chain_aromatic_patch_candidates(chain):
    """
    Build candidate residues from a single chain:
    - standard amino acid residue record
    - aromatic residue
    - rel_sasa >= SURFACE_EXPOSURE_THRESHOLD
    - has anchor coordinate (CB or CA)
    """

    chain_aromatic_patch_candidates = []

    # Process all standard residues from a single chain
    for res in chain.get_residues():
        if res.id[0] != " ":
            continue

        resname = res.get_resname().upper()
        rel_sasa = getattr(res, "rel_sasa", 0.0)        # CHECK ON THIS???

        if resname not in AROMATIC_RESIDUES:
            continue
        if rel_sasa < SURFACE_EXPOSURE_THRESHOLD:       # CHECK ON THIS??
            continue

        anchor = get_patch_anchor_coord(res)
        if anchor is None:
            continue

        abs_sasa = getattr(res, "abs_sasa", 0.0)        # CHECK ON THIS??

        chain_aromatic_patch_candidates.append({
            "chain_id": chain.get_id(),
            "resname": resname,
            "resseq": res.id[1],
            "icode": str(res.id[2]).strip() if res.id[2] is not None else "",
            "coord": anchor,
            "abs_sasa": abs_sasa,
            })

    return chain_aromatic_patch_candidates


def build_aromatic_patch_graph(aromatic_patch_candidates):
    """Connected components over residue anchors with patch cutoff radius."""
    n = len(aromatic_patch_candidates)
    aromatic_patch_graph = [[] for _ in range(n)]

    for i in range(n):
        ci = aromatic_patch_candidates[i]["coord"]
        for j in range(i + 1, n):
            cj = aromatic_patch_candidates[j]["coord"]
            if np.linalg.norm(ci - cj) <= PATCH_RADIUS:
                aromatic_patch_graph[i].append(j)
                aromatic_patch_graph[j].append(i)

    return aromatic_patch_graph


def build_chain_aromatic_patch_graph(chain_aromatic_patch_candidates):
    """Connected components over residue anchors with patch cutoff radius."""
    n = len(chain_aromatic_patch_candidates)
    chain_aromatic_patch_graph = [[] for _ in range(n)]

    for i in range(n):
        ci = chain_aromatic_patch_candidates[i]["coord"]
        for j in range(i + 1, n):
            cj = chain_aromatic_patch_candidates[j]["coord"]
            if np.linalg.norm(ci - cj) <= PATCH_RADIUS:
                chain_aromatic_patch_graph[i].append(j)
                chain_aromatic_patch_graph[j].append(i)

    return chain_aromatic_patch_graph


def enumerate_aromatic_patches(aromatic_patch_graph):
    n = len(aromatic_patch_graph)
    visited = [False] * n
    aromatic_patches = []

    for i in range(n):
        if visited[i]:
            continue

        stack = [i]
        visited[i] = True
        patch = []

        while stack:
            u = stack.pop()
            patch.append(u)
            for v in aromatic_patch_graph[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)

        aromatic_patches.append(patch)

    return aromatic_patches


def enumerate_chain_aromatic_patches(chain_aromatic_patch_graph):
    n = len(chain_aromatic_patch_graph)
    visited = [False] * n
    chain_aromatic_patches = []

    for i in range(n):
        if visited[i]:
            continue

        stack = [i]
        visited[i] = True
        patch = []

        while stack:
            u = stack.pop()
            patch.append(u)
            for v in chain_aromatic_patch_graph[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)

        chain_aromatic_patches.append(patch)

    return chain_aromatic_patches


def compute_aromatic_patch_metrics(aromatic_patches, aromatic_patch_candidates):
    aromatic_patch_metrics = []
    
    for patch in aromatic_patches:
        patch_size = len(patch)
        patch_sasa = sum(aromatic_patch_candidates[i]["abs_sasa"] for i in patch)
        
        #Check on this logic since there is no ww scale like for hydrophobic patches!!!
        patch_intensity = patch_sasa / patch_size if patch_size > 0 else 0.0
                
        patch_burden = patch_intensity * patch_sasa

        aromatic_patch_metrics.append({
            "patch_size": patch_size,
            "patch_sasa": patch_sasa,
            "patch_intensity": patch_intensity,
            "patch_burden": patch_burden,
        })

    return aromatic_patch_metrics


def compute_chain_aromatic_patch_metrics(chain_aromatic_patches, chain_aromatic_patch_candidates):
    chain_aromatic_patch_metrics = []
    
    for patch in chain_aromatic_patches:
        patch_size = len(patch)
        patch_sasa = sum(chain_aromatic_patch_candidates[i]["abs_sasa"] for i in patch)
        
        #Check on this logic since there is no ww scale like for hydrophobic patches!!!
        patch_intensity = patch_sasa / patch_size if patch_size > 0 else 0.0
                
        patch_burden = patch_intensity * patch_sasa

        chain_aromatic_patch_metrics.append({
            "patch_size": patch_size,
            "patch_sasa": patch_sasa,
            "patch_intensity": patch_intensity,
            "patch_burden": patch_burden,
        })

    return chain_aromatic_patch_metrics


#CHECK ON THE FILTER THRESHOLD FOR AROMATIC PATCHES!!
def filter_valid_aromatic_patches(aromatic_patch_metrics):
    valid_aromatic_patches = [
        p for p in aromatic_patch_metrics
        if p["patch_size"] >= MIN_PATCH_SIZE and p["patch_sasa"] >= PATCH_SASA_THRESHOLD
    ]
    return valid_aromatic_patches


#CHECK ON THE FILTER THRESHOLD FOR AROMATIC PATCHES!!
def filter_valid_chain_aromatic_patches(chain_aromatic_patch_metrics):
    valid_chain_aromatic_patches = [
        p for p in chain_aromatic_patch_metrics
        if p["patch_size"] >= MIN_PATCH_SIZE and p["patch_sasa"] >= PATCH_SASA_THRESHOLD
    ]
    return valid_chain_aromatic_patches


def summarize_aromatic_patches(valid_aromatic_patches):
    if not valid_aromatic_patches:
        return {
            "Num_Aromatic_Patches": 0,
            "Largest_Aromatic_Patch_Size": 0,
            "Largest_Aromatic_Patch_SASA": 0.0,
            "Max_Aromatic_Patch_Intensity": 0.0,
            "Mean_Aromatic_Patch_Intensity": 0.0,
            "Sum_Total_Aromatic_Patch_SASA": 0.0,
            "Top_Aromatic_Patch_Burden": 0.0,
        }

    sizes = [p["patch_size"] for p in valid_aromatic_patches]
    sasas = [p["patch_sasa"] for p in valid_aromatic_patches]
    intensities = [p["patch_intensity"] for p in valid_aromatic_patches]
    burdens = [p["patch_burden"] for p in valid_aromatic_patches]

    return {
        "Num_Aromatic_Patches": len(valid_aromatic_patches),
        "Largest_Aromatic_Patch_Size": int(max(sizes)),
        "Largest_Aromatic_Patch_SASA": round(max(sasas), 2),
        "Max_Aromatic_Patch_Intensity": round(max(intensities), 4),
        "Mean_Aromatic_Patch_Intensity": round(float(np.mean(intensities)), 4),
        "Sum_Total_Aromatic_Patch_SASA": round(sum(sasas), 2),
        "Top_Aromatic_Patch_Burden": round(max(burdens), 4),
    }


def summarize_chain_aromatic_patches(valid_chain_aromatic_patches):
    if not valid_chain_aromatic_patches:
        return {
            "Chain_Num_Aromatic_Patches": 0,
            "Chain_Largest_Aromatic_Patch_Size": 0,
            "Chain_Largest_Aromatic_Patch_SASA": 0.0,
            "Chain_Max_Aromatic_Patch_Intensity": 0.0,
            "Chain_Mean_Aromatic_Patch_Intensity": 0.0,
            "Chain_Sum_Total_Aromatic_Patch_SASA": 0.0,
            "Chain_Top_Aromatic_Patch_Burden": 0.0,
        }

    sizes = [p["patch_size"] for p in valid_chain_aromatic_patches]
    sasas = [p["patch_sasa"] for p in valid_chain_aromatic_patches]
    intensities = [p["patch_intensity"] for p in valid_chain_aromatic_patches]
    burdens = [p["patch_burden"] for p in valid_chain_aromatic_patches]

    return {
        "Chain_Num_Aromatic_Patches": len(valid_chain_aromatic_patches),
        "Chain_Largest_Aromatic_Patch_Size": int(max(sizes)),
        "Chain_Largest_Aromatic_Patch_SASA": round(max(sasas), 2),
        "Chain_Max_Aromatic_Patch_Intensity": round(max(intensities), 4),
        "Chain_Mean_Aromatic_Patch_Intensity": round(float(np.mean(intensities)), 4),
        "Chain_Sum_Total_Aromatic_Patch_SASA": round(sum(sasas), 2),
        "Chain_Top_Aromatic_Patch_Burden": round(max(burdens), 4),
    }


def calculate_net_charge(struct, pH=7.0):
    """
    Calculate the net charge of the structure (all chains) at a given pH.
    Uses residue-specific pKa values for charged residues.
    """

    # Residue pKa values (approximate)
    pKa_values = {
        "ASP": 3.9,
        "GLU": 4.2,
        "HIS": 6.0,
        "LYS": 10.5, 
        "ARG": 12.5,
        "CYS": 8.3,
        "TYR": 10.1
    }
    net_charge = 0.0

    # Collect all standard residues from all chains
    for chain in struct[0]:
        for res in chain.get_residues():
            if res.id[0] != " ":
                continue

            resname = res.get_resname().upper()
            if resname in pKa_values:
                pKa = pKa_values[resname]
                if resname in ["ASP", "GLU"]:
                    net_charge -= 1 / (1 + 10**(pH - pKa))
                elif resname in ["LYS", "ARG"]:
                    net_charge += 1 / (1 + 10**(pKa - pH))
                elif resname == "HIS":
                    net_charge += 1 / (1 + 10**(pKa - pH))

    return round(net_charge, 2)


def calculate_chain_net_charge(chain, pH=7.0):
    """
    Calculate the net charge of a single chain at a given pH.
    Uses residue-specific pKa values for charged residues.
    """

    # Residue pKa values (approximate)
    pKa_values = {
        "ASP": 3.9,
        "GLU": 4.2,
        "HIS": 6.0,
        "LYS": 10.5, 
        "ARG": 12.5,
        "CYS": 8.3,
        "TYR": 10.1
    }
    chain_net_charge = 0.0

    # Collect all standard residues from a single chain
    for res in chain.get_residues():
        if res.id[0] != " ":
            continue

        resname = res.get_resname().upper()
        if resname in pKa_values:
            pKa = pKa_values[resname]
            if resname in ["ASP", "GLU"]:
                chain_net_charge -= 1 / (1 + 10**(pH - pKa))
            elif resname in ["LYS", "ARG"]:
                chain_net_charge += 1 / (1 + 10**(pKa - pH))
            elif resname == "HIS":
                chain_net_charge += 1 / (1 + 10**(pKa - pH))

    return round(chain_net_charge, 2)


def calculate_dipole_moment(struct):
    """
    Calculates the dipole moment of the structure based on the 3D distribution of charges.
    Returns: dipole magnitude in Debye.
    """

    positive_coords = []
    negative_coords = []
    all_coords = []
  
    # Process all standard residues from all chains
    for chain in struct[0]:
        for res in chain.get_residues():
            if res.id[0] != " ":
                continue

            # Assign charges and collect coordinates
            res_name = res.get_resname().upper()
            atoms = list(res.get_atoms())
            if not atoms:
                continue

            com = np.array([atom.get_coord() for atom in res.get_atoms()]).mean(axis=0)

            if res_name in POSITIVE_RESIDUES:
                charge = 1.0 if res_name in ['ARG', 'LYS'] else 0.1
                positive_coords.append((com, charge))
            elif res_name in NEGATIVE_RESIDUES:
                charge = -1.0
                negative_coords.append((com, -1.0))
            all_coords.extend([atom.get_coord() for atom in atoms])
    
    if not positive_coords and not negative_coords:
        return np.zeros(3)
    
    # Calculate center of mass for the chain
    struct_com = np.array(all_coords).mean(axis=0)
    
    # Calculate dipole vector
    dipole_vector = np.zeros(3)
    for coord, charge in positive_coords:
        dipole_vector += charge * (coord - struct_com)
    for coord, charge in negative_coords:
        dipole_vector += charge * (coord - struct_com)
    
    # Convert to Debye (1 Debye = 0.2082 e·Å)
    dipole_vector_debye = dipole_vector / 0.2082
    return dipole_vector_debye


def calculate_chain_dipole_moment(chain):
    """
    Calculates the dipole moment of a single chain based on the 3D distribution of charges.
    Returns: dipole magnitude in Debye.
    """

    chain_positive_coords = []
    chain_negative_coords = []
    chain_all_coords = []

   
    # Process all standard residues from all chains
    for res in chain.get_residues():
        if res.id[0] != " ":
            continue

    # Assign charges and collect coordinates
        res_name = res.get_resname().upper()
        atoms = list(res.get_atoms())
        if not atoms:
            continue

        com = np.array([atom.get_coord() for atom in res.get_atoms()]).mean(axis=0)

        if res_name in POSITIVE_RESIDUES:
            charge = 1.0 if res_name in ['ARG', 'LYS'] else 0.1
            chain_positive_coords.append((com, charge))
        elif res_name in NEGATIVE_RESIDUES:
            charge = -1.0
            chain_negative_coords.append((com, -1.0))
        chain_all_coords.extend([atom.get_coord() for atom in atoms])
    
    if not chain_positive_coords and not chain_negative_coords:
        return np.zeros(3)
    
    # Calculate center of mass for the chain
    struct_com = np.array(chain_all_coords).mean(axis=0)
    
    # Calculate dipole vector
    chain_dipole_vector = np.zeros(3)
    for coord, charge in chain_positive_coords:
        chain_dipole_vector += charge * (coord - struct_com)
    for coord, charge in chain_negative_coords:
        chain_dipole_vector += charge * (coord - struct_com)
    
    # Convert to Debye (1 Debye = 0.2082 e·Å)
    chain_dipole_vector_debye = chain_dipole_vector / 0.2082
    return chain_dipole_vector_debye


def calculate_complementary_charge_patches(struct):
    """
    Finds spatial proximity of positive and negative patches (struct-level).
    Returns: number of complementary patch pairs
    """

    CHARGE_NEIGHBOR_DIST       = 15.0   # Å 
    CHARGE_CLOSE_NEIGHBOR_DIST = 4.0    # Å 
    CHARGE_MIN_NEIGHBORS       = 3      # residues 

    positive_residues = []
    negative_residues = []
    
    for chain in struct[0]:
        for res in chain.get_residues():
            if res.id[0] != " ":
                continue

            resname =res.get_resname().upper()
            if resname in POSITIVE_RESIDUES:
                positive_residues.append(res)
            elif resname in NEGATIVE_RESIDUES:
                negative_residues.append(res)
    
    if not positive_residues or not negative_residues:
        return 0
    
    # Build atom lists
    pos_atoms = []
    for res in positive_residues:
        pos_atoms.extend(list(res.get_atoms()))
    
    neg_atoms = []
    for res in negative_residues:
        neg_atoms.extend(list(res.get_atoms()))

    # Neighbor searches
    ns_pos = NeighborSearch(pos_atoms)
    ns_neg = NeighborSearch(neg_atoms)
    
    # Count complementary patches within 15Å (interaction range), excluding close contacts
    complementary_charge_patches = 0
    
    # Positive residues → check nearby negative patches
    for pos_res in positive_residues:
        for atom in pos_res.get_atoms():
            neighbors = ns_neg.search(atom.get_coord(), CHARGE_NEIGHBOR_DIST, level='R')
            close_neighbors = ns_neg.search(atom.get_coord(), CHARGE_CLOSE_NEIGHBOR_DIST, level='R')
           
            if len(neighbors) >= CHARGE_MIN_NEIGHBORS and len(close_neighbors) == 0:
                complementary_charge_patches += 1
                break

    # Negative residues → check nearby positive patches
    for neg_res in negative_residues:
        for atom in neg_res.get_atoms():
            neighbors = ns_pos.search(atom.get_coord(), CHARGE_NEIGHBOR_DIST, level='R')
            close_neighbors = ns_pos.search(atom.get_coord(), CHARGE_CLOSE_NEIGHBOR_DIST, level='R')
           
            if len(neighbors) >= CHARGE_MIN_NEIGHBORS and len(close_neighbors) == 0:
                complementary_charge_patches += 1
                break        
    
    return complementary_charge_patches


def calculate_chain_complementary_charge_patches(chain):
    """
    Finds spatial proximity of positive and negative patches (chain-level).
    Returns: number of complementary patch pairs
    """

    CHAIN_CHARGE_NEIGHBOR_DIST       = 15.0   # Å 
    CHAIN_CHARGE_CLOSE_NEIGHBOR_DIST = 4.0    # Å 
    CHAIN_CHARGE_MIN_NEIGHBORS       = 3      # residues 

    positive_residues = []
    negative_residues = []
    
    for res in chain.get_residues():
        if res.id[0] != " ":
            continue

        resname =res.get_resname().upper()
        if resname in POSITIVE_RESIDUES:
            positive_residues.append(res)
        elif resname in NEGATIVE_RESIDUES:
            negative_residues.append(res)
    
    if not positive_residues or not negative_residues:
        return 0
    
    # Build atom lists
    pos_atoms = []
    for res in positive_residues:
        pos_atoms.extend(list(res.get_atoms()))
    
    neg_atoms = []
    for res in negative_residues:
        neg_atoms.extend(list(res.get_atoms()))

    # Neighbor searches
    ns_pos = NeighborSearch(pos_atoms)
    ns_neg = NeighborSearch(neg_atoms)
    
    # Count complementary patches within 15Å (interaction range), excluding close contacts
    chain_complementary_charge_patches = 0
    
    # Positive residues → check nearby negative patches
    for pos_res in positive_residues:
        for atom in pos_res.get_atoms():
            neighbors = ns_neg.search(atom.get_coord(), CHAIN_CHARGE_NEIGHBOR_DIST, level='R')
            close_neighbors = ns_neg.search(atom.get_coord(), CHAIN_CHARGE_CLOSE_NEIGHBOR_DIST, level='R')
           
            if len(neighbors) >= CHAIN_CHARGE_MIN_NEIGHBORS and len(close_neighbors) == 0:
                chain_complementary_charge_patches += 1
                break

    # Negative residues → check nearby positive patches
    for neg_res in negative_residues:
        for atom in neg_res.get_atoms():
            neighbors = ns_pos.search(atom.get_coord(), CHAIN_CHARGE_NEIGHBOR_DIST, level='R')
            close_neighbors = ns_pos.search(atom.get_coord(), CHAIN_CHARGE_CLOSE_NEIGHBOR_DIST, level='R')
           
            if len(neighbors) >= CHAIN_CHARGE_MIN_NEIGHBORS and len(close_neighbors) == 0:
                chain_complementary_charge_patches += 1
                break        
    
    return chain_complementary_charge_patches


# --- MAIN PIPELINE ---
def run_structure_hydrophobicity_aggregation_analysis():
    MASTER_CSV = Path("/home/bunsree/projects/multispecific_Abs/TheraSAbDab_SeqStruc_07Dec2025.csv")
    # Read MASTER_CSV
    df_master = pd.read_csv(MASTER_CSV)
    df_master['key'] = df_master['Therapeutic'].str.lower().str.strip()

    for BASE_DIR in BASE_DIRS:
        if not BASE_DIR.exists():
            print(f"ERROR: PDB directory not found at {BASE_DIR}")
            return

        print(f"\n=== Running analysis for {BASE_DIR} ===")
    
        OUTPUT_CSV = BASE_DIR / "Structure_Based_Hydrophobicity_Aggregation_Charge_Module.csv"
    
        # Recursively find all PDB files under BASE_DIR with minimal duplicate-key protection.
        pdb_map = {}
        duplicate_key_count = 0
        for f in sorted(BASE_DIR.rglob("*.pdb"), key=lambda p: str(p).lower()):
            stem = f.stem.lower()
            if stem.endswith("_fab1") or stem.endswith("_fab2"):
                key = stem
            else:
                key = (
                    stem
                    .replace("_fab", "")            
                    .replace("_fv_bite", "")
                    .replace("_fv1", "")
                    .replace("_scfv", "")
                )
            if key in pdb_map:
                duplicate_key_count += 1
                print(f"WARNING: duplicate PDB key '{key}' -> keeping {pdb_map[key].name}, skipping {f.name}")
                continue
            pdb_map[key] = f

        if duplicate_key_count:
            print(f"WARNING: skipped {duplicate_key_count} duplicate normalized PDB file(s).")

        print(f"Found {len(pdb_map)} PDB files for HYDROPHOBICITY AND AGGREGATION analysis.")
        
        # Setup tools
        parser = PDBParser(QUIET=True)
        master_features = []
        failed_entries = []
        
        # Process every PDB found
        print("--- CALCULATING STRUCTURE-BASED HYDROPHOBICITY AND AGGREGATION ANALYSIS ---")
        for ab_name, pdb_path in tqdm(pdb_map.items()):
            try:
                # Load pdb structure
                struct = parser.get_structure(ab_name, str(pdb_path))

                # Compute abs SASA first, then rel SASA from masked abs SASA
                chain_residues = {}
                for chain in struct[0]:
                    compute_abs_sasa(chain, chain.get_id())
                    count = 0
                    for res in chain:
                        max_asa = get_max_asa(res.get_resname())
                        res.rel_sasa = (res.abs_sasa / max_asa) if max_asa > 0 else 0.0
                        if res.id[0] == " ":
                            count += 1
                    chain_residues[chain.get_id()] = count
                
                # Stucture-level spatial aggregation propensity
                fab_total_sap = calculate_spatial_agg_propensity(struct, WW_SCALE, R=5.0)

                # Structure-level metrics (Fab or Fv structure)
                hydrophobic_patch_candidates = build_hydrophobic_patch_candidates(struct)
                hydrophobic_patch_graph = build_hydrophobic_patch_graph(hydrophobic_patch_candidates)
                hydrophobic_patches = enumerate_hydrophobic_patches(hydrophobic_patch_graph)
                hydrophobic_patch_metrics = compute_hydrophobic_patch_metrics(hydrophobic_patches, hydrophobic_patch_candidates)
                valid_hydrophobic_patches = filter_valid_hydrophobic_patches(hydrophobic_patch_metrics)
                hydrophobic_patch_summary = summarize_hydrophobic_patches(valid_hydrophobic_patches)
                    
                hydro_dipole_moment = calculate_hydrophobic_dipole_moment(struct, WW_SCALE)

                aromatic_patch_candidates = build_aromatic_patch_candidates(struct)
                aromatic_patch_graph = build_aromatic_patch_graph(aromatic_patch_candidates)
                aromatic_patches = enumerate_aromatic_patches(aromatic_patch_graph)
                aromatic_patch_metrics = compute_aromatic_patch_metrics(aromatic_patches, aromatic_patch_candidates)
                valid_aromatic_patches = filter_valid_aromatic_patches(aromatic_patch_metrics)
                aromatic_patch_summary = summarize_aromatic_patches(valid_aromatic_patches)

                net_charge = calculate_net_charge(struct)
                dipole_moment = calculate_dipole_moment(struct)
                complementary_charge_patches = calculate_complementary_charge_patches(struct)
                
                total_residues = sum(chain_residues.values())
                chA_length = chain_residues.get('A', 0)
                chB_length = chain_residues.get('B', 0)
                
                fab_total_sasa = 0.0
                for chain in struct[0]:
                    fab_total_sasa += sum(getattr(res, "abs_sasa", 0.0) for res in chain.get_residues() if res.id[0] == " ") 
                
                # Compile data for each chain
                for chain in struct[0]:
                    residues = []
                    for res in chain.get_residues():
                        if res.id[0] == " ":
                            residues.append(res)
                    
                    chain_sasa = sum(getattr(res, "abs_sasa", 0.0) for res in residues)
                
                    # Chain-level hydrophobic patch evaluation
                    chain_hydrophobic_patch_candidates = build_chain_hydrophobic_patch_candidates(chain)
                    chain_hydrophobic_patch_graph = build_chain_hydrophobic_patch_graph(chain_hydrophobic_patch_candidates)
                    chain_hydrophobic_patches = enumerate_chain_hydrophobic_patches(chain_hydrophobic_patch_graph)
                    chain_hydrophobic_patch_metrics = compute_chain_hydrophobic_patch_metrics(chain_hydrophobic_patches, chain_hydrophobic_patch_candidates)
                    valid_chain_hydrophobic_patches = filter_valid_chain_hydrophobic_patches(chain_hydrophobic_patch_metrics)
                    chain_hydrophobic_patch_summary = summarize_chain_hydrophobic_patches(valid_chain_hydrophobic_patches)

                    # Chain-level spatial aggregation propensity
                    chain_total_sap = calculate_chain_spatial_agg_propensity(chain, WW_SCALE, R=5.0)

                    # Chain-level hydrophobic dipole moment
                    chain_hydro_dipole_moment = calculate_chain_hydrophobic_dipole_moment(chain, WW_SCALE)

                    # Chain-level aromatic patch metrics
                    chain_aromatic_patch_candidates = build_chain_aromatic_patch_candidates(chain)
                    chain_aromatic_patch_graph = build_chain_aromatic_patch_graph(chain_aromatic_patch_candidates)
                    chain_aromatic_patches = enumerate_chain_aromatic_patches(chain_aromatic_patch_graph)
                    chain_aromatic_patch_metrics = compute_chain_aromatic_patch_metrics(chain_aromatic_patches, chain_aromatic_patch_candidates)
                    valid_chain_aromatic_patches = filter_valid_chain_aromatic_patches(chain_aromatic_patch_metrics)
                    chain_aromatic_patch_summary = summarize_chain_aromatic_patches(valid_chain_aromatic_patches)

                    # Chain-level charge-related metrics
                    chain_net_charge = calculate_chain_net_charge(chain)
                    chain_dipole_moment = calculate_chain_dipole_moment(chain)
                    chain_complementary_charge_patches = calculate_chain_complementary_charge_patches(chain)      

                # Detect if Fab1/Fab2 explicitly present
                suffix = ""
                base = ab_name

                if ab_name.endswith("_fab1"):
                    suffix = " Fab1"
                    base = ab_name[:-5]
                elif ab_name.endswith("_fab2"):
                    suffix = " Fab2"
                    base = ab_name[:-5]
                elif ab_name.endswith("_fab"):
                    base = ab_name[:-4]
                         
                base = (
                    base           
                    .replace("_fv_bite", "")
                    .replace("_fv1", "")
                    .replace("_scfv", "")
                ) 

                ab_name_clean = base.capitalize() + suffix
                
                # Compile overall data for the whole Fab (struct-level and per-chain metrics)
                entry = {
                    "Therapeutic": ab_name_clean,
                    "Fab_Total_SASA": round(fab_total_sasa, 2),
                    "Fab_Total_SAP": round(fab_total_sap, 2),
                    "Ch_A_Length": chA_length,
                    "Ch_B_Length": chB_length,
                    "Fab_Total_AAs": total_residues,
                    "Chain_SASA": round(chain_sasa, 2),
                    "Chain_SAP": round(chain_total_sap, 2),

                    # Hydrophobic patch summary (struct-level)
                    "Num_Hydrophobic_Patches": hydrophobic_patch_summary["Num_Hydrophobic_Patches"],
                    "Largest_Hydrophobic_Patch_Size": hydrophobic_patch_summary["Largest_Hydrophobic_Patch_Size"],
                    "Largest_Hydrophobic_Patch_SASA": hydrophobic_patch_summary["Largest_Hydrophobic_Patch_SASA"],
                    "Max_Hydrophobic_Patch_Intensity": hydrophobic_patch_summary["Max_Hydrophobic_Patch_Intensity"],
                    "Mean_Hydrophobic_Patch_Intensity": hydrophobic_patch_summary["Mean_Hydrophobic_Patch_Intensity"],
                    "Sum_Total_Hydrophobic_Patch_SASA": hydrophobic_patch_summary["Sum_Total_Hydrophobic_Patch_SASA"],
                    "Top_Hydrophobic_Patch_Burden": hydrophobic_patch_summary["Top_Hydrophobic_Patch_Burden"],
                    "Hydro_Dipole_Moment_X": round(float(hydro_dipole_moment[0]), 3),
                    "Hydro_Dipole_Moment_Y": round(float(hydro_dipole_moment[1]), 3),
                    "Hydro_Dipole_Moment_Z": round(float(hydro_dipole_moment[2]), 3),
                    # Aromatic patch summary (struct-level)
                    "Num_Aromatic_Patches": aromatic_patch_summary["Num_Aromatic_Patches"],
                    "Largest_Aromatic_Patch_Size": aromatic_patch_summary["Largest_Aromatic_Patch_Size"],
                    "Largest_Aromatic_Patch_SASA": aromatic_patch_summary["Largest_Aromatic_Patch_SASA"],
                    "Max_Aromatic_Patch_Intensity": aromatic_patch_summary["Max_Aromatic_Patch_Intensity"],
                    "Mean_Aromatic_Patch_Intensity": aromatic_patch_summary["Mean_Aromatic_Patch_Intensity"],
                    "Sum_Total_Aromatic_Patch_SASA": aromatic_patch_summary["Sum_Total_Aromatic_Patch_SASA"],
                    "Top_Aromatic_Patch_Burden": aromatic_patch_summary["Top_Aromatic_Patch_Burden"],
                    # Aggregation metrics
                    "Net_Charge": net_charge,
                    "Dipole_Moment_X": round(float(dipole_moment[0]), 3),
                    "Dipole_Moment_Y": round(float(dipole_moment[1]), 3),
                    "Dipole_Moment_Z": round(float(dipole_moment[2]), 3),           
                    "Complementary_Charge_Patches": int(complementary_charge_patches),
                    # Hydrophobic patch summary (chain-level)
                    "Chain_Num_Hydrophobic_Patches": chain_hydrophobic_patch_summary["Chain_Num_Hydrophobic_Patches"],
                    "Chain_Largest_Hydrophobic_Patch_Size": chain_hydrophobic_patch_summary["Chain_Largest_Hydrophobic_Patch_Size"],
                    "Chain_Largest_Hydrophobic_Patch_SASA": chain_hydrophobic_patch_summary["Chain_Largest_Hydrophobic_Patch_SASA"],
                    "Chain_Max_Hydrophobic_Patch_Intensity": chain_hydrophobic_patch_summary["Chain_Max_Hydrophobic_Patch_Intensity"],
                    "Chain_Mean_Hydrophobic_Patch_Intensity": chain_hydrophobic_patch_summary["Chain_Mean_Hydrophobic_Patch_Intensity"],
                    "Chain_Sum_Total_Hydrophobic_Patch_SASA": chain_hydrophobic_patch_summary["Chain_Sum_Total_Hydrophobic_Patch_SASA"],
                    "Chain_Top_Hydrophobic_Patch_Burden": chain_hydrophobic_patch_summary["Chain_Top_Hydrophobic_Patch_Burden"],
                    "Chain_Hydro_Dipole_Moment_X": round(float(chain_hydro_dipole_moment[0]), 3),
                    "Chain_Hydro_Dipole_Moment_Y": round(float(chain_hydro_dipole_moment[1]), 3),
                    "Chain_Hydro_Dipole_Moment_Z": round(float(chain_hydro_dipole_moment[2]), 3),
                    # Aromatic patch summary (chain-level)
                    "Chain_Num_Aromatic_Patches": chain_aromatic_patch_summary["Chain_Num_Aromatic_Patches"],
                    "Chain_Largest_Aromatic_Patch_Size": chain_aromatic_patch_summary["Chain_Largest_Aromatic_Patch_Size"],
                    "Chain_Largest_Aromatic_Patch_SASA": chain_aromatic_patch_summary["Chain_Largest_Aromatic_Patch_SASA"],
                    "Chain_Max_Aromatic_Patch_Intensity": chain_aromatic_patch_summary["Chain_Max_Aromatic_Patch_Intensity"],
                    "Chain_Mean_Aromatic_Patch_Intensity": chain_aromatic_patch_summary["Chain_Mean_Aromatic_Patch_Intensity"],
                    "Chain_Sum_Total_Aromatic_Patch_SASA": chain_aromatic_patch_summary["Chain_Sum_Total_Aromatic_Patch_SASA"],
                    "Chain_Top_Aromatic_Patch_Burden": chain_aromatic_patch_summary["Chain_Top_Aromatic_Patch_Burden"],
                    # Aggregation metrics (chain-level)
                    "Chain_Net_Charge": chain_net_charge,
                    "Chain_Dipole_Moment_X": round(float(chain_dipole_moment[0]), 3),
                    "Chain_Dipole_Moment_Y": round(float(chain_dipole_moment[1]), 3),
                    "Chain_Dipole_Moment_Z": round(float(chain_dipole_moment[2]), 3),           
                    "Chain_Complementary_Charge_Patches": int(chain_complementary_charge_patches),
                }

                # Update with additional summaries
                entry.update(hydrophobic_patch_summary)
                entry.update(aromatic_patch_summary)
                entry.update(chain_hydrophobic_patch_summary)
                entry.update(chain_aromatic_patch_summary)
                
                # Append to master features                
                master_features.append(entry)
           
            except Exception as e:
                print(f"Error processing {ab_name}: {e}")
                failed_entries.append((ab_name, str(e)))

        if failed_entries:
            print(f"WARNING: {len(failed_entries)} structure(s) failed during analysis.")
            for ab_name, err in failed_entries[:10]:
                print(f"  - {ab_name}: {err}")
            if len(failed_entries) > 10:
                print(f"  ... and {len(failed_entries) - 10} more failures")
        
        if master_features:
            df = pd.DataFrame(master_features)

            # create matching keys (strip _fab|_fab1|_fab2|_fv1|_scfv|_fv_bite, lowercase, remove spaces)
            df['key'] = (
                df['Therapeutic']
                .str.lower()
                .str.replace(r'_(fab1|fab2|fab\b|fv_bite|fv1|scfv)', '', regex=True)
                .str.replace(" ", "", regex=False)
                .str.strip()
            )

            #create matching keys
            df_master['key'] = (
                df_master['Therapeutic']
                .str.lower()
                .str.replace(r'_(fab1|fab2|fab\b|fv_bite|fv1|scfv)', '', regex=True)
                .str.replace(" ", "", regex=False)
                .str.strip()
            )

            #df['Therapeutic'].str.lower().str.replace("_fab1", "", regex=False).str.replace("_fab2", "", regex=False).str.replace("_fab", "", regex=False).str.replace("_fv_bite", "", regex=False).str.replace("_fv1", "", regex=False).str.replace("_scfv", "", regex=False).str.replace(" ", "", regex=False).str.strip()
            #df_master['key'] = df_master['Therapeutic'].str.lower().str.replace("_fab1", "", regex=False).str.replace("_fab2", "", regex=False).str.replace("_fab", "", regex=False).str.replace("_fv_bite", "", regex=False).str.replace("_fv1", "", regex=False).str.replace("_scfv", "", regex=False).str.replace(" ", "", regex=False).str.strip()

            # merge on the cleaned 'key' column and report unmatched keys
            df = pd.merge(df, df_master[['key', 'CH1 Isotype', 'VD LC']], on='key', how='left', indicator=True)

            unmatched = df.loc[df['_merge'] == 'left_only', 'Therapeutic'].dropna().astype(str).unique().tolist()
            if unmatched:
                print(f"WARNING: {len(unmatched)} therapeutic key mismatch(es) in merge.")
                print(f"Examples: {', '.join(unmatched[:10])}")
                if len(unmatched) > 10:
                    print(f"... and {len(unmatched) - 10} more")

            # drop temporary merge columns
            df.drop(columns=['key', '_merge'], inplace=True)

            # reorder columns
            cols = ['Therapeutic', 'CH1 Isotype', 'VD LC'] + [c for c in df.columns if c not in ['Therapeutic', 'CH1 Isotype', 'VD LC']]
            df = df[cols]

            # save final CSV
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"\n=== SUCCESS: Hydrophobicity and Aggregation analysis complete for {len(df)} antibodies ===")
            print(f"File saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_structure_hydrophobicity_aggregation_analysis()