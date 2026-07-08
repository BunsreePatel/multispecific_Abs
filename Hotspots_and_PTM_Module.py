import os
import re
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.DSSP import DSSP
from Bio.Data.PDBData import residue_sasa_scales
from Bio.PDB.Polypeptide import is_aa

warnings.filterwarnings("ignore")

# --- PATHS ---
BASE_DIRS = [
    Path("/home/bunsree/projects/multispecific_Abs/scFv"),
    Path("/home/bunsree/projects/multispecific_Abs/Bispecific_scFv"),
    Path("/home/bunsree/projects/multispecific_Abs/BiTE (Bispecific T-Cell Engager)"),
    Path("/home/bunsree/projects/multispecific_Abs/Bispecific_mAb"),
    Path("/home/bunsree/projects/multispecific_Abs/Whole_mAb")
]

MASTER_CSV = Path("/home/bunsree/projects/rosalind-bioinformatics/multispecific_antibodies/TheraSAbDab_SeqStruc_07Dec2025.csv")

"""
THEORY: STRUCTURE-BASED HOT SPOTS AND PTM SUSCEPTIBILITY MODULE
=============================================================
 
HOT SPOTS AND PTM SUSCEPTIBILITY FEATURES (3D-Dependent):
 
1. WILKE MAX-ASA
    - A dictionary of maximum solvent accessible surface area (ASA) values for the 20 standard amino acids.
    - In Biopython, these values are used as normalization factors to calculate Relative Solvent Accessibility (RSA) from absolute SASA values.
    - This scale is often preferred over older scales because it provides a tighter upper bound.
    - Literature: Tien, M. Z., Meyer, A. G., Sydykova, D. K., Spielman, S. J., & Wilke, C. O. (2013). Maximum allowed solvent accessibilities of residues in proteins. PLoS One, 8(11), e80635.

2. N_MASK
    - Fab-only structures (VH + CH1 / VL + CL) have artificially exposed C-termini because CH1 and CL C-terminal regions are interface residues designed to be buried in full IgG.
      When Fc is removed, these hydrophobic residues become artificially exposed, causing computational tools to incorrectly flag them as aggregation-prone patches.
    - Masking these C-terminal residues prevents false positive hydrophobic/charge patch calls at truncation boundaries, restoring biological relevance to developability predictions by focusing analysis on the actual Fab surface exposed in solution.
      Zeroing out the last N residues of each chain prevents false positive hydrophobic/charge patch calls at the CH1/CL truncation interface before any feature extraction.
    - N_MASK values are conservative engineering estimates based on domain architecture
    - Literature: Röthisberger, D., Honegger, A., & Plückthun, A. (2005). Domain interactions in the Fab fragment: A comparative evaluation of the single-chain Fv and Fab format engineered with variable domains of different stability. Journal of Molecular Biology, 347(4), 773-789.

3. SOLVENT-ACCESSIBLE SURFACE AREA (SASA)
    - Folded (Hydrophobic and Polar) calculated using Shrake-Rupley rolling-probe algorithm (Biopython)
    - Unfolded (Hydrophobic and Polar) approximated from MAX_ASA (Wilke scale)
    - Literature: Shrake, A., & Rupley, J. A. (1973). Environment and exposure to solvent of protein atoms. Lysozyme and insulin. Journal of Molecular Biology, 79(2), 351-371.

4. SECONDARY STRUCTURE FEATURES (DSSP)
    - DSSP (Dictionary of Protein Secondary Structure) assigns secondary structure (helix, sheet, coil) to each residue using hydrogen-bond geometry and backbone dihedral angles.
    - CDR-Proxy uses the three longest contiguous DSSP loop/turn/bend stretches per chain and annotates as CDR_like_1/2/3, a structure-only proxy for true CDR loops since no sequence numbering scheme is used.
    - B-Factors are computed as the mean B-factor across all atoms in a residue and stored with each PTM candidate.
    - Literature: Kabsch, W., & Sander, C. (1983). Dictionary of protein secondary structure: Pattern recognition of hydrogen-bonded and geometrical features. Biopolymers, 22(12), 2577-2637.

5. HOTSPOT & PTM "MOTIFS"
    - Some PTMs are best predicted by sequence, but this tool is intentionally structure-only.
    - N-Glycosylation, ASN Deamidation, ASP Isomerization, GLN Deamidation candidate residues are identified using defined motifs based on 3D proximity, not sequence; Spatial Cutoff of R = 5.0 Å (sidechain neighbor search radius).
    - MET Oxidation, HIS Oxidation, TRP Oxidation, CYS Oxidation surface-exposed residues are identified based on 3D proximity, not sequence; Spatial Cutoff of R = 5.0 Å (sidechain neighbor search radius).
    - Free CYS residues are identified structurally as exposed unpaired cysteines (SG not within DISULFIDE_DIST of another SG).
    - Literature: 
"""

# --- BIOPHYSICAL CONSTANTS AND SCALES ---
HYDROPHOBIC_RESIDUES        = ['ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'PRO', 'TRP']     # UPDATED SINCE TYR IS AMPHIPATHIC, AT PH 7, -OH IS PROTONATED AND NEUTRAL, MAKING IT BEHAVE AS POLAR
POLAR_RESIDUES              = ['SER', 'THR', 'CYS', 'ASN', 'GLN', 'TYR']
AROMATIC_RESIDUES           = ['PHE', 'TYR', 'TRP']
POSITIVE_RESIDUES           = ['ARG', 'LYS', 'HIS']
NEGATIVE_RESIDUES           = ['ASP', 'GLU']

# --- MODULE-SPECIFIC SCALES/THRESHOLDS (from literature) ---
SURFACE_EXPOSURE_THRESHOLD  = 0.1   # balanced, higher threshold would be 0.2; also possible to make it 0 an remove as a factor
DISULFIDE_DIST              = 2.2   # Å SG–SG (literature has it ~ 2.05)

MAX_ASA = residue_sasa_scales["Wilke"]

# --- TRUNCATION MASKING PARAMETERS ---
"""
Since pdb structures are Fab or Fv structures, the C-termini for the Heavy and Light Chains will be masked.
"""

N_MASK_HEAVY = 10   # Chain A (heavy, CH1 C-terminus)
N_MASK_LIGHT = 5    # Chain B (light, CL C-terminus)

# --- HOTSPOTS AND PTM HELPER FUNCTIONS ---
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


def get_patch_anchor_coord(res):
    """Use beta carbon (CB) atom as anchor; fallback to alpha carbon (CA)."""
    
    if res.has_id("CB"):
        return res["CB"].get_coord()
    if res.has_id("CA"):
        return res["CA"].get_coord()
    return None


# --- FAB/FV LEVEL STRUCTURE-BASED CDR MAPPING (LARGEST LOOPS) ---
def annotate_structure_cdr_loops(struct, pdb_path, dssp_executable="/home/bunsree/miniconda3/envs/thermaldssp_env/bin/mkdssp"):
    """
    Annotate the three largest DSSP loop regions (coils/turns/bends) in each chain as a structure-based CDR-proxy.
    Adds .is_cdr and .cdr_name attributes to residues.
    """
    
    dssp = DSSP(struct[0], str(pdb_path), dssp=dssp_executable)
    for chain in struct[0]:
        chain_id = chain.get_id()
        residues = [res for res in chain.get_residues() if res.id[0] == " "]
        dssp_keys = [(chain_id, res.id) for res in residues]
        ss_codes = [dssp[k][2] if k in dssp else "-" for k in dssp_keys]
        loops = []
        current_loop = []
        for res, ss in zip(residues, ss_codes):
            if ss in ("-", "S", "T"):
                current_loop.append(res)
            else:
                if current_loop:
                    loops.append(list(current_loop))
                    current_loop = []
        if current_loop:
            loops.append(list(current_loop))
        loops = sorted(loops, key=len, reverse=True)[:3]
        for idx, loop in enumerate(loops):
            cdr_name = f"CDR_proxy_{idx+1}"
            for res in loop:
                setattr(res, "is_cdr", True)
                setattr(res, "cdr_name", cdr_name)
        for res in residues:
            if not hasattr(res, "is_cdr"):
                setattr(res, "is_cdr", False)
                setattr(res, "cdr_name", None)


def get_cdr_residues(struct):
    """
    Return a list of all residues in the structure that are annotated as CDR-proxies.
    """
    
    cdr_residues = []
    for chain in struct[0]:
        for res in chain.get_residues():
            if hasattr(res, "is_cdr") and res.is_cdr:
                cdr_residues.append(res)
    return cdr_residues


def count_cdr_ptm_sites(ptm_candidates, struct):
    """
    Given a list of PTM candidate dicts and a structure,
    return the number of candidates that are in the CDR-proxies.
    """
    
    count = 0
    for cand in ptm_candidates:
        chain_id = cand["chain_id"]
        resseq = cand["resseq"]
        for res in struct[0][chain_id]:
            if res.id[0] == " " and res.id[1] == resseq:
                if hasattr(res, "is_cdr") and res.is_cdr:
                    count += 1
                break
    return count


# --- FAB/FV LEVEL SECONDARY STRUCTURE HELPER FUNCTIONS ---
def calculate_secondary_structure_features(struct, pdb_path, dssp_executable="/home/bunsree/miniconda3/envs/thermaldssp_env/bin/mkdssp"):
    """
    Calculate DSSP-based secondary structure features.

    Parameters:
    - struct: Biopython structure object (parsed from a .pdb file using PDBParser).
    - pdb_path: Path to the .pdb file
    - dssp_executable: Path to the DSSP executable.

    Returns:
    - secondary_structure_counts: Dictionary with counts of secondary structure elements.
      Example: {"alpha_helix": 45, "beta_sheet": 30, "coil": 25}
    - secondary_structure_percentages: Dictionary with percentages of secondary structure elements.
      Example: {"alpha_helix": 45.0, "beta_sheet": 30.0, "coil": 25.0}
    """
    
    print(f"Running DSSP on {pdb_path} with executable {dssp_executable}")
    
    # Run DSSP on the structure
    dssp = DSSP(struct[0], str(pdb_path), dssp=dssp_executable)

    # Initialize counters
    secondary_structure_counts = {"alpha_helix": 0, "beta_sheet": 0, "coil": 0}
    total_residues = 0

    # Map DSSP codes to secondary structure categories
    dssp_to_category = {
        "H": "alpha_helix",  # Alpha helix
        "G": "alpha_helix",  # 3-10 helix
        "I": "alpha_helix",  # Pi helix
        "E": "beta_sheet",   # Extended strand
        "B": "beta_sheet",   # Isolated beta-bridge
        "T": "coil",         # Turn
        "S": "coil",         # Bend
        "-": "coil",         # Loop/irregular
    }

    # Iterate over DSSP results
    for _, dssp_data in dssp.property_dict.items():
        ss_code = dssp_data[2]  # Secondary structure code
        if ss_code in dssp_to_category:
            category = dssp_to_category[ss_code]
            secondary_structure_counts[category] += 1
            total_residues += 1

    # Calculate percentages
    secondary_structure_percentages = {
        category: (count / total_residues * 100) if total_residues > 0 else 0.0
        for category, count in secondary_structure_counts.items()
    }

    return secondary_structure_counts, secondary_structure_percentages


# --- FAB/FV LEVEL PTM AND HOTSPOT ANALYSIS FUNCTIONS ---
def analyze_n_glycosylation(struct, radius=5.0):
    """
    Identify N-glycosylation motif sites (N-X-S/T, X != Pro) based on 3D proximity, not sequence.
    The most prominent N-glycosylation hotspot is found in immunoglobulin Kappa light chains, 
    heavily associated with the pathogenesis of AL amyloidosis.
    Returns a list of dicts with residue info.
    """
    
    n_glycosylation_candidates = []

    for chain in struct[0]:
        residues = [res for res in chain.get_residues() if is_aa(res, standard=True) and res.id[0] == " "]
        for res_n in residues:
            if res_n.get_resname().upper() != "ASN":
                continue
            anchor_n = get_patch_anchor_coord(res_n)
            if anchor_n is None:
                continue
            rel_sasa = getattr(res_n, "rel_sasa", 0.0)
            if rel_sasa < SURFACE_EXPOSURE_THRESHOLD:
                continue
            # Find all non-Pro neighbors within radius
            for res_x in residues:
                if res_x is res_n or res_x.get_resname().upper() == "PRO":
                    continue
                anchor_x = get_patch_anchor_coord(res_x)
                if anchor_x is None or np.linalg.norm(anchor_n - anchor_x) > radius:
                    continue
                # Find all SER/THR neighbors within radius of both ASN and X
                for res_st in residues:
                    if res_st in (res_n, res_x):
                        continue
                    if res_st.get_resname().upper() not in ("SER", "THR"):
                        continue
                    anchor_st = get_patch_anchor_coord(res_st)
                    if anchor_st is None:
                        continue
                    if (np.linalg.norm(anchor_n - anchor_st) <= radius and
                        np.linalg.norm(anchor_x - anchor_st) <= radius):
                        n_glycosylation_candidates.append({
                            "chain_id": chain.get_id(),
                            "resname": res_n.get_resname(),
                            "resseq": res_n.id[1],
                            "icode": str(res_n.id[2]).strip() if res_n.id[2] is not None else "",
                            "coord": anchor_n,
                            "rel_sasa": rel_sasa,
                            "abs_sasa": getattr(res_n, "abs_sasa", 0.0),
                            "bfactor": np.mean([atom.get_bfactor() for atom in res_n.get_atoms()]),
                        })
                        break
    
    result = {
        "N_Glycosylation_Count": len(n_glycosylation_candidates)
    }

    for i, site in enumerate(n_glycosylation_candidates, start=1):
        result[f"NGLY{i}_Chain"] = site["chain_id"]
        result[f"NGLY{i}_ResName"] = site["resname"]
        result[f"NGLY{i}_ResSeq"] = site["resseq"]
        result[f"NGLY{i}_ICode"] = site["icode"]
        result[f"NGLY{i}_X"] = site["coord"][0]
        result[f"NGLY{i}_Y"] = site["coord"][1]
        result[f"NGLY{i}_Z"] = site["coord"][2]
        result[f"NGLY{i}_RelSASA"] = site["rel_sasa"]
        result[f"NGLY{i}_AbsSASA"] = site["abs_sasa"]
        result[f"NGLY{i}_BFactor"] = site["bfactor"]

    return n_glycosylation_candidates, result
    

def analyze_asn_deamidation(struct, radius=5.0):
    """
    Identify ASN deamidation motif sites (N-G) based on 3D proximity, not sequence.
    NG is highest-risk deamidation motif; succinimide intermediate converts
    ASN to ASP/isoASP causing charge heterogeneity and potency loss.
    An ammonia molecule is expelled.
    Returns a list of dicts with residue info.
    """
    
    asn_deamidation_candidates = []
    for chain in struct[0]:
        residues = [res for res in chain.get_residues() if is_aa(res, standard=True) and res.id[0] == " "]
        for res_n in residues:
            if res_n.get_resname().upper() != "ASN":
                continue
            anchor_n = get_patch_anchor_coord(res_n)
            if anchor_n is None:
                continue
            rel_sasa = getattr(res_n, "rel_sasa", 0.0)
            if rel_sasa < SURFACE_EXPOSURE_THRESHOLD:
                continue
            # Find all GLY neighbors within radius
            for res_g in residues:
                if res_g is res_n or res_g.get_resname().upper() != "GLY":
                    continue
                anchor_g = get_patch_anchor_coord(res_g)
                if anchor_g is None or np.linalg.norm(anchor_n - anchor_g) > radius:
                    continue
                asn_deamidation_candidates.append({
                    "chain_id": chain.get_id(),
                    "resname": res_n.get_resname(),
                    "resseq": res_n.id[1],
                    "icode": str(res_n.id[2]).strip() if res_n.id[2] is not None else "",
                    "coord": anchor_n,
                    "rel_sasa": rel_sasa,
                    "abs_sasa": getattr(res_n, "abs_sasa", 0.0),
                    "bfactor": np.mean([atom.get_bfactor() for atom in res_n.get_atoms()]),
                })
                break  # Only need one valid motif per ASN
    
    result = {
        "ASN_Deamidation_Count": len(asn_deamidation_candidates)
    }

    for i, site in enumerate(asn_deamidation_candidates, start=1):
        result[f"ASN{i}_Chain"] = site["chain_id"]
        result[f"ASN{i}_ResName"] = site["resname"]
        result[f"ASN{i}_ResSeq"] = site["resseq"]
        result[f"ASN{i}_ICode"] = site["icode"]
        result[f"ASN{i}_X"] = site["coord"][0]
        result[f"ASN{i}_Y"] = site["coord"][1]
        result[f"ASN{i}_Z"] = site["coord"][2]
        result[f"ASN{i}_RelSASA"] = site["rel_sasa"]
        result[f"ASN{i}_AbsSASA"] = site["abs_sasa"]
        result[f"ASN{i}_BFactor"] = site["bfactor"]

    return asn_deamidation_candidates, result        


def analyze_asp_isomerization(struct, radius=5.0):
    """
    Identify ASP isomerization motif sites (D-G) based on 3D proximity, not sequence.
    ASP cyclizes via succinimide intermediate to isoASP directly (not deamidation).
    IsoASP introduces backbone shift that can disrupt CDR binding geometry.
    A water molecule is expelled via a dehydration reaction.
    Returns a list of dicts with residue info.
    """
    
    asp_isomerization_candidates = []
    for chain in struct[0]:
        residues = [res for res in chain.get_residues() if is_aa(res, standard=True) and res.id[0] == " "]
        for res_d in residues:
            if res_d.get_resname().upper() != "ASP":
                continue
            anchor_d = get_patch_anchor_coord(res_d)
            if anchor_d is None:
                continue
            rel_sasa = getattr(res_d, "rel_sasa", 0.0)
            if rel_sasa < SURFACE_EXPOSURE_THRESHOLD:
                continue
            # Find all GLY neighbors within radius
            for res_g in residues:
                if res_g is res_d or res_g.get_resname().upper() != "GLY":
                    continue
                anchor_g = get_patch_anchor_coord(res_g)
                if anchor_g is None or np.linalg.norm(anchor_d - anchor_g) > radius:
                    continue
                asp_isomerization_candidates.append({
                    "chain_id": chain.get_id(),
                    "resname": res_d.get_resname(),
                    "resseq": res_d.id[1],
                    "icode": str(res_d.id[2]).strip() if res_d.id[2] is not None else "",
                    "coord": anchor_d,
                    "rel_sasa": rel_sasa,
                    "abs_sasa": getattr(res_d, "abs_sasa", 0.0),
                    "bfactor": np.mean([atom.get_bfactor() for atom in res_d.get_atoms()]),
                })
                break  # Only need one valid motif per ASP

    result = {
        "ASP_Isomerization_Count": len(asp_isomerization_candidates)
    }

    for i, site in enumerate(asp_isomerization_candidates, start=1):
        result[f"ASP{i}_Chain"] = site["chain_id"]
        result[f"ASP{i}_ResName"] = site["resname"]
        result[f"ASP{i}_ResSeq"] = site["resseq"]
        result[f"ASP{i}_ICode"] = site["icode"]
        result[f"ASP{i}_X"] = site["coord"][0]
        result[f"ASP{i}_Y"] = site["coord"][1]
        result[f"ASP{i}_Z"] = site["coord"][2]
        result[f"ASP{i}_RelSASA"] = site["rel_sasa"]
        result[f"ASP{i}_AbsSASA"] = site["abs_sasa"]
        result[f"ASP{i}_BFactor"] = site["bfactor"]

    return asp_isomerization_candidates, result   


def analyze_gln_deamidation(struct, radius=5.0):
    """
    Identify GLN deamidation motif sites (Q-N/G/S) based on 3D proximity, not sequence.
    GLN deamidation is slower than ASN.
    Returns a list of dicts with residue info and structural features.
    """
    
    gln_deamidation_candidates = []
    for chain in struct[0]:
        residues = [res for res in chain.get_residues() if is_aa(res, standard=True) and res.id[0] == " "]
        for res_q in residues:
            if res_q.get_resname().upper() != "GLN":
                continue
            anchor_q = get_patch_anchor_coord(res_q)
            if anchor_q is None:
                continue
            rel_sasa = getattr(res_q, "rel_sasa", 0.0)
            if rel_sasa < SURFACE_EXPOSURE_THRESHOLD:
                continue
            # Find all ASN/GLY/SER neighbors within radius
            for res_next in residues:
                if res_next is res_q or res_next.get_resname().upper() not in ("ASN", "GLY", "SER"):
                    continue
                anchor_next = get_patch_anchor_coord(res_next)
                if anchor_next is None or np.linalg.norm(anchor_q - anchor_next) > radius:
                    continue
                gln_deamidation_candidates.append({
                    "chain_id": chain.get_id(),
                    "resname": res_q.get_resname(),
                    "resseq": res_q.id[1],
                    "icode": str(res_q.id[2]).strip() if res_q.id[2] is not None else "",
                    "coord": anchor_q,
                    "rel_sasa": rel_sasa,
                    "abs_sasa": getattr(res_q, "abs_sasa", 0.0),
                    "bfactor": np.mean([atom.get_bfactor() for atom in res_q.get_atoms()]),
                })
                break  # Only need one valid motif per GLN
    
    result = {
        "GLN_Isomerization_Count": len(gln_deamidation_candidates)
    }

    for i, site in enumerate(gln_deamidation_candidates, start=1):
        result[f"GLN{i}_Chain"] = site["chain_id"]
        result[f"GLN{i}_ResName"] = site["resname"]
        result[f"GLN{i}_ResSeq"] = site["resseq"]
        result[f"GLN{i}_ICode"] = site["icode"]
        result[f"GLN{i}_X"] = site["coord"][0]
        result[f"GLN{i}_Y"] = site["coord"][1]
        result[f"GLN{i}_Z"] = site["coord"][2]
        result[f"GLN{i}_RelSASA"] = site["rel_sasa"]
        result[f"GLN{i}_AbsSASA"] = site["abs_sasa"]
        result[f"GLN{i}_BFactor"] = site["bfactor"]

    return gln_deamidation_candidates, result   


def analyze_met_oxidation(struct):
    """
    Identify MET oxidation sites (M) that are surface-exposed.
    Oxidation to methionine sulfoxide causes potency loss and heterogeneity.
    Returns a list of dicts with residue info and structural features.
    """
    
    met_oxidation_candidates = []
    for chain in struct[0]:
        residues = [res for res in chain.get_residues() if is_aa(res, standard=True) and res.id[0] == " "]
        for res in residues:
            if res.get_resname().upper() != "MET":
                continue
            anchor = get_patch_anchor_coord(res)
            if anchor is None:
                continue       
            rel_sasa = getattr(res, "rel_sasa", 0.0)
            if rel_sasa < SURFACE_EXPOSURE_THRESHOLD:
                continue
            abs_sasa = getattr(res, "abs_sasa", 0.0)
            met_oxidation_candidates.append({
                "chain_id": chain.get_id(),
                "resname": res.get_resname(),
                "resseq": res.id[1],
                "icode": str(res.id[2]).strip() if res.id[2] is not None else "",
                "coord": anchor,
                "rel_sasa": rel_sasa,
                "abs_sasa": abs_sasa,
                "bfactor": np.mean([atom.get_bfactor() for atom in res.get_atoms()]),
            })

    result = {
        "MET_Oxidation_Count": len(met_oxidation_candidates)
    }

    for i, site in enumerate(met_oxidation_candidates, start=1):
        result[f"MET{i}_Chain"] = site["chain_id"]
        result[f"MET{i}_ResName"] = site["resname"]
        result[f"MET{i}_ResSeq"] = site["resseq"]
        result[f"MET{i}_ICode"] = site["icode"]
        result[f"MET{i}_X"] = site["coord"][0]
        result[f"MET{i}_Y"] = site["coord"][1]
        result[f"MET{i}_Z"] = site["coord"][2]
        result[f"MET{i}_RelSASA"] = site["rel_sasa"]
        result[f"MET{i}_AbsSASA"] = site["abs_sasa"]
        result[f"MET{i}_BFactor"] = site["bfactor"]

    return met_oxidation_candidates, result           


def analyze_his_oxidation(struct):
    """
    Identify HIS oxidation sites (H) that are surface-exposed.
    HIS is susceptible to oxidation and pH-sensitive charge state changes,
    both relevant liabilities in CDR regions.
    Returns a list of dicts with residue info and structural features.
    """
    
    his_oxidation_candidates = []
    for chain in struct[0]:
        residues = [res for res in chain.get_residues() if is_aa(res, standard=True) and res.id[0] == " "]
        for res in residues:
            if res.get_resname().upper() != "HIS":
                continue
            anchor = get_patch_anchor_coord(res)
            if anchor is None:
                continue
            rel_sasa = getattr(res, "rel_sasa", 0.0)
            if rel_sasa < SURFACE_EXPOSURE_THRESHOLD:
                continue
            abs_sasa = getattr(res, "abs_sasa", 0.0)      
            his_oxidation_candidates.append({
                "chain_id": chain.get_id(),
                "resname": res.get_resname(),
                "resseq": res.id[1],
                "icode": str(res.id[2]).strip() if res.id[2] is not None else "",
                "coord": anchor,
                "rel_sasa": rel_sasa,
                "abs_sasa": abs_sasa,
                "bfactor": np.mean([atom.get_bfactor() for atom in res.get_atoms()]),
            })

    result = {
        "HIS_Oxidation_Count": len(his_oxidation_candidates)
    }

    for i, site in enumerate(his_oxidation_candidates, start=1):
        result[f"HIS{i}_Chain"] = site["chain_id"]
        result[f"HIS{i}_ResName"] = site["resname"]
        result[f"HIS{i}_ResSeq"] = site["resseq"]
        result[f"HIS{i}_ICode"] = site["icode"]
        result[f"HIS{i}_X"] = site["coord"][0]
        result[f"HIS{i}_Y"] = site["coord"][1]
        result[f"HIS{i}_Z"] = site["coord"][2]
        result[f"HIS{i}_RelSASA"] = site["rel_sasa"]
        result[f"HIS{i}_AbsSASA"] = site["abs_sasa"]
        result[f"HIS{i}_BFactor"] = site["bfactor"]

    return his_oxidation_candidates, result  


def analyze_trp_oxidation(struct):      # check if I need to calculate buried?
    """
    Identify TRP oxidation sites (W) that are surface-exposed.
    Oxidizes to kynurenine or oxolactone under oxidative stress????
    Returns a list of dicts with residue info and structural features.
    """
    
    trp_oxidation_candidates = []
    for chain in struct[0]:
        residues = [res for res in chain.get_residues() if is_aa(res, standard=True) and res.id[0] == " "]
        for res in residues:
            if res.get_resname().upper() != "TRP":
                continue
            anchor = get_patch_anchor_coord(res)
            if anchor is None:
                continue
            rel_sasa = getattr(res, "rel_sasa", 0.0)
            if rel_sasa < SURFACE_EXPOSURE_THRESHOLD:
                continue
            abs_sasa = getattr(res, "abs_sasa", 0.0)
            trp_oxidation_candidates.append({
                "chain_id": chain.get_id(),
                "resname": res.get_resname(),
                "resseq": res.id[1],
                "icode": str(res.id[2]).strip() if res.id[2] is not None else "",
                "coord": anchor,
                "rel_sasa": rel_sasa,
                "abs_sasa": abs_sasa,
                "bfactor": np.mean([atom.get_bfactor() for atom in res.get_atoms()]),
            })

    result = {
        "TRP_Oxidation_Count": len(trp_oxidation_candidates)
    }

    for i, site in enumerate(trp_oxidation_candidates, start=1):
        result[f"TRP{i}_Chain"] = site["chain_id"]
        result[f"TRP{i}_ResName"] = site["resname"]
        result[f"TRP{i}_ResSeq"] = site["resseq"]
        result[f"TRP{i}_ICode"] = site["icode"]
        result[f"TRP{i}_X"] = site["coord"][0]
        result[f"TRP{i}_Y"] = site["coord"][1]
        result[f"TRP{i}_Z"] = site["coord"][2]
        result[f"TRP{i}_RelSASA"] = site["rel_sasa"]
        result[f"TRP{i}_AbsSASA"] = site["abs_sasa"]
        result[f"TRP{i}_BFactor"] = site["bfactor"]

    return trp_oxidation_candidates, result  
    

def analyze_cys_oxidation(struct):
    """
    Identify CYS oxidation sites (C) that are surface-exposed.
    Cysteine features a highly reactive terminal thiol (-SH) group.
    Returns a list of dicts with residue info.
    """
    
    cys_oxidation_candidates = []
    for chain in struct[0]:
        residues = [res for res in chain.get_residues() if is_aa(res, standard=True) and res.id[0] == " "]
        for res in residues:
            if res.get_resname().upper() != "CYS":
                continue
            anchor = get_patch_anchor_coord(res)
            if anchor is None:
                continue
            rel_sasa = getattr(res, "rel_sasa", 0.0)
            if rel_sasa < SURFACE_EXPOSURE_THRESHOLD:
                continue
            abs_sasa = getattr(res, "abs_sasa", 0.0)
            cys_oxidation_candidates.append({
                    "chain_id": chain.get_id(),
                    "resname": res.get_resname(),
                    "resseq": res.id[1],
                    "icode": str(res.id[2]).strip() if res.id[2] is not None else "",
                    "coord": anchor,
                    "rel_sasa": rel_sasa,
                    "abs_sasa": abs_sasa,
                    "bfactor": np.mean([atom.get_bfactor() for atom in res.get_atoms()]),
                })

    result = {
        "CYS_Oxidation_Count": len(cys_oxidation_candidates)
    }

    for i, site in enumerate(cys_oxidation_candidates, start=1):
        result[f"CYS_OX{i}_Chain"] = site["chain_id"]
        result[f"CYS_OX{i}_ResName"] = site["resname"]
        result[f"CYS_OX{i}_ResSeq"] = site["resseq"]
        result[f"CYS_OX{i}_ICode"] = site["icode"]
        result[f"CYS_OX{i}_X"] = site["coord"][0]
        result[f"CYS_OX{i}_Y"] = site["coord"][1]
        result[f"CYS_OX{i}_Z"] = site["coord"][2]
        result[f"CYS_OX{i}_RelSASA"] = site["rel_sasa"]
        result[f"CYS_OX{i}_AbsSASA"] = site["abs_sasa"]
        result[f"CYS_OX{i}_BFactor"] = site["bfactor"]

    return cys_oxidation_candidates, result          


def analyze_free_cys(struct):
    """
    Exposed CYS with free thiol (SG not within disulfide distance of another SG).
    Free thiols can form unintended intermolecular disulfides and contribute to aggregation.
    Disulfide detection threshold: SG-SG distance < DISULFIDE_DIST (2.2 Å).
    Returns a list of dicts with residue info.
    """
    
    free_cys_candidates = []

    all_sg_atoms = []
    for chain in struct[0]:
        for res in chain.get_residues():
            if is_aa(res, standard=True) and res.id[0] == " ":
                if res.get_resname().upper() == "CYS" and res.has_id("SG"):
                    all_sg_atoms.append(res["SG"])

    if not all_sg_atoms:
        return free_cys_candidates
    
    ns = NeighborSearch(all_sg_atoms)

    for chain in struct[0]:
        residues = [res for res in chain.get_residues() if is_aa(res, standard=True) and res.id[0] == " "]
        for res in residues:
            if res.get_resname().upper() != "CYS":
                continue
            if not res.has_id("SG"):
                continue

            sg = res["SG"]
            neighbors   = ns.search(sg.get_coord(), DISULFIDE_DIST)
            in_disulfide = any(
                atom != sg and atom.get_parent() != res
                for atom in neighbors
            )
 
            if in_disulfide:
                continue

            rel_sasa = getattr(res, "rel_sasa", 0.0)
            abs_sasa = getattr(res, "abs_sasa", 0.0)
            anchor   = get_patch_anchor_coord(res)
            if rel_sasa >= SURFACE_EXPOSURE_THRESHOLD and anchor is not None:
            
                free_cys_candidates.append({
                    "chain_id": chain.get_id(),
                    "resname":  res.get_resname(),
                    "resseq":   res.id[1],
                    "icode":    str(res.id[2]).strip() if res.id[2] is not None else "",
                    "coord":    anchor,
                    "rel_sasa": rel_sasa,
                    "abs_sasa": abs_sasa,
                    "bfactor":  np.mean([atom.get_bfactor() for atom in res.get_atoms()]),
                })

    result = {
        "Free_CYS_Count": len(free_cys_candidates)
    }

    for i, site in enumerate(free_cys_candidates, start=1):
        result[f"Free_CYS{i}_Chain"] = site["chain_id"]
        result[f"Free_CYS{i}_ResName"] = site["resname"]
        result[f"Free_CYS{i}_ResSeq"] = site["resseq"]
        result[f"Free_CYS{i}_ICode"] = site["icode"]
        result[f"Free_CYS{i}_X"] = site["coord"][0]
        result[f"Free_CYS{i}_Y"] = site["coord"][1]
        result[f"Free_CYS{i}_Z"] = site["coord"][2]
        result[f"Free_CYS{i}_RelSASA"] = site["rel_sasa"]
        result[f"Free_CYS{i}_AbsSASA"] = site["abs_sasa"]
        result[f"Free_CYS{i}_BFactor"] = site["bfactor"]

    return free_cys_candidates, result   


# --- MAIN PIPELINE ---
def run_structure_hotspots_and_ptm_analysis():
    MASTER_CSV = Path("/home/bunsree/projects/rosalind-bioinformatics/multispecific_antibodies/TheraSAbDab_SeqStruc_07Dec2025.csv")
    # Read MASTER_CSV
    df_master = pd.read_csv(MASTER_CSV)
    df_master['key'] = df_master['Therapeutic'].str.lower().str.strip()
   
    for BASE_DIR in BASE_DIRS:
        if not BASE_DIR.exists():
            print(f"ERROR: PDB directory not found at {BASE_DIR}")
            return

        print(f"\n=== Running analysis for {BASE_DIR} ===")
    
        OUTPUT_CSV = BASE_DIR / "Structure_Based_Hotspots_and_PTM_Module.csv"
    
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

        print(f"Found {len(pdb_map)} PDB files for HOTSPOTS AND PTM analysis.")
        
        # Setup tools
        parser = PDBParser(QUIET=True)
        master_features = []
        failed_entries = []
        
        # Process every PDB found
        print("--- CALCULATING STRUCTURE-BASED HOTSPOTS AND PTM ANALYSIS ---")
        for ab_name, pdb_path in tqdm(pdb_map.items()):
            try:
                # Load pdb structure
                struct = parser.get_structure(ab_name, str(pdb_path))

                # Annotate with structure-based CDRs (largest loops)
                annotate_structure_cdr_loops(struct, pdb_path)

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
                
                # Total SASA across all chains
                fab_total_sasa =0.0
                for chain in struct[0]:
                    residues = []
                    for res in chain.get_residues():
                        if res.id[0] == " ":
                            residues.append(res)
                    chain_sasa = sum(getattr(res, "abs_sasa", 0.0) for res in residues)
                    fab_total_sasa += chain_sasa

                total_residues = sum(chain_residues.values())
                chA_length = chain_residues.get('A', 0)
                chB_length = chain_residues.get('B', 0)

                # Fab/Fv-level secondary structure features (DSSP, 3D H-bond geometry based)
                secondary_structure_counts, secondary_structure_percentages = calculate_secondary_structure_features(struct, pdb_path)

                # Fab/Fv Level metrics
                n_glyco_list, n_glyco_result   = analyze_n_glycosylation(struct)
                asn_deam_list, asn_deam_result  = analyze_asn_deamidation(struct)
                asp_iso_list, asp_iso_result   = analyze_asp_isomerization(struct)
                gln_deam_list, gln_deam_result  = analyze_gln_deamidation(struct)
                met_ox_list, met_ox_result    = analyze_met_oxidation(struct)
                his_ox_list, his_ox_result    = analyze_his_oxidation(struct)
                trp_ox_list, trp_ox_result    = analyze_trp_oxidation(struct)
                cys_ox_list, cys_ox_result    = analyze_cys_oxidation(struct)
                free_cys_list, free_cys_result  = analyze_free_cys(struct)

                # Fab/Fv Level Count CDR-localized PTM sites
                n_cdr_n_glyco = count_cdr_ptm_sites(n_glyco_list, struct)
                n_cdr_asn_deam = count_cdr_ptm_sites(asn_deam_list, struct)
                n_cdr_asp_iso = count_cdr_ptm_sites(asp_iso_list, struct)
                n_cdr_gln_deam = count_cdr_ptm_sites(gln_deam_list, struct)
                n_cdr_met_ox = count_cdr_ptm_sites(met_ox_list, struct)
                n_cdr_his_ox = count_cdr_ptm_sites(his_ox_list, struct)
                n_cdr_trp_ox = count_cdr_ptm_sites(trp_ox_list, struct)
                n_cdr_cys_ox = count_cdr_ptm_sites(cys_ox_list, struct)
                n_cdr_free_cys = count_cdr_ptm_sites(free_cys_list, struct)
                
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
                    "Ch_A_Length": chA_length,
                    "Ch_B_Length": chB_length,
                    "Fab_Total_AAs": total_residues,

                    # Fab/Fv Level Secondary structure features (DSSP)
                    "N_Helix": secondary_structure_counts["alpha_helix"],
                    "N_Sheet": secondary_structure_counts["beta_sheet"],
                    "N_Coil": secondary_structure_counts["coil"],
                    "Pct_Helix": round(secondary_structure_percentages["alpha_helix"], 2),
                    "Pct_Sheet": round(secondary_structure_percentages["beta_sheet"], 2),
                    "Pct_Coil": round(secondary_structure_percentages["coil"], 2),

                    # Fab/Fv Level Hotspots and PTMs
                    "N_Glycosylation_Count": n_glyco_result["N_Glycosylation_Count"],
                    "ASN_Deamidation_Count": asn_deam_result["ASN_Deamidation_Count"],
                    "ASP_Isomerization_Count": asp_iso_result["ASP_Isomerization_Count"],
                    "GLN_Deamidation_Count":gln_deam_result["GLN_Isomerization_Count"],
                    "MET_Oxidation_Count": met_ox_result["MET_Oxidation_Count"],
                    "HIS_Oxidation_Count": his_ox_result["HIS_Oxidation_Count"],
                    "TRP_Oxidation_Count": trp_ox_result["TRP_Oxidation_Count"],
                    "CYS_Oxidation_Count": cys_ox_result["CYS_Oxidation_Count"],
                    "Free_CYS_Count": free_cys_result["Free_CYS_Count"],

                    # Fab/Fv Level CDR-localized PTM counts
                    "CDR_N_Glycosylation": n_cdr_n_glyco,
                    "CDR_ASN_Deamidation": n_cdr_asn_deam,
                    "CDR_ASP_Isomerization": n_cdr_asp_iso,
                    "CDR_GLN_Deamidation": n_cdr_gln_deam,
                    "CDR_MET_Oxidation": n_cdr_met_ox,
                    "CDR_HIS_Oxidation": n_cdr_his_ox,
                    "CDR_TRP_Oxidation": n_cdr_trp_ox,
                    "CDR_CYS_Oxidation": n_cdr_cys_ox,
                    "CDR_Free_CYS": n_cdr_free_cys,

                }

                # Hotspots and PTMS Structured Outputs
                entry.update(n_glyco_result)
                entry.update(asn_deam_result)
                entry.update(asp_iso_result)
                entry.update(gln_deam_result)
                entry.update(met_ox_result)
                entry.update(his_ox_result)
                entry.update(trp_ox_result)
                entry.update(cys_ox_result)
                entry.update(free_cys_result)
               
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

            # Reorder PTM columns in natural order (ASN1, ASN2, ASN3, ... instead of the order in which pandas first encountered them)
            def natural_key(col):
                m = re.match(r'([A-Za-z_]+?)(\d+)_(.*)', col)
                if m:
                    prefix, number, suffix = m.groups()
                    return (prefix, int(number), suffix)
                return (col, 0, "")
            
            fixed_cols = ['Therapeutic', 'CH1 Isotype', 'VD LC']
            other_cols = [c for c in df.columns if c not in fixed_cols]
            df = df[fixed_cols + sorted(other_cols, key=natural_key)]

            # save final CSV
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"\n=== SUCCESS: Hotspots and PTM analysis complete for {len(df)} antibodies ===")
            print(f"File saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_structure_hotspots_and_ptm_analysis()