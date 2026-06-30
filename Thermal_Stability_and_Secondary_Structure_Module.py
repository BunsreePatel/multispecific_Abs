import os
import warnings
import pandas as pd
import numpy as np
import math  # For mathematical operations like logarithms
import scipy # For numerical computations (e.g., scipy.optimize.newton)
from pathlib import Path
from tqdm import tqdm
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.DSSP import DSSP
from Bio.Data.PDBData import residue_sasa_scales
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

# --- PATHS ---
BASE_DIRS = [
    Path("/home/bunsree/projects/rosalind-bioinformatics/multispecific_antibodies/scFv"),
    Path("/home/bunsree/projects/rosalind-bioinformatics/multispecific_antibodies/Bispecific_scFv"),
    Path("/home/bunsree/projects/rosalind-bioinformatics/multispecific_antibodies/BiTE (Bispecific T-Cell Engager)"),
    Path("/home/bunsree/projects/rosalind-bioinformatics/multispecific_antibodies/Bispecific_mAb"),
    Path("/home/bunsree/projects/rosalind-bioinformatics/multispecific_antibodies/Whole_mAb")
]

MASTER_CSV = Path("/home/bunsree/projects/rosalind-bioinformatics/multispecific_antibodies/TheraSAbDab_SeqStruc_07Dec2025.csv")

"""
THEORY: STRUCTURE-BASED THERMAL STABILITY AND SECONDARY STRUCTURE MODULE
=============================================================
 
THERMAL STABILITY FEATURES (3D-Dependent):
 
1. WILKE MAX-ASA
    - A dictionary of maximum solvent accessible surface area (ASA) values for the 20 standard amino acids.
    - In Biopython, these values are used as normalization factors to calculate Relative Solvent Accessibility (RSA) from absolute SASA values.
    - This scale is often preferred over older scales because it provides a tighter upper bound.
    - Literature: Tien, M. Z., Meyer, A. G., Sydykova, D. K., et al. (2013). Maximum allowed solvent accessibilities of residues in proteins. PLoS One, 8(11), e80635.

2. N_MASK
    - Fab-only structures (VH + CH1 / VL + CL) have artificially exposed C-termini because CH1 and CL C-terminal regions are interface residues designed to be buried in full IgG.
      When Fc is removed, these hydrophobic residues become artificially exposed, causing computational tools to incorrectly flag them as aggregation-prone patches.
    - Masking these C-terminal residues prevents false positive hydrophobic/charge patch calls at truncation boundaries, restoring biological relevance to developability predictions by focusing analysis on the actual Fab surface exposed in solution.
      Zeroing out the last N residues of each chain prevents false positive hydrophobic/charge patch calls at the CH1/CL truncation interface before any feature extraction.
    - N_MASK values are conservative engineering estimates based on domain architecture
    - Literature: Röthisberger, D., Honegger, A., & Plückthun, A. (2005). Domain interactions in the Fab fragment: A comparative evaluation of the single-chain Fv and Fab format engineered with variable domains of different stability. Journal of Molecular Biology, 347(4), 773-789.

3. SOLVENT-ACCESSIBLE SURFACE AREA (SASA)
    - Folded (Hydrophobic and Polar) calculated using Shrake-Rupley  rolling-probe algorithm
    - Unfolded (Hydrophobic and Polar) approximated from MAX_ASA (Wilke scale)
    - Literature: Shrake, A., & Rupley, J. A. (1973). Environment and exposure to solvent of protein atoms. Lysozyme and insulin. Journal of Molecular Biology, 79(2), 351-371.
    
4. STRUCTURAL SUBFEATURES (ENTHALPY AND ENTROPY) (Van der Waals??? MIGHT NEED TO ADD THIS!!!)
    - Interaction Geometry Cutoffs:
        DISULFIDE_DIST              = 2.2           # Å SG-SG (literature has it ~ 2.05)
        SALT_DIST                   = 4.0           # Å interaction between positively charged AA (LYS or ARG) and negatively charged AA (ASP or GLU)
        HBOND_DIST                  = 3.5           # Å N or O donor-acceptor (literature has it ~2.7-3.5)
    - Literature: McAuley, A., Jacob, J., Kolvenbach, C. G., et al. (2008). Contributions of a disulfide bond to the structure, stability, and dimerization of human IgG1 antibody CH3 domain. Protein Science, 17(1), 95-106.
    - Literature: Donald, J. E., Kulp, D. W., & DeGrado, W. F. (2011). Salt bridges: Geometrically specific, designable interactions. Proteins, 79(3), 898-915.
    - Literature: Kabsch, W., & Sander, C. (1983). Dictionary of protein secondary structure: Pattern recognition of hydrogen-bonded and geometrical features. Biopolymers, 22(12), 2577-2637.

    - Enthalpy Constants:
        DELTAH_DISULFIDE            = -4.0          # kcal/mol per disulfide
        DELTAH_SALTBRIDGE           = -4.0          # kcal/mol per salt bridge
        DELTAH_HBOND                = -1.0          # kcal/mol per backbone H-bond
        HYDRO_CP_ALPHA              = 0.00034       # kcal/(mol·K·Å²) (hydrophobic heat capacity coefficient)   (Streit, 2024)
        POLAR_CP_BETA               = -0.00012      # kcal/(mol·K·Å²) (polar heat capacity coefficient)         (Streit, 2024)
        GAMMA_HYDROPHOBIC_ENERGY    = 0.025         # kcal/mol/Å²  (solvation coefficient)
    
    - Literature: Pace, C. N., Grimsley, G. R., Thomson, J. A., et al. (1988). Conformational stability and activity of ribonuclease T1 with zero, one, and two intact disulfide bonds. The Journal of Biological Chemistry, 263(24), 11620-11625.
    - Literature: Wimley, W. C., Gawrisch, K., Creamer, T. P., et al. (1996). Direct measurement of salt-bridge solvation energies using a peptide model system: Implications for protein stability. Proceedings of the National Academy of Sciences USA, 93(8), 2985-2990.
    - Literature: Scholtz, J. M., Marqusee, S., Baldwin, R. L., et al. (1991). Calorimetric determination of the enthalpy change for the alpha-helix to coil transition of an alanine peptide in water. Proceedings of the National Academy of Sciences USA, 88(7), 2854-2858.
    - Literature: Streit, J. O., Bukvin, I. V., Chan, S. H. S., et al. (2024). The ribosome lowers the entropic penalty of protein folding. Nature, 633, 232-239.   
    - Literature:
    
    - Entropy Constants:
        DELTAS_CONF_SIDECHAIN       = 1.0           # kcal/mol  PLACEHOLDER
        DELTAS_CONF_BACKBONE        = 1.0           # kcal/mol  PLACEHOLDER
        DELTAS_DISULFIDE            = 1.0           # kcal/mol  PLACEHOLDER
        DELTAS_SALTBRIDGE           = 1.0           # kcal/mol per salt bridge PLACEHOLDER
        DELTAS_HBOND                = 1.0           # kcal/mol per backbone H-bond PLACEHOLDER
        GAMMA_HYDROPHOBIC_ENTROPY   = 0.025         # kcal/mol/Å²  (entropy coefficient)??   PLACEHOLDER
    - Literature:
    - Literature: 
    - Literature: 
    - Literature: 
    
5. GIBBS-HELMHOLTZ FOR TWO-STATE EQUILIBRIUM (FOLDED/UNFOLDED)
    - Assumption: N <-> U (two-state, cooperative unfolding)
    - Equation:
        ΔG_NU(T)                    = ΔH_REF + ΔCp_proxy*(T - T_REF) - T*[ΔS_REF + ΔCp_proxy*ln(T/T_REF)]

    - Terms:
        ΔG_NU(T):                   change in Gibbs free energy for unfolding       # kcal/mol
        ΔH_REF:                     enthalpy proxy at T_REF                         # kcal/mol
        ΔCp_proxy:                  heat-capacity proxy from ΔSASA                  # kcal/mol·K
        ΔS_REF:                     entropy proxy at T_REF                          # kcal/mol·K

        ΔG_NU_at_T_REF              = ΔH_REF - T_REF*ΔS_REF (optional but recommended as a QC/reference feature)

    - Two-state equilibrium:
        K_NU(T) = exp(-ΔG_NU(T)/(R*T))
        fraction_unfolded = K_NU/(1 + K_NU)
    - Thermal proxies:
        Tm_proxy: scan T in [T_LOW, T_HIGH] (1000 points) and return T where |ΔG_NU(T)| is minimal (≈ ΔG_NU(T)=0)
        T_onset_proxy: use fraction_unfolded directly; compute ln_K = ln(fraction_unfolded)/(1-fraction_unfolded));
                       scan T in [T_LOW, T_HIGH] (1000 points) and return T where |ΔG_NU(T) + RT·ln(K_NU)| is minimal.
    - Literature: Pace, C. N., & Laurents, D. V. (1989). A new method for determining the heat capacity change for protein folding. Biochemistry, 28(6), 2520-2525.
    - Literature: Murphy, K. P., & Freire, E. (1992). Thermodynamics of structural stability and cooperative folding behavior in proteins. Advances in Protein Chemistry, 43, 313-361.

6. SECONDARY STRUCTURE FEATURES
    - DSSP (Dictionary of Protein Secondary Structure)
    - Literature: Kabsch, W., & Sander, C. (1983). Dictionary of protein secondary structure: Pattern recognition of hydrogen-bonded and geometrical features. Biopolymers, 22(12), 2577-2637.

"""

# --- BIOPHYSICAL CONSTANTS AND SCALES ---
HYDROPHOBIC_RESIDUES = ['ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'PRO', 'TRP']     # UPDATED SINCE TYR IS AMPHIPATHIC, AT PH 7, -Oh IS PROTONATED AND NEUTRAL, MAKING IT BEHAVE AS POLAR
POLAR_RESIDUES = ['SER', 'THR', 'CYS', 'ASN', 'GLN', 'TYR']
AROMATIC_RESIDUES = ['PHE', 'TYR', 'TRP']
POSITIVE_RESIDUES = ['ARG', 'LYS', 'HIS']
NEGATIVE_RESIDUES = ['ASP', 'GLU']

# check on Are you using DSSP to differentiate between surface-exposed Tyr (acting as a polar sticker) and buried Tyr (acting as a hydrophobic core residue)?

# --- THERMODYNAMIC CONSTANTS --- # DOUBLE CHECK ON THESE WITH LITERATURE!!!
T_REF= 298.15           # Standard temperature in Kelvin (25°C)
T_LOW = 273.15          # 0°C in Kelvin
T_HIGH = 403.15         # 130°C in Kelvin
R = 8.314               # Gas constant in J/(mol·K)
R_CAL = 1.987           # cal/(mol·K)
R_KCAL = 0.001987       # kcal/(mol·K)

# --- INTERACTION GEOMETRY CUTOFFS ---
DISULFIDE_DIST              = 2.2       # Å SG–SG (literature has it ~ 2.05)
SALT_DIST                   = 4.0       # Å interaction between positively charged AA (LYS or ARG) and negatively charged AA (ASP or GLU)
HBOND_DIST                  = 3.5       # Å N or O donor–acceptor (literature has it ~2.7-3.5)

# --- INTERACTION ENTHALPY CONSTANTS ---
DELTAH_DISULFIDE            = -4.0      # kcal/mol per disulfide
DELTAH_SALTBRIDGE           = -4.0      # kcal/mol per salt bridge
DELTAH_HBOND                = -1.0      # kcal/mol per backbone H-bond
HYDRO_CP_ALPHA              = 0.00034   # kcal/(mol·K·Å²) (hydrophobic heat capacity coefficient)   (Streit, 2024)
POLAR_CP_BETA               = -0.00012  # kcal/(mol·K·Å²) (polar heat capacity coefficient)         (Streit, 2024)
GAMMA_HYDROPHOBIC_ENERGY    = 0.025     # kcal/mol/Å²  (solvation coefficient)

# --- INTERACTION ENTROPY CONSTANTS ---
DELTAS_CONF_SIDECHAIN       = 1.0       # kcal/mol  PLACEHOLDER
DELTAS_CONF_BACKBONE        = 1.0       # kcal/mol  PLACEHOLDER
DELTAS_DISULFIDE            = 1.0       # kcal/mol  PLACEHOLDER
DELTAS_SALTBRIDGE           = 1.0       # kcal/mol per salt bridge PLACEHOLDER
DELTAS_HBOND                = 1.0       # kcal/mol per backbone H-bond PLACEHOLDER
GAMMA_HYDROPHOBIC_ENTROPY   = 0.025     # kcal/mol/Å²  (entropy coefficient)??   PLACEHOLDER

MAX_ASA = residue_sasa_scales["Wilke"]

# --- TRUNCATION MASKING PARAMETERS ---
"""
Since pdb structures are Fab or Fv structures, the C-termini for the Heavy and Light Chains will be masked. # NEED TO CHECK ON THIS LOGIC FOR Fv structures with linkers!!!
"""
N_MASK_HEAVY = 10   # Chain A (heavy, CH1 C-terminus)
N_MASK_LIGHT = 5    # Chain B (light, CL C-terminus)

# --- THERMAL STABILITY HELPER FUNCTIONS ---

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


def _dist(a, b):
    diff = a.get_vector() - b.get_vector()
    return diff.norm()


# MAY WANT TO ADD DIHEDRAL ANGLE CHECK
def count_disulfide_bonds(struct):
    """
    Count intrachain disulfide bonds in a Fab/Fv structure.
    Criterion: SG-SG distance < DISULFIDE_DIST (2.2 Å).

    Parameters:
    - struct: Biopython structure object

    Returns:
    - count of disulfide bonds (int)
    """
    cys_residues = []

    for chain in struct[0]:
        for res in chain.get_residues():
            if res.id[0] != " ":
                continue
            if res.get_resname().upper() == "CYS":
                cys_residues.append(res)

    if not cys_residues:
        return 0

    # SG = sulfur gamma atom (on the CYS side chain that forms covalent S-S bond in a disulfide)
    cys_sg_atoms = []
    for res in cys_residues:
        if res.has_id("SG"):
            cys_sg_atoms.append(res["SG"])

    if not cys_sg_atoms:
        return 0

    ns = NeighborSearch(cys_sg_atoms)
    dsbond_count = 0

    for sg in cys_sg_atoms:
        neighbors = ns.search(sg.get_vector().get_array(), DISULFIDE_DIST)
        for n in neighbors:
            if n == sg:  # Skip self
                continue
            if n.get_parent() == sg.get_parent():  # Skip same residue
                continue
            if n.get_parent().get_parent() != sg.get_parent().get_parent():  # Skip interchain (Fab and Fv structures are intrachain only)
                continue
            if _dist(sg, n) < DISULFIDE_DIST:
                dsbond_count += 1
        
    return dsbond_count // 2


# MAY WANT TO ADD DIHEDRAL ANGLE CHECK
def count_chain_disulfide_bonds(chain):
    """
    Count intrachain disulfide bonds of a single chain.
    Criterion: SG-SG distance < DISULFIDE_DIST (2.2 Å).
    """
    cys_residues = []

    for res in chain.get_residues():
        if res.id[0] != " ":
            continue
        if res.get_resname().upper() == "CYS":
            cys_residues.append((chain, res))

    if not cys_residues:
        return 0

    # SG = sulfur gamma atom (on the CYS side chain that forms covalent S-S bond in a disulfide)
    cys_sg_atoms = []
    for chain, res in cys_residues:
        if res.has_id("SG"):
            cys_sg_atoms.append(res["SG"])

    if not cys_sg_atoms:
        return 0

    ns = NeighborSearch(cys_sg_atoms)
    chain_dsbond_count = 0

    for sg in cys_sg_atoms:
        neighbors = ns.search(sg.get_vector().get_array(), DISULFIDE_DIST)
        for n in neighbors:
            if n == sg:  # Skip self
                continue
            if n.get_parent() == sg.get_parent():  # Skip same residue
                continue
            if n.get_parent().get_parent() != sg.get_parent().get_parent():  # Skip interchain (Fab and Fv structures are intrachain only)
                continue
            if _dist(sg, n) < DISULFIDE_DIST:
                chain_dsbond_count += 1
        
    return chain_dsbond_count // 2


def count_salt_bridges(struct):
    """
    Count salt bridges in a Fab/Fv structure.
    Criterion: distance < SALT_DIST (4.0 Å) between oppositely charged side-chain atoms.
    Positive: ARG (NH1, NH2), LYS (NZ - epsilon-amino group) acts as the cation.
    Negative: ASP (OD1, OD2 within the carboxylate group), GLU (OE1, OE2 within the carboxylate group) acts as the anion.

    Parameters:
    - struct: Biopython structure object

    Returns:
    - count of salt bridges (int)
    """
    positive_residues = []
    negative_residues = []

    for chain in struct[0]:
        for res in chain.get_residues():
            if res.id[0] != " ":
                continue
            resname = res.get_resname().upper()
            if resname in POSITIVE_RESIDUES:
                positive_residues.append(res)
            elif resname in NEGATIVE_RESIDUES:
                negative_residues.append(res)

    if not positive_residues or not negative_residues:
        return 0

    pos_atoms = []
    for res in positive_residues:
        for atom_id in ["NH1", "NH2"] if res.get_resname() == "ARG" else ["NZ"] if res.get_resname() == "LYS" else []:
            if res.has_id(atom_id):
                pos_atoms.append(res[atom_id])

    neg_atoms = []
    for res in negative_residues:
        for atom_id in ["OD1", "OD2"] if res.get_resname() == "ASP" else ["OE1", "OE2"] if res.get_resname() == "GLU" else []:
            if res.has_id(atom_id):
                neg_atoms.append(res[atom_id])

    if not pos_atoms or not neg_atoms:
        return 0
    
    ns_pos = NeighborSearch(pos_atoms)
    ns_neg = NeighborSearch(neg_atoms)
    count  = 0

    for pos in pos_atoms:
        neighbors = ns_neg.search(pos.get_vector().get_array(), SALT_DIST)
        count += sum(1 for n in neighbors if _dist(pos, n) < SALT_DIST)

    for neg in neg_atoms:
        neighbors = ns_pos.search(neg.get_vector().get_array(), SALT_DIST)
        count += sum(1 for n in neighbors if _dist(neg, n) < SALT_DIST)

    return count // 2


def count_chain_salt_bridges(chain):
    """
    Count salt bridges of a single chain.
    Criterion: distance < SALT_DIST (4.0 Å) between oppositely charged side-chain atoms.
    Positive: ARG (NH1, NH2), LYS (NZ - epsilon-amino group) acts as the cation.
    Negative: ASP (OD1, OD2 within the carboxylate group), GLU (OE1, OE2 within the carboxylate group) acts as the anion.
    """
    chain_positive_residues = []
    chain_negative_residues = []

    for res in chain.get_residues():
        if res.id[0] != " ":
            continue
        resname = res.get_resname().upper()
        if resname in POSITIVE_RESIDUES:
            chain_positive_residues.append((chain, res))
        elif resname in NEGATIVE_RESIDUES:
            chain_negative_residues.append((chain, res))

    if not chain_positive_residues or not chain_negative_residues:
        return 0

    chain_pos_atoms = []
    for chain, res in chain_positive_residues:
        for atom_id in ["NH1", "NH2"] if res.get_resname() == "ARG" else ["NZ"] if res.get_resname() == "LYS" else []:
            if res.has_id(atom_id):
                chain_pos_atoms.append(res[atom_id])

    chain_neg_atoms = []
    for chain, res in chain_negative_residues:
        for atom_id in ["OD1", "OD2"] if res.get_resname() == "ASP" else ["OE1", "OE2"] if res.get_resname() == "GLU" else []:
            if res.has_id(atom_id):
                chain_neg_atoms.append(res[atom_id])

    if not chain_pos_atoms or not chain_neg_atoms:
        return 0
    
    ns_pos = NeighborSearch(chain_pos_atoms)
    ns_neg = NeighborSearch(chain_neg_atoms)
    chain_salt_bridge_count = 0

    for pos in chain_pos_atoms:
        neighbors = ns_neg.search(pos.get_vector().get_array(), SALT_DIST)
        chain_salt_bridge_count += sum(1 for n in neighbors if _dist(pos, n) < SALT_DIST)

    for neg in chain_neg_atoms:
        neighbors = ns_pos.search(neg.get_vector().get_array(), SALT_DIST)
        chain_salt_bridge_count += sum(1 for n in neighbors if _dist(neg, n) < SALT_DIST)

    return chain_salt_bridge_count // 2


def count_hydrogen_bonds(struct):       # new, double check the angle in literature
    """
    Count hydrogen bonds in a Fab/Fv structure using NeighborSearch.
    Criterion: donor-acceptor distance < HBOND_DIST (3.5 Å) and D-CA···A angle >= 120°.
    Donors and acceptors: backbone and sidechain N and O atoms.

    Parameters:
    - struct: Biopython structure object

    Returns:
    - count of hydrogen bonds (int)
    """
    atoms = []

    for chain in struct[0]:
        for res in chain.get_residues():
            if res.id[0] != " ":
                continue
            for atom in res.get_atoms():
                if atom.element in ['N', 'O']:
                    atoms.append(atom)

    if not atoms:
        return 0

    ns = NeighborSearch(atoms)
    hybond_count = 0
    seen = set()

    for donor in atoms:
        d_res = donor.get_parent()
        if not d_res.has_id("CA"):
            continue
        ca_coord = d_res["CA"].get_vector().get_array()

        neighbors = ns.search(donor.get_vector().get_array(), HBOND_DIST)
        for acceptor in neighbors:
            if donor == acceptor:
                continue
            a_res = acceptor.get_parent()
            if d_res == a_res:
                continue

            pair = tuple(sorted([id(donor), id(acceptor)]))
            if pair in seen:
                continue
            seen.add(pair)

            vec_ref = donor.get_vector().get_array() - ca_coord
            vec_da  = acceptor.get_vector().get_array() - donor.get_vector().get_array()

            norm_ref = np.linalg.norm(vec_ref)
            norm_da  = np.linalg.norm(vec_da)
            if norm_ref == 0 or norm_da == 0:
                continue

            cos_angle = np.dot(vec_ref, vec_da) / (norm_ref * norm_da)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle     = np.degrees(np.arccos(cos_angle))

            if angle >= 120:
                hybond_count += 1

    return hybond_count


def count_chain_hydrogen_bonds(chain):       # new, double check the angle in literature
    """
    Count hydrogen bonds of a single chain using NeighborSearch.
    Criterion: donor-acceptor distance < HBOND_DIST (3.5 Å) and D-CA···A angle >= 120°.
    Donors and acceptors: backbone and sidechain N and O atoms.
    """
    chain_atoms = []

    for res in chain.get_residues():
        if res.id[0] != " ":
            continue
        for atom in res.get_atoms():
            if atom.element in ['N', 'O']:
                chain_atoms.append(atom)

    if not chain_atoms:
        return 0

    ns = NeighborSearch(chain_atoms)
    chain_hybond_count = 0
    seen = set()

    for donor in chain_atoms:
        d_res = donor.get_parent()
        if not d_res.has_id("CA"):
            continue
        ca_coord = d_res["CA"].get_vector().get_array()

        neighbors = ns.search(donor.get_vector().get_array(), HBOND_DIST)
        for acceptor in neighbors:
            if donor == acceptor:
                continue
            a_res = acceptor.get_parent()
            if d_res == a_res:
                continue

            pair = tuple(sorted([id(donor), id(acceptor)]))
            if pair in seen:
                continue
            seen.add(pair)

            vec_ref = donor.get_vector().get_array() - ca_coord
            vec_da  = acceptor.get_vector().get_array() - donor.get_vector().get_array()

            norm_ref = np.linalg.norm(vec_ref)
            norm_da  = np.linalg.norm(vec_da)
            if norm_ref == 0 or norm_da == 0:
                continue

            cos_angle = np.dot(vec_ref, vec_da) / (norm_ref * norm_da)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle     = np.degrees(np.arccos(cos_angle))

            if angle >= 120:
                chain_hybond_count  += 1

    return chain_hybond_count


def calc_hydrophobic_sasa_folded(struct):
    """
    Calculate total solvent-accessible surface area of hydrophobic residues (folded state).

    Parameters:
    - struct: Biopython structure object

    Returns:
    - hydrophobic_sasa_folded in Å²
    """
    residues = []

    for chain in struct[0]:
        for res in chain.get_residues():
            if res.id[0] != " ":
                continue
            residues.append(res)

    hydrophobic_sasa_folded = 0.0

    for res in residues:
        resname = res.get_resname().upper()
        if resname not in HYDROPHOBIC_RESIDUES:
            continue
        hydrophobic_sasa_folded += getattr(res, "abs_sasa", 0.0)

    return hydrophobic_sasa_folded


def calc_chain_hydrophobic_sasa_folded(chain):
    """
    Calculate total solvent-accessible surface area of hydrophobic residues (folded state) of a single chain.
    """
    residues = []

    for res in chain.get_residues():
        if res.id[0] != " ":
            continue
        residues.append((chain, res))

    chain_hydrophobic_sasa_folded = 0.0

    for chain, res in residues:
        resname = res.get_resname().upper()
        if resname not in HYDROPHOBIC_RESIDUES:
            continue
        chain_hydrophobic_sasa_folded += getattr(res, "abs_sasa", 0.0)

    return chain_hydrophobic_sasa_folded


def calc_polar_sasa_folded(struct):
    """
    Calculate total solvent-accessible surface area of polar residues (folded state).

    Parameters:
    - struct: Biopython structure object

    Returns:
    - polar_sasa_folded in Å²
    """
    residues = []

    for chain in struct[0]:
        for res in chain.get_residues():
            if res.id[0] != " ":
                continue
            residues.append(res)

    polar_sasa_folded = 0.0

    for res in residues:
        resname = res.get_resname().upper()
        if resname not in POLAR_RESIDUES:
            continue
        polar_sasa_folded += getattr(res, "abs_sasa", 0.0)

    return polar_sasa_folded


def calc_chain_polar_sasa_folded(chain):
    """
    Calculate total solvent-accessible surface area of polar residues (folded state) of a single chain.
    """
    residues = []

    for res in chain.get_residues():
        if res.id[0] != " ":
            continue
        residues.append((chain, res))

    chain_polar_sasa_folded = 0.0

    for chain, res in residues:
        resname = res.get_resname().upper()
        if resname not in POLAR_RESIDUES:
            continue
        chain_polar_sasa_folded += getattr(res, "abs_sasa", 0.0)

    return chain_polar_sasa_folded


def calc_hydrophobic_sasa_unfolded(struct):
    """
    Calculate total solvent-accessible surface area of hydrophobic residues (unfolded state).

    Parameters:
    - struct: Biopython structure object

    Returns:
    - hydrophobic_sasa_unfolded in Å²
    """
    residues = []

    for chain in struct[0]:
        for res in chain.get_residues():
            if res.id[0] != " ":
                continue
            residues.append(res)

    hydrophobic_sasa_unfolded = 0.0

    for res in residues:
        resname = res.get_resname().upper()
        max_asa = get_max_asa(resname)
        if resname not in HYDROPHOBIC_RESIDUES:
            continue
        hydrophobic_sasa_unfolded += max_asa

    return hydrophobic_sasa_unfolded


def calc_chain_hydrophobic_sasa_unfolded(chain):
    """
    Calculate total solvent-accessible surface area of hydrophobic residues (unfolded state) of a single chain.
    """
    residues = []

    for res in chain.get_residues():
        if res.id[0] != " ":
            continue
        residues.append((chain, res))

    chain_hydrophobic_sasa_unfolded = 0.0

    for chain, res in residues:
        resname = res.get_resname().upper()
        max_asa = get_max_asa(resname)
        if resname not in HYDROPHOBIC_RESIDUES:
            continue
        chain_hydrophobic_sasa_unfolded += max_asa

    return chain_hydrophobic_sasa_unfolded


def calc_polar_sasa_unfolded(struct):
    """
    Calculate total solvent-accessible surface area of polar residues (unfolded state).

    Parameters:
    - struct: Biopython structure object

    Returns:
    - polar_sasa_unfolded in Å²
    """
    residues = []

    for chain in struct[0]:
        for res in chain.get_residues():
            if res.id[0] != " ":
                continue
            residues.append(res)

    polar_sasa_unfolded = 0.0

    for res in residues:
        resname = res.get_resname().upper()
        max_asa = get_max_asa(resname)
        if resname not in POLAR_RESIDUES:
            continue
        polar_sasa_unfolded += max_asa

    return polar_sasa_unfolded


def calc_chain_polar_sasa_unfolded(chain):
    """
    Calculate total solvent-accessible surface area of polar residues (unfolded state) of a single chain.
    """
    residues = []

    for res in chain.get_residues():
        if res.id[0] != " ":
            continue
        residues.append((chain, res))

    chain_polar_sasa_unfolded = 0.0

    for chain, res in residues:
        resname = res.get_resname().upper()
        max_asa = get_max_asa(resname)
        if resname not in POLAR_RESIDUES:
            continue
        chain_polar_sasa_unfolded += max_asa

    return chain_polar_sasa_unfolded


def calc_delta_H_REF(struct):
    """
    Estimate ΔH_REF from discrete structural interactions.
    ΔH_total = N_HB x DELTAH_HBOND + N_SB x DELTAH_SALTBRIDGE + N_DS x DELTAH_DISULFIDE + GAMMA_HYDROPHOBIC_ENERGY x hydrophobic_sasa_folded

    Parameters:
    - struct: Biopython structure object

    Returns:
    - ΔH_REF in kcal/mol
    """
    n_hbond      = count_hydrogen_bonds(struct)
    n_saltbridge = count_salt_bridges(struct)
    n_disulfide  = count_disulfide_bonds(struct)

    dH_hbond       = n_hbond      * DELTAH_HBOND
    dH_saltbridge  = n_saltbridge * DELTAH_SALTBRIDGE
    dH_disulfide   = n_disulfide  * DELTAH_DISULFIDE
    dH_hydrophobic = calc_hydrophobic_sasa_folded(struct) * GAMMA_HYDROPHOBIC_ENERGY    # Å² × kcal/mol/Å² → kcal/mol

    delta_H_REF = dH_hbond + dH_saltbridge + dH_disulfide + dH_hydrophobic

    return delta_H_REF


def calc_chain_delta_H_REF(chain):
    """
    Estimate ΔH_REF from discrete structural interactions.
    ΔH_total = N_HB x DELTAH_HBOND + N_SB x DELTAH_SALTBRIDGE + N_DS x DELTAH_DISULFIDE + GAMMA_HYDROPHOBIC_ENERGY x hydrophobic_sasa_folded
    """
    n_chain_hbond      = count_chain_hydrogen_bonds(chain)
    n_chain_saltbridge = count_chain_salt_bridges(chain)
    n_chain_disulfide  = count_chain_disulfide_bonds(chain)

    dH_chain_hbond       = n_chain_hbond      * DELTAH_HBOND
    dH_chain_saltbridge  = n_chain_saltbridge * DELTAH_SALTBRIDGE
    dH_chain_disulfide   = n_chain_disulfide  * DELTAH_DISULFIDE
    dH_chain_hydrophobic = calc_chain_hydrophobic_sasa_folded(chain) * GAMMA_HYDROPHOBIC_ENERGY    # Å² × kcal/mol/Å² → kcal/mol

    chain_delta_H_REF = dH_chain_hbond + dH_chain_saltbridge + dH_chain_disulfide + dH_chain_hydrophobic

    return chain_delta_H_REF


def calc_delta_Cp_proxy(struct):
    """
    Estimate ΔCp_proxy from the change in SASA between folded (native) and unfolded states.
    Cp is the increase in heat capacity between native and unfolded protein.
    
    ΔSASA_hydrophobic = hydrophobic_sasa_unfolded - hydrophobic_sasa_folded
    ΔSASA_polar = polar_sasa_unfolded - polar_sasa_folded  
      
    Parameters:
    - struct: Biopython structure object

    Returns:
    - delta_Cp_proxy as HYDRO_CP_ALPHA·delta_sasa_hydrophobic + POLAR_CP_BETA·delta_sasa_polar in kcal/(mol·K)
    - delta_sasa_hydrophobic in Å²
    - delta_sasa_polar in Å²
    - hydrophobic_sasa_folded in Å²
    - polar_sasa_folded in Å²
    - hydrophobic_sasa_unfolded in Å²
    - polar_sasa_unfolded in Å²
    """
    # Folded state SASA from pdb structure
    hydrophobic_sasa_folded = calc_hydrophobic_sasa_folded(struct)
    polar_sasa_folded = calc_polar_sasa_folded(struct)

    # Unfolded state SASA from pdb structure MAX_ASA
    hydrophobic_sasa_unfolded = calc_hydrophobic_sasa_unfolded(struct)
    polar_sasa_unfolded = calc_polar_sasa_unfolded(struct)

    # ΔSASA
    delta_sasa_hydrophobic = hydrophobic_sasa_unfolded - hydrophobic_sasa_folded
    delta_sasa_polar = polar_sasa_unfolded - polar_sasa_folded

    # ΔCp proxy
    delta_Cp_proxy = (
        (delta_sasa_hydrophobic * HYDRO_CP_ALPHA) + 
        (delta_sasa_polar * POLAR_CP_BETA)
    )

    return (
        delta_Cp_proxy,
        delta_sasa_hydrophobic,
        delta_sasa_polar,
        hydrophobic_sasa_folded,
        polar_sasa_folded,
        hydrophobic_sasa_unfolded,
        polar_sasa_unfolded
    )


def calc_chain_delta_Cp_proxy(chain):
    """
    Estimate ΔCp_proxy from the change in SASA between folded (native) and unfolded states of a single chain.
    Cp is the increase in heat capacity between native and unfolded protein.
    
    ΔSASA_hydrophobic = hydrophobic_sasa_unfolded - hydrophobic_sasa_folded
    ΔSASA_polar = polar_sasa_unfolded - polar_sasa_folded  
    """
    # Folded state SASA from pdb structure
    chain_hydrophobic_sasa_folded = calc_chain_hydrophobic_sasa_folded(chain)
    chain_polar_sasa_folded = calc_chain_polar_sasa_folded(chain)

    # Unfolded state SASA from pdb structure MAX_ASA
    chain_hydrophobic_sasa_unfolded = calc_chain_hydrophobic_sasa_unfolded(chain)
    chain_polar_sasa_unfolded = calc_chain_polar_sasa_unfolded(chain)

    # ΔSASA
    chain_delta_sasa_hydrophobic = chain_hydrophobic_sasa_unfolded - chain_hydrophobic_sasa_folded
    chain_delta_sasa_polar = chain_polar_sasa_unfolded - chain_polar_sasa_folded

    # ΔCp proxy
    chain_delta_Cp_proxy = (
        (chain_delta_sasa_hydrophobic * HYDRO_CP_ALPHA) + 
        (chain_delta_sasa_polar * POLAR_CP_BETA)
    )

    return (
        chain_delta_Cp_proxy,
        chain_delta_sasa_hydrophobic,
        chain_delta_sasa_polar,
        chain_hydrophobic_sasa_folded,
        chain_polar_sasa_folded,
        chain_hydrophobic_sasa_unfolded,
        chain_polar_sasa_unfolded
    )

def calc_delta_S_REF(struct):
    """
    Estimate ΔS_REF from discrete structural entropy contributions.

    ΔS_total =
        N_HB * DELTAS_HBOND +
        N_SB * DELTAS_SALTBRIDGE +
        N_DS * DELTAS_DISULFIDE +
        N_sidechain * DELTAS_CONF_SIDECHAIN +
        N_backbone * DELTAS_CONF_BACKBONE +
        GAMMA_HYDROPHOBIC_ENTROPY * hydrophobic_sasa_folded
    
    Parameters:
    - struct: Biopython structure object

    Returns:
    - ΔS_REF in kcal/(mol·K)
    """

    # Collect all standard residues from all chains
    residues = []
    for chain in struct[0]:
        for res in chain.get_residues():
            if res.id[0] != " ":
                continue
            residues.append(res)

    n_hbond      = count_hydrogen_bonds(struct)
    n_saltbridge = count_salt_bridges(struct)
    n_disulfide  = count_disulfide_bonds(struct)
    n_sidechain  = len(residues)        #PLACEHOLDER
    n_backbone   = len(residues)        #PLACEHOLDER

    dS_hbond                        = n_hbond      * DELTAS_HBOND
    dS_saltbridge                   = n_saltbridge * DELTAS_SALTBRIDGE
    dS_disulfide                    = n_disulfide  * DELTAS_DISULFIDE
    dS_hydrophobic_sasa_folded      = calc_hydrophobic_sasa_folded(struct) * GAMMA_HYDROPHOBIC_ENTROPY    # Å² × kcal/mol/Å² → kcal/mol
    dS_sidechain                    = n_sidechain * DELTAS_CONF_SIDECHAIN
    dS_backbone                     = n_backbone * DELTAS_CONF_BACKBONE

    delta_S_REF = dS_hbond + dS_saltbridge + dS_disulfide + dS_hydrophobic_sasa_folded + dS_sidechain + dS_backbone
    
    return delta_S_REF


def calc_chain_delta_S_REF(chain):
    """
    Estimate ΔS_REF from discrete structural entropy contributions of a single chain.

    ΔS_total =
        N_HB * DELTAS_HBOND +
        N_SB * DELTAS_SALTBRIDGE +
        N_DS * DELTAS_DISULFIDE +
        N_sidechain * DELTAS_CONF_SIDECHAIN +
        N_backbone * DELTAS_CONF_BACKBONE +
        GAMMA_HYDROPHOBIC_ENTROPY * chain_hydrophobic_sasa_folded
    """

    # Collect all standard residues of a single chain
    residues = []
    for res in chain.get_residues():
        if res.id[0] != " ":
            continue
        residues.append((chain, res))

    n_chain_hbond      = count_chain_hydrogen_bonds(chain)
    n_chain_saltbridge = count_chain_salt_bridges(chain)
    n_chain_disulfide  = count_chain_disulfide_bonds(chain)
    n_chain_sidechain  = len(residues)        #PLACEHOLDER
    n_chain_backbone   = len(residues)        #PLACEHOLDER

    dS_chain_hbond                      = n_chain_hbond      * DELTAS_HBOND
    dS_chain_saltbridge                 = n_chain_saltbridge * DELTAS_SALTBRIDGE
    dS_chain_disulfide                  = n_chain_disulfide  * DELTAS_DISULFIDE
    dS_chain_hydrophobic_sasa_folded    = calc_chain_hydrophobic_sasa_folded(chain) * GAMMA_HYDROPHOBIC_ENTROPY    # Å² × kcal/mol/Å² → kcal/mol
    dS_chain_sidechain                  = n_chain_sidechain * DELTAS_CONF_SIDECHAIN
    dS_chain_backbone                   = n_chain_backbone * DELTAS_CONF_BACKBONE

    chain_delta_S_REF = dS_chain_hbond + dS_chain_saltbridge + dS_chain_disulfide + dS_chain_hydrophobic_sasa_folded + dS_chain_sidechain + dS_chain_backbone
    
    return chain_delta_S_REF


def calc_delta_G_NU(T, delta_H_REF, delta_Cp_proxy, delta_S_REF):
    """
    Gibbs-Helmholtz ΔG(T) parameterized at T_REF = 298.15 K.
    ΔG_NU(T) =  ΔH_REF + ΔCp(T - T_REF) - T[ΔS_REF + ΔCpln(T/T_REF)]
    
    Parameters:
    - T: temperature (K)
    - delta_H_REF: enthalpy term
    - delta_Cp_proxy: heat capacity term
    - delta_S_REF: unfolding entropy term
        
    Returns:
    - ΔG_NU(T)
    """

    delta_G_NU = (delta_H_REF
            + delta_Cp_proxy * (T - T_REF)
            - T * (delta_S_REF + delta_Cp_proxy * math.log(T / T_REF)))
    
    return delta_G_NU


def calc_delta_G_NU_at_T_REF(T_REF, delta_H_REF, delta_S_REF):      #is this needed or used?
    """
    Gibbs-Helmholtz ΔG(T) parameterized at T_REF = 298.15 K.
    ΔG_NU(T_REF) =  ΔH_REF + ΔCp(T_REF - T_REF) - T_REF[ΔS_REF + ΔCpln(T_REF/T_REF)]
    ΔG_NU(T_REF) =  ΔH_REF + 0 - T_REF[ΔS_REF + 0]

    Parameters:
    - delta_H_REF: enthalpy term (kcal/mol)
    - delta_S_REF: unfolding entropy term (kcal/(mol·K))
    
    Returns:
    - ΔG_NU_at_T_REF: Gibbs free energy at T_REF (kcal/mol)
    """
    
    return (delta_H_REF
            - T_REF * (delta_S_REF))


def calculate_tm_proxy(struct):
    """
    Estimate melting temperature (Tm) proxy via two-state Gibbs-Helmholtz.
    Approximates T where ΔG_NU(T) ≈ 0 by scanning T from T_LOW to T_HIGH
    and selecting the T with the smallest |ΔG_NU(T)|.

    ΔG_NU(T) =  ΔH_REF + ΔCp(T - T_REF) - T[ΔS_REF + ΔCpln(T/T_REF)]
    0 = ΔH_REF + ΔCp(T - T_REF) - T[ΔS_REF + ΔCpln(T/T_REF)]
    
    Expand:
    0 = ΔH_REF + ΔCpT - ΔCpT_REF - TΔS_REF + TΔCpln(T/T_REF)
    
    Rearrange:
    TΔS_REF + TΔCpln(T/T_REF) = ΔH_REF + ΔCpT - ΔCpT_REF

    T_REF = 298.15 K (reference temperature for static pdb);
    Tm_proxy is the T in [T_LOW, T_HIGH] that minimizes |ΔG_NU(T)|.

    Parameters:
    - struct: Biopython structure object

    Returns:
    - Tm proxy in Kelvin
    """
    
    # Collect all standard residues from all chains
    residues = []
    for chain in struct[0]:
        for res in chain.get_residues():
            if res.id[0] != " ":
                continue
            residues.append(res)

    delta_H_REF = float(calc_delta_H_REF(struct))
    delta_Cp_proxy = float(calc_delta_Cp_proxy(struct)[0])
    delta_S_REF = float(calc_delta_S_REF(struct))
    
    temps = np.linspace(T_LOW, T_HIGH, 1000)
    dg_values = [calc_delta_G_NU(T, delta_H_REF, delta_Cp_proxy, delta_S_REF) for T in temps]
    min_idx = np.argmin(np.abs(dg_values))
    tm_proxy = temps[min_idx]
    return tm_proxy


def calculate_chain_tm_proxy(chain):
    """
    Estimate melting temperature (Tm) proxy via two-state Gibbs-Helmholtz.
    Approximates T where ΔG_NU(T) ≈ 0 by scanning T from T_LOW to T_HIGH
    and selecting the T with the smallest |ΔG_NU(T)|.

    ΔG_NU(T) =  ΔH_REF + ΔCp(T - T_REF) - T[ΔS_REF + ΔCpln(T/T_REF)]
    0 = ΔH_REF + ΔCp(T - T_REF) - T[ΔS_REF + ΔCpln(T/T_REF)]
    
    Expand:
    0 = ΔH_REF + ΔCpT - ΔCpT_REF - TΔS_REF + TΔCpln(T/T_REF)
    
    Rearrange:
    TΔS_REF + TΔCpln(T/T_REF) = ΔH_REF + ΔCpT - ΔCpT_REF

    T_REF = 298.15 K (reference temperature for static pdb);
    Tm_proxy is the T in [T_LOW, T_HIGH] that minimizes |ΔG_NU(T)|.
    """
    
    # Collect all standard residues from all chains
    residues = []
    
    for res in chain.get_residues():
        if res.id[0] != " ":
            continue
        residues.append((chain, res))

    chain_delta_H_REF = float(calc_chain_delta_H_REF(chain))
    chain_delta_Cp_proxy = float(calc_chain_delta_Cp_proxy(chain)[0])
    chain_delta_S_REF = float(calc_chain_delta_S_REF(chain))
    
    chain_temps = np.linspace(T_LOW, T_HIGH, 1000)
    chain_dg_values = [calc_delta_G_NU(T, chain_delta_H_REF, chain_delta_Cp_proxy, chain_delta_S_REF) for T in chain_temps]
    chain_min_idx = np.argmin(np.abs(chain_dg_values))
    chain_tm_proxy = chain_temps[chain_min_idx]
    return chain_tm_proxy


def calculate_t_onset_proxy(struct, fraction_unfolded=0.02):
    """
    Estimate onset temperature (T_onset) proxy via two-state Gibbs-Helmholtz.
    Approximates T where ΔG_NU(T) + RTln(K_NU) ≈ 0 by scanning T from T_LOW to T_HIGH
    and selecting the T with the smallest |ΔG_NU(T) + RT·ln(K_NU)|.

    Q_U(T) = K_NU(T) / (1 + K_NU(T))         [extent of unfolding, from literature]

    Rerrange:
    Q_U · (1 + K_NU) = K_NU
    Q_U + Q_U·K_NU = K_NU
    Q_U = K_NU - Q_U·K_NU
    Q_U = K_NU·(1 - Q_U)
    K_NU = Q_U / (1 - Q_U)                   [rearranged; Q_U = fraction_unfolded at T_onset]

    ΔG_NU(T) =  ΔH_REF + ΔCp(T - T_REF) - T[ΔS_REF + ΔCpln(T/T_REF)]
    0 = ΔH_REF + ΔCp(T - T_REF) - T[ΔS_REF + ΔCp·ln(T/T_REF)] + RT·ln(K_NU)

    T_REF = 298.15 K (reference temperature for static pdb);
    T_onset is the T in [T_LOW, T_HIGH] that minimizes |ΔG_NU(T) + RT·ln(K_NU)|.

    Parameters:
    - struct: Biopython structure object
    - fraction_unfolded: unfolding_extent at onset (default 0.02 = 2%)

    Returns:
    - T_onset proxy in Kelvin
    """

    #Collect all standard residues from all chains
    residues = []
    for chain in struct[0]:
        for res in chain.get_residues():
            if res.id[0] != " ":
                continue
            residues.append(res)

    delta_H_REF = float(calc_delta_H_REF(struct))
    delta_Cp_proxy = float(calc_delta_Cp_proxy(struct)[0])
    delta_S_REF = float(calc_delta_S_REF(struct))
    
    ln_K = math.log(fraction_unfolded / (1.0 - fraction_unfolded))

    temps = np.linspace(T_LOW, T_HIGH, 1000)
    dg_values = [calc_delta_G_NU(T, delta_H_REF, delta_Cp_proxy, delta_S_REF) + R_KCAL * T * ln_K for T in temps]
    min_idx = np.argmin(np.abs(dg_values))
    t_onset_proxy = temps[min_idx]
    return t_onset_proxy


def calculate_chain_t_onset_proxy(chain, fraction_unfolded=0.02):
    """
    Estimate onset temperature (T_onset) proxy via two-state Gibbs-Helmholtz of a single chain.
    Approximates T where ΔG_NU(T) + RTln(K_NU) ≈ 0 by scanning T from T_LOW to T_HIGH
    and selecting the T with the smallest |ΔG_NU(T) + RT·ln(K_NU)|.

    Q_U(T) = K_NU(T) / (1 + K_NU(T))         [extent of unfolding, from literature]

    Rerrange:
    Q_U · (1 + K_NU) = K_NU
    Q_U + Q_U·K_NU = K_NU
    Q_U = K_NU - Q_U·K_NU
    Q_U = K_NU·(1 - Q_U)
    K_NU = Q_U / (1 - Q_U)                   [rearranged; Q_U = fraction_unfolded at T_onset]

    ΔG_NU(T) =  ΔH_REF + ΔCp(T - T_REF) - T[ΔS_REF + ΔCpln(T/T_REF)]
    0 = ΔH_REF + ΔCp(T - T_REF) - T[ΔS_REF + ΔCp·ln(T/T_REF)] + RT·ln(K_NU)

    T_REF = 298.15 K (reference temperature for static pdb);
    T_onset is the T in [T_LOW, T_HIGH] that minimizes |ΔG_NU(T) + RT·ln(K_NU)|.
    """

    #Collect all standard residues from all chains
    residues = []
    
    for res in chain.get_residues():
        if res.id[0] != " ":
            continue
        residues.append((chain, res))

    chain_delta_H_REF = float(calc_chain_delta_H_REF(chain))
    chain_delta_Cp_proxy = float(calc_chain_delta_Cp_proxy(chain)[0])
    chain_delta_S_REF = float(calc_chain_delta_S_REF(chain))
    
    ln_K = math.log(fraction_unfolded / (1.0 - fraction_unfolded))

    chain_temps = np.linspace(T_LOW, T_HIGH, 1000)
    chain_dg_values = [calc_delta_G_NU(T, chain_delta_H_REF, chain_delta_Cp_proxy, chain_delta_S_REF) + R_KCAL * T * ln_K for T in chain_temps]
    chain_min_idx = np.argmin(np.abs(chain_dg_values))
    chain_t_onset_proxy = chain_temps[chain_min_idx]
    return chain_t_onset_proxy


# --- SECONDARY STRUCTURE HELPER FUNCTIONS ---

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


# --- MAIN PIPELINE ---
def run_structure_thermal_stability_secondary_structure_analysis():
    MASTER_CSV = Path("/home/bunsree/projects/rosalind-bioinformatics/multispecific_antibodies/TheraSAbDab_SeqStruc_07Dec2025.csv")
    # Read MASTER_CSV
    df_master = pd.read_csv(MASTER_CSV)
    df_master['key'] = df_master['Therapeutic'].str.lower().str.strip()

    for BASE_DIR in BASE_DIRS:
        if not BASE_DIR.exists():
            print(f"ERROR: PDB directory not found at {BASE_DIR}")
            return

        print(f"\n=== Running analysis for {BASE_DIR} ===")
    
        OUTPUT_CSV = BASE_DIR / "Structure_Based_Thermal_Stability_and_Secondary_Structure_Module.csv"
    
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

        print(f"Found {len(pdb_map)} PDB files for THERMAL STABILITY AND SECONDARY STRUCTURE analysis.")
        
        # Setup tools
        parser = PDBParser(QUIET=True)
        master_features = []
        failed_entries = []
        
        # Process every PDB found
        print("--- CALCULATING STRUCTURE-BASED THERMAL STABILITY AND SECONDARY STRUCTURE ANALYSIS ---")
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
                
                # Structure-level thermodynamic calculations (Fab or Fv structure)
                delta_H_REF = calc_delta_H_REF(struct)                
                delta_S_REF = calc_delta_S_REF(struct)
                delta_Cp_proxy = calc_delta_Cp_proxy(struct)[0]
                delta_G_NU = calc_delta_G_NU(
                    T_REF, 
                    delta_H_REF,
                    delta_Cp_proxy,
                    delta_S_REF
                )
                
                delta_G_NU_at_T_REF = calc_delta_G_NU_at_T_REF(
                    T_REF, 
                    delta_H_REF,
                    delta_S_REF
                )

                tm_proxy = calculate_tm_proxy(struct)
                t_onset_proxy = calculate_t_onset_proxy(struct)
                
                # Unpack the returned values from calc_delta_Cp_proxy
                delta_Cp_proxy, delta_sasa_hydrophobic, delta_sasa_polar, hydrophobic_sasa_folded, polar_sasa_folded, hydrophobic_sasa_unfolded, polar_sasa_unfolded = calc_delta_Cp_proxy(struct)
                
                # Structural Fab/Fv-level secondary structure features (DSSP, 3D H-bond geometry based)
                secondary_structure_counts, secondary_structure_percentages = calculate_secondary_structure_features(struct, pdb_path)
                
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
                
                # Compile data for each chain
                chain_delta_H_REF = calc_chain_delta_H_REF(chain)                
                chain_delta_S_REF = calc_chain_delta_S_REF(chain)
                chain_delta_Cp_proxy = calc_chain_delta_Cp_proxy(chain)[0]
                delta_G_NU = calc_delta_G_NU(
                    T_REF, 
                    chain_delta_H_REF,
                    chain_delta_Cp_proxy,
                    chain_delta_S_REF
                )
                
                delta_G_NU_at_T_REF = calc_delta_G_NU_at_T_REF(
                    T_REF, 
                    chain_delta_H_REF,
                    chain_delta_S_REF
                )
            
                chain_tm_proxy = calculate_chain_tm_proxy(chain)
                chain_t_onset_proxy = calculate_chain_t_onset_proxy(chain)

                # Unpack the returned values from calc_chain_delta_Cp_proxy
                chain_delta_Cp_proxy, chain_delta_sasa_hydrophobic, chain_delta_sasa_polar, chain_hydrophobic_sasa_folded, chain_polar_sasa_folded, chain_hydrophobic_sasa_unfolded, chain_polar_sasa_unfolded = calc_chain_delta_Cp_proxy(chain)
                              
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
                    "Ch_A_length": chA_length,
                    "Ch_B_length": chB_length,
                    "Fab_Total_AAs": total_residues,
                    # Struct-level Interaction counts (inputs to Tm/T_onset)
                    "N_Disulfide_Bonds": count_disulfide_bonds(struct),
                    "N_Salt_Bridges": count_salt_bridges(struct),
                    "N_Hydrogen_Bonds": count_hydrogen_bonds(struct),
                    "Delta_H_REF": round(calc_delta_H_REF(struct), 3),
                    "Delta_Cp_Proxy": round(delta_Cp_proxy, 3),
                    "Hydrophobic_SASA_Folded": round(hydrophobic_sasa_folded, 3),
                    "Polar_SASA_Folded": round(polar_sasa_folded, 3),
                    "Hydrophobic_SASA_Unfolded": round(hydrophobic_sasa_unfolded, 3),
                    "Polar_SASA_Unfolded": round(polar_sasa_unfolded, 3),
                    "Delta_SASA_Hydrophobic": round(delta_sasa_hydrophobic, 3),
                    "Delta_SASA_Polar": round(delta_sasa_polar, 3),
                    "Delta_G_NU": round(delta_G_NU, 3),
                    "Delta_G_NU_at_T_REF": round(delta_G_NU_at_T_REF, 3),
                    "Delta_S_REF": round(delta_S_REF, 3),
                    # Struct-level Thermal stability proxies
                    "Tm_Proxy_K": round(float(tm_proxy), 3) if tm_proxy is not None else None,
                    "Tm_Proxy_C": round(float(tm_proxy - 273.15), 3) if tm_proxy is not None else None,
                    "T_Onset_Proxy_K": round(float(t_onset_proxy), 3) if t_onset_proxy is not None else None,
                    "T_Onset_Proxy_C": round(float(t_onset_proxy - 273.15), 3) if t_onset_proxy is not None else None,
                    # Struct-level Secondary structure features (DSSP)
                    "N_Helix": secondary_structure_counts["alpha_helix"],
                    "N_Sheet": secondary_structure_counts["beta_sheet"],
                    "N_Coil": secondary_structure_counts["coil"],
                    "Pct_Helix": round(secondary_structure_percentages["alpha_helix"], 2),
                    "Pct_Sheet": round(secondary_structure_percentages["beta_sheet"], 2),
                    "Pct_Coil": round(secondary_structure_percentages["coil"], 2),
                    # Chain-level Interaction counts (inputs to Tm/T_onset)
                    "Chain_N_Disulfide_Bonds": count_chain_disulfide_bonds(chain),
                    "Chain_N_Salt_Bridges": count_chain_salt_bridges(chain),
                    "Chain_N_Hydrogen_Bonds": count_chain_hydrogen_bonds(chain),
                    "Chain_Delta_H_REF": round(calc_chain_delta_H_REF(chain), 3),
                    "Chain_Delta_Cp_Proxy": round(chain_delta_Cp_proxy, 3),
                    "Chain_Hydrophobic_SASA_Folded": round(chain_hydrophobic_sasa_folded, 3),
                    "Chain_Polar_SASA_Folded": round(chain_polar_sasa_folded, 3),
                    "Chain_Hydrophobic_SASA_Unfolded": round(chain_hydrophobic_sasa_unfolded, 3),
                    "Chain_Polar_SASA_Unfolded": round(chain_polar_sasa_unfolded, 3),
                    "Chain_Delta_SASA_Hydrophobic": round(chain_delta_sasa_hydrophobic, 3),
                    "Chain_Delta_SASA_Polar": round(chain_delta_sasa_polar, 3),
                    "Chain_Delta_G_NU": round(delta_G_NU, 3),
                    "Chain_Delta_G_NU_at_T_REF": round(delta_G_NU_at_T_REF, 3),
                    "Chain_Delta_S_REF": round(chain_delta_S_REF, 3),
                    # Chain-level Thermal stability proxies
                    "Chain_Tm_Proxy_K": round(float(chain_tm_proxy), 3) if chain_tm_proxy is not None else None,
                    "Chain_Tm_Proxy_C": round(float(chain_tm_proxy - 273.15), 3) if chain_tm_proxy is not None else None,
                    "Chain_T_Onset_Proxy_K": round(float(chain_t_onset_proxy), 3) if chain_t_onset_proxy is not None else None,
                    "Chain_T_Onset_Proxy_C": round(float(chain_t_onset_proxy - 273.15), 3) if chain_t_onset_proxy is not None else None,
                }

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

            # create matching keys
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
            print(f"\n=== SUCCESS: Thermal Stability and Secondary Structure analysis complete for {len(df)} antibodies ===")
            print(f"File saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_structure_thermal_stability_secondary_structure_analysis()