# ============================================================================
# COLABFOLD FAB BUILDER - BATCH MODE
# ============================================================================

import os
import pandas as pd
import re
import warnings
import logging
from pathlib import Path
from tqdm import tqdm
import subprocess
import json
from google.colab import files
import time

# Suppress library noise
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# --- COLAB PATHS ---
BASE_DIR = Path("/content/drive/MyDrive/antibody_data")
OUTPUT_ROOT = Path("/content/drive/MyDrive/PDB_Output_ColabFold_Fab_Structures")
CSV_PATH = Path("/content/drive/MyDrive/TheraSAbDab_SeqStruc_07Dec2025.csv")
TEMP_FASTA_DIR = Path("/content/drive/MyDrive/temp_fastas")
COLABFOLD_BIN = "/usr/local/bin/colabfold_batch"

# IUPAC Standard 20 Amino Acids
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

# ============================================================================
# CONSTANTS — CH1 ONLY + CL ONLY (Fab level modeling)
# ============================================================================

# Human Constant Region Sequences (Standard Reference Library)
CONSTANTS = {
    'HEAVY_CH1': {
        'G1': (
            "ASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKV"
        ), 
        'G2': (
            "ASTKGPSVFPLAPCSRSTSESTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSNFGTQTYTCNVDHKPSNTKVDKTV"
        ),
        'G4': (
            "ASTKGPSVFPLAPCSRSTSESTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSSLGTKTYTCNVDHKPSNTKVDKRV"
        ),
    },

    'LIGHT_CL': {
        'Kappa': (
            "RTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"
        ),
        'Lambda': (
            "GQPKAAPSVTLFPPSSEELQANKATLVCLISDFYPGAVTVAWKADSSPVKAGVETTTPSKQSNNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTVAPTECS"
        ),
    }
}

def clean_sequence(seq):
    """Removes any characters not in the standard 20 AA alphabet."""
    return "".join([aa for aa in seq.upper() if aa in AA_ALPHABET])

def detect_isotype(file_path, lookup_dict):
    """Scans the source file for isotype keywords."""
    stem_lower = file_path.stem.lower()
    # Priority 1: CSV Lookup
    if stem_lower in lookup_dict:
        h_raw, l_raw = lookup_dict[stem_lower]
        h_iso = 'G1'
        if 'G2' in str(h_raw): h_iso = 'G2'
        elif 'G4' in str(h_raw): h_iso = 'G4'
        l_iso = 'Lambda' if 'LAMBDA' in str(l_raw).upper() else 'Kappa'
        return h_iso, l_iso
    
    # Priority 2: Keyword Scan
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().upper()
        h_iso = 'G1'
        if 'G2' in content or 'IGG2' in content: h_iso = 'G2'
        elif 'G4' in content or 'IGG4' in content: h_iso = 'G4'
        l_iso = 'Lambda' if 'LAMBDA' in content else 'Kappa'
        return h_iso, l_iso
    except: 
        return 'G1', 'Kappa'
    
# EXTRACT SEQUENCES and return 2 Fab pairs as ((H1,L1),(H2,L2)) from FASTA/python file.
def extract_chains_dynamic(file_path):
    
    # Dictionary to store chains explicitly by index
    chain_dict = {
        1: {"H": None, "L": None},
        2: {"H": None, "L": None}
    }

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into FASTA blocks
        blocks = content.split('>')
        for block in blocks[1:]:  # skip anything before the first '>'
            lines = block.strip().split('\n')
            if len(lines) < 2:
                continue

            header = lines[0].upper()
            raw_seq = "".join(lines[1:]).strip()
            clean_seq = clean_sequence(raw_seq)

            if not clean_seq or len(clean_seq) < 20:
                continue

            # Determine chain type and index explicitly
            if re.search(r'(HEAVY_CHAIN_1|_H1\b)', header):
                chain_dict[1]["H"] = clean_seq
            elif re.search(r'(HEAVY_CHAIN_2|_H2\b)', header):
                chain_dict[2]["H"] = clean_seq
            elif re.search(r'(LIGHT_CHAIN_1|_L1\b)', header):
                chain_dict[1]["L"] = clean_seq
            elif re.search(r'(LIGHT_CHAIN_2|_L2\b)', header):
                chain_dict[2]["L"] = clean_seq
            else:
                # If header does not match H1/H2/L1/L2 explicitly, skip it
                continue

        # Validate that all chains are found
        missing = []
        for idx in [1, 2]:
            if not chain_dict[idx]["H"]:
                missing.append(f"H{idx}")
            if not chain_dict[idx]["L"]:
                missing.append(f"L{idx}")

        if missing:
            print(f"Warning: {file_path.stem} missing chains: {', '.join(missing)}")
            return None

        # Return explicit Fab pairs
        return (chain_dict[1]["H"], chain_dict[1]["L"]), (chain_dict[2]["H"], chain_dict[2]["L"])

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# ============================================================================
# CREATE FAB FASTA
# ============================================================================
def create_fab_fasta(heavy_fv: str, light_fv: str, h_iso: str, l_iso: str, output_fasta: Path):
# Create a FASTA with VH + CH1 and VL + CL for one Fab
    heavy_fab = heavy_fv + CONSTANTS['HEAVY_CH1'][h_iso]
    light_fab = light_fv + CONSTANTS['LIGHT_CL'][l_iso]
        
    with open(output_fasta, 'w') as f:
        f.write(">Fab\n")
        f.write(f"{heavy_fab}:{light_fab}\n")

# ============================================================================
# RUN COLABFOLD
# ============================================================================
def run_colabfold(fasta_path: Path, output_dir: Path, antibody_name: str):
    """Run ColabFold batch for Fab multimer modeling using Google AlphaFold2.ipynb notebook."""
    cmd = [
        COLABFOLD_BIN,
        str(fasta_path),
        str(output_dir),
        "--msa-mode", "single_sequence",
        "--num-models", "1",
        "--num-recycle", "3",
        "--model-type", "alphafold2_multimer_v3",
        "--amber",
        "--rank", "multimer",
    ]

    log_path = output_dir / f"{antibody_name}_colabfold.log"

    try:
        with open(log_path, 'w') as log_file:
            result = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=1800,
            )

        if result.returncode != 0:
            print(f"[FAILED] {antibody_name}")
            return False

        print(f"[SUCCESS] {antibody_name}")
        return True

    except Exception as e:
        print(f"[EXCEPTION] {antibody_name}: {e}")
        return False

# Rename Output .pdb — save as *_Fab.pdb
def rename_output_pdb(output_dir: Path, antibody_name: str, final_output_path: Path):

    pdb_files = list(output_dir.glob("*_relaxed_rank_001_*.pdb"))
    if not pdb_files:
        pdb_files = list(output_dir.glob("*_unrelaxed_rank_001_*.pdb"))
    
    if pdb_files:
        source_pdb = pdb_files[0]
        source_pdb.rename(final_output_path)
        print(f"[SAVED] {final_output_path}")
    else:
        print(f"[ERROR] No PDB for {antibody_name}")

def run_pipeline():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(TEMP_FASTA_DIR, exist_ok=True)

    # Load Master Isotype Data from CSV
    isotype_lookup = {}
    if CSV_PATH.exists():
        try:
            df = pd.read_csv(CSV_PATH)
            isotype_lookup = dict(zip(df['Therapeutic'].str.lower(), zip(df['CH1 Isotype'], df['VD LC'])))
            print(f"--- LOADED {len(isotype_lookup)} ISOTYPES FROM CSV ---")
        except Exception as e:
            print(f"--- ERROR LOADING CSV: {e}. Falling back to keyword search only. ---")
    else:
        print("--- NO CSV FOUND. Using keyword search for isotypes. ---")

    # Get antibody files   
    antibody_files = sorted([
        f for f in BASE_DIR.rglob("*.py") 
        if not f.name.startswith("._")
    ])
    
    if not antibody_files:
        print("[ERROR] No antibody files found")
        return

    print(f"\nSTARTING COLABFOLD ASSEMBLY: {len(antibody_files)} antibody file(s)\n")

    for f_path in tqdm(antibody_files, desc="Building Fab Structures with ColabFold"):
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"FOLDING: {f_path.stem}")
        tqdm.write(f"{'='*60}")
        
        # Extract sequences
        result = extract_chains_dynamic(f_path)
        if not result:
            tqdm.write(f"[ERROR] No valid sequences found in {f_path.stem}")
            continue

        # Unpack bispecific Fab pairs
        (fab1_VH, fab1_VL), (fab2_VH, fab2_VL) = result
        
        # Detect isotype
        h_iso, l_iso = detect_isotype(f_path, isotype_lookup)
        tqdm.write(f"Isotype: Heavy={h_iso}, Light={l_iso}")
        
        # Loop over each distinct Fab
        fab_pairs = [
            ("Fab1", fab1_VH, fab1_VL),
            ("Fab2", fab2_VH, fab2_VL)
        ]

        for fab_name, VH_chain, VL_chain in fab_pairs:
            tqdm.write(f"Processing {fab_name} — VH: {len(VH_chain)} aa, VL: {len(VL_chain)} aa")

            # Create Fab FASTA
            fasta_path = TEMP_FASTA_DIR / f"{f_path.stem}_{fab_name}.fasta"
            create_fab_fasta(VH_chain, VL_chain, h_iso, l_iso, fasta_path)
            tqdm.write(f"Created FASTA: {fasta_path}")

            # Create output directory
            antibody_output = OUTPUT_ROOT / f"{f_path.stem}_{fab_name}"
            os.makedirs(antibody_output, exist_ok=True)

            # Run ColabFold
            tqdm.write(f"Running ColabFold for {fab_name}...")
            success = run_colabfold(fasta_path, antibody_output, f"{f_path.stem}_{fab_name}")
            if success:
                final_pdb = OUTPUT_ROOT / f"{f_path.stem}_{fab_name}.pdb"
                rename_output_pdb(antibody_output, f"{f_path.stem}_{fab_name}", final_pdb)
            tqdm.write("")

    print("\n" + "="*60)
    print("=== FAB STRUCTURE GENERATION COMPLETE ===")
    print("="*60)

    # Show results
    pdb_files = list(OUTPUT_ROOT.glob("*_Fab*.pdb"))
    if pdb_files:
        print(f"\n Generated {len(pdb_files)} PDB file(s):")
        for pdb in pdb_files:
            print(f"  - {pdb.name}")
        print(f"\n Download PDB files from: {OUTPUT_ROOT}")
    else:
        print("\n No PDB files generated. Check logs above for errors.")

# ============================================================================
# STEP 1: UPLOAD FILES
# ============================================================================
print("="*60)
print("STEP 1: UPLOAD FILES")
print("="*60)
print("\n1. Upload antibody.py files")
print("2. Upload CSV file (TheraSAbDab_SeqStruc_07Dec2025.csv)")
print("\nClick 'Choose Files' below:\n")

uploaded = files.upload()

# Create directories and move files
os.makedirs(BASE_DIR, exist_ok=True)

for filename, content in uploaded.items():
    if filename.endswith('.py'):
        # Save antibody file
        dest = BASE_DIR / filename
        with open(dest, 'wb') as f:
            f.write(content)
        print(f"✓ Saved antibody file: {dest}")
    elif filename.endswith('.csv'):
        # Save CSV
        with open(CSV_PATH, 'wb') as f:
            f.write(content)
        print(f"✓ Saved CSV: {CSV_PATH}")

print("\n" + "="*60)
print("FILES UPLOADED SUCCESSFULLY")
print("="*60)

# ============================================================================
# STEP 2: RUN PIPELINE
# ============================================================================
print("\n\nStarting pipeline in 3 seconds...\n")
time.sleep(3)

if __name__ == "__main__":
    run_pipeline()