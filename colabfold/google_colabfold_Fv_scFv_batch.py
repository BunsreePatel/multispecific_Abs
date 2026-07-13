# ============================================================================
# COLABFOLD Fv BUILDER - BATCH MODE
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
OUTPUT_ROOT = Path("/content/drive/MyDrive/PDB_Output_ColabFold_Fv_scFv_Structures")
TEMP_FASTA_DIR = Path("/content/drive/MyDrive/temp_fastas")
COLABFOLD_BIN = "/usr/local/bin/colabfold_batch"

# IUPAC Standard 20 Amino Acids
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

# ============================================================================
# CONSTANTS — Linker Only (Fv level modeling)
# ============================================================================

CONSTANTS = {
    'Linker': {
        'Brolucizumab':  ("GGGGGSGGGGSGGGGSGGGGS",  "VL-VH"),
        'Efungumab':     ("GGGGSGGGGSGGGGS",        "VH-VL"),
        'Gancotamab':    ("GGGGSGGGGSGGGGS",        "VH-VL"),
        'Licaminlimab':  ("GGGGGSGGGGSGGGGSGGGGS",  "VL-VH"),
        'Pexelizumab':   ("RTGGGGSGGGGSGGGGS",      "VL-VH"),
    }
}

# BROLUCIZUMAB HAS ORIENTATION: VL-linker-VH
# EFUNGUMAB HAS ORIENTATION: VH-linker-VL
# GANCOTAMAB HAS ORIENTATION: VH-linker-VL
# LICAMINLIMAB HAS ORIENTATION: VL-linker-VH
# PEXELIZUMAB HAS ORIENTATION: VL-linker-VH, includes RT linker spacer residues

def build_fv_sequence(VH: str, VL: str, antibody_name: str) -> str:
    linker, orientation = CONSTANTS['Linker'][antibody_name]
    if orientation == "VH-VL":
        return VH + linker + VL
    elif orientation == "VL-VH":
        return VL + linker + VH
    else:
        raise ValueError(f"Unknown orientation '{orientation}' for {antibody_name}")
    
def clean_sequence(seq):
    """Removes any characters not in the standard 20 AA alphabet."""
    return "".join([aa for aa in seq.upper() if aa in AA_ALPHABET])
   
# EXTRACT SEQUENCES and return (VH, VL) from FASTA/python file.
def extract_chains_dynamic(file_path):
    
    # Dictionary to store chains explicitly by index
    chain_dict = {
        1: {"H": None, "L": None}
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
            elif re.search(r'(LIGHT_CHAIN_1|_L1\b)', header):
                chain_dict[1]["L"] = clean_seq
            else:
                # If header does not match H1/L1 explicitly, skip it
                continue

        # Validate that all chains are found
        missing = []
        for idx in [1]:
            if not chain_dict[idx]["H"]:
                missing.append(f"H{idx}")
            if not chain_dict[idx]["L"]:
                missing.append(f"L{idx}")

        if missing:
            print(f"Warning: {file_path.stem} missing chains: {', '.join(missing)}")
            return None

        return (chain_dict[1]["H"], chain_dict[1]["L"])

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# ============================================================================
# CREATE Fv FASTA
# ============================================================================
def create_Fv_fasta(VH: str, VL: str, antibody_name: str, output_fasta: Path):
# Create a FASTA with VH + linkers and/or spacers for one Fv, using the antibody-specific linker from CONSTANTS.
    
    fv_sequence = build_fv_sequence(VH, VL, antibody_name)

    with open(output_fasta, 'w') as f:
        f.write(">Fv\n")
        f.write(f"{fv_sequence}\n")

# ============================================================================
# RUN COLABFOLD
# ============================================================================
def run_colabfold(fasta_path: Path, output_dir: Path, antibody_name: str):
    """Run ColabFold batch for Fv modeling using Google AlphaFold2.ipynb notebook."""
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

# Rename Output .pdb — save as *_Fv.pdb
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

    # Get antibody files   
    antibody_files = sorted([
        f for f in BASE_DIR.rglob("*.py") 
        if not f.name.startswith("._")
    ])
    
    if not antibody_files:
        print("[ERROR] No antibody files found")
        return

    print(f"\nSTARTING COLABFOLD ASSEMBLY: {len(antibody_files)} antibody file(s)\n")

    for f_path in tqdm(antibody_files, desc="Building Fv Structures with ColabFold"):
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"FOLDING: {f_path.stem}")
        tqdm.write(f"{'='*60}")
        
        # Extract sequences
        result = extract_chains_dynamic(f_path)
        if not result:
            tqdm.write(f"[ERROR] No valid sequences found in {f_path.stem}")
            continue

        (VH, VL) = result
        Fv_name = "Fv1"

        tqdm.write(f"Processing {Fv_name} — VH: {len(VH)} aa, VL: {len(VL)} aa")

        # Create Fv FASTA
        fasta_path = TEMP_FASTA_DIR / f"{f_path.stem}_{Fv_name}.fasta"
        create_Fv_fasta(VH, VL, f_path.stem.capitalize(), fasta_path)
        tqdm.write(f"Created FASTA: {fasta_path}")

        # Create output directory
        antibody_output = OUTPUT_ROOT / f"{f_path.stem}_{Fv_name}"
        os.makedirs(antibody_output, exist_ok=True)

        # Run ColabFold
        tqdm.write(f"Running ColabFold for {Fv_name}...")
        success = run_colabfold(fasta_path, antibody_output, f"{f_path.stem}_{Fv_name}")
        if success:
            final_pdb = OUTPUT_ROOT / f"{f_path.stem}_{Fv_name}.pdb"
            rename_output_pdb(antibody_output, f"{f_path.stem}_{Fv_name}", final_pdb)
        tqdm.write("")

    print("\n" + "="*60)
    print("=== Fv STRUCTURE GENERATION COMPLETE ===")
    print("="*60)

    # Show results
    pdb_files = list(OUTPUT_ROOT.glob("*_Fv*.pdb"))
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