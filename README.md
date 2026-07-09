## Multispecific Antibody Predictor

This GitHub repository presents an end-to-end pipeline for multispecific antibody prediction using ColabFold, extracting 3D protein features, and engineering datasets for machine learning models to assess developability and biophysical risk.

### Project Roadmap
Phase 1: Collect antibody FASTA sequences and organize different formats

Phase 2: Fold Fab and Fv structures with ColabFold and compute all four structure-
based feature engineering modules

Phase 3: Execute ML model development and validation

Phase 4: Deploy Streamlit app for developability and biophysical risk profiling

#### Phase 1
Multispecific Antibodies Sequence Database - A collection of therapeutic multispecific and monospecific antibody FASTA sequences. Formats include Whole mAb, Bispecific mAb, scFv, Bispecific scFv, and BiTE (Bispecific T-cell Engager).

##### Data Source
Antibody sequences obtained from [Thera-SAbDab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/therasabdab/search/)

##### References:
- Schneider, C., Raybould, M.I.J., Deane, C.M. (2022) SAbDab in the Age of Biotherapeutics: Updates including SAbDab-Nano, the Nanobody Structure Tracker. *Nucleic Acids Res.* 50(D1):D1368-D1372
- Raybould, M.I.J., Marks, C. et al (2019) Thera-SAbDab: the Therapeutic Structural Antibody Database. *Nucleic Acids Res.* gkz827
- Dunbar, J., Krawczyk, K. et al (2014) SAbDab: the Structural Antibody Database. *Nucleic Acids Res.* 42:D1140-D1146

#### Phase 2
Antibody Structure-Based Feature Engineering - Phase 2 converts antibody sequence data from Phase 1 into PDB structure files using ColabFold and extracts quantitative biophysical structure-based features.

- Hydrophobicity, Aggregation, Charge Module

- Thermal Stability and Secondary Structure Module

- Hotspots and PTM Susceptibility Module

- Format-Specific Attributes Module

#### Phase 3 - In Progress

#### Phase 4 - Planned

#### License
- Code: MIT License
- Antibody Sequences: Sourced from Thera-SAbDab (publicly available data)

#### Disclaimer
These sequences are from publicly available therapeutic antibodies as curated by Thera-SAbDab. For commercial use, please verify current patent status and regulatory information.

#### Statistics
- Total antibodies: 628