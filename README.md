# Multispecific Antibodies Sequence Database

A collection of therapeutic multispecific and monospecific antibody sequences in FASTA format.

## Data Source
Antibody sequences obtained from [Thera-SAbDab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/therasabdab/search/)

**Citations:**
- Schneider, C., Raybould, M.I.J., Deane, C.M. (2022) SAbDab in the Age of Biotherapeutics: Updates including SAbDab-Nano, the Nanobody Structure Tracker. *Nucleic Acids Res.* 50(D1):D1368-D1372
- Raybould, M.I.J., Marks, C. et al (2019) Thera-SAbDab: the Therapeutic Structural Antibody Database. *Nucleic Acids Res.* gkz827
- Dunbar, J., Krawczyk, K. et al (2014) SAbDab: the Structural Antibody Database. *Nucleic Acids Res.* 42:D1140-D1146

## Repository Contents
- Individual `.py` files containing FASTA sequences for each therapeutic antibody
- `sabdabconverter.py` - Tool for converting Thera-SAbDab data to FASTA format
- `readme_count.py` - Script to update antibody count in README

## Usage
Each antibody file contains sequences in FASTA format stored as a string variable:
```python
from navicixizumab import navicixizumab
print(navicixizumab)
```

Or parse the FASTA:
```python
with open('navicixizumab.py', 'r') as f:
    content = f.read()
    # Extract sequences from triple-quoted string
```

## License
- **Code** (sabdabconverter.py, readme_count.py): MIT License
- **Data** (antibody sequences): Sourced from Thera-SAbDab (publicly available data)

## Disclaimer
These sequences are from publicly available therapeutic antibodies as curated by Thera-SAbDab. For commercial use, please verify current patent status and regulatory information.

## Statistics
- Data last updated: June 2026
- Total antibodies: 633