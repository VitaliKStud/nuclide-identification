# nuclide-identification

This repository is about to identify nuclides via neural network techniques for gamma-ray 
measurements. 

---

TODO: 

- Fill Documentation Environment and Setup section.
- Init folder-structure via poetry.

---

## Environment and Setup

- Python version: 3.13.1
- pip: 24.3.1
- poetry as package-manager

```
pip install poetry
```

## Folder Structure

---
```
.
├── data\
│   ├── measurements\       # Measurements of Gamma-Ray.
│   └── nuclides\           # Scraped Nuclide-Data from API (https://www-nds.iaea.org).
├── docs\                   # Documentation files.
├── plots\                  # Folder for created plots.
├── src\                    # Source-Code.
└── tests\                  # Unit-Tests.
```

---

## NOTES

Am-241: 59,5 keV
Co-60: 1173,2 und 1332,5 keV
Cs-137: 661,7 keV
Eu-152: 121,8 keV / 344,3 keV / 778,9 keV / 964 keV / 1085,8 keV / 1112,1 keV  / 1408 keV
Eu-154: 123,1 keV / 1274,4 keV / 723,32 keV / 1004,8 keV / 873,18 keV / 996,29 keV

8160 Channels 

# 2017-07-03_15-51-58_Summenspektrum.txt


- 1246 different Measurements


https://www-nds.iaea.org/

- Cool page with API, need to be explored
- https://de.wikipedia.org/wiki/Internationale_Atomenergie-Organisation
- https://www-nds.iaea.org/relnsd/v1/data?fields=decay_rads&nuclides=am241&rad_types=g
fields=decay_rads&nuclides=135xe&rad_types=e
- gammas