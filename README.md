# SMILES in Chemistry — Neon Lab

A Streamlit educational chemistry dashboard for SRM IST, Tiruchirappalli.

## Files
- `app.py` — dashboard and navigation
- `utils.py` — PubChem, RDKit, descriptor calculations and 3D structure generation
- `assets/style.css` — dark glassmorphism theme

## Runtime dependencies
Install the project's existing dependencies plus:
`streamlit`, `requests`, `pandas`, `rdkit`

The browser 3D viewer loads 3Dmol.js from its public CDN. PubChem access requires outbound HTTPS.

## Run
`streamlit run app.py`

## Scientific scope
ADME-related fields are calculated physicochemical descriptors, not validated clinical pharmacokinetic outputs. Toxicity sections are source-linked public-record text. The app intentionally does not generate patient dosing or human LD50 recommendations.
