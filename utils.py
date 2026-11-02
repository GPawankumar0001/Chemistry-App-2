"""
Utilities for SMILES in Chemistry — Neon Lab.
Network requests are bounded by timeouts and all external data is treated as untrusted.
"""
from __future__ import annotations

import math
import re
import time
from typing import Any

import pandas as pd
import requests

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors
except Exception:  # Allow app to show a useful dependency error.
    Chem = None
    AllChem = Descriptors = Lipinski = rdMolDescriptors = None

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUG_VIEW = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "SMILES-in-Chemistry-SRMIST/1.0 (educational project)"})

class PubChemError(RuntimeError):
    """Raised when a PubChem record cannot be retrieved or interpreted."""

def _require_rdkit() -> None:
    if Chem is None:
        raise RuntimeError("RDKit is unavailable. Ensure rdkit is installed in the deployment environment.")

def _get_json(url: str, params: dict | None = None, timeout: int = 18) -> dict:
    try:
        response = SESSION.get(url, params=params, timeout=timeout)
        if response.status_code == 404:
            raise PubChemError("No matching PubChem record was found.")
        response.raise_for_status()
        return response.json()
    except requests.Timeout as exc:
        raise PubChemError("PubChem request timed out. Please try again.") from exc
    except requests.RequestException as exc:
        raise PubChemError(f"PubChem network error: {exc}") from exc
    except ValueError as exc:
        raise PubChemError("PubChem returned an unexpected response.") from exc

def validate_smiles(smiles: str) -> tuple[bool, str]:
    _require_rdkit()
    if not isinstance(smiles, str) or not smiles.strip():
        return False, "SMILES is empty."
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return False, "The supplied SMILES could not be parsed by RDKit."
    return True, "Valid SMILES."

def _resolve_cid(query: str) -> int:
    cleaned = query.strip()
    if cleaned.isdigit():
        return int(cleaned)
    # First interpret as SMILES, then as a compound name.
    if Chem is not None and Chem.MolFromSmiles(cleaned) is not None:
        url = f"{PUBCHEM}/compound/smiles/cids/JSON"
        data = _get_json(url, params={"smiles": cleaned, "name_type": "smiles"})
    else:
        url = f"{PUBCHEM}/compound/name/{requests.utils.quote(cleaned, safe='')}/cids/JSON"
        data = _get_json(url)
    cids = data.get("IdentifierList", {}).get("CID", [])
    if not cids:
        raise PubChemError("No PubChem CID was returned for this query.")
    return int(cids[0])

def fetch_pubchem_compound(query: str) -> dict[str, Any]:
    cid = _resolve_cid(query)
    props = (
        "CanonicalSMILES,IsomericSMILES,MolecularFormula,MolecularWeight,"
        "IUPACName,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount,RotatableBondCount"
    )
    url = f"{PUBCHEM}/compound/cid/{cid}/property/{props}/JSON"
    data = _get_json(url)
    rows = data.get("PropertyTable", {}).get("Properties", [])
    if not rows:
        raise PubChemError("PubChem returned no molecular properties.")
    p = rows[0]
    return {
        "cid": cid,
        "name": p.get("IUPACName") or f"PubChem CID {cid}",
        "iupac_name": p.get("IUPACName"),
        "molecular_formula": p.get("MolecularFormula"),
        "molecular_weight": p.get("MolecularWeight"),
        "canonical_smiles": p.get("ConnectivitySMILES") or p.get("CanonicalSMILES"),
        "isomeric_smiles": p.get("SMILES") or p.get("IsomericSMILES"),
        "xlogp": p.get("XLogP"),
        "tpsa": p.get("TPSA"),
        "hbd": p.get("HBondDonorCount"),
        "hba": p.get("HBondAcceptorCount"),
        "rotatable_bonds": p.get("RotatableBondCount"),
    }

def calculate_descriptors(smiles: str) -> dict[str, Any]:
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit could not parse this SMILES.")
    def safe_round(value: Any, digits: int = 2):
        try:
            number = float(value)
            return round(number, digits) if math.isfinite(number) else None
        except (TypeError, ValueError):
            return None
    return {
        "MolecularWeight": safe_round(Descriptors.MolWt(mol), 2),
        "ExactMass": safe_round(Descriptors.ExactMolWt(mol), 4),
        "XLogP": safe_round(Descriptors.MolLogP(mol), 2),
        "TPSA": safe_round(rdMolDescriptors.CalcTPSA(mol), 2),
        "NumHDonors": int(Lipinski.NumHDonors(mol)),
        "NumHAcceptors": int(Lipinski.NumHAcceptors(mol)),
        "RotatableBonds": int(Lipinski.NumRotatableBonds(mol)),
        "HeavyAtoms": int(mol.GetNumHeavyAtoms()),
        "FormalCharge": int(Chem.GetFormalCharge(mol)),
        "RingCount": int(rdMolDescriptors.CalcNumRings(mol)),
        "FractionCSP3": safe_round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        "MolarRefractivity": safe_round(Descriptors.MolMR(mol), 2),
    }

def classify_lipinski(desc: dict[str, Any]) -> str:
    """Educational rule-of-five screen, not a prediction of drug-likeness or efficacy."""
    checks = []
    for key, limit, label, greater in [
        ("MolecularWeight", 500, "molecular weight ≤ 500", False),
        ("XLogP", 5, "LogP ≤ 5", False),
        ("NumHDonors", 5, "H-bond donors ≤ 5", False),
        ("NumHAcceptors", 10, "H-bond acceptors ≤ 10", False),
    ]:
        value = desc.get(key)
        if value is not None:
            checks.append((value > limit if greater else value <= limit, label))
    passed = sum(1 for ok, _ in checks if ok)
    if len(checks) < 4:
        return f"Rule-of-five screen: {passed}/{len(checks)} available criteria within conventional limits. Incomplete data."
    return f"Rule-of-five screen: {passed}/4 criteria within conventional limits. This heuristic is not a clinical or efficacy assessment."

def render_descriptor_table(desc: dict[str, Any]) -> pd.DataFrame:
    labels = {
        "MolecularWeight": ("Molecular weight", "g/mol"),
        "ExactMass": ("Exact mass", "Da"),
        "XLogP": ("Calculated LogP", "dimensionless"),
        "TPSA": ("Topological polar surface area", "Å²"),
        "NumHDonors": ("H-bond donors", "count"),
        "NumHAcceptors": ("H-bond acceptors", "count"),
        "RotatableBonds": ("Rotatable bonds", "count"),
        "HeavyAtoms": ("Heavy atoms", "count"),
        "FormalCharge": ("Formal charge", "e"),
        "RingCount": ("Ring count", "count"),
        "FractionCSP3": ("Fraction Csp3", "fraction"),
        "MolarRefractivity": ("Molar refractivity", "cm³/mol"),
    }
    rows = []
    for key, (label, unit) in labels.items():
        if key in desc:
            rows.append({"Descriptor": label, "Value": desc[key], "Unit": unit})
    return pd.DataFrame(rows)

def generate_3d_structure(smiles: str, cid: int | None = None) -> str | None:
    """
    Try PubChem's 3D SDF first, then generate coordinates locally with RDKit.
    Returns an SDF mol block suitable for 3Dmol.js.
    """
    if cid is not None:
        try:
            url = f"{PUBCHEM}/compound/cid/{int(cid)}/record/SDF"
            response = SESSION.get(url, params={"record_type": "3d"}, timeout=18)
            if response.ok and "V2000" in response.text or "V3000" in response.text:
                return response.text
        except requests.RequestException:
            pass

    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    params.maxIterations = 200
    result = AllChem.EmbedMolecule(mol, params)
    if result != 0:
        return None
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=250)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=250)
    except Exception:
        pass
    return Chem.MolToMolBlock(mol)

def _flatten_pubchem_sections(node: dict, output: list[dict], cid: int) -> None:
    title = node.get("TOCHeading") or node.get("Name")
    if title and any(word in title.lower() for word in (
        "toxicity", "pharmacokinetics", "safety", "acute effects",
        "ld50", "carcinogenicity", "toxicity summary"
    )):
        snippets = []
        for info in node.get("Information", []) or []:
            value = info.get("Value", {})
            if "StringWithMarkup" in value:
                snippets.extend(x.get("String", "") for x in value["StringWithMarkup"] if x.get("String"))
            elif "String" in value:
                snippets.append(str(value["String"]))
        summary = " ".join(s for s in snippets if s).strip()
        if summary:
            output.append({
                "title": str(title),
                "summary": summary[:3500],
                "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}#section=Safety-and-Hazards",
            })
    for child in node.get("Section", []) or []:
        _flatten_pubchem_sections(child, output, cid)

def fetch_pubchem_safety_sections(cid: int) -> list[dict[str, str]]:
    url = f"{PUG_VIEW}/data/compound/{int(cid)}/JSON"
    data = _get_json(url, params={"heading": "Toxicity"})
    record = data.get("Record", {})
    output: list[dict[str, str]] = []
    _flatten_pubchem_sections(record, output, int(cid))
    # Avoid duplicate headings.
    seen = set()
    unique = []
    for item in output:
        key = item["title"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique
