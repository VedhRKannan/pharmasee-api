from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import logging
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import RDLogger


# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# ✅ Create the Flask app
app = Flask(__name__)


# ✅ Load models at startup
models = {}
model_names = ["lipophilicity (logD)", "solubility (logS)"]

try:
    for name in model_names:
        model_path = os.path.join("saved_models", f"{name}.pkl")
        models[name] = joblib.load(model_path)
    logger.info("✅ Models loaded successfully.")
except Exception as e:
    logger.error(f"❌ Failed to load models: {e}")

def mol_to_fp(smiles, radius=2, nBits=1024):
    """Convert SMILES to a Morgan fingerprint (as a numpy array)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits, dtype=int)  # Return zero array if invalid SMILES
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# ✅ Health check route
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Pharmasee API is running!"})

# ✅ Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse the JSON request
        data = request.get_json(force=True)
        smiles = data.get("smiles", "")

        if not smiles:
            return jsonify({"error": "No SMILES provided"}), 400

        # Validate SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return jsonify({"error": "Invalid SMILES format"}), 400

        # Generate predictions
        predictions = {}
        for name, model in models.items():
            fp = mol_to_fp(smiles)
            predictions[name] = float(model.predict([fp])[0])

        return jsonify(predictions)

    except Exception as e:
        logger.error(f"❌ Error processing prediction: {e}")
        return jsonify({"error": str(e)}), 500



# ==== hERG template-fit scorer (RDKit only) ==================================
import urllib.request, tempfile, base64

HERG_ASSETS = os.path.abspath("./herg_assets")
os.makedirs(HERG_ASSETS, exist_ok=True)
HERG_PDB = os.path.join(HERG_ASSETS, "8ZYP.pdb")
TEMPLATE_SDF = os.path.join(HERG_ASSETS, "template_ligand.sdf")

def _download_8ZYP():
    if not os.path.exists(HERG_PDB):
        urllib.request.urlretrieve("https://files.rcsb.org/download/8ZYP.pdb", HERG_PDB)

def _extract_template_sdf():
    """
    Extract the largest non-water HET group from 8ZYP and save as SDF for use as template ligand.
    RCSB PDB files include CONECT records so RDKit can perceive bonds.
    """
    if os.path.exists(TEMPLATE_SDF):
        return
    _download_8ZYP()
    with open(HERG_PDB, "r") as f:
        pdb = f.read()
    # keep only HETATM + matching CONECT for largest HET group
    lines = [ln for ln in pdb.splitlines() if ln.startswith(("HETATM","CONECT","END"))]
    # group HETATMs by residue name+id
    groups = {}
    for ln in lines:
        if ln.startswith("HETATM"):
            resn = ln[17:20].strip()
            resid = ln[22:26].strip()
            key = f"{resn}:{resid}"
            groups.setdefault(key, []).append(ln)
    if not groups:
        raise RuntimeError("No hetero ligand found in 8ZYP.")
    key = max(groups, key=lambda k: len(groups[k]))
    het_lines = groups[key]
    # collect atom serials for CONECT
    atom_ids = set(int(ln[6:11]) for ln in het_lines)
    conect = [ln for ln in lines if ln.startswith("CONECT") and int(ln[6:11]) in atom_ids]
    pdb_block = "\n".join(het_lines + conect) + "\nEND\n"

    # Build RDKit mol from PDB and write SDF
    tmpl = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=True)
    if tmpl is None:
        raise RuntimeError("Failed to parse template ligand from 8ZYP.")
    # ensure a conformer exists
    if tmpl.GetNumConformers() == 0:
        AllChem.EmbedMolecule(tmpl, AllChem.ETKDGv3())
    w = Chem.SDWriter(TEMPLATE_SDF); w.write(tmpl); w.close()

def _load_template_mol():
    _extract_template_sdf()
    suppl = Chem.SDMolSupplier(TEMPLATE_SDF, removeHs=False)
    mol = next((m for m in suppl if m is not None), None)
    if mol is None:
        raise RuntimeError("Template SDF could not be loaded.")
    return mol

def _build_query_3d(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    # try multiple conformers and keep the best after a fast UFF minimize
    nconf = 10
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=nconf, params=params)
    energies = []
    for cid in ids:
        AllChem.UFFOptimizeMolecule(mol, confId=cid, maxIters=200)
        e = AllChem.UFFGetMoleculeForceField(mol, confId=cid).CalcEnergy()
        energies.append((e, cid))
    energies.sort()
    # keep lowest-energy conformer only
    best = energies[0][1]
    # drop others
    conf = mol.GetConformer(best)
    newmol = Chem.Mol(mol)
    newmol.RemoveAllConformers()
    newmol.AddConformer(conf, assignId=True)
    return newmol

@app.route("/herg_quickscore", methods=["POST"])
def herg_quickscore():
    """
    Body: { "smiles": "..." }
    Returns: { shape_tanimoto, aligned_pose_sdf_b64, template: "8ZYP ligand" }
    """
    try:
        data = request.get_json(force=True) or {}
        smiles = (data.get("smiles") or "").strip()
        if not smiles:
            return jsonify({"error":"No SMILES provided"}), 400

        template = _load_template_mol()
        query = _build_query_3d(smiles)
        if query is None:
            return jsonify({"error":"Invalid SMILES or failed 3D build"}), 400

        # Open3DAlign (MMFF types) then compute shape Tanimoto
        o3a = AllChem.GetO3A(query, template, prbCid=0, refCid=0, atomMap=None)
        o3a.Align()
        # Shape Tanimoto (0 = no overlap, 1 = identical shape)
        from rdShapeHelpers import ShapeTanimotoDist
        dist = ShapeTanimotoDist(query, 0, template, 0)  # distance in [0..1]
        shape_tanimoto = 1.0 - float(dist)

        # Export aligned pose SDF
        sdf = Chem.MolToMolBlock(query, confId=0)
        sdf_b64 = base64.b64encode(sdf.encode("utf-8")).decode("ascii")

        return jsonify({
            "shape_tanimoto": shape_tanimoto,
            "aligned_pose_sdf_b64": sdf_b64,
            "template": "hERG 8ZYP co-crystal ligand (E-4031)"
        })
    except Exception as e:
        logger.exception("hERG quickscore error")
        return jsonify({"error": str(e)}), 500
# ============================================================================ #




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
