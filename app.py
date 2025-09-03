from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import logging
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import RDLogger


# ---- hERG DOCKING (GNINA + Meeko) ----
import subprocess, tempfile, base64, urllib.request

DOCK_ROOT = os.path.abspath("./herg_assets")
os.makedirs(DOCK_ROOT, exist_ok=True)
REC_PDB = os.path.join(DOCK_ROOT, "8ZYP.pdb")         # hERG E-4031-bound (good autobox ref)
REC_PDBQT = os.path.join(DOCK_ROOT, "herg_rec.pdbqt")
REF_LIG_PDB = os.path.join(DOCK_ROOT, "ref_ligand.pdb")
REF_LIG_PDBQT = os.path.join(DOCK_ROOT, "ref_ligand.pdbqt")

def _run(cmd, cwd=None):
    logger.info("RUN: %s", " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(cmd)}\n{res.stdout}")
    return res.stdout

def ensure_receptor_prepared():
    # 1) Download 8ZYP once
    if not os.path.exists(REC_PDB):
        url = "https://files.rcsb.org/download/8ZYP.pdb"
        urllib.request.urlretrieve(url, REC_PDB)

    # 2) Extract largest hetero ligand as autobox reference
    if not os.path.exists(REF_LIG_PDB):
        het_by_res = {}
        with open(REC_PDB) as f:
            for line in f:
                if not line.startswith("HETATM"): continue
                resn = line[17:20].strip()
                if resn in ("HOH", "WAT", "NA", "K", "CL", "ZN", "MG"):  # skip waters/ions
                    continue
                het_by_res.setdefault(resn, []).append(line)
        if not het_by_res:
            raise RuntimeError("No hetero ligand found in 8ZYP for autoboxing.")
        resn_max = max(het_by_res, key=lambda k: len(het_by_res[k]))
        with open(REF_LIG_PDB, "w") as out:
            out.writelines(het_by_res[resn_max])

    # 3) Prepare receptor and ref ligand with Meeko CLIs
    if not os.path.exists(REC_PDBQT):
        _run(["mk_prepare_receptor.py", "-i", REC_PDB, "-o", REC_PDBQT, "-U", "waters"])
    if not os.path.exists(REF_LIG_PDBQT):
        _run(["mk_prepare_ligand.py", "-i", REF_LIG_PDB, "-o", REF_LIG_PDBQT])

def smiles_to_rdkit_3d(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    return mol

def rdkit_to_pdbqt_with_meeko(mol, out_path):
    from rdkit.Chem import SDWriter
    tmp_sdf = out_path + ".sdf"
    w = SDWriter(tmp_sdf); w.write(mol); w.close()
    _run(["mk_prepare_ligand.py", "-i", tmp_sdf, "-o", out_path])
    os.remove(tmp_sdf)

@app.route("/herg_dock", methods=["POST"])
def herg_dock():
    try:
        data = request.get_json(force=True) or {}
        smiles = (data.get("smiles") or "").strip()
        ex = str(data.get("exhaustiveness") or "8")
        modes = str(data.get("num_modes") or "5")

        if not smiles:
            return jsonify({"error": "No SMILES provided"}), 400

        ensure_receptor_prepared()

        mol = smiles_to_rdkit_3d(smiles)
        if mol is None:
            return jsonify({"error": "Could not build 3D for SMILES"}), 400

        with tempfile.TemporaryDirectory() as tmpd:
            lig_pdbqt = os.path.join(tmpd, "ligand.pdbqt")
            rdkit_to_pdbqt_with_meeko(mol, lig_pdbqt)

            out_sdf = os.path.join(tmpd, "out.sdf")
            log_txt = os.path.join(tmpd, "gnina.log")

            cmd = [
                "gnina",
                "--receptor", REC_PDBQT,
                "--ligand", lig_pdbqt,
                "--autobox_ligand", REF_LIG_PDBQT,
                "--exhaustiveness", ex,
                "--num_modes", modes,
                "--seed", "0",
                "--cpu",
                "--out", out_sdf,
                "--log", log_txt
            ]
            _run(cmd)

            from rdkit.Chem import SDMolSupplier
            suppl = SDMolSupplier(out_sdf, removeHs=False)
            poses = [m for m in suppl if m is not None]
            if not poses:
                return jsonify({"error": "Docking produced no poses"}), 500

            # Top pose props
            top = poses[0]
            props = top.GetPropsAsDict()
            cnn_score = float(props.get("CNNscore")) if "CNNscore" in props else None
            cnn_aff   = float(props.get("CNNaffinity")) if "CNNaffinity" in props else None
            vina_aff  = float(props.get("minimizedAffinity")) if "minimizedAffinity" in props else None

            # Return top pose SDF (base64). You can also gzip & return all poses if you want.
            from rdkit import Chem as RDc
            sdf_str = RDc.MolToMolBlock(top)
            sdf_b64 = base64.b64encode(sdf_str.encode("utf-8")).decode("ascii")

            return jsonify({
                "cnn_score": cnn_score,                 # 0..1 (higher = better)
                "cnn_affinity": cnn_aff,                # GNINA learned affinity
                "vina_affinity_kcal_mol": vina_aff,     # Vina kcal/mol
                "pose_sdf_b64": sdf_b64,
                "receptor": "hERG 8ZYP (prepared)",
                "autobox_ref": os.path.basename(REF_LIG_PDB),
                "modes_returned": len(poses)
            })

    except Exception as e:
        logger.exception("hERG docking error")
        return jsonify({"error": str(e)}), 500


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
