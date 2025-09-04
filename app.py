# app.py
from flask import Flask, request, jsonify
import os, logging, joblib, numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen
from rdkit import DataStructs, RDLogger

# ──────────────────────────────────────────────────────────────────────────────
# Logging / RDKit
# ──────────────────────────────────────────────────────────────────────────────
RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Flask app
# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Load ML models for /predict
# ──────────────────────────────────────────────────────────────────────────────
models = None
model_names = ["lipophilicity (logD)", "solubility (logS)"]

def get_models():
    global models
    if models is not None:
        return models
    loaded = {}
    for name in model_names:
        path = os.path.join("saved_models", f"{name}.pkl")
        loaded[name] = joblib.load(path)
    logging.getLogger(__name__).info("✅ Models loaded (lazy).")
    models = loaded
    return models

def mol_to_fp(smiles, radius=2, nBits=1024):
    """Morgan fingerprint as numpy array for sklearn models."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits, dtype=int)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Pharmasee API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True) or {}
        smiles = (data.get("smiles") or "").strip()
        if not smiles:
            return jsonify({"error": "No SMILES provided"}), 400
        if Chem.MolFromSmiles(smiles) is None:
            return jsonify({"error": "Invalid SMILES format"}), 400

        mdl = get_models()  # <-- load once on first call

        fp = mol_to_fp(smiles)
        predictions = { name: float(model.predict([fp])[0]) for name, model in mdl.items() }
        return jsonify(predictions)
    except Exception as e:
        logger.exception("❌ Error processing prediction")
        return jsonify({"error": str(e)}), 500



# ──────────────────────────────────────────────────────────────────────────────
# RDKit-only hERG demo scorer (no external tools)
# ──────────────────────────────────────────────────────────────────────────────
HERG_BLOCKERS = [
    ("Dofetilide",  "CN(CCC1=CC=C(C=C1)NS(=O)(=O)C)CCOC2=CC=C(C=C2)NS(=O)(=O)C"),
    ("Terfenadine", "CC(C)(C)C1=CC=C(C=C1)C(CCCN2CCC(CC2)C(C3=CC=CC=C3)(C4=CC=CC=C4)O)O"),
    ("Astemizole",  "COc1ccc(CCN2CCC(CC2)Nc1nc2c(n1Cc1ccc(cc1)F)cccc2)cc1"),
    ("Cisapride",   "Clc1cc(c(OC)cc1N)C(=O)NC3CCN(CCCOc2ccc(F)cc2)CC3OC"),
]

def fp_morgan(smiles, radius=2, nBits=2048):
    m = Chem.MolFromSmiles(smiles)
    if m is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nBits)

@app.route("/herg_demo", methods=["POST"])
def herg_demo():
    """
    Body: { "smiles": "..." }
    Returns:
      {
        "ok": true,
        "logP": float,
        "best_ref": {"name": str, "smiles": str, "tanimoto": float},
        "similarities": [ {"name": str, "tanimoto": float}, ... ],
        "affinity_demo_kcal_mol": float   # purely demo: -5 - 4*best_sim
      }
    """
    try:
        data = request.get_json(force=True) or {}
        smiles = (data.get("smiles") or "").strip()
        if not smiles:
            return jsonify({"error": "No SMILES provided"}), 400

        qfp = fp_morgan(smiles)
        if qfp is None:
            return jsonify({"error": "Invalid SMILES"}), 400

        sims = []
        best = {"name": None, "smiles": None, "tanimoto": 0.0}
        for name, s in HERG_BLOCKERS:
            bfp = fp_morgan(s)
            if bfp is None:
                continue
            tan = DataStructs.TanimotoSimilarity(qfp, bfp)
            sims.append({"name": name, "tanimoto": tan})
            if tan > best["tanimoto"]:
                best = {"name": name, "smiles": s, "tanimoto": tan}

        mol = Chem.MolFromSmiles(smiles)
        logP = float(Crippen.MolLogP(mol))
        affinity_demo = float(-5.0 - 4.0 * best["tanimoto"])  # demo-only number

        return jsonify({
            "ok": True,
            "logP": logP,
            "best_ref": best,
            "similarities": sims,
            "affinity_demo_kcal_mol": affinity_demo
        })
    except Exception as e:
        logger.exception("herg_demo error")
        return jsonify({"error": str(e)}), 500

# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
