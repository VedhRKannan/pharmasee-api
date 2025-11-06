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

def smith_waterman_score(a: str, b: str, match: int = 2, mismatch: int = -1, gap: int = -2):
    """
    Smith–Waterman local alignment on raw strings (e.g., SMILES).
    Returns (raw_score, normalized_score_in_[0,1]).
    Normalization: raw_score / (match * min(len(a), len(b))) clipped to [0,1].
    """
    if not a or not b:
        return 0.0, 0.0

    la, lb = len(a), len(b)
    # DP matrix (la+1) x (lb+1)
    H = [[0] * (lb + 1) for _ in range(la + 1)]
    best = 0

    for i in range(1, la + 1):
        ai = a[i - 1]
        for j in range(1, lb + 1):
            bj = b[j - 1]
            s = match if ai == bj else mismatch
            # Smith–Waterman recurrence
            diag = H[i - 1][j - 1] + s
            up   = H[i - 1][j] + gap
            left = H[i][j - 1] + gap
            H[i][j] = max(0, diag, up, left)
            if H[i][j] > best:
                best = H[i][j]

    # Normalize to [0,1] by the max possible all-match local segment (shorter length)
    denom = match * min(la, lb)
    norm = float(best) / float(denom) if denom > 0 else 0.0
    if norm < 0: norm = 0.0
    if norm > 1: norm = 1.0
    return float(best), norm

@app.route("/herg_demo", methods=["POST"])
def herg_demo():
    """
    Body: { "smiles": "..." }
    Returns:
      {
        "ok": true,
        "logP": float,
        "best_ref": {"name": str, "smiles": str, "sw_raw": float, "sw_norm": float},
        "similarities": [ {"name": str, "sw_raw": float, "sw_norm": float}, ... ],
        "affinity_demo_kcal_mol": float   # demo: -5 - 4*best_sw_norm
      }
    """
    try:
        data = request.get_json(force=True) or {}
        smiles = (data.get("smiles") or "").strip()
        if not smiles:
            return jsonify({"error": "No SMILES provided"}), 400

        # Validate ligand SMILES early
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return jsonify({"error": "Invalid SMILES"}), 400

        # Compute Smith–Waterman similarity to each reference SMILES (HERG_BLOCKERS)
        sims = []
        best = {"name": None, "smiles": None, "sw_raw": 0.0, "sw_norm": 0.0}
        for name, s in HERG_BLOCKERS:
            if not s:
                continue
            sw_raw, sw_norm = smith_waterman_score(smiles, s, match=2, mismatch=-1, gap=-2)
            sims.append({"name": name, "sw_raw": sw_raw, "sw_norm": sw_norm})
            if sw_norm > best["sw_norm"]:
                best = {"name": name, "smiles": s, "sw_raw": sw_raw, "sw_norm": sw_norm}

        # Context metric (unchanged)
        logP = float(Crippen.MolLogP(mol))

        # Demo affinity: use SW normalized similarity instead of Tanimoto
        affinity_demo = float(-5.0 - 4.0 * best["sw_norm"])

        return jsonify({
            "ok": True,
            "logP": logP,
            "best_ref": best,
            "similarities": sims,
            "affinity_demo_kcal_mol": affinity
        })
    except Exception as e:
        logger.exception("herg_demo error")
        return jsonify({"error": str(e)}), 500
# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
