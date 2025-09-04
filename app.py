# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import logging
from shutil import which
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import RDLogger

import subprocess, tempfile, base64

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
# Models (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
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
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits, dtype=int)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Pharmasee API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        smiles = data.get("smiles", "")
        if not smiles:
            return jsonify({"error": "No SMILES provided"}), 400

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return jsonify({"error": "Invalid SMILES format"}), 400

        predictions = {}
        for name, model in models.items():
            fp = mol_to_fp(smiles)
            predictions[name] = float(model.predict([fp])[0])

        return jsonify(predictions)
    except Exception as e:
        logger.exception("❌ Error processing prediction")
        return jsonify({"error": str(e)}), 500

# ──────────────────────────────────────────────────────────────────────────────
# hERG docking (Smina) — receptor PDBQT only, no prep
# ──────────────────────────────────────────────────────────────────────────────

# Find a receptor PDBQT file (try common names/locations)
def find_receptor_pdbqt():
    here = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.join(here, "9CHQ.pdbqt"),
        os.path.join(here, "9chq.pdbqt"),
        os.path.join(here, "herg_rec.pdbqt"),
        os.path.join(here, "herg_vina_assets", "9CHQ.pdbqt"),
        os.path.join(here, "herg_vina_assets", "9chq.pdbqt"),
        os.path.join(here, "herg_vina_assets", "herg_rec.pdbqt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

REC_PDBQT = find_receptor_pdbqt()

def _run(cmd, cwd=None):
    logger.info("RUN: %s", " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(cmd)}\n{res.stdout}")
    return res.stdout

def smiles_to_rdkit_3d_min(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    return mol

def rdkit_to_pdbqt_obabel(mol, out_path):
    # Write temp SDF, convert via obabel -> PDBQT (adds Gasteiger)
    from rdkit.Chem import SDWriter
    tmp_sdf = out_path + ".sdf"
    w = SDWriter(tmp_sdf); w.write(mol); w.close()
    _run([
        "obabel",
        "-isdf", tmp_sdf,
        "-opdbqt", "-O", out_path,
        "--partialcharge", "gasteiger"
    ])
    os.remove(tmp_sdf)

def compute_box_center_from_pdbqt(pdbqt_path):
    """
    Compute a coarse docking box center from Cα atoms in a receptor PDBQT.
    Falls back to all ATOM coords if no CA present.
    """
    xs, ys, zs = [], [], []
    with open(pdbqt_path, "r") as f:
        for ln in f:
            if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
                try:
                    x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
                    xs.append(x); ys.append(y); zs.append(z)
                except Exception:
                    continue
    if not xs:
        with open(pdbqt_path, "r") as f:
            for ln in f:
                if ln.startswith("ATOM"):
                    try:
                        x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
                        xs.append(x); ys.append(y); zs.append(z)
                    except Exception:
                        continue
    if not xs:
        raise RuntimeError("Could not compute receptor center (no ATOM records).")
    cx = sum(xs)/len(xs); cy = sum(ys)/len(ys); cz = sum(zs)/len(zs)
    return cx, cy, cz

# Default box size (Å). Center will be computed per-request from receptor unless the user overrides.
DEFAULT_SIZE = (28.0, 28.0, 28.0)

@app.route("/herg_vina", methods=["POST"])
def herg_vina():
    """
    Body JSON:
      {
        "smiles": "...",                     # required
        "exhaustiveness": 8,                 # optional (default 4)
        "num_modes": 9,                      # optional (default 5)
        "center": [cx,cy,cz], "size": [sx,sy,sz]  # optional: override fixed box
      }
    Returns JSON:
      {
        "vina_affinity_kcal_mol": float|null,
        "modes_returned": int,
        "poses_sdf_b64": "base64 SDF of all poses",
        "receptor": "<file.pdbqt>",
        "box_used": { "center":[...], "size":[...] }
      }
    """
    try:
        data = request.get_json(force=True) or {}
        smiles = (data.get("smiles") or "").strip()
        ex     = str(data.get("exhaustiveness") or "4")
        modes  = str(data.get("num_modes") or "5")
        user_center = data.get("center")
        user_size   = data.get("size")

        # Dependency & file checks
        missing = [exe for exe in ("smina", "obabel") if which(exe) is None]
        if missing:
            return jsonify({"error": f"Dependency missing: {', '.join(missing)} not found in PATH"}), 500

        if REC_PDBQT is None or not os.path.exists(REC_PDBQT):
            return jsonify({"error": "Receptor PDBQT not found. Place 9CHQ.pdbqt (or herg_rec.pdbqt) next to app.py or in ./herg_vina_assets/"}), 500

        if not smiles:
            return jsonify({"error": "No SMILES provided"}), 400

        # Build ligand 3D and convert to PDBQT
        lig3d = smiles_to_rdkit_3d_min(smiles)
        if lig3d is None:
            return jsonify({"error": "Failed to build 3D for SMILES"}), 400

        # Determine box
        if user_center and user_size:
            try:
                cx, cy, cz = map(float, user_center)
                sx, sy, sz = map(float, user_size)
            except Exception:
                return jsonify({"error": "Invalid center/size values; must be arrays of 3 numbers"}), 400
            box_used = {"center": [cx,cy,cz], "size": [sx,sy,sz]}
        else:
            cx, cy, cz = compute_box_center_from_pdbqt(REC_PDBQT)
            sx, sy, sz = DEFAULT_SIZE
            box_used = {"center": [cx,cy,cz], "size": [sx,sy,sz]}

        with tempfile.TemporaryDirectory() as td:
            lig_pdbqt = os.path.join(td, "ligand.pdbqt")
            rdkit_to_pdbqt_obabel(lig3d, lig_pdbqt)
            out_sdf   = os.path.join(td, "out.sdf")
            log_txt   = os.path.join(td, "smina.log")

            cmd = [
                "smina",
                "--receptor", REC_PDBQT,
                "--ligand", lig_pdbqt,
                "--exhaustiveness", ex,
                "--num_modes", modes,
                "--seed", "0",
                "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
                "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
                "--out", out_sdf,
                "--log", log_txt
            ]
            _run(cmd)

            from rdkit.Chem import SDMolSupplier, SDWriter
            suppl = SDMolSupplier(out_sdf, removeHs=False)
            poses = [m for m in suppl if m is not None]
            if not poses:
                log_tail = ""
                try:
                    with open(log_txt, "r") as lf:
                        log_tail = lf.read()[-2000:]
                except Exception:
                    pass
                return jsonify({"error": "Docking produced no poses",
                                "dock_log_tail": log_tail,
                                "box_used": box_used}), 500

            # Best (first) pose affinity
            top = poses[0]
            vina_aff = None
            for key in ("minimizedAffinity", "affinity"):
                try:
                    vina_aff = float(top.GetProp(key))
                    break
                except Exception:
                    pass

            # Bundle all poses to one SDF (base64)
            tmp_all = os.path.join(td, "all.sdf")
            w = SDWriter(tmp_all)
            for p in poses:
                w.write(p)
            w.close()
            with open(tmp_all, "r") as f:
                sdf_str = f.read()
            sdf_b64 = base64.b64encode(sdf_str.encode("utf-8")).decode("ascii")

            return jsonify({
                "vina_affinity_kcal_mol": vina_aff,
                "modes_returned": len(poses),
                "poses_sdf_b64": sdf_b64,
                "receptor": os.path.basename(REC_PDBQT),
                "box_used": box_used
            })

    except Exception as e:
        logger.exception("hERG smina error")
        return jsonify({"error": str(e)}), 500

# ──────────────────────────────────────────────────────────────────────────────
# Debug helpers (optional but handy)
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/routes", methods=["GET"])
def list_routes():
    return jsonify(sorted([str(r) for r in app.url_map.iter_rules()]))

@app.route("/herg_debug", methods=["GET"])
def herg_debug():
    deps = {
        "smina": which("smina") or None,
        "obabel": which("obabel") or None,
        "receptor_pdbqt": REC_PDBQT,
        "receptor_exists": bool(REC_PDBQT and os.path.exists(REC_PDBQT)),
    }
    return jsonify(deps)

# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
