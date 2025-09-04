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

# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Create the Flask app
app = Flask(__name__)

# ===================== ML MODELS (unchanged) =====================
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

# ===================== hERG DOCKING (SMINA/Vina) =====================
import subprocess, tempfile, base64, urllib.request

DOCK_VINA_ROOT = os.path.abspath("./herg_vina_assets")
os.makedirs(DOCK_VINA_ROOT, exist_ok=True)
REC_PDB       = os.path.join(DOCK_VINA_ROOT, "8ZYP.pdb")
REC_PDBQT     = os.path.join(DOCK_VINA_ROOT, "herg_rec.pdbqt")
REF_LIG_PDB   = os.path.join(DOCK_VINA_ROOT, "ref_ligand.pdb")
REF_LIG_PDBQT = os.path.join(DOCK_VINA_ROOT, "ref_ligand.pdbqt")

def _run(cmd, cwd=None):
    logger.info("RUN: %s", " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(cmd)}\n{res.stdout}")
    return res.stdout

def _download_file(url, dest):
    # robust download with timeout
    logger.info("Downloading %s -> %s", url, dest)
    with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as out:
        out.write(resp.read())

def ensure_receptor_prepared_vina():
    # 1) Download ligand-bound hERG (8ZYP) once
    if not os.path.exists(REC_PDB):
        _download_file("https://files.rcsb.org/download/8ZYP.pdb", REC_PDB)

    # 2) Extract largest non-water HET as autobox reference
    if not os.path.exists(REF_LIG_PDB):
        het_by_res = {}
        with open(REC_PDB) as f:
            for ln in f:
                if not ln.startswith("HETATM"):
                    continue
                resn = ln[17:20].strip()
                if resn in ("HOH", "WAT", "NA", "K", "CL", "ZN", "MG"):
                    continue
                het_by_res.setdefault(resn, []).append(ln)
        if not het_by_res:
            raise RuntimeError("No hetero ligand found in 8ZYP for autoboxing.")
        resn_max = max(het_by_res, key=lambda k: len(het_by_res[k]))
        with open(REF_LIG_PDB, "w") as out:
            out.writelines(het_by_res[resn_max])
            out.write("\nEND\n")

    # 3) Prepare receptor & ref ligand to PDBQT (Meeko CLIs)
    if not os.path.exists(REC_PDBQT):
        _run(["mk_prepare_receptor.py", "-i", REC_PDB, "-o", REC_PDBQT, "-U", "waters"])
    if not os.path.exists(REF_LIG_PDBQT):
        _run(["mk_prepare_ligand.py", "-i", REF_LIG_PDB, "-o", REF_LIG_PDBQT])

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

def rdkit_to_pdbqt_meeko(mol, out_path):
    from rdkit.Chem import SDWriter
    tmp_sdf = out_path + ".sdf"
    w = SDWriter(tmp_sdf); w.write(mol); w.close()
    _run(["mk_prepare_ligand.py", "-i", tmp_sdf, "-o", out_path])
    os.remove(tmp_sdf)

@app.route("/herg_vina", methods=["POST"])
def herg_vina():
    """
    Body: { "smiles": "...", "exhaustiveness": 8, "num_modes": 9 }
    Returns: { vina_affinity_kcal_mol, poses_sdf_b64, modes_returned, receptor, autobox_ref }
    """
    try:
        data = request.get_json(force=True) or {}
        smiles = (data.get("smiles") or "").strip()
        ex     = str(data.get("exhaustiveness") or "4")  # modest defaults while testing
        modes  = str(data.get("num_modes") or "5")
        if not smiles:
            return jsonify({"error": "No SMILES provided"}), 400

        # 🔎 Clear dependency check with explicit error (so you never get an empty response)
        missing = [exe for exe in ("smina", "mk_prepare_ligand.py", "mk_prepare_receptor.py") if which(exe) is None]
        if missing:
            return jsonify({"error": f"Dependency missing: {', '.join(missing)} not found in PATH"}), 500

        ensure_receptor_prepared_vina()

        lig3d = smiles_to_rdkit_3d_min(smiles)
        if lig3d is None:
            return jsonify({"error": "Failed to build 3D for SMILES"}), 400

        with tempfile.TemporaryDirectory() as td:
            lig_pdbqt = os.path.join(td, "ligand.pdbqt")
            rdkit_to_pdbqt_meeko(lig3d, lig_pdbqt)
            out_sdf   = os.path.join(td, "out.sdf")
            log_txt   = os.path.join(td, "smina.log")

            # Run smina with autobox around co-crystal ligand
            cmd = [
                "smina",
                "--receptor", REC_PDBQT,
                "--ligand", lig_pdbqt,
                "--autobox_ligand", REF_LIG_PDBQT,
                "--exhaustiveness", ex,
                "--num_modes", modes,
                "--seed", "0",
                "--out", out_sdf,
                "--log", log_txt
            ]
            _run(cmd)

            # Read poses
            from rdkit.Chem import SDMolSupplier, SDWriter
            suppl = SDMolSupplier(out_sdf, removeHs=False)
            poses = [m for m in suppl if m is not None]
            if not poses:
                # include smina log in error for easier debugging
                log_tail = ""
                try:
                    with open(log_txt, "r") as lf:
                        log_tail = lf.read()[-2000:]
                except Exception:
                    pass
                return jsonify({"error": "Docking produced no poses", "dock_log_tail": log_tail}), 500

            # Top pose affinity
            top = poses[0]
            vina_aff = None
            for key in ("minimizedAffinity", "affinity"):
                try:
                    vina_aff = float(top.GetProp(key))
                    break
                except Exception:
                    pass

            # Bundle all poses into a single SDF (base64)
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
                "receptor": "hERG 8ZYP (prepared via Meeko)",
                "autobox_ref": os.path.basename(REF_LIG_PDB)
            })

    except Exception as e:
        logger.exception("hERG smina error")
        # On any error, return structured JSON (so your Next proxy/browser never sees an empty body)
        return jsonify({"error": str(e)}), 500

# Optional: quick dependency probe
@app.route("/herg_debug", methods=["GET"])
def herg_debug():
    deps = {
        "smina": which("smina") or None,
        "mk_prepare_ligand.py": which("mk_prepare_ligand.py") or None,
        "mk_prepare_receptor.py": which("mk_prepare_receptor.py") or None,
        "rec_exists": os.path.exists(REC_PDB),
        "rec_pdbqt_exists": os.path.exists(REC_PDBQT),
        "ref_lig_exists": os.path.exists(REF_LIG_PDB),
        "ref_lig_pdbqt_exists": os.path.exists(REF_LIG_PDBQT),
    }
    return jsonify(deps)

# ===============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
