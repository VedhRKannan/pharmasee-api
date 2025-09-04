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

# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Flask app
# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# ML models (your existing predict)
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

# ──────────────────────────────────────────────────────────────────────────────
# hERG docking via SMINA (Vina) using PDB 9CHQ for autobox (with robust fallbacks)
# ──────────────────────────────────────────────────────────────────────────────
import subprocess, tempfile, base64, urllib.request

DOCK_VINA_ROOT = os.path.abspath("./herg_vina_assets")
os.makedirs(DOCK_VINA_ROOT, exist_ok=True)

# Use 9CHQ as primary receptor source (co-crystal ligand present)
REC_PDB       = os.path.join(DOCK_VINA_ROOT, "9CHQ.pdb")     # we write any downloaded legacy/ent as this file name
REC_PDBQT     = os.path.join(DOCK_VINA_ROOT, "herg_rec.pdbqt")
REF_LIG_PDB   = os.path.join(DOCK_VINA_ROOT, "ref_ligand.pdb")
REF_LIG_PDBQT = os.path.join(DOCK_VINA_ROOT, "ref_ligand.pdbqt")

def _run(cmd, cwd=None):
    logger.info("RUN: %s", " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(cmd)}\n{res.stdout}")
    return res.stdout

def _try_download(urls, dest):
    last_err = None
    for url in urls:
        try:
            logger.info("Downloading %s -> %s", url, dest)
            with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as out:
                out.write(resp.read())
            return True
        except Exception as e:
            last_err = e
            logger.warning("Download failed from %s: %s", url, e)
    if last_err:
        raise last_err
    return False

def ensure_receptor_prepared_vina(ligand_resn=None):
    """
    Ensures a 9CHQ receptor is present & prepared.
    - Downloads 9CHQ (tries RCSB .pdb then PDBe .ent); if all fail, falls back to 5VA1.
    - Extracts a co-crystal ligand (largest HET group by default, or specific RESN if provided).
    - Prepares receptor and ligand to PDBQT via Meeko.
    Returns dict with:
      { "have_ref_ligand": bool, "receptor_file": REC_PDB, "ligand_resn": <str|None> }
    """
    # 1) Download receptor once (prefer 9CHQ)
    if not os.path.exists(REC_PDB):
        try:
            _try_download([
                "https://files.rcsb.org/download/9CHQ.pdb",
                "https://www.ebi.ac.uk/pdbe/entry-files/download/pdb9chq.ent",
                # fallback to older hERG (apo) only if needed
                "https://files.rcsb.org/download/5VA1.pdb",
                "https://www.ebi.ac.uk/pdbe/entry-files/download/pdb5va1.ent",
            ], REC_PDB)
        except Exception as e:
            raise RuntimeError(f"Failed to download receptor (9CHQ/5VA1): {e}")

    # 2) Try to extract a reference ligand (HET group) for autoboxing
    have_ref = False
    chosen_resn = None
    if not os.path.exists(REF_LIG_PDB):
        het_by_key = {}  # key = (resn, resid)
        with open(REC_PDB) as f:
            for ln in f:
                if not ln.startswith("HETATM"):
                    continue
                resn = ln[17:20].strip()
                if resn in ("HOH","WAT","NA","K","CL","ZN","MG"):
                    continue
                resid = ln[22:26].strip()
                het_by_key.setdefault((resn, resid), []).append(ln)

        if het_by_key:
            if ligand_resn:
                candidates = [(k, v) for (k, v) in het_by_key.items() if k[0].upper() == str(ligand_resn).upper()]
                if candidates:
                    sel_key, lines = max(candidates, key=lambda kv: len(kv[1]))
                    chosen_resn = sel_key[0]
                else:
                    sel_key, lines = max(het_by_key.items(), key=lambda kv: len(kv[1]))
                    chosen_resn = sel_key[0]
            else:
                sel_key, lines = max(het_by_key.items(), key=lambda kv: len(kv[1]))
                chosen_resn = sel_key[0]

            with open(REF_LIG_PDB, "w") as out:
                out.writelines(lines)
                out.write("\nEND\n")
            have_ref = True

    # 3) Prepare receptor & (if present) ref ligand to PDBQT
    if not os.path.exists(REC_PDBQT):
        _run([
    "mk_prepare_receptor.py",
    "--read_pdb", REC_PDB,          # read PDB/ENT directly
    "-p", REC_PDBQT,                # write receptor PDBQT to this file
    "--delete_residues", "HOH,WAT", # strip waters
    "-a"                            # allow missing bits
])
    if have_ref and not os.path.exists(REF_LIG_PDBQT):
        _run(["mk_prepare_ligand.py", "-i", REF_LIG_PDB, "-o", REF_LIG_PDBQT])

    return {"have_ref_ligand": have_ref, "receptor_file": REC_PDB, "ligand_resn": chosen_resn}

def compute_box_center_from_receptor(pdb_path):
    """
    Compute a rough docking box center from protein Cα atoms (fallback when no ligand).
    Returns (cx, cy, cz) as floats.
    """
    xs, ys, zs = [], [], []
    with open(pdb_path) as f:
        for ln in f:
            if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
                try:
                    x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
                    xs.append(x); ys.append(y); zs.append(z)
                except Exception:
                    continue
    if not xs:
        with open(pdb_path) as f:
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
    Body JSON:
      {
        "smiles": "...",                     # required
        "exhaustiveness": 8,                 # optional (default 4 for speed)
        "num_modes": 9,                      # optional (default 5)
        "ligand_resn": "XXX",                # optional: choose specific HET RESN from 9CHQ to autobox on
        "center": [cx,cy,cz], "size": [sx,sy,sz]  # optional: override fixed box
      }
    Returns JSON:
      {
        "vina_affinity_kcal_mol": float|null,
        "modes_returned": int,
        "poses_sdf_b64": "base64 SDF of all poses",
        "receptor": "9CHQ.pdb",
        "box_used": { "mode": "autobox_ligand" | "auto_center" | "fixed", ... }
      }
    """
    try:
        data = request.get_json(force=True) or {}
        smiles = (data.get("smiles") or "").strip()
        ex     = str(data.get("exhaustiveness") or "4")  # modest defaults during testing
        modes  = str(data.get("num_modes") or "5")
        user_center = data.get("center")
        user_size   = data.get("size")
        ligand_resn = data.get("ligand_resn")  # e.g., "E41", "LIG", etc.

        if not smiles:
            return jsonify({"error": "No SMILES provided"}), 400

        # Dependency check (clear JSON if missing)
        missing = [exe for exe in ("smina", "mk_prepare_ligand.py", "mk_prepare_receptor.py")
                   if which(exe) is None]
        if missing:
            return jsonify({"error": f"Dependency missing: {', '.join(missing)} not found in PATH"}), 500

        prep = ensure_receptor_prepared_vina(ligand_resn=ligand_resn)
        have_ref = bool(prep.get("have_ref_ligand"))

        lig3d = smiles_to_rdkit_3d_min(smiles)
        if lig3d is None:
            return jsonify({"error": "Failed to build 3D for SMILES"}), 400

        # Decide autobox vs fixed box
        box_used = {}
        use_autobox = have_ref and not (user_center or user_size)

        if user_center and user_size:
            try:
                cx, cy, cz = map(float, user_center)
                sx, sy, sz = map(float, user_size)
                box_used = {"mode": "fixed", "center": [cx, cy, cz], "size": [sx, sy, sz]}
                use_autobox = False
            except Exception:
                return jsonify({"error": "Invalid center/size values; must be arrays of 3 numbers"}), 400

        if not use_autobox and not box_used:
            # No ref ligand available → compute a coarse center and use generous size
            cx, cy, cz = compute_box_center_from_receptor(REC_PDB)
            sx, sy, sz = 28.0, 28.0, 28.0
            box_used = {"mode": "auto_center", "center": [cx, cy, cz], "size": [sx, sy, sz]}

        with tempfile.TemporaryDirectory() as td:
            lig_pdbqt = os.path.join(td, "ligand.pdbqt")
            rdkit_to_pdbqt_meeko(lig3d, lig_pdbqt)
            out_sdf   = os.path.join(td, "out.sdf")
            log_txt   = os.path.join(td, "smina.log")

            cmd = [
                "smina",
                "--receptor", REC_PDBQT,
                "--ligand", lig_pdbqt,
                "--exhaustiveness", ex,
                "--num_modes", modes,
                "--seed", "0",
                "--out", out_sdf,
                "--log", log_txt
            ]
            if use_autobox:
                cmd.extend(["--autobox_ligand", REF_LIG_PDBQT])
                box_used = {"mode": "autobox_ligand", "ligand": os.path.basename(REF_LIG_PDB)}
            else:
                cx, cy, cz = box_used["center"]
                sx, sy, sz = box_used["size"]
                cmd.extend([
                    "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
                    "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz)
                ])

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
                "receptor": os.path.basename(REC_PDB),
                "box_used": box_used
            })

    except Exception as e:
        logger.exception("hERG smina error")
        return jsonify({"error": str(e)}), 500

# ──────────────────────────────────────────────────────────────────────────────
# Helpful debug endpoints (keep during bring-up)
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/routes", methods=["GET"])
def list_routes():
    return jsonify(sorted([str(r) for r in app.url_map.iter_rules()]))

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

# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # For Railway; change port if your platform injects $PORT
    app.run(host="0.0.0.0", port=8080)
