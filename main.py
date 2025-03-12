# api/predict_smiles.py
from http.server import BaseHTTPRequestHandler
import json
import joblib
import numpy as np
import warnings

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import RDLogger

# Disable RDKit warnings
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

# -----------------------
# Load your models here
# -----------------------
names = ["lip", "sol"]
loaded_models = {}
for name in names:
    loaded_models[name] = joblib.load(f"saved_models/{name}.pkl")

def mol_to_fp(smiles, radius=2, nBits=1024):
    """Convert SMILES to Morgan fingerprint as a numpy array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Return a zero fingerprint if invalid
        return np.zeros(nBits, dtype=int)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # parse JSON, do something, return JSON
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"hello": "world"}).encode())
