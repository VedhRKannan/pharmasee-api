from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import logging
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import RDLogger
from flask_cors import CORS

CORS(app, origins="*")

# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# ✅ Create the Flask app
app = Flask(__name__)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)


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

# ✅ Required for local testing
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
