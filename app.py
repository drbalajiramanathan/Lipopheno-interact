import uvicorn
import joblib
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel, Field

# --- 1. Define Constants and Feature Order ---

# This order MUST match the order used to train the scaler.pkl
# These are the *string* names of the features.
FEATURE_COLUMNS = [
    # Genomic
    'LDLR_variant', 'APOB_variant', 'PCSK9_variant', 'LPA_variant', 'Polygenic_Score',
    # Proteomic
    'hs-CRP', 'Lp-PLA2', 'IL-6', 'MPO',
    # Lipidomic
    'LDL-C', 'Lp(a)', 'ApoB', 'TG', 'HDL-C'
]


# --- 2. Define Pydantic Input Model ---
# This ensures the API receives all 14 features.
# We use valid Python names (hs_CRP) and map them to the
# incoming JSON keys ("hs-CRP") using Field(alias=...).

class PatientInput(BaseModel):
    LDLR_variant: float
    APOB_variant: float
    PCSK9_variant: float
    LPA_variant: float
    Polygenic_Score: float
    hs_CRP: float = Field(alias="hs-CRP")
    Lp_PLA2: float = Field(alias="Lp-PLA2")
    IL_6: float = Field(alias="IL-6")
    MPO: float
    LDL_C: float = Field(alias="LDL-C")
    Lp_a: float = Field(alias="Lp(a)")
    ApoB: float
    TG: float
    HDL_C: float = Field(alias="HDL-C")


# --- 3. Load Artifacts at Startup ---
# ... (rest of the file is the same as before) ...
app = FastAPI(title="LipoPheno-Interact API")

try:
    scaler = joblib.load("scaler.pkl")
    graph_data = np.load("graph_features.npz")

    # Extract static graph features
    DEGREE_FEATURES = graph_data['degree'].astype(np.float32).reshape(1, -1)
    BETWEENNESS_FEATURES = graph_data['betweenness'].astype(np.float32).reshape(1, -1)

    # Initialize ONNX inference session
    ort_session = ort.InferenceSession("model.onnx")
    INPUT_NAME = ort_session.get_inputs()[0].name

    print("INFO: All model artifacts loaded successfully.")

except FileNotFoundError as e:
    print(f"ERROR: Missing artifact. {e}")
    scaler = None
    ort_session = None


# --- 4. Define API Endpoints ---

@app.get("/")
def root():
    return {"message": "LipoPheno-Interact API is running."}


@app.post("/predict")
def predict_pis(patient_data: PatientInput):
    """
    Predicts the Plaque Instability Score (PIS) from 14 patient features.
    """
    if not ort_session or not scaler:
        return {"error": "Model artifacts not loaded."}, 500

    # 1. Create 1D array from Pydantic model in the correct feature order
    # We call .dict(by_alias=True) to get a dictionary with the
    # original string names (e.g., "hs-CRP") as keys.
    patient_data_dict = patient_data.dict(by_alias=True)

    try:
        patient_values = np.array([
            patient_data_dict[col] for col in FEATURE_COLUMNS
        ])
    except KeyError as e:
        return {"error": f"Missing expected feature: {e}. Check FEATURE_COLUMNS list."}, 500

    # 2. Scale the 14 patient-specific features
    scaled_patient_values = scaler.transform(patient_values.reshape(1, -1)).astype(np.float32)

    # 3. Create the 42-feature "Graph-Informed" vector
    full_feature_vector = np.concatenate([
        scaled_patient_values,
        DEGREE_FEATURES,
        BETWEENNESS_FEATURES
    ], axis=1)

    # 4. Run ONNX Inference
    ort_inputs = {INPUT_NAME: full_feature_vector}
    ort_outs = ort_session.run(None, ort_inputs)

    pis_score = float(ort_outs[0][0][0])
    pis_score_100 = round(pis_score * 100, 2)

    return {"plaque_instability_score": pis_score_100}


# --- 5. Run the App (for local testing) ---

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)