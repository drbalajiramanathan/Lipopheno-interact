import uvicorn
import joblib
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware  # <-- IMPORT FOR CORS

# --- 1. Define Constants and Feature Order ---

# This order MUST match the order used to train the .onnx model
FEATURE_COLUMNS = [
    # Genomic
    'LDLR_variant', 'APOB_variant', 'PCSK9_variant', 'LPA_variant', 'Polygenic_Score',
    # Proteomic
    'hs-CRP', 'Lp-PLA2', 'IL-6', 'MPO',
    # Lipidomic
    'LDL-C', 'Lp(a)', 'ApoB', 'TG', 'HDL-C'
]


# --- 2. Define Pydantic Input Model ---
# Maps incoming JSON keys (e.g., "hs-CRP") to valid Python names (e.g., hs_CRP)

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
app = FastAPI(title="LipoPheno-Interact API")

# --- ADD THE CORS MIDDLEWARE BLOCK ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],
)
# ------------------------------------

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
    DEGREE_FEATURES = None
    BETWEENNESS_FEATURES = None


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

    # --- FIX ---
    # The scaler.pkl expects 9 features (Proteomic + Lipidomic), not 14.

    # Define the 9 features that must be scaled
    SCALER_COLUMNS = [
        'hs-CRP', 'Lp-PLA2', 'IL-6', 'MPO',
        'LDL-C', 'Lp(a)', 'ApoB', 'TG', 'HDL-C'
    ]

    # Define the 5 features that are NOT scaled (Genomic)
    UNSCALED_COLUMNS = [
        'LDLR_variant', 'APOB_variant', 'PCSK9_variant', 'LPA_variant', 'Polygenic_Score'
    ]

    # 1. Create arrays from Pydantic model
    patient_data_dict = patient_data.dict(by_alias=True)

    try:
        # Create 1D array of the 9 features to be scaled
        values_to_scale = np.array([
            patient_data_dict[col] for col in SCALER_COLUMNS
        ])

        # Create 1D array of the 5 unscaled features
        unscaled_values = np.array([
            patient_data_dict[col] for col in UNSCALED_COLUMNS
        ]).astype(np.float32)

    except KeyError as e:
        return {"error": f"Missing expected feature: {e}."}, 500

    # 2. Scale the 9 features
    scaled_values = scaler.transform(values_to_scale.reshape(1, -1)).astype(np.float32)

    # 3. Re-combine to create the 14-feature "patient-specific" vector
    # We must match the original 14 'FEATURE_COLUMNS' order.

    scaled_dict = dict(zip(SCALER_COLUMNS, scaled_values.flatten()))
    unscaled_dict = dict(zip(UNSCALED_COLUMNS, unscaled_values.flatten()))

    processed_dict = {**unscaled_dict, **scaled_dict}

    # Rebuild the 14-feature vector in the *exact* original order
    final_14_features = np.array([
        processed_dict[col] for col in FEATURE_COLUMNS
    ]).astype(np.float32).reshape(1, -1)

    # 4. Create the final 42-feature "Graph-Informed" vector
    full_feature_vector = np.concatenate([
        final_14_features,
        DEGREE_FEATURES,
        BETWEENNESS_FEATURES
    ], axis=1)

    # 5. Run ONNX Inference
    ort_inputs = {INPUT_NAME: full_feature_vector}
    ort_outs = ort_session.run(None, ort_inputs)

    pis_score = float(ort_outs[0][0][0])
    pis_score_100 = round(pis_score * 100, 2)

    return {"plaque_instability_score": pis_score_100}


# --- 5. Run the App (for local testing) ---

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)