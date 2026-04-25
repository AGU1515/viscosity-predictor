import streamlit as st
import pandas as pd
import numpy as np
import joblib

from rdkit import Chem
from rdkit.Chem import Descriptors, Draw


st.set_page_config(
    page_title="Molecular Viscosity Predictor",
    layout="centered"
)


@st.cache_resource
def load_model():
    saved = joblib.load("viscosity_xgboost_model.joblib")
    return saved["model"], saved["descriptor_names"], saved["feature_medians"]


model, descriptor_names, feature_medians = load_model()


def generate_features(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None, None, "Invalid SMILES string."

    if mol.GetNumAtoms() == 0:
        return None, None, "SMILES string does not contain a valid molecule."

    features = []

    for name, function in Descriptors._descList:
        try:
            value = function(mol)
        except Exception:
            value = np.nan

        features.append(value)

    X_user = pd.DataFrame([features], columns=descriptor_names)
    X_user = X_user.replace([np.inf, -np.inf], np.nan)
    X_user = X_user.fillna(feature_medians)

    return X_user, mol, None


st.title("Molecular Viscosity Predictor")

st.write(
    "Enter a valid SMILES string to predict molecular viscosity at 25 °C. "
    "The model predicts log η using RDKit molecular descriptors and XGBoost."
)

user_smiles = st.text_input(
    "SMILES string",
    placeholder="Example: CCO"
)

if st.button("Predict viscosity"):
    if not user_smiles.strip():
        st.warning("Please enter a SMILES string.")

    else:
        X_user, mol, error_message = generate_features(user_smiles.strip())

        if X_user is None:
            st.error(error_message)

        else:
            prediction = model.predict(X_user)[0]

            st.success(
                f"The predicted viscosity of your molecule at 25 °C is **{prediction:.2f} log η units**."
            )

            st.subheader("Molecular structure")
            image = Draw.MolToImage(mol, size=(350, 350))
            st.image(image)
