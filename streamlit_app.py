import streamlit as st
import pandas as pd
import numpy as np
import joblib
from Bio import Entrez, SeqIO
import torch
from transformers import BertTokenizer, BertModel, BioGptTokenizer, BioGptModel, BigBirdTokenizer, BigBirdModel

# Solvent mapping
solvent_mapping = {
    'Methanol': 1.0, 'Water': 2.0, 'Ethanol': 3.0, 'n-Hexane': 4.0, 'Cloroform': 5.0,
    'Acetone': 6.0, 'Acetonitrile': 7.0, 'Ethyl Acetate': 8.0, 'Methylene Chloride': 20.0,
    'n-Butanol': 10.0, 'None': 0.0
}

# Used part components
main_components_1 = ['Root', 'Stem', 'Leaf', 'Flower', 'Fruit', 'Seed', 'Fruit skin', 'Bark', 'Branch', "Whole plant", "Aerial parts"]
main_components = ['Root', 'Stem', 'Leaf', 'Flower', 'Fruit', 'Seed', 'Fruit skin', 'Bark', 'Branch']

# Set your email (required for NCBI access)
Entrez.email = "phuongnv@hup.edu.vn"

# Function to fetch ITS2 from NCBI
def get_ITS2_sequence(organism_name):
    try:
        query = f'"{organism_name}"[Organism] AND "ITS2"[All Fields]'
        handle = Entrez.esearch(db="nucleotide", term=query, retmax=5)
        record = Entrez.read(handle)
        handle.close()
        
        if record["IdList"]:
            sequence_id = record["IdList"][0]
            print(f"Found sequence ID: {sequence_id}")
            handle = Entrez.efetch(db="nucleotide", id=sequence_id, rettype="gb", retmode="text")
            seq_record = SeqIO.read(handle, "genbank")
            handle.close()
            
            print(f"Organism: {seq_record.annotations['organism']}")
            print(f"Definition: {seq_record.description}")
            print(f"Sequence Length: {len(seq_record.seq)}")
            
            for feature in seq_record.features:
                if feature.type == "misc_RNA" and "ITS2" in feature.qualifiers.get("note", [""])[0]:
                    its2_seq = feature.extract(seq_record.seq)
                    print(f"Extracted ITS2 sequence: {its2_seq}")
                    return str(its2_seq)
            return str(seq_record.seq)
        raise ValueError(f"No ITS2 sequence linked to gene record for {organism_name}")
    except Exception as e:
        raise ValueError(f"Error retrieving ITS2 from NCBI: {str(e)}")

# Function to encode solvents
def encode_solvent(solvent1, solvent2, ratio1, ratio2):
    total = ratio1 + ratio2
    if total == 0:
        return 0.0
    percent1 = ratio1 / total
    percent2 = ratio2 / total
    val1 = solvent_mapping.get(solvent1, 0.0) * percent1
    val2 = solvent_mapping.get(solvent2, 0.0) * percent2 if solvent2 != 'None' else 0.0
    return val1 + val2

# Function to encode used parts
def encode_used_part(part):
    encoded = [0] * len(main_components)
    if 'Whole plant' in part:
        return [1] * len(main_components)
    elif 'Aerial parts' in part:
        return [0 if comp == 'Root' else 1 for comp in main_components]
    else:
        for p in part:
            for i, comp in enumerate(main_components):
                if comp == p:
                    encoded[i] = 1
    return encoded

# Prediction function with lazy loading of transformer models
def predict(its2_sequence, solvent1, solvent2, ratio1, ratio2, used_parts):
    # Load pre-trained models and tokenizers
    tokenizer_dnabert = BertTokenizer.from_pretrained("zhihan1996/DNA_bert_6")
    model_dnabert = BertModel.from_pretrained("zhihan1996/DNA_bert_6")
    tokenizer_biogpt = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    model_biogpt = BioGptModel.from_pretrained("microsoft/biogpt")
    tokenizer_bigbird = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")
    model_bigbird = BigBirdModel.from_pretrained("google/bigbird-roberta-base")

    # Move models to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dnabert.to(device)
    model_biogpt.to(device)
    model_bigbird.to(device)

    # DNA-BERT encoding
    def encode_its2_dnabert(its2_sequence):
        def tokenize_sequence(sequence, k=6):
            k_mers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
            return ' '.join(k_mers)
        tokenized_sequence = tokenize_sequence(its2_sequence)
        inputs = tokenizer_dnabert(tokenized_sequence, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model_dnabert(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()

    # BioGPT encoding
    def encode_its2_biogpt(its2_sequence):
        inputs = tokenizer_biogpt(its2_sequence, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model_biogpt(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()

    # BigBird encoding
    def encode_its2_bigbird(its2_sequence):
        def tokenize_sequence(sequence, k=6):
            k_mers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
            return ' '.join(k_mers)
        tokenized_sequence = tokenize_sequence(its2_sequence)
        inputs = tokenizer_bigbird(tokenized_sequence, return_tensors='pt', truncation=True, padding=True, max_length=4096)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model_bigbird(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()

    # Encode inputs
    solvent_encoded = encode_solvent(solvent1, solvent2, ratio1, ratio2)
    used_part_encoded = encode_used_part(used_parts)
    its2_dnabert = encode_its2_dnabert(its2_sequence)
    its2_biogpt = encode_its2_biogpt(its2_sequence)
    its2_bigbird = encode_its2_bigbird(its2_sequence)

    # Reshape to 2D
    solvent_encoded_2d = np.array([[solvent_encoded]])
    used_part_encoded_2d = np.array([used_part_encoded])

    # Combine features
    X_dnabert = np.hstack((its2_dnabert, solvent_encoded_2d, used_part_encoded_2d))
    X_biogpt = np.hstack((its2_biogpt, solvent_encoded_2d, used_part_encoded_2d))
    X_bigbird = np.hstack((its2_bigbird, solvent_encoded_2d, used_part_encoded_2d))

    # Load XGBoost models inside the function
    model1 = joblib.load('dnabertXGB_model.pkl')
    model2 = joblib.load('bigbirdXGB_model.pkl')
    model3 = joblib.load('biogptXGB_model.pkl')
    final_model = joblib.load('stack_model.pkl')

    # Get intermediate predictions
    y1 = model1.predict(X_dnabert).reshape(-1, 1)
    y2 = model2.predict(X_bigbird).reshape(-1, 1)
    y3 = model3.predict(X_biogpt).reshape(-1, 1)
    X_final = np.concatenate((y1, y2, y3), axis=1)

    # Final prediction
    y_pred = final_model.predict(X_final)
    probabilities = final_model.predict_proba(X_final)
    return y_pred[0], probabilities[0]

# Streamlit UI
st.title("Screening of Xanthine Oxidase Inhibitory Plants")
st.write("Provide the required inputs below to predict xanthine oxidase inhibitory activity.")
st.write("We recommend using your ITS2 sequence directly instead of retrieving from NCBI for greater accuracy.")

# Section 1: ITS2 Sequence or Plant Name
st.header("1. ITS2 Sequence or Plant Name")
input_method = st.radio("Choose Input Method", ("ITS2 Sequence", "Plant Name"))
if input_method == "Plant Name":
    plant_name = st.text_input("Plant Name", placeholder="e.g., Angelica dahurica", key="plant_name")
    its2_sequence = None
else:
    its2_sequence = st.text_area("ITS2 Sequence", placeholder="e.g., atgc...", height=100, key="its2_sequence")
    plant_name = None

# Section 2: Solvent Selection
st.header("2. Solvent Selection")
solvent_options = list(solvent_mapping.keys())
col1, col2 = st.columns(2)
with col1:
    solvent1 = st.selectbox("Solvent 1", solvent_options, index=solvent_options.index('Methanol'), key="solvent1")
    ratio1 = st.number_input("Ratio of Solvent 1", min_value=0, max_value=100, value=100, key="ratio1")
with col2:
    solvent2 = st.selectbox("Solvent 2", solvent_options, index=solvent_options.index('None'), key="solvent2")
    ratio2 = st.number_input("Ratio of Solvent 2", min_value=0, max_value=100, value=0, key="ratio2")

# Section 3: Used Parts
st.header("3. Used Parts")
used_parts = st.multiselect("Select Used Parts", main_components_1, key="used_parts")

# Predict button
if st.button("Predict", key="predict_button"):
    if (input_method == "Plant Name" and plant_name) or (input_method == "ITS2 Sequence" and its2_sequence):
        with st.spinner("Processing..." if input_method == "ITS2 Sequence" else "Fetching ITS2 from NCBI and processing..."):
            try:
                # Determine ITS2 sequence based on input method
                if input_method == "Plant Name":
                    its2_sequence = get_ITS2_sequence(plant_name).lower()
                else:
                    its2_sequence = its2_sequence.lower()

                prediction, probs = predict(its2_sequence, solvent1, solvent2, ratio1, ratio2, used_parts)
                st.subheader("Prediction Result")
                if prediction == 1:
                    result = "Active"
                    st.success(f"Prediction: **{result}**")
                    st.write(f"Probability of Activity: {probs[1] * 100:.2f}%")
                    st.write(r"Note: Active means that the $IC_{50}$ is below 100 µg/mL.")
                else:
                    result = "Inactive"
                    st.success(f"Prediction: **{result}**")
                    st.write(f"Probability of Inactivity: {probs[0] * 100:.2f}%")
                    st.write(r"Note: Inactive means that the $IC_{50}$ is greater than 100 µg/mL.")
            except ValueError as e:
                st.error(str(e))
    else:
        st.error("Please provide a plant name or ITS2 sequence based on your selected input method.")
