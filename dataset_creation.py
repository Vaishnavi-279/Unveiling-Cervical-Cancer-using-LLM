import os
import numpy as np
from Bio import PDB
from Bio.Data import IUPACData
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scipy.spatial.distance import pdist, squareform
import pickle

# Set your Hugging Face token
token = "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"

# Load the SeqVec (ESM-1b) model for sequence embedding
tokenizer = AutoTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S", token=token)
model = AutoModel.from_pretrained("facebook/esm1b_t33_650M_UR50S", token=token)

# Ensure 'models' directory exists to save the model file
os.makedirs("models", exist_ok=True)

# Set fixed size for contact maps
FIXED_SIZE = 500

# Calculate contact maps from PDB files
def calculate_contact_map(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    coords = [residue['CA'].get_coord() for model in structure for chain in model for residue in chain if residue.has_id('CA')]
    
    distances = pdist(coords)
    contact_map = squareform(distances)
    
    padded_contact_map = np.zeros((FIXED_SIZE, FIXED_SIZE))
    size = min(contact_map.shape[0], FIXED_SIZE)
    padded_contact_map[:size, :size] = contact_map[:size, :size]
    return padded_contact_map

# Generate SeqVec embedding for a sequence
def generate_seqvec_embedding(sequence, max_length=1024):
    valid_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    cleaned_sequence = "".join([aa for aa in sequence if aa in valid_amino_acids])
    
    inputs = tokenizer(cleaned_sequence, return_tensors="pt", padding=True, truncation=True, max_length=max_length, add_special_tokens=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Extract sequence from PDB file
def extract_sequence_from_pdb(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    sequence = "".join([IUPACData.protein_letters_3to1.get(residue.get_resname().capitalize(), 'X') for model in structure for chain in model for residue in chain if PDB.is_aa(residue, standard=True)])
    return sequence

# Define paths to directories for healthy and unhealthy protein data
healthy_proteins = "data/Healthy_proteins"
unhealthy_proteins = "data/Unhealthy_proteins"

# List files in directory
def list_files_in_directory(directory_path):
    file_paths = []
    for filename in os.listdir(directory_path):
        full_path = os.path.join(directory_path, filename)
        if os.path.isfile(full_path) and full_path.endswith(".pdb"):
            file_paths.append(full_path)
    return file_paths

# Prepare data
X_contact_maps, X_embeddings, y = [], [], []

# Process healthy proteins
healthy_files = list_files_in_directory(healthy_proteins)
for pdb_file in healthy_files:
    contact_map = calculate_contact_map(pdb_file)
    sequence = extract_sequence_from_pdb(pdb_file)
    embedding = generate_seqvec_embedding(sequence)
    
    X_contact_maps.append(contact_map.flatten())
    X_embeddings.append(embedding)
    y.append(0)  # Label for healthy

# Process unhealthy proteins
unhealthy_files = list_files_in_directory(unhealthy_proteins)
for pdb_file in unhealthy_files:
    contact_map = calculate_contact_map(pdb_file)
    sequence = extract_sequence_from_pdb(pdb_file)
    embedding = generate_seqvec_embedding(sequence)
    
    X_contact_maps.append(contact_map.flatten())
    X_embeddings.append(embedding)
    y.append(1)  # Label for unhealthy (cancerous)

# Convert lists to arrays
X_contact_maps = np.array(X_contact_maps)
X_embeddings = np.array(X_embeddings)
y = np.array(y)

# Combine features (contact maps and embeddings) and split data
X = np.concatenate((X_contact_maps, X_embeddings), axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM model
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)

# Evaluate and save the model
accuracy = accuracy_score(y_test, svm.predict(X_test))
print(f"Model Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, svm.predict(X_test)))

# Save the trained model
with open("models/cancer_detection_svm_model.pkl", "wb") as f:
    pickle.dump(svm, f)
