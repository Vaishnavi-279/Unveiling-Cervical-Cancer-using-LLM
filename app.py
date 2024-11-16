# app.py
from flask import Flask, redirect, render_template, request, jsonify, url_for
import numpy as np
import pickle
from Bio import PDB
from Bio.Data import IUPACData
from scipy.spatial.distance import pdist, squareform
from transformers import AutoTokenizer, AutoModel
import torch
import os

app = Flask(__name__)

# Load the Hugging Face token and model
token = "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"
tokenizer = AutoTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S", token=token)
model = AutoModel.from_pretrained("facebook/esm1b_t33_650M_UR50S", token=token)

# Load the trained SVM model
with open("svm_model.pkl", "rb") as f:
    svm = pickle.load(f)

# Ensure the directory exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define function to process PDB files and get predictions
def calculate_contact_map(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    coords = [residue['CA'].get_coord() for model in structure for chain in model for residue in chain if residue.has_id('CA')]
    
    distances = pdist(coords)
    contact_map = squareform(distances)
    padded_contact_map = np.zeros((500, 500))
    size = min(contact_map.shape[0], 500)
    padded_contact_map[:size, :size] = contact_map[:size, :size]
    return padded_contact_map.flatten()

def extract_sequence_from_pdb(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    sequence = "".join([IUPACData.protein_letters_3to1.get(residue.get_resname().capitalize(), 'X') for model in structure for chain in model for residue in chain if PDB.is_aa(residue, standard=True)])
    return sequence

def generate_seqvec_embedding(sequence, max_length=1024):
    inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


@app.route('/')
def upload():
    return render_template('index.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'pdb_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['pdb_file']
    pdb_file_path = os.path.join(UPLOAD_FOLDER, 'uploaded_pdb.pdb')
    file.save(pdb_file_path)

    contact_map = calculate_contact_map(pdb_file_path)
    sequence = extract_sequence_from_pdb(pdb_file_path)
    embedding = generate_seqvec_embedding(sequence)

    features = np.concatenate((contact_map, embedding)).reshape(1, -1)
    probability = svm.predict_proba(features)[0][1]
    diagnosis = 'cancerous' if probability >= 0.5 else 'healthy'

    return redirect(url_for('result', probability=probability, diagnosis=diagnosis))

@app.route('/result')
def result():
    probability = request.args.get('probability', None, type=float)
    diagnosis = request.args.get('diagnosis', None)
    return render_template('result.html', probability=probability, diagnosis=diagnosis)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('styles', exist_ok=True) 
    os.makedirs('static', exist_ok=True)  # Ensure static folder exists
    app.run(debug=True)
