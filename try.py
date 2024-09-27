import os
import numpy as np
import pandas as pd

# Dictionary for three-letter to one-letter amino acid code conversion
three_to_one = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
    "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
    "TRP": "W", "TYR": "Y"
}

def parse_pdb_sequence(pdb_file):
    """Extract amino acid sequence from a PDB file."""
    sequence = []
    with open(pdb_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Filter lines containing amino acid residues
            if line.startswith("ATOM") and " CA " in line:
                resname = line[17:20].strip()
                if resname in three_to_one:
                    sequence.append(three_to_one[resname])
    return ''.join(sequence)

def parse_c_alpha_atoms(pdb_file):
    """Extract C-alpha atom coordinates from a PDB file."""
    c_alpha_atoms = []
    with open(pdb_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("ATOM") and " CA " in line:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                c_alpha_atoms.append((x, y, z))
    return c_alpha_atoms

def calculate_distance(atom1, atom2):
    """Calculate the Euclidean distance between two atoms."""
    return np.sqrt((atom1[0] - atom2[0])**2 + (atom1[1] - atom2[1])**2 + (atom1[2] - atom2[2])**2)

def generate_contact_map(c_alpha_atoms, threshold=8.0):
    """Generate a contact map using C-alpha atom coordinates."""
    n_atoms = len(c_alpha_atoms)
    distances = np.zeros((n_atoms, n_atoms))

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            distance = calculate_distance(c_alpha_atoms[i], c_alpha_atoms[j])
            distances[i, j] = distance
            distances[j, i] = distance

    contact_map = (distances <= threshold).astype(int)
    return contact_map

def flatten_contact_map(contact_map):
    """Flatten the contact map to a string format for CSV saving."""
    return ','.join(map(str, contact_map.flatten()))

def create_protein_dataset(pdb_files, output_csv):
    """Create a dataset with protein sequences and contact maps, saving to CSV."""
    data = []

    for pdb_file in pdb_files:
        try:
            sequence = parse_pdb_sequence(pdb_file)
            c_alpha_atoms = parse_c_alpha_atoms(pdb_file)
            contact_map = generate_contact_map(c_alpha_atoms)
            contact_map_flattened = flatten_contact_map(contact_map)

            # Append the sequence and flattened contact map to the dataset
            data.append({
                'PDB_File': os.path.basename(pdb_file),
                'Sequence': sequence,
                'Contact_Map': contact_map_flattened
            })
        except Exception as e:
            print(f"Error processing {pdb_file}: {e}")

    # Convert data to DataFrame and save as CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=True)
    print(f"Dataset created and saved as '{output_csv}'")

# Example usage: List of PDB files to process
pdb_files = ['Proteins/7pgl.pdb', 'Proteins/7pgm.pdb', 'Proteins/7pgn.pdb', 'Proteins/451c.pdb']  # Replace with actual PDB file paths
output_csv = 'protein_dataset.csv'
create_protein_dataset(pdb_files, output_csv)
