import numpy as np

# Converted three-letter to one-letter code
three_to_one = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G", 
    "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N", 
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V", 
    "TRP": "W", "TYR": "Y"
}

# Found the position of carbon aplha atoms and created a list and also created a list of amino acids present in the protein
def parse_pdb(pdb_file):
    with open(pdb_file, 'r') as file:
        lines = file.readlines()
    
    c_alpha_atoms = []
    amino_acids = []
    for line in lines:
        if line.startswith("ATOM") and " CA " in line:
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            c_alpha_atoms.append((x, y, z))
            

            three_letter_code = line[17:20].strip()
            one_letter_code = three_to_one.get(three_letter_code, 'X')
            amino_acids.append(one_letter_code)
    
    return c_alpha_atoms, amino_acids

# Found the distance between two carbon atoms
def calculate_distance(atom1, atom2):
    return np.sqrt((atom1[0] - atom2[0])**2 + (atom1[1] - atom2[1])**2 + (atom1[2] - atom2[2])**2)

# Function to generate contact map
def generate_contact_map(c_alpha_atoms, threshold=8.0):
    n_atoms = len(c_alpha_atoms)
    distances = np.zeros((n_atoms, n_atoms))

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            distance = calculate_distance(c_alpha_atoms[i], c_alpha_atoms[j])
            distances[i, j] = distance
            distances[j, i] = distance
    contact_map = (distances <= threshold).astype(int)
    return contact_map


pdb_file = 'Proteins/7pgl.pdb'
c_alpha_atoms, amino_acids = parse_pdb(pdb_file)
contact_map = generate_contact_map(c_alpha_atoms)

# Created a file and imported the contact map to the file
output_file = '7pgl_contact_map.txt'
with open(output_file, 'w') as file:

    # file.write(''.join(amino_acids) + '\n')

    for row in contact_map:
        file.write(''.join(map(str, row)) + '\n')
