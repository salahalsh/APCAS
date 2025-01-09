"""
Enhanced PDB Loading Module with Robust Structure Handling
-------------------------------------------------------
"""

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def load_and_process_pdb(pdb_path):
    """
    Load PDB file with enhanced error handling and structure processing.
    
    Args:
        pdb_path (str): Path to PDB file
        
    Returns:
        RDKit molecule or None
    """
    try:
        # First attempt: Standard PDB loading
        mol = Chem.MolFromPDBFile(pdb_path, 
                                 removeHs=False,
                                 sanitize=False,
                                 proximityBonding=False)
        
        if mol is None or mol.GetNumAtoms() == 0:
            print("Standard loading failed, trying alternative approach...")
            
            # Read PDB content
            with open(pdb_path, 'r') as f:
                pdb_lines = f.readlines()
            
            # Process ATOM/HETATM records
            atoms = []
            atom_positions = []
            
            for line in pdb_lines:
                if line.startswith(('ATOM', 'HETATM')):
                    # Extract atom information
                    atom_symbol = line[12:16].strip()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    
                    atoms.append(atom_symbol)
                    atom_positions.append([x, y, z])
            
            if not atoms:
                raise ValueError("No atoms found in PDB file")
            
            # Create molecule from scratch
            mol = Chem.RWMol()
            
            # Add atoms
            atom_map = {}
            for i, (symbol, pos) in enumerate(zip(atoms, atom_positions)):
                # Clean up atom symbol (remove numbers)
                clean_symbol = ''.join(c for c in symbol if not c.isdigit()).strip()
                atom = Chem.Atom(clean_symbol)
                idx = mol.AddAtom(atom)
                atom_map[i+1] = idx
            
            # Process CONECT records for bonds
            for line in pdb_lines:
                if line.startswith('CONECT'):
                    # Parse CONECT record
                    fields = line[6:].strip().split()
                    if len(fields) >= 2:
                        try:
                            from_atom = int(fields[0])
                            for to_atom in map(int, fields[1:]):
                                if from_atom in atom_map and to_atom in atom_map:
                                    mol.AddBond(atom_map[from_atom],
                                             atom_map[to_atom],
                                             Chem.BondType.SINGLE)
                        except ValueError:
                            continue
            
            # Convert to regular molecule
            mol = mol.GetMol()
            
            # Add 3D coordinates
            conf = Chem.Conformer(mol.GetNumAtoms())
            for i, pos in enumerate(atom_positions):
                conf.SetAtomPosition(i, pos)
            mol.AddConformer(conf)
            
        # Try to sanitize
        try:
            Chem.SanitizeMol(mol,
                sanitizeOps=Chem.SANITIZE_ALL^Chem.SANITIZE_KEKULIZE)
            print("Full sanitization successful")
        except:
            print("Full sanitization failed, trying minimal sanitization")
            try:
                Chem.SanitizeMol(mol,
                    sanitizeOps=Chem.SANITIZE_ADJUSTHS|Chem.SANITIZE_CLEANUP)
                print("Minimal sanitization successful")
            except:
                print("Warning: Could not sanitize molecule")
        
        # Validate structure
        if mol.GetNumAtoms() == 0:
            raise ValueError("No atoms in final molecule")
            
        print(f"Successfully loaded structure with {mol.GetNumAtoms()} atoms")
        print(f"Number of bonds: {mol.GetNumBonds()}")
        
        # Print atom details
        print("\nAtom details:")
        for atom in mol.GetAtoms():
            print(f"Atom {atom.GetIdx()}: {atom.GetSymbol()} "
                  f"(Valence: {atom.GetTotalValence()}, "
                  f"Formal Charge: {atom.GetFormalCharge()})")
        
        return mol
        
    except Exception as e:
        print(f"Error loading structure: {e}")
        return None

