#!/usr/bin/env python3
"""
Enhanced Polymer Builder with Proper 3D Structure Generation
--------------------------------------------------------
Builds polymer structures with validated 3D coordinates and proper atom naming.
"""

import os
import sys
from pathlib import Path
import logging
from datetime import datetime
import json
from typing import Dict, List, Optional, Tuple, Any

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, SanitizeMol
import numpy as np
from polymer_monitor import PolymerMonitor

class EnhancedPolymerBuilder:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.setup_logging()
        self.setup_directories()
        self.monomers = {}
        self.atom_counter = 1


        # Initialize monitoring
        self.monitor = PolymerMonitor(self.output_dir)
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        for dirname in ['polymer', 'monomers', 'trajectory']:
            dirpath = self.output_dir / dirname
            dirpath.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dirpath}")




    def set_monitor(self, monitor):
        self.monitor = monitor
        
    def _update_monitor(self, step, current_monomer):
        if hasattr(self, 'monitor'):
            self.monitor.monitor_step(self.polymer, step, current_monomer)

    def _generate_3d_structure(self, mol, optimize=True):
        """Generate 3D structure with minimal optimization"""
        try:
            mol = Chem.AddHs(mol)
            
            # Use minimal parameters for fastest embedding
            params = AllChem.ETKDGv2()
            params.maxIterations = 50  # Further reduced iterations
            params.randomSeed = 42
            
            success = AllChem.EmbedMolecule(mol, params)
            if success == -1:
                params.useRandomCoords = True
                success = AllChem.EmbedMolecule(mol, params)
                if success == -1:
                    raise ValueError("Failed to generate 3D coordinates")
            
            if optimize:
                # Minimal UFF optimization
                AllChem.UFFOptimizeMolecule(mol, maxIters=50)
            
            return mol
            
        except Exception as e:
            self.logger.error(f"Error in 3D structure generation: {e}")
            return None

    def _prepare_connection_points(self, mol: Chem.Mol, 
                                 connection_indices: List[int]) -> Tuple[Chem.Mol, List[int]]:
        """Prepare connection points with proper valence handling"""
        try:
            # Create editable molecule
            edit_mol = Chem.RWMol(mol)
            new_connection_points = []
            
            for idx in connection_indices:
                atom = edit_mol.GetAtomWithIdx(idx)
                
                # Ensure proper valence at connection points
                current_valence = atom.GetExplicitValence()
                target_valence = atom.GetImplicitValence()
                
                # Calculate available valence
                available_valence = target_valence - current_valence
                
                if available_valence < 1:
                    # Add temporary atom to handle connection
                    temp_idx = edit_mol.AddAtom(Chem.Atom('H'))
                    edit_mol.AddBond(idx, temp_idx, Chem.BondType.SINGLE)
                    new_connection_points.append(temp_idx)
                else:
                    new_connection_points.append(idx)
            
            return edit_mol.GetMol(), new_connection_points
            
        except Exception as e:
            print(f"Error preparing connection points: {e}")
            return mol, connection_indices

    def process_monomer(self, name, smiles):
        """Process monomer with proper structure initialization"""
        print(f"\nProcessing monomer: {name}")
        print(f"SMILES: {smiles}")
        
        try:
            # Convert [R] to * for RDKit
            rdkit_smiles = smiles.replace('[R]', '*')
            
            # Create initial molecule with proper initialization
            mol = Chem.MolFromSmiles(rdkit_smiles, sanitize=False)
            if mol is None:
                raise ValueError("Failed to create molecule from SMILES")
            
            # Proper initialization sequence
            SanitizeMol(mol)
            Chem.GetSymmSSSR(mol)
            
            # Generate 3D structure
            mol = self._generate_3d_structure(mol)
            if mol is None:
                raise ValueError("Failed to generate 3D structure")
            
            # Find and validate connection points
            connection_points = []
            connecting_atoms = []
            
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == '*':
                    neighbors = atom.GetNeighbors()
                    if len(neighbors) != 1:
                        raise ValueError(f"Invalid connection point at atom {atom.GetIdx()}")
                    connection_points.append(atom.GetIdx())
                    connecting_atoms.append(neighbors[0].GetIdx())
            
            if len(connection_points) != 2:
                raise ValueError(f"Found {len(connection_points)} connection points, need exactly 2")
            
            # Store monomer info
            self.monomers[name] = {
                'mol': mol,
                'connecting_atoms': connecting_atoms,
                'connection_points': connection_points,
                'smiles': smiles
            }
            
            # Save monomer files
            self.save_monomer_files(name, mol)
            print(f"Successfully processed monomer: {name}")
            return True
            
        except Exception as e:
            print(f"Error processing monomer {name}: {str(e)}")
            return False

    def save_monomer_files(self, name, mol):
        """Save monomer files with proper atom naming"""
        try:
            monomer_dir = self.output_dir / 'monomers' / name
            monomer_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate proper atom names
            for atom in mol.GetAtoms():
                atom_symbol = atom.GetSymbol()
                if atom_symbol == '*':
                    continue
                atom.SetProp('_Name', f"{atom_symbol}{self.atom_counter}")
                self.atom_counter += 1
            
            # Save structure files
            Chem.MolToMolFile(mol, str(monomer_dir / f'{name}.mol'))
            Chem.MolToPDBFile(mol, str(monomer_dir / f'{name}.pdb'))
            
            print(f"Saved monomer files in: {monomer_dir}")
            
        except Exception as e:
            print(f"Error saving monomer files: {str(e)}")

    def _prepare_first_monomer(self, monomer_data):
        """Prepare first monomer with proper hydrogen handling"""
        try:
            # Remove hydrogens first
            mol = Chem.RemoveHs(monomer_data['mol'])
            
            # Create editable molecule
            edit_mol = Chem.RWMol(mol)
            
            # Find and remove connection points (*)
            r_indices = []
            for atom in edit_mol.GetAtoms():
                if atom.GetSymbol() == '*':
                    r_indices.append(atom.GetIdx())
            
            # Remove in reverse order
            for idx in sorted(r_indices, reverse=True):
                edit_mol.RemoveAtom(idx)
            
            # Get final molecule
            mol = edit_mol.GetMol()
            
            # Clean up the molecule
            mol.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(mol)
            
            # Add hydrogens back and generate 3D structure
            mol = Chem.AddHs(mol)
            mol = self._generate_3d_structure(mol, optimize=True)
            
            if mol is None:
                raise ValueError("Failed to generate 3D structure")
            
            return mol
            
        except Exception as e:
            print(f"Error preparing first monomer: {str(e)}")
            return None

    def connect_monomers(self, polymer: Chem.Mol, monomer_data: Dict) -> Optional[Chem.Mol]:
        """Connect monomers with robust index handling and valence checking"""
        try:
            # Remove hydrogens before connecting
            poly = Chem.RemoveHs(polymer)
            mono = Chem.RemoveHs(monomer_data['mol'])
            
            # Create editable molecules
            poly_edit = Chem.RWMol(poly)
            mono_edit = Chem.RWMol(mono)
            
            # Get connection points
            poly_atoms = poly.GetNumAtoms()
            mono_atoms = mono.GetNumAtoms()
            
            # Find and remove connection points from monomer
            r_indices = []
            connecting_atoms = []
            for atom in mono_edit.GetAtoms():
                if atom.GetSymbol() == '*':
                    neighbors = atom.GetNeighbors()
                    if len(neighbors) == 1:
                        connecting_atoms.append(neighbors[0].GetIdx())
                    r_indices.append(atom.GetIdx())
            
            # Remove R groups in reverse order to maintain valid indices
            for idx in sorted(r_indices, reverse=True):
                mono_edit.RemoveAtom(idx)
                
            # Get clean molecules
            mono = mono_edit.GetMol()
            
            # Find attachment point on polymer
            poly_connection = None
            for atom in poly.GetAtoms():
                if len(atom.GetNeighbors()) < atom.GetImplicitValence():
                    poly_connection = atom.GetIdx()
                    break
            
            if poly_connection is None:
                poly_connection = poly_atoms - 1  # Use last atom as fallback
                
            # Combine molecules
            combined = Chem.CombineMols(poly, mono)
            
            # Create editable combined molecule
            edit_mol = Chem.RWMol(combined)
            
            # Calculate offset for second molecule indices
            offset = poly_atoms
            
            # Add connecting bond with index checking
            if connecting_atoms:
                # Use first valid connecting atom from monomer
                mono_connection = connecting_atoms[0]
                
                # Verify indices are valid
                if poly_connection < poly_atoms and (mono_connection + offset) < edit_mol.GetNumAtoms():
                    # Add bond with valence checking
                    try:
                        edit_mol.AddBond(poly_connection, 
                                       mono_connection + offset,
                                       Chem.BondType.SINGLE)
                    except Exception as e:
                        print(f"Warning in bond formation: {e}")
                        # Try alternative connection if primary fails
                        for alt_connection in connecting_atoms[1:]:
                            try:
                                edit_mol.AddBond(poly_connection,
                                               alt_connection + offset,
                                               Chem.BondType.SINGLE)
                                break
                            except Exception:
                                continue
                else:
                    raise ValueError("Invalid connection indices")
            
            # Get final molecule
            mol = edit_mol.GetMol()
            
            # Clean up the molecule
            mol.UpdatePropertyCache(strict=False)
            try:
                Chem.SanitizeMol(mol,
                    sanitizeOps=Chem.SANITIZE_ALL^Chem.SANITIZE_KEKULIZE)
            except Exception as e:
                print(f"Warning during sanitization: {e}")
                try:
                    Chem.SanitizeMol(mol,
                        sanitizeOps=Chem.SANITIZE_FINDRADICALS|
                                   Chem.SANITIZE_SETAROMATICITY|
                                   Chem.SANITIZE_SETCONJUGATION|
                                   Chem.SANITIZE_SETHYBRIDIZATION|
                                   Chem.SANITIZE_SYMMRINGS,
                        catchErrors=True)
                except Exception as e:
                    print(f"Warning during minimal sanitization: {e}")
            
            # Add hydrogens back and generate 3D structure
            mol = Chem.AddHs(mol)
            mol = self._generate_3d_structure(mol, optimize=True)
            
            if mol is None:
                raise ValueError("Failed to generate 3D structure")
            
            # Validate the structure
            if not self._validate_coordinates(mol):
                raise ValueError("Invalid coordinates after connection")
            
            return mol
            
        except Exception as e:
            print(f"Error connecting monomers: {str(e)}")
            return None

    def _find_connection_points(self, mol: Chem.Mol) -> List[int]:
        """Find valid connection points in molecule"""
        connection_points = []
        for atom in mol.GetAtoms():
            # Check for available valence
            if len(atom.GetNeighbors()) < atom.GetImplicitValence():
                connection_points.append(atom.GetIdx())
            # Check for R groups
            elif atom.GetSymbol() == '*':
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:
                    connection_points.append(neighbors[0].GetIdx())
        return connection_points


    def _validate_coordinates(self, mol):
        """Validate atomic coordinates"""
        try:
            if mol.GetNumConformers() == 0:
                return False
                
            conf = mol.GetConformer()
            positions = []
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                # Check for zero or invalid coordinates
                if pos.x == 0 and pos.y == 0 and pos.z == 0:
                    return False
                if abs(pos.x) > 999.999 or abs(pos.y) > 999.999 or abs(pos.z) > 999.999:
                    return False
                positions.append([pos.x, pos.y, pos.z])
            
            # Check for atomic clashes
            positions = np.array(positions)
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < 0.5:  # Atoms too close
                        return False
            
            return True
            
        except Exception:
            return False

    def build_polymer(self, sequence):
        """
        Build polymer with enhanced validation, optimization, and trajectory saving.
        Includes comprehensive error handling and structure verification at each step.
        """
        print("\nStarting polymer construction...")
        
        try:
            if not sequence:
                raise ValueError("Empty sequence")
            
            polymer = None
            total_units = 0
            save_interval = 25
            
            # Initialize progress tracking
            total_monomers = sum(count for count, _ in sequence)
            
            for count, name in sequence:
                print(f"\nAdding {count} units of {name}")
                
                if name not in self.monomers:
                    raise ValueError(f"Monomer not found: {name}")
                
                monomer_data = self.monomers[name]
                
                for i in range(count):
                    if polymer is None:
                        # Initialize first monomer with validation
                        polymer = self._prepare_first_monomer(monomer_data)
                        if polymer is None:
                            raise ValueError("Failed to prepare first monomer")
                        
                        # Validate initial structure
                        if not self._validate_monomer_structure(polymer):
                            raise ValueError("Initial monomer validation failed")
                        
                        print("Added first monomer unit")
                        
                    else:
                        # Connect and validate each new monomer
                        polymer = self.connect_monomers(polymer, monomer_data)
                        if polymer is None:
                            raise ValueError(f"Failed to connect monomer {i+1} of {name}")
                        
                        # Intermediate structure validation
                        if not self._validate_growing_structure(polymer):
                            raise ValueError(f"Structure validation failed at unit {total_units + 1}")
                    
                    total_units += 1


                    # Update monitoring
                    self.monitor.monitor_step(polymer, total_units, name)                   
                    
                    # Progress reporting
                    if total_units % 5 == 0:
                        print(f"Built {total_units}/{total_monomers} units "
                              f"({(total_units/total_monomers)*100:.1f}% complete)...")
                    
                    # Periodic structure saving and validation
                    if total_units % save_interval == 0:
                        self._periodic_optimization(polymer)
                        self.save_trajectory(polymer, total_units)
            


            if polymer:
                print("\nOptimizing final structure...")
                
                # Comprehensive final optimization
                final_mol = self._optimize_final_structure(polymer)
                
                if final_mol is None:
                    raise ValueError("Final optimization failed")
                
                # Final structure validation
                validation_result = self._validate_final_structure(final_mol)
                if not validation_result[0]:
                    raise ValueError(f"Final validation failed: {validation_result[1]}")
                
                # Save final structure and properties
                self.save_polymer_files(final_mol)
                print("\nPolymer construction completed successfully!")
                return final_mol
            
            raise ValueError("Failed to build polymer")
            
        except Exception as e:
            print(f"\nError building polymer: {str(e)}")
            return None

    def _validate_monomer_structure(self, mol):
        """Validate individual monomer structure"""
        try:
            if mol is None or mol.GetNumAtoms() == 0:
                return False
                
            # Basic structure checks
            Chem.SanitizeMol(mol)
            
            # Validate 3D coordinates
            if mol.GetNumConformers() == 0:
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol)
            
            return self._validate_coordinates(mol)
            
        except Exception as e:
            print(f"Monomer validation error: {str(e)}")
            return False

    def _validate_growing_structure(self, mol):
        """Validate growing polymer structure"""
        try:
            if not self._validate_coordinates(mol):
                return False
            
            # Check for disconnected fragments
            fragments = Chem.GetMolFrags(mol)
            if len(fragments) > 1:
                return False
            
            # Validate bond lengths
            conf = mol.GetConformer()
            for bond in mol.GetBonds():
                start = conf.GetAtomPosition(bond.GetBeginAtomIdx())
                end = conf.GetAtomPosition(bond.GetEndAtomIdx())
                length = (start - end).Length()
                if length < 0.7 or length > 2.0:  # Angstroms
                    return False
            
            return True
            
        except Exception:
            return False

    def _periodic_optimization(self, mol):
        """Perform periodic structure optimization"""
        try:
            # Quick MMFF optimization
            mp = AllChem.MMFFGetMoleculeProperties(mol)
            if mp is None:
                return
                
            ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
            if ff is None:
                return
                
            ff.Minimize(maxIts=200)
            
        except Exception:
            pass

    def _validate_final_structure(self, mol):
        """Comprehensive final structure validation"""
        try:
            if not self._validate_coordinates(mol):
                return False, "Coordinate validation failed"
            
            # Structure connectivity check
            fragments = Chem.GetMolFrags(mol)
            if len(fragments) > 1:
                return False, "Structure contains disconnected fragments"
            
            # Energy validation
            mp = AllChem.MMFFGetMoleculeProperties(mol)
            if mp is None:
                return False, "Failed to calculate molecular properties"
                
            ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
            if ff is None:
                return False, "Failed to initialize force field"
                
            energy = ff.CalcEnergy()
            if energy > 1e6:  # Unreasonably high energy
                return False, "Structure has excessive strain energy"
            
            return True, None
            
        except Exception as e:
            return False, str(e)

    def _optimize_final_structure(self, mol):
        """Final structure optimization"""
        try:
            # Initialize the molecule
            SanitizeMol(mol)
            Chem.GetSymmSSSR(mol)
            
            # Thorough optimization
            mol = self._generate_3d_structure(mol, optimize=True)
            
            # Additional UFF optimization with more iterations
            if mol:
                AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
            
            if mol and self._validate_coordinates(mol):
                return mol
            return None
            
        except Exception as e:
            print(f"Error in final optimization: {str(e)}")
            return None

    def save_trajectory(self, mol, step):
        """Save intermediate structures"""
        try:
            if not self._validate_coordinates(mol):
                return
                
            traj_dir = self.output_dir / 'trajectory'
            Chem.MolToMolFile(mol, str(traj_dir / f'step_{step:04d}.mol'))
            
        except Exception as e:
            print(f"Warning saving trajectory: {str(e)}")

    def save_polymer_files(self, polymer):
        """Save final polymer structure with enhanced PDB formatting"""
        try:
            if not self._validate_coordinates(polymer):
                raise ValueError("Invalid coordinates in final structure")
                
            polymer_dir = self.output_dir / 'polymer'
            
            print("\nSaving polymer files:")
            
            # Save MOL file
            mol_path = str(polymer_dir / 'polymer.mol')
            print(f"MOL file: {mol_path}")
            Chem.MolToMolFile(polymer, mol_path)
            
            # Save enhanced PDB file
            pdb_path = polymer_dir / 'polymer.pdb'
            print(f"PDB file: {pdb_path}")
            
            # Write PDB with enhanced formatting
            with open(pdb_path, 'w') as f:
                # Write header information
                now = datetime.now()
                f.write("TITLE     BLOCK COPOLYMER STRUCTURE\n")
                f.write(f"REMARK   1 CREATED BY POLYMER BUILDER ON {now.strftime('%Y-%m-%d')}\n")
                f.write("REMARK   2 SEQUENCE: 32xPHBV - 1xPHBV_N_PEG - 41xPEG\n")
                f.write("REMARK   3 POLYMER COMPOSITION:\n")
                f.write("REMARK   3  PHBV: POLY(3-HYDROXYBUTYRATE-CO-3-HYDROXYVALERATE)\n")
                f.write("REMARK   3  PHBV_N_PEG: PHBV-PEG JUNCTION UNIT\n")
                f.write("REMARK   3  PEG: POLYETHYLENE GLYCOL\n")
                f.write("AUTHOR    GENERATED BY ENHANCED POLYMER BUILDER\n")
                f.write("REVDAT   1   " + now.strftime('%d-%b-%y') + "\n")
                
                # Write coordinates with enhanced residue names and chain IDs
                conf = polymer.GetConformer()
                atom_count = 0
                chain_id = 'A'  # Start with chain A
                
                # Track monomer boundaries for residue numbering
                atoms_per_phbv = 9  # Adjust based on your monomer size
                atoms_per_phbv_n_peg = 16
                atoms_per_peg = 7
                
                for atom in polymer.GetAtoms():
                    atom_count += 1
                    pos = conf.GetAtomPosition(atom_count-1)
                    
                    # Determine residue name and number based on position
                    if atom_count <= 32 * atoms_per_phbv:
                        resname = "PHB"  # PHBV units
                        chain_id = 'A'
                        resnum = (atom_count - 1) // atoms_per_phbv + 1
                    elif atom_count <= (32 * atoms_per_phbv + atoms_per_phbv_n_peg):
                        resname = "JCT"  # Junction unit
                        chain_id = 'B'
                        resnum = 1
                    else:
                        resname = "PEG"  # PEG units
                        chain_id = 'C'
                        resnum = (atom_count - (32 * atoms_per_phbv + atoms_per_phbv_n_peg)) // atoms_per_peg + 1
                    
                    # Write ATOM/HETATM record with proper formatting
                    record = "HETATM" if atom.GetSymbol() != 'C' else "ATOM  "
                    f.write(f"{record}{atom_count:5d}  {atom.GetSymbol():<3}{resname:3} {chain_id}{resnum:4d}    "
                           f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00  0.00           {atom.GetSymbol():>2}\n")
                
                # Write connectivity records
                for bond in polymer.GetBonds():
                    f.write(f"CONECT{bond.GetBeginAtomIdx()+1:5d}{bond.GetEndAtomIdx()+1:5d}\n")
                
                # Write end record
                f.write("END\n")
            
            # Calculate and save properties
            properties = {
                'formula': rdMolDescriptors.CalcMolFormula(polymer),
                'molecular_weight': Descriptors.ExactMolWt(polymer),
                'num_atoms': polymer.GetNumAtoms(),
                'num_bonds': polymer.GetNumBonds(),
                'num_rings': rdMolDescriptors.CalcNumRings(polymer),
                'rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(polymer)
            }
            
            # Calculate end-to-end distance
            if polymer.GetNumConformers() > 0:
                conf = polymer.GetConformer()
                pos_0 = conf.GetAtomPosition(0)
                pos_n = conf.GetAtomPosition(polymer.GetNumAtoms() - 1)
                end_to_end = np.sqrt(
                    (pos_0.x - pos_n.x)**2 + 
                    (pos_0.y - pos_n.y)**2 +
                    (pos_0.z - pos_n.z)**2)
                properties['end_to_end_distance'] = float(end_to_end)
            
            with open(polymer_dir / 'polymer_properties.json', 'w') as f:
                json.dump(properties, f, indent=2)
            
            print("\nPolymer properties:")
            for key, value in properties.items():
                print(f"{key}: {value}")
            
        except Exception as e:
            print(f"Error saving polymer files: {str(e)}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced polymer structure builder")
    parser.add_argument("-i", "--input", required=True, help="Input monomer definition file")
    parser.add_argument("-o", "--output", default="polymer_output", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        print(f"\nInitializing polymer builder...")
        print(f"Input file: {args.input}")
        print(f"Output directory: {args.output}")
        
        builder = EnhancedPolymerBuilder(args.output)
        
        print("\nReading monomer definitions...")
        
        monomers = {}
        sequence = []
        current_section = None
        
        with open(args.input) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if line.startswith('[') and line.endswith(']'):
                    current_section = line[1:-1].upper()
                    print(f"\nFound section: {current_section}")
                    continue
                
                if current_section != 'SEQUENCE' and ':' in line:
                    name, smiles = [x.strip() for x in line.split(':')]
                    if name and smiles:
                        monomers[name] = smiles
                        print(f"Found monomer: {name}")
                
                elif current_section == 'SEQUENCE':
                    parts = [x.strip() for x in line.split(',')]
                    if len(parts) == 2:
                        count = int(parts[0])
                        name = parts[1]
                        sequence.append((count, name))
                        print(f"Added to sequence: {count} x {name}")
        
        if not monomers:
            raise ValueError("No valid monomers found in input file")
        
        if not sequence:
            raise ValueError("No valid sequence found in input file")
        
        print("\nProcessing monomers...")
        for name, smiles in monomers.items():
            if not builder.process_monomer(name, smiles):
                raise ValueError(f"Failed to process monomer: {name}")
        
        polymer = builder.build_polymer(sequence)
        
        if polymer:
            print("\nPolymer construction completed successfully!")
            print(f"Output files saved in: {args.output}")
            return 0
        
        return 1
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())