#!/usr/bin/env python3
"""
Enhanced Monomer Processing Module
--------------------------------
Processes monomers and generates accurate structure files with improved 2D/3D visualization 
and correct MOL2/PDB file generation. Can run standalone or as part of the polymer builder.


python polymer_builder_monomer.py -i "SMILES_STRING" -n monomer_name -o output_dir

python polymer_builder_monomer.py -i monomer_data.txt -o output_dir

"""

import os
import sys
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdDepictor
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdMolTransforms
from rdkit.Chem.Draw.MolDrawing import DrawingOptions

@dataclass
class ProcessedMonomer:
    """Enhanced container for processed monomer data with validation"""
    name: str
    smiles: str
    mol: Optional[Chem.Mol] = None
    clean_mol: Optional[Chem.Mol] = None
    connection_points: List[int] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate monomer data completeness and structure"""
        try:
            if not all([
                self.name,
                self.smiles,
                isinstance(self.mol, Chem.Mol),
                isinstance(self.clean_mol, Chem.Mol),
                len(self.connection_points) == 2,
                isinstance(self.properties, dict)
            ]):
                return False
                
            # Additional structure validation
            if self.clean_mol.GetNumConformers() == 0:
                return False
                
            # Validate connection points are within molecule size
            num_atoms = self.clean_mol.GetNumAtoms()
            if any(idx >= num_atoms for idx in self.connection_points):
                return False
                
            return True
            
        except Exception:
            return False

class MonomerProcessor:
    """Enhanced monomer processing with improved structure generation"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize MonomerProcessor with output directory setup and logging"""
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / 'polymer_output'
        self.temp_dir = Path(tempfile.mkdtemp(prefix='monomer_processing_'))
        self.logger = self._setup_logging()
        self._setup_directories()
        
        # Initialize RDKit drawing options
        self.drawing_options = DrawingOptions()
        self.drawing_options.bondLineWidth = 2.0
        self.drawing_options.atomLabelFontSize = 16
        self.drawing_options.includeAtomNumbers = True
        
        # Store processed monomers
        self.processed_monomers: Dict[str, ProcessedMonomer] = {}

    def _setup_logging(self) -> logging.Logger:
        """Configure detailed logging system with both file and console output"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Remove any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create logs directory
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'monomer_processing_{timestamp}.log'
        
        # File handler with detailed format
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console handler with simpler format
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        return logger

    def _setup_directories(self):
        """Create required directory structure with proper permissions"""
        try:
            dirs = [
                'monomers',
                'logs',
                'temp',
                'visualization'
            ]
            
            for dir_name in dirs:
                dir_path = self.output_dir / dir_name
                dir_path.mkdir(parents=True, exist_ok=True)
                
                # Test write permissions
                test_file = dir_path / '.test'
                try:
                    test_file.touch()
                    test_file.unlink()
                except Exception as e:
                    self.logger.error(f"Directory {dir_path} is not writable: {e}")
                    raise
                    
            self.logger.info("Directory structure created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create directory structure: {e}")
            raise

    def cleanup(self):
        """Clean up temporary files and directories"""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
            self.logger.info("Temporary files cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error cleaning up temporary files: {e}")

    def validate_smiles(self, smiles: str) -> bool:
        """
        Validate SMILES string with comprehensive chemistry checks.
        Ensures proper connection points and valid chemical structure.
        """
        try:
            if not isinstance(smiles, str) or not smiles.strip():
                self.logger.error("Invalid SMILES: Empty or not a string")
                return False
            
            # Check for balanced brackets
            if smiles.count('[') != smiles.count(']'):
                self.logger.error("Invalid SMILES: Unbalanced brackets")
                return False
            
            # Check for exactly two [R] connection points
            if smiles.count('[R]') != 2:
                self.logger.error(f"Invalid SMILES: Found {smiles.count('[R]')} [R] connection points, need exactly 2")
                return False
            
            # Convert [R] to * for RDKit compatibility
            rdkit_smiles = smiles.replace('[R]', '*')
            
            # Validate with RDKit
            mol = Chem.MolFromSmiles(rdkit_smiles)
            if mol is None:
                self.logger.error("Invalid SMILES: Failed RDKit validation")
                return False
            
            # Check for valid chemistry
            for atom in mol.GetAtoms():
                # Check for invalid valences
                if atom.GetSymbol() != '*' and not atom.GetImplicitValence() >= 0:
                    self.logger.error(f"Invalid valence on atom {atom.GetIdx()}")
                    return False
                    
                # Check connection points
                if atom.GetSymbol() == '*':
                    # Connection points should have exactly one neighbor
                    if len(atom.GetNeighbors()) != 1:
                        self.logger.error(f"Connection point {atom.GetIdx()} has {len(atom.GetNeighbors())} neighbors, expected 1")
                        return False
                        
                    # Connection point neighbor should be carbon or oxygen
                    neighbor = atom.GetNeighbors()[0]
                    if neighbor.GetSymbol() not in ['C', 'O']:
                        self.logger.error(f"Connection point {atom.GetIdx()} connected to invalid atom type {neighbor.GetSymbol()}")
                        return False

            self.logger.info(f"SMILES validation successful: {smiles}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating SMILES: {str(e)}")
            return False

    def process_monomer(self, name: str, smiles: str) -> Optional[ProcessedMonomer]:
        """
        Process a single monomer with comprehensive error handling and structure generation.
        This is the main entry point for monomer processing.
        """
        try:
            self.logger.info(f"Starting processing of monomer: {name}")
            
            # Validate inputs
            if not name or not smiles:
                raise ValueError("Name and SMILES string are required")
                
            if not self.validate_smiles(smiles):
                raise ValueError("SMILES validation failed")
                
            # Convert [R] to * for RDKit compatibility
            rdkit_smiles = smiles.replace('[R]', '*')
            
            # Create initial RDKit mol object
            mol = Chem.MolFromSmiles(rdkit_smiles)
            if mol is None:
                raise ValueError("Failed to create initial RDKit molecule")
            
            # Process structure and get connection points
            clean_mol, connection_points = self._process_monomer_structure(mol)
            if clean_mol is None or len(connection_points) != 2:
                raise ValueError("Structure processing failed")
            
            # Generate optimized 3D structure
            clean_mol = self._generate_3d_structure(clean_mol)
            if clean_mol is None:
                raise ValueError("3D structure generation failed")
            
            # Calculate comprehensive properties
            properties = self._calculate_properties(clean_mol)
            if not properties:
                raise ValueError("Property calculation failed")
            
            # Create ProcessedMonomer instance
            processed = ProcessedMonomer(
                name=name,
                smiles=smiles,
                mol=mol,
                clean_mol=clean_mol,
                connection_points=connection_points,
                properties=properties
            )
            
            # Validate processed monomer
            if not processed.validate():
                raise ValueError("Processed monomer validation failed")
            
            # Save all output files
            if not self._save_monomer_files(processed):
                raise ValueError("Failed to save output files")
            
            # Store in processed monomers dictionary
            self.processed_monomers[name] = processed
            
            self.logger.info(f"Successfully processed monomer: {name}")
            return processed
            
        except Exception as e:
            self.logger.error(f"Error processing monomer {name}: {str(e)}")
            return None

    def _process_monomer_structure(self, mol: Chem.Mol) -> Tuple[Optional[Chem.Mol], List[int]]:
        """
        Process monomer structure with improved connection point handling.
        Returns cleaned molecule and connection point indices.
        """
        try:
            connection_points = []
            atoms_to_remove = []
            
            # Find connection points and their neighbors
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == '*':
                    neighbors = list(atom.GetNeighbors())
                    if len(neighbors) != 1:
                        raise ValueError(f"Invalid connection point: {atom.GetIdx()}")
                    connection_points.append(neighbors[0].GetIdx())
                    atoms_to_remove.append(atom.GetIdx())
            
            if len(connection_points) != 2:
                raise ValueError(f"Found {len(connection_points)} connection points, need exactly 2")
            
            # Create editable molecule and remove connection point atoms
            edit_mol = Chem.RWMol(mol)
            for idx in sorted(atoms_to_remove, reverse=True):
                edit_mol.RemoveAtom(idx)
            
            clean_mol = edit_mol.GetMol()
            
            # Sanitize and update properties
            Chem.SanitizeMol(clean_mol)
            clean_mol.UpdatePropertyCache(strict=False)
            
            # Update ring info and stereochemistry
            Chem.GetSSSR(clean_mol)
            Chem.AssignStereochemistry(clean_mol, cleanIt=True, force=True)
            Chem.SetAromaticity(clean_mol)
            
            return clean_mol, connection_points
            
        except Exception as e:
            self.logger.error(f"Error processing structure: {str(e)}")
            return None, []

    def _generate_3d_structure(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """
        Generate optimized 3D structure with improved conformer generation.
        Uses basic RDKit parameters for better compatibility.
        """
        try:
            # Initialize properties and add hydrogens
            mol = Chem.AddHs(mol)
            
            # Generate a single conformer first
            status = AllChem.EmbedMolecule(mol, 
                                        randomSeed=42,
                                        useRandomCoords=True,
                                        enforceChirality=True)
            
            if status == -1:
                raise ValueError("Initial conformer generation failed")
                
            # Energy minimize the structure
            AllChem.MMFFOptimizeMolecule(mol)
            
            # Generate additional conformers if needed
            if mol.GetNumAtoms() > 5:  # Only for larger molecules
                confs = AllChem.EmbedMultipleConfs(mol, 
                                                numConfs=10,
                                                randomSeed=42,
                                                clearConfs=False)
                
                # Optimize all conformers
                energies = []
                for conf in confs:
                    energy = AllChem.MMFFOptimizeMolecule(mol, confId=conf)
                    energies.append((energy, conf))
                
                # Select lowest energy conformer
                if energies:
                    energies.sort()
                    best_conf = energies[0][1]
                    
                    # Create new molecule with only the best conformer
                    best_mol = Chem.Mol(mol)
                    best_mol.RemoveAllConformers()
                    best_mol.AddConformer(mol.GetConformer(best_conf))
                    return best_mol
            
            return mol
            
        except Exception as e:
            self.logger.error(f"Error generating 3D structure: {str(e)}")
            return None

    def _calculate_properties(self, mol: Chem.Mol) -> Dict[str, Any]:
        """
        Calculate comprehensive molecular properties including 3D descriptors.
        Returns a dictionary of property names and values.
        """
        properties: Dict[str, Any] = {}
        try:
            # Basic molecular properties
            properties.update({
                'molecular_weight': Descriptors.ExactMolWt(mol),
                'heavy_atom_count': mol.GetNumHeavyAtoms(),
                'total_atom_count': mol.GetNumAtoms(),
                'ring_count': rdMolDescriptors.CalcNumRings(mol),
                'aromatic_ring_count': rdMolDescriptors.CalcNumAromaticRings(mol),
                'rotatable_bond_count': rdMolDescriptors.CalcNumRotatableBonds(mol),
                'hydrogen_bond_donors': rdMolDescriptors.CalcNumHBD(mol),
                'hydrogen_bond_acceptors': rdMolDescriptors.CalcNumHBA(mol),
                'topological_polar_surface_area': Descriptors.TPSA(mol),
                'formal_charge': rdMolDescriptors.CalcMolFormula(mol),
                'logp': Descriptors.MolLogP(mol),
                'number_of_stereoisomers': len(rdMolDescriptors.GetMorganFingerprint(mol, 2).GetNonzeroElements())
            })
            
            # Calculate 3D properties if conformer exists
            if mol.GetNumConformers() > 0:
                conf = mol.GetConformer()
                
                # Get atomic coordinates
                positions = []
                for i in range(mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(i)
                    positions.append([pos.x, pos.y, pos.z])
                positions = np.array(positions)
                
                # Calculate center of mass
                center = np.mean(positions, axis=0)
                
                # Calculate radius of gyration
                rg = np.sqrt(np.mean(np.sum((positions - center)**2, axis=1)))
                properties['radius_of_gyration'] = float(rg)
                
                # Calculate principal moments of inertia
                inertia_tensor = np.zeros((3, 3))
                for i in range(3):
                    for j in range(3):
                        inertia_tensor[i,j] = np.sum(
                            (positions[:,i] - center[i]) * (positions[:,j] - center[j])
                        )
                eigenvalues = np.linalg.eigvals(inertia_tensor)
                properties['principal_moments'] = eigenvalues.tolist()
                
                # Calculate molecular volume using grid-based approach
                try:
                    mol_volume = AllChem.ComputeMolVolume(mol, confId=0)
                    properties['molecular_volume'] = mol_volume
                except:
                    self.logger.warning("Failed to calculate molecular volume")
            
            return properties
            
        except Exception as e:
            self.logger.error(f"Error calculating properties: {str(e)}")
            return properties

    def _save_monomer_files(self, monomer: ProcessedMonomer) -> bool:
        """
        Save all monomer-related files with improved structure output.
        Generates MOL2, PDB, SMI files and 2D visualization.
        """
        try:
            # Create monomer-specific directory
            monomer_dir = self.output_dir / 'monomers' / monomer.name
            monomer_dir.mkdir(parents=True, exist_ok=True)
            
            # Save MOL file with 3D coordinates
            mol_path = monomer_dir / f"{monomer.name}.mol"
            Chem.MolToMolFile(monomer.clean_mol, str(mol_path), includeStereo=True)
            
            # Save PDB file with proper atom typing
            pdb_path = monomer_dir / f"{monomer.name}.pdb"
            Chem.MolToPDBFile(monomer.clean_mol, str(pdb_path))
            
            # Generate and save MOL2 file with proper atom typing
            mol2_success = self._save_mol2_file(monomer.clean_mol, 
                                              monomer_dir / f"{monomer.name}.mol2")
            if not mol2_success:
                self.logger.warning(f"Failed to generate MOL2 file for {monomer.name}")
            
            # Save SMILES
            with open(monomer_dir / f"{monomer.name}.smi", 'w') as f:
                f.write(f"{monomer.smiles}\n")
            
            # Save connection points
            with open(monomer_dir / f"{monomer.name}_connections.json", 'w') as f:
                json.dump({'connection_points': monomer.connection_points}, f, indent=2)
            
            # Save properties
            with open(monomer_dir / f"{monomer.name}_properties.json", 'w') as f:
                json.dump(monomer.properties, f, indent=2)
            
            # Generate and save 2D structure visualization
            vis_success = self._save_2d_structure(monomer.clean_mol, 
                                                monomer_dir / f"{monomer.name}_2d.png",
                                                highlight_atoms=monomer.connection_points)
            if not vis_success:
                self.logger.warning(f"Failed to generate 2D visualization for {monomer.name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving monomer files: {str(e)}")
            return False

    def _save_mol2_file(self, mol: Chem.Mol, path: Path) -> bool:
        """
        Generate MOL2 file with proper atom typing and 3D coordinates.
        Includes Gasteiger charge calculation and SYBYL atom typing.
        """
        try:
            # Ensure hydrogens are present
            mol = Chem.AddHs(mol)
            if not mol.GetNumConformers():
                raise ValueError("Molecule has no 3D coordinates")

            # Calculate Gasteiger charges
            AllChem.ComputeGasteigerCharges(mol)
            
            # Get conformer
            conf = mol.GetConformer()

            with open(path, 'w') as f:
                # Write header
                f.write("@<TRIPOS>MOLECULE\n")
                f.write(f"{path.stem}\n")
                f.write(f"{mol.GetNumAtoms()} {mol.GetNumBonds()} 0 0 0\n")
                f.write("SMALL\nUSER_CHARGES\n\n")

                # Write atom block
                f.write("@<TRIPOS>ATOM\n")
                for idx, atom in enumerate(mol.GetAtoms(), 1):
                    # Get position
                    pos = conf.GetAtomPosition(idx-1)
                    
                    # Get atom type
                    sybyl_type = self._get_sybyl_atom_type(atom)
                    
                    # Get charge
                    charge = atom.GetDoubleProp('_GasteigerCharge') if atom.HasProp('_GasteigerCharge') else 0.0
                    
                    # Write atom line
                    f.write(f"{idx:>7} {atom.GetSymbol()}{idx:<4} "
                           f"{pos.x:>9.4f} {pos.y:>9.4f} {pos.z:>9.4f} "
                           f"{sybyl_type:<5} 1 UNL {charge:>9.4f}\n")

                # Write bond block
                f.write("\n@<TRIPOS>BOND\n")
                for idx, bond in enumerate(mol.GetBonds(), 1):
                    # Determine bond type
                    if bond.GetIsAromatic():
                        bond_type = "ar"
                    else:
                        bond_type = str(int(bond.GetBondTypeAsDouble()))
                    
                    # Write bond line
                    f.write(f"{idx:>7} {bond.GetBeginAtomIdx()+1:>7} "
                           f"{bond.GetEndAtomIdx()+1:>7} {bond_type:>4}\n")

                # Write substructure block
                f.write("\n@<TRIPOS>SUBSTRUCTURE\n")
                f.write("     1 UNL     1 TEMP              0 ****  ****    0 ROOT\n")

            return True
            
        except Exception as e:
            self.logger.error(f"Error saving MOL2 file: {str(e)}")
            return False

    def _get_sybyl_atom_type(self, atom: Chem.Atom) -> str:
        """
        Determine SYBYL atom type based on detailed atomic environment.
        Considers hybridization, aromaticity, and bonding patterns.
        """
        # Get basic atomic properties
        symbol = atom.GetSymbol()
        hyb = atom.GetHybridization()
        is_aromatic = atom.GetIsAromatic()
        in_ring = atom.IsInRing()
        
        # Carbon types
        if symbol == 'C':
            if is_aromatic:
                return 'C.ar'
            elif hyb == Chem.HybridizationType.SP3:
                return 'C.3'
            elif hyb == Chem.HybridizationType.SP2:
                return 'C.2'
            elif hyb == Chem.HybridizationType.SP:
                return 'C.1'
            return 'C.3'  # Default
            
        # Nitrogen types
        elif symbol == 'N':
            if is_aromatic:
                return 'N.ar'
            elif hyb == Chem.HybridizationType.SP3:
                return 'N.3'
            elif hyb == Chem.HybridizationType.SP2:
                return 'N.2'
            elif hyb == Chem.HybridizationType.SP:
                return 'N.1'
            return 'N.3'
            
        # Oxygen types
        elif symbol == 'O':
            if is_aromatic:
                return 'O.ar'
            elif hyb == Chem.HybridizationType.SP3:
                return 'O.3'
            elif hyb == Chem.HybridizationType.SP2:
                return 'O.2'
            return 'O.3'
            
        # Sulfur types
        elif symbol == 'S':
            if is_aromatic:
                return 'S.ar'
            elif hyb == Chem.HybridizationType.SP3:
                return 'S.3'
            elif hyb == Chem.HybridizationType.SP2:
                return 'S.2'
            return 'S.3'
            
        # Phosphorus types
        elif symbol == 'P':
            if hyb == Chem.HybridizationType.SP3:
                return 'P.3'
            elif hyb == Chem.HybridizationType.SP2:
                return 'P.2'
            return 'P.3'
            
        # Hydrogen
        elif symbol == 'H':
            return 'H'
            
        # Default to atomic symbol
        return symbol


    def draw_2d_from_smiles(self, smiles: str, output_path: str, size: Tuple[int, int] = (400, 400)) -> bool:
        """
        Generate and save a 2D molecular structure drawing from SMILES string.
        
        Args:
            smiles: SMILES representation of the molecule
            output_path: Path where the PNG image will be saved
            size: Tuple of (width, height) for the image size
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Failed to create molecule from SMILES: {smiles}")
                return False
                
            # Compute 2D coordinates
            AllChem.Compute2DCoords(mol)
            
            # Set up drawing options
            d2d = Draw.rdDepictor.GetPreferredDepiction(mol)
            drawer = Draw.rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
            
            # Configure drawing options for clarity
            opts = drawer.drawOptions()
            opts.bondLineWidth = 2.0
            opts.minFontSize = 14
            opts.maxFontSize = 16
            opts.addAtomIndices = True
            
            # Draw and save
            img = Draw.MolToImage(mol, size=size)
            img.save(output_path)
            
            return True
        except Exception as e:
            print(f"Error generating 2D structure: {str(e)}")
            return False

    def _save_2d_structure(self, mol: Chem.Mol, path: Path, highlight_atoms: List[int] = None,
                            size: Tuple[int, int] = (400, 400)) -> bool:
        """
        Generate and save a 2D structure visualization as PNG image.
        
        Args:
            mol: RDKit molecule object
            path: Output file path (should be a Path object)
            highlight_atoms: List of atom indices to highlight
            size: Image dimensions (width, height)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Remove hydrogens for clearer visualization
            mol_2d = Chem.RemoveHs(Chem.Mol(mol))
            
            # Generate 2D coordinates
            AllChem.Compute2DCoords(mol_2d)
            
            # Create the drawing
            img = Draw.MolToImage(mol_2d, size=size)
            
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the image
            img.save(str(path))
            
            # Verify file was created
            if not path.exists():
                raise ValueError(f"Failed to create image file at {path}")
            
            self.logger.info(f"Successfully saved 2D structure to: {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving 2D structure: {str(e)}")
            return False

    def get_processed_monomer(self, name: str) -> Optional[ProcessedMonomer]:
        """
        Retrieve previously processed monomer data from disk.
        
        Args:
            name: Name of the monomer to retrieve
            
        Returns:
            ProcessedMonomer object if found, None otherwise
        """
        try:
            monomer_dir = self.output_dir / 'monomers' / name
            if not monomer_dir.exists():
                return None
                
            # Load SMILES
            smiles_path = monomer_dir / f"{name}.smi"
            if not smiles_path.exists():
                raise ValueError(f"SMILES file not found: {smiles_path}")
            with open(smiles_path) as f:
                smiles = f.read().strip()
            
            # Load connection points
            conn_path = monomer_dir / f"{name}_connections.json"
            if not conn_path.exists():
                raise ValueError(f"Connections file not found: {conn_path}")
            with open(conn_path) as f:
                connection_data = json.load(f)
            
            # Load properties
            prop_path = monomer_dir / f"{name}_properties.json"
            if not prop_path.exists():
                raise ValueError(f"Properties file not found: {prop_path}")
            with open(prop_path) as f:
                properties = json.load(f)
            
            # Load molecule from MOL file
            mol_path = monomer_dir / f"{name}.mol"
            if not mol_path.exists():
                raise ValueError(f"MOL file not found: {mol_path}")
            clean_mol = Chem.SDMolSupplier(str(mol_path), removeHs=False)[0]
            if clean_mol is None:
                raise ValueError(f"Failed to load MOL file: {mol_path}")
            
            # Create processed monomer
            processed = ProcessedMonomer(
                name=name,
                smiles=smiles,
                mol=Chem.MolFromSmiles(smiles.replace('[R]', '*')),
                clean_mol=clean_mol,
                connection_points=connection_data['connection_points'],
                properties=properties
            )
            
            # Validate
            if not processed.validate():
                raise ValueError(f"Validation failed for loaded monomer: {name}")
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Error retrieving processed monomer {name}: {str(e)}")
            return None



def process_monomer_file(file_path: str, output_dir: str = "output") -> bool:
    """
    Process a monomer definition file containing multiple monomers.
    
    Args:
        file_path: Path to the monomer definition file
        output_dir: Directory for output files
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        # Verify file exists
        file_path = Path(file_path)
        if not file_path.exists():
            logging.error(f"File not found: {file_path}")
            return False

        logging.info(f"Processing file: {file_path}")
        
        # Initialize the processor
        processor = MonomerProcessor(Path(output_dir))
        
        # Read and process the monomer file
        with open(file_path, 'r') as f:
            lines = f.readlines()
            logging.info(f"Read {len(lines)} lines from file")
            
            current_section = None
            processed_count = 0
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                    
                # Check for section markers
                if line.startswith('[') and line.endswith(']'):
                    current_section = line[1:-1].upper()
                    logging.info(f"Found section: {current_section}")
                    continue
                
                # Process monomer definitions
                if current_section != 'SEQUENCE' and ':' in line:
                    try:
                        name, smiles = [x.strip() for x in line.split(':')]
                        if name and smiles:
                            # Convert * to [R] for the processor
                            smiles = smiles.replace('*', '[R]')
                            logging.info(f"\nProcessing monomer: {name}")
                            logging.info(f"SMILES: {smiles}")
                            
                            # Process the monomer
                            result = processor.process_monomer(name, smiles)
                            if result:
                                processed_count += 1
                                logging.info(f"Successfully processed monomer: {name}")
                                logging.info(f"Connection points: {result.connection_points}")
                            else:
                                logging.error(f"Failed to process monomer: {name}")
                    except Exception as e:
                        logging.error(f"Error processing line {line_num}: {str(e)}")
            
            logging.info(f"\nProcessing complete. Successfully processed {processed_count} monomers")
            return processed_count > 0
            
    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        return False

def setup_logging(verbose: bool = False):
    """Configure logging with appropriate format and level"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def parse_arguments():
    """Parse command line arguments for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process monomer structures")
    parser.add_argument("-i", "--input", help="Input SMILES, monomer file, or directory", required=True)
    parser.add_argument("-n", "--name", help="Monomer name (required for single SMILES)", required=False)
    parser.add_argument("-o", "--output", help="Output directory", default="polymer_output")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--no-3d", action="store_true", help="Skip 3D structure generation")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    
    return parser.parse_args()

def main():
    """Enhanced main execution function supporting both single SMILES and file processing"""
    args = parse_arguments()
    setup_logging(args.verbose)
    
    # Set up output directory
    output_dir = Path(args.output)
    
    input_path = Path(args.input)
    
    # Check if input is a file
    if input_path.is_file():
        # Check if it's a monomer definition file
        if input_path.suffix.lower() in ['.txt', '.dat']:
            success = process_monomer_file(input_path, args.output)
            return 0 if success else 1
        
        # Assume it's a SMILES string if not a recognized file type
        if not args.name:
            print("Error: --name is required when processing a single SMILES")
            return 1
            
        with open(input_path) as f:
            smiles = f.read().strip()
    else:
        # Treat input as direct SMILES string
        if not args.name:
            print("Error: --name is required when processing a single SMILES")
            return 1
        smiles = args.input
    
    # Initialize processor
    processor = MonomerProcessor(output_dir)
    
    # Process single monomer if we have SMILES and name
    if args.name and smiles:
        processed = processor.process_monomer(args.name, smiles)
        if processed:
            print(f"\nSuccessfully processed monomer: {args.name}")
            print("\nProperties:")
            for key, value in processed.properties.items():
                print(f"  {key}: {value}")
            print(f"\nConnection points: {processed.connection_points}")
            print(f"\nOutput files saved in: {output_dir}/monomers/{args.name}/")
            return 0
    
    print(f"\nFailed to process monomer: {args.name}")
    return 1

if __name__ == "__main__":
    sys.exit(main())