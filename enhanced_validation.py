"""
Enhanced Polymer Validation System
--------------------------------
Provides comprehensive validation for polymer structures with improved error handling
and detailed structure verification.
"""

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
import numpy as np
import logging
from typing import Dict, Optional, Tuple

class EnhancedPolymerValidator:
    """Enhanced validation system for polymer structures"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_polymer(self, polymer: Chem.Mol) -> Tuple[bool, Optional[str]]:
        """
        Comprehensive polymer validation with detailed error reporting.
        
        Args:
            polymer: RDKit molecule object to validate
            
        Returns:
            Tuple of (success: bool, error_message: Optional[str])
        """
        try:
            if polymer is None:
                return False, "Polymer structure is None"
            
            # Basic structure validation
            if not self._validate_basic_structure(polymer):
                return False, "Basic structure validation failed"
            
            # Validate atom properties
            if not self._validate_atoms(polymer):
                return False, "Atom validation failed"
            
            # Validate bonds
            if not self._validate_bonds(polymer):
                return False, "Bond validation failed"
            
            # Validate 3D structure if present
            if polymer.GetNumConformers() > 0:
                if not self._validate_3d_structure(polymer):
                    return False, "3D structure validation failed"
            
            # Validate polymer connectivity
            if not self._validate_connectivity(polymer):
                return False, "Polymer connectivity validation failed"
            
            return True, None
            
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            return False, f"Validation error: {str(e)}"
    
    def _validate_basic_structure(self, mol: Chem.Mol) -> bool:
        """Validate basic molecular structure"""
        try:
            # Check for minimum structure requirements
            if mol.GetNumAtoms() < 2:
                self.logger.error("Too few atoms in structure")
                return False
            
            if mol.GetNumBonds() < 1:
                self.logger.error("No bonds in structure")
                return False
            
            # Verify molecular sanitization
            Chem.SanitizeMol(mol)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Basic structure validation failed: {e}")
            return False
    
    def _validate_atoms(self, mol: Chem.Mol) -> bool:
        """Validate atom properties and configurations"""
        try:
            for atom in mol.GetAtoms():
                # Check for invalid atomic numbers
                if atom.GetAtomicNum() <= 0:
                    self.logger.error(f"Invalid atomic number at atom {atom.GetIdx()}")
                    return False
                
                # Check for proper valence
                if not atom.GetImplicitValence() >= 0:
                    self.logger.error(f"Invalid valence at atom {atom.GetIdx()}")
                    return False
                
                # Check for proper formal charge
                if abs(atom.GetFormalCharge()) > 2:
                    self.logger.error(f"Unusual formal charge at atom {atom.GetIdx()}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Atom validation failed: {e}")
            return False
    
    def _validate_bonds(self, mol: Chem.Mol) -> bool:
        """Validate bond properties and configurations"""
        try:
            for bond in mol.GetBonds():
                # Check for invalid bond types
                if bond.GetBondType() not in [
                    Chem.rdchem.BondType.SINGLE,
                    Chem.rdchem.BondType.DOUBLE,
                    Chem.rdchem.BondType.TRIPLE,
                    Chem.rdchem.BondType.AROMATIC
                ]:
                    self.logger.error(f"Invalid bond type at bond {bond.GetIdx()}")
                    return False
                
                # Check for proper bond length if 3D structure exists
                if mol.GetNumConformers() > 0:
                    length = self._calculate_bond_length(mol, bond)
                    if length < 0.7 or length > 2.0:  # Angstroms
                        self.logger.error(f"Unusual bond length ({length:.2f} Å) at bond {bond.GetIdx()}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Bond validation failed: {e}")
            return False
    
    def _validate_3d_structure(self, mol: Chem.Mol) -> bool:
        """Validate 3D structural properties"""
        try:
            conf = mol.GetConformer()
            
            # Check for valid coordinates
            positions = []
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                
                # Check for invalid coordinates
                if any(abs(x) > 1000 for x in [pos.x, pos.y, pos.z]):
                    self.logger.error(f"Invalid coordinate range at atom {i}")
                    return False
                
                # Check for origin-centered coordinates
                if all(abs(x) < 1e-6 for x in [pos.x, pos.y, pos.z]):
                    self.logger.error(f"Zero coordinates at atom {i}")
                    return False
                
                positions.append([pos.x, pos.y, pos.z])
            
            # Check for atomic clashes
            positions = np.array(positions)
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < 0.5:  # Angstroms
                        self.logger.error(f"Atomic clash between atoms {i} and {j}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"3D structure validation failed: {e}")
            return False
    
    def _validate_connectivity(self, mol: Chem.Mol) -> bool:
        """Validate polymer chain connectivity"""
        try:
            # Check for disconnected fragments
            fragments = Chem.GetMolFrags(mol)
            if len(fragments) > 1:
                self.logger.error("Structure contains disconnected fragments")
                return False
            
            # Verify linear polymer chain
            for atom in mol.GetAtoms():
                # Terminal atoms should have exactly one neighbor
                if atom.GetDegree() == 1:
                    continue
                    
                # Internal atoms should typically have 2-4 neighbors
                if atom.GetDegree() > 4:
                    self.logger.error(f"Unusual connectivity at atom {atom.GetIdx()}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Connectivity validation failed: {e}")
            return False
    
    def _calculate_bond_length(self, mol: Chem.Mol, bond: Chem.Bond) -> float:
        """Calculate bond length between two atoms"""
        try:
            conf = mol.GetConformer()
            pos1 = conf.GetAtomPosition(bond.GetBeginAtomIdx())
            pos2 = conf.GetAtomPosition(bond.GetEndAtomIdx())
            
            return np.sqrt(
                (pos1.x - pos2.x)**2 +
                (pos1.y - pos2.y)**2 +
                (pos1.z - pos2.z)**2
            )
            
        except Exception:
            return 0.0


    def _validate_and_fix_valence(mol: Chem.Mol) -> Tuple[Chem.Mol, bool]:
        """Validate and fix valence issues in polymer structure"""
        try:
            # Create editable molecule
            rwmol = Chem.RWMol(mol)
            fixed = False

            # Check each atom's valence
            for atom in rwmol.GetAtoms():
                # Skip dummy atoms
                if atom.GetSymbol() == '*':
                    continue
                    
                # Get current valence
                explicit_valence = atom.GetExplicitValence()
                allowed_valence = atom.GetImplicitValence()
                
                # Handle common valence issues
                if explicit_valence > allowed_valence:
                    atomic_num = atom.GetAtomicNum()
                    
                    # Handle oxygen (common issue)
                    if atomic_num == 8:  # Oxygen
                        if explicit_valence > 2:
                            # Modify bonds to respect oxygen's valence
                            for bond in atom.GetBonds():
                                if bond.GetBondType() != Chem.BondType.SINGLE:
                                    bond.SetBondType(Chem.BondType.SINGLE)
                            fixed = True
                    
                    # Handle carbon
                    elif atomic_num == 6:  # Carbon
                        if explicit_valence > 4:
                            # Adjust bond orders to respect carbon's valence
                            bonds = list(atom.GetBonds())
                            bonds.sort(key=lambda x: x.GetBondTypeAsDouble(), reverse=True)
                            for bond in bonds:
                                if explicit_valence > 4:
                                    if bond.GetBondType() != Chem.BondType.SINGLE:
                                        bond.SetBondType(Chem.BondType.SINGLE)
                                        explicit_valence -= 1
                            fixed = True
                    
                    # Handle nitrogen
                    elif atomic_num == 7:  # Nitrogen
                        if explicit_valence > 3:
                            # Adjust bonds for nitrogen
                            for bond in atom.GetBonds():
                                if bond.GetBondType() != Chem.BondType.SINGLE:
                                    bond.SetBondType(Chem.BondType.SINGLE)
                            fixed = True

            # If fixes were made, update properties
            if fixed:
                try:
                    rwmol.UpdatePropertyCache(strict=False)
                    Chem.SanitizeMol(rwmol, 
                        sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS|
                                   Chem.SanitizeFlags.SANITIZE_KEKULIZE|
                                   Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|
                                   Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|
                                   Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|
                                   Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                        catchErrors=True)
                except Exception as e:
                    print(f"Warning: Minor issues in structure cleanup: {e}")

            return rwmol.GetMol(), fixed

        except Exception as e:
            print(f"Error in valence validation: {e}")
            return mol, False
             

def validate_polymer_structure(polymer: Chem.Mol) -> Tuple[bool, Optional[str]]:
    """
    Convenience function for polymer validation
    
    Args:
        polymer: RDKit molecule object to validate
        
    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    validator = EnhancedPolymerValidator()
    return validator.validate_polymer(polymer)
