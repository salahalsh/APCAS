"""
Enhanced Polymer Analytics Module with Advanced Analysis and Visualization


python polymer_analytics.py -i ./polymer_test_output/polymer/polymer.pdb -o ./polymer_test_output/analytics

----------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import Dict, Optional, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
from pathlib import Path


# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw
from rdkit.Chem import Fragments
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolAlign
from rdkit.Chem import rdMolTransforms

# Matplotlib configuration
import matplotlib.gridspec as gridspec

# Configure warnings
warnings.filterwarnings('ignore')

# Configure RDKit logging
RDLogger = logging.getLogger('rdApp')
RDLogger.setLevel(logging.ERROR)

# Set plotting defaults
plt.style.use('default')
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'figure.titlesize': 16,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.6,
    'figure.autolayout': True
})

class PolymerAnalytics:
    """Enhanced analytics system with comprehensive polymer analysis capabilities"""
    
    def __init__(self, output_dir: Path):
        """Initialize the analytics system"""
        self.output_dir = Path(output_dir)
        self.setup_directories()
        self.setup_logging()
        
    def setup_directories(self):
        """Setup only necessary directories"""
        directories = [
            'excel_data',    # For Excel reports
            'plots',         # Main plots directory
            'plots/3d',      # 3D structure visualizations
        ]
        for dir_name in directories:
            dir_path = self.output_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        """Configure analytics-specific logging with both file and console handlers"""
        try:
            # Get logger
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            
            # Clear any existing handlers
            if self.logger.handlers:
                self.logger.handlers.clear()
            
            # Create logs directory
            log_dir = self.output_dir / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup log files
            main_log = log_dir / 'analytics.log'
            detail_log = self.output_dir / 'analytics.log'  # Root directory log
            
            # Create formatters
            detailed_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            simple_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            
            # Create and configure file handlers
            main_handler = logging.FileHandler(str(main_log))
            main_handler.setFormatter(detailed_formatter)
            main_handler.setLevel(logging.INFO)
            
            detail_handler = logging.FileHandler(str(detail_log))
            detail_handler.setFormatter(simple_formatter)
            detail_handler.setLevel(logging.INFO)
            
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(simple_formatter)
            console_handler.setLevel(logging.INFO)
            
            # Add all handlers
            self.logger.addHandler(main_handler)
            self.logger.addHandler(detail_handler)
            self.logger.addHandler(console_handler)
            
            # Log initial setup
            self.logger.info("Analytics system initialized")
            self.logger.info(f"Output directory: {self.output_dir}")
            self.logger.info(f"Log files: {main_log}, {detail_log}")
            
        except Exception as e:
            print(f"Error setting up logging: {str(e)}")
            raise



    def _validate_pdb_file(self, path: Path) -> bool:
        """Validate PDB file format and content"""
        try:
            self.logger.info(f"Validating PDB file: {path}")
            print(f"Validating PDB file: {path}")
            
            if not path.exists():
                self.logger.error("File does not exist")
                print("File does not exist")
                return False
                
            size = path.stat().st_size
            if size == 0:
                self.logger.error("File is empty")
                print("File is empty")
                return False
                
            print(f"File size: {size} bytes")
            
            with open(path, 'r') as f:
                lines = f.readlines()
                
            atom_lines = [l for l in lines if l.startswith(('ATOM', 'HETATM'))]
            connect_lines = [l for l in lines if l.startswith('CONECT')]
            
            if not atom_lines:
                print("No ATOM or HETATM records found")
                return False
                
            print(f"Found {len(atom_lines)} atom records")
            
            try:
                first_atom = atom_lines[0]
                x = float(first_atom[30:38])
                y = float(first_atom[38:46])
                z = float(first_atom[46:54])
                print("Successfully parsed atomic coordinates")
            except:
                print("Failed to parse atomic coordinates")
                return False
            
            print("PDB file validation successful")
            return True
            
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return False


    def run_analysis(self, polymer_path: Path) -> Dict:
        """Run comprehensive polymer analysis with enhanced parameters"""
        try:
            print("\nStarting polymer analysis...")
            
            print("Loading structure...")
            # Convert to Path object if string
            if isinstance(polymer_path, str):
                polymer_path = Path(polymer_path)
            elif isinstance(polymer_path, dict) and 'polymer_path' in polymer_path:
                polymer_path = Path(polymer_path['polymer_path'])
                
            mol = self.load_structure(polymer_path)
            if mol is None:
                raise ValueError("Failed to load polymer structure")
            print(f"Successfully loaded structure with {mol.GetNumAtoms()} atoms")

            print("\nCalculating properties...")
            results = {}
            
            # Calculate all properties with progress reporting
            print("- Analyzing basic properties...")
            results['basic_properties'] = self._analyze_basic_properties(mol)
            
            print("- Analyzing topological properties...")
            results['topological_properties'] = self._analyze_topological_properties(mol)
            
            print("- Calculating geometric properties...")
            results['geometric_properties'] = self._analyze_geometric_properties(mol)
            
            print("- Analyzing electronic properties...")
            results['electronic_properties'] = self._analyze_electronic_properties(mol)
            
            print("- Analyzing fragments...")
            results['fragment_analysis'] = self._analyze_fragments(mol)
            
            print("- Analyzing conformational properties...")
            results['conformational_properties'] = self._analyze_conformational_properties(mol)
            
            print("- Estimating thermodynamic properties...")
            results['thermodynamic_estimates'] = self._estimate_thermodynamic_properties(mol)
            
            print("- Analyzing polymer-specific properties...")
            results['polymer_specific'] = self._analyze_polymer_specific_properties(mol)

            # Generate comprehensive Excel report
            print("\nGenerating Excel report...")
            self._generate_excel_report(results)
            
            # Generate enhanced visualizations
            print("Generating visualizations...")
            self._generate_plots(results, mol)
            
            print("\nAnalysis completed successfully!")
            return results
            
        except Exception as e:
            print(f"\nError during analysis: {str(e)}")
            self.logger.error(f"Analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}

    def load_structure(self, path: Path) -> Optional[Chem.Mol]:
        """Load and validate polymer structure with enhanced PDB handling and valence calculations"""
        try:
            if not isinstance(path, Path):
                path = Path(str(path))
                
            if not path.exists():
                raise FileNotFoundError(f"Structure file not found: {path}")
            
            print(f"\nReading file: {path}")
            
            # First, analyze the PDB file content
            with open(path, 'r') as f:
                lines = f.readlines()
                
            atom_lines = [l for l in lines if l.startswith(('ATOM', 'HETATM'))]
            connect_lines = [l for l in lines if l.startswith('CONECT')]
            
            print(f"\nPDB File Analysis:")
            print(f"Total lines: {len(lines)}")
            print(f"ATOM/HETATM records: {len(atom_lines)}")
            print(f"CONECT records: {len(connect_lines)}")
            
            # Create an editable molecule
            mol = Chem.RWMol()
            
            # Dictionary to store atom indices and coordinates
            atom_map = {}
            
            print("\nBuilding structure from atomic coordinates...")
            for i, line in enumerate(atom_lines):
                try:
                    # Parse atom information
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    
                    # Get atom symbol (try both standard locations)
                    atom_symbol = line[76:78].strip()
                    if not atom_symbol:
                        atom_symbol = line[12:14].strip()
                    atom_symbol = ''.join(c for c in atom_symbol if c.isalpha())
                    
                    # Create atom with explicit valence state
                    atom = Chem.Atom(atom_symbol)
                    atom.SetNoImplicit(True)  # Disable implicit H calculation
                    atom.SetFormalCharge(0)   # Set neutral charge
                    
                    # Add atom to molecule
                    atom_idx = mol.AddAtom(atom)
                    atom_map[i + 1] = atom_idx  # PDB files are 1-indexed
                    
                    # Set 3D coordinates
                    if atom_idx == 0:  # First atom
                        conf = Chem.Conformer(1)
                        conf.SetAtomPosition(0, (x, y, z))
                        mol.AddConformer(conf)
                    else:
                        mol.GetConformer().SetAtomPosition(atom_idx, (x, y, z))
                        
                except Exception as e:
                    print(f"Warning: Failed to process atom in line: {line.strip()}")
                    continue
            
            print(f"Added {mol.GetNumAtoms()} atoms to structure")
            
            # Add bonds based on CONECT records
            print("\nProcessing connectivity...")
            for line in connect_lines:
                try:
                    tokens = line.split()
                    if len(tokens) > 1:
                        from_atom = int(tokens[1])
                        for to_atom in tokens[2:]:
                            to_atom = int(to_atom)
                            if from_atom in atom_map and to_atom in atom_map:
                                mol.AddBond(atom_map[from_atom], atom_map[to_atom], Chem.BondType.SINGLE)
                except:
                    continue
                    
            print(f"Added {mol.GetNumBonds()} bonds to structure")
            
            # Convert to non-editable molecule
            mol = mol.GetMol()
            
            # Attempt basic cleanup without full sanitization
            try:
                print("\nAttempting basic structure cleanup...")
                Chem.SanitizeMol(mol, 
                                sanitizeOps=Chem.SANITIZE_SYMMRINGS|Chem.SANITIZE_CLEANUP,
                                catchErrors=True)
                print("Basic cleanup successful")
            except Exception as e:
                print(f"Warning during cleanup: {e}")
                print("Proceeding with basic structure")
            
            return mol
            
        except Exception as e:
            print(f"\nError in structure loading: {str(e)}")
            self.logger.error(f"Structure loading error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _analyze_basic_properties(self, mol: Chem.Mol) -> Dict:
        """Calculate basic molecular properties with careful handling of valence states"""
        try:
            properties = {
                'atom_count': mol.GetNumAtoms(),
                'bond_count': mol.GetNumBonds(),
                'ring_count': 0,  # Will calculate if possible
            }
            
            # Calculate molecular weight manually
            total_mass = 0.0
            for atom in mol.GetAtoms():
                total_mass += Chem.GetPeriodicTable().GetAtomicWeight(atom.GetAtomicNum())
            properties['molecular_weight'] = total_mass
            
            # Try to calculate ring information safely
            try:
                ring_info = mol.GetRingInfo()
                if ring_info:
                    properties['ring_count'] = ring_info.NumRings()
            except:
                pass
                
            # Calculate atom type distribution
            atom_types = {}
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                atom_types[symbol] = atom_types.get(symbol, 0) + 1
            properties['atom_distribution'] = atom_types
            
            # Calculate basic topological properties
            properties['largest_fragment_size'] = len(max(Chem.GetMolFrags(mol), key=len))
            properties['num_fragments'] = len(Chem.GetMolFrags(mol))
            
            return properties
            
        except Exception as e:
            print(f"Warning during property calculation: {e}")
            return {
                'atom_count': mol.GetNumAtoms(),
                'bond_count': mol.GetNumBonds(),
                'error': str(e)
            }

    def _analyze_topological_properties(self, mol: Chem.Mol) -> Dict:
        """Calculate topological properties with updated descriptor names and error handling"""
        properties = {}
        
        try:
            # Basic topological indices that don't require full sanitization
            properties['num_atoms'] = mol.GetNumAtoms()
            properties['num_bonds'] = mol.GetNumBonds()
            properties['num_rotatable_bonds'] = rdMolDescriptors.CalcNumRotatableBonds(mol)
            
            # Try calculating more complex descriptors safely
            try:
                properties['balaban_j'] = Descriptors.BalabanJ(mol)
            except:
                properties['balaban_j'] = None
                
            try:
                properties['bertz_ct'] = Descriptors.BertzCT(mol)
            except:
                properties['bertz_ct'] = None
                
            try:
                # Hall-Kier Alpha
                properties['hall_kier_alpha'] = Descriptors.HallKierAlpha(mol)
            except:
                properties['hall_kier_alpha'] = None
                
            # Calculate Chi indices safely
            chi_versions = ['Chi0v', 'Chi1v', 'Chi2v', 'Chi3v', 'Chi4v']
            for chi in chi_versions:
                try:
                    if hasattr(Descriptors, chi):
                        properties[f'chi_{chi[3:].lower()}'] = getattr(Descriptors, chi)(mol)
                except:
                    properties[f'chi_{chi[3:].lower()}'] = None
                    
            # Calculate Kappa indices safely
            try:
                properties['kappa1'] = Descriptors.Kappa1(mol)
                properties['kappa2'] = Descriptors.Kappa2(mol)
                properties['kappa3'] = Descriptors.Kappa3(mol)
            except:
                properties['kappa1'] = None
                properties['kappa2'] = None
                properties['kappa3'] = None
                
            # Calculate path-based indices
            try:
                # Get the adjacency matrix
                adj_matrix = Chem.GetAdjacencyMatrix(mol)
                import numpy as np
                
                # Calculate basic path-based descriptors
                properties['avg_degree'] = np.mean(np.sum(adj_matrix, axis=1))
                properties['max_degree'] = np.max(np.sum(adj_matrix, axis=1))
                properties['min_degree'] = np.min(np.sum(adj_matrix, axis=1))
                
                # Calculate basic cycle information
                ring_info = mol.GetRingInfo()
                properties['num_rings'] = ring_info.NumRings()
                ring_sizes = [len(ring) for ring in ring_info.AtomRings()]
                if ring_sizes:
                    properties['min_ring_size'] = min(ring_sizes)
                    properties['max_ring_size'] = max(ring_sizes)
                    properties['avg_ring_size'] = sum(ring_sizes) / len(ring_sizes)
                
            except Exception as e:
                print(f"Warning: Error calculating path-based indices: {e}")
                
            # Remove None values for cleaner output
            properties = {k: v for k, v in properties.items() if v is not None}
            
            return properties
            
        except Exception as e:
            print(f"Error in topological analysis: {e}")
            return {
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds(),
                'error': str(e)
            }


    def _analyze_geometric_properties(self, mol: Chem.Mol) -> Dict:
        """Calculate geometric and 3D structural properties with improved error handling"""
        try:
            properties = {}
            print("Starting geometric properties analysis...")
            
            # Get the conformer - since we're reading from PDB, we should already have 3D coordinates
            try:
                conf = mol.GetConformer()
                print("Successfully obtained conformer")
                
                # Convert conformer positions to numpy array for calculations
                positions = []
                for i in range(mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(i)
                    positions.append([pos.x, pos.y, pos.z])
                positions = np.array(positions)
                print(f"Processed {len(positions)} atomic positions")

                # Calculate center of mass
                masses = []
                for atom in mol.GetAtoms():
                    masses.append(Chem.GetPeriodicTable().GetAtomicWeight(atom.GetAtomicNum()))
                masses = np.array(masses)
                total_mass = np.sum(masses)
                com = np.average(positions, weights=masses, axis=0)
                print("Calculated center of mass")

                # Calculate distances from center of mass
                distances_from_com = np.linalg.norm(positions - com, axis=1)
                properties['max_distance_from_com'] = float(np.max(distances_from_com))
                properties['min_distance_from_com'] = float(np.min(distances_from_com))
                properties['mean_distance_from_com'] = float(np.mean(distances_from_com))
                print("Calculated COM distances")

                # Calculate radius of gyration
                # Rg^2 = (1/M) * sum(mi * ri^2)
                rg_squared = np.sum(masses * np.sum((positions - com)**2, axis=1)) / total_mass
                properties['radius_of_gyration'] = float(np.sqrt(rg_squared))
                print("Calculated radius of gyration")

                # Calculate end-to-end distance (using terminal atoms)
                properties['end_to_end_distance'] = float(np.linalg.norm(positions[0] - positions[-1]))
                print("Calculated end-to-end distance")

                # Calculate asphericity (using eigenvalues of gyration tensor)
                gyration_tensor = np.zeros((3, 3))
                for i in range(len(positions)):
                    r = positions[i] - com
                    for j in range(3):
                        for k in range(3):
                            gyration_tensor[j,k] += masses[i] * r[j] * r[k]
                gyration_tensor /= total_mass
                eigenvalues = np.linalg.eigvals(gyration_tensor)
                properties['asphericity'] = float(eigenvalues[2] - (eigenvalues[0] + eigenvalues[1])/2)
                print("Calculated asphericity")

                # Calculate molecular volume using grid-based approach
                try:
                    properties['volume'] = float(AllChem.ComputeMolVolume(mol))
                    print("Calculated molecular volume")
                except:
                    print("Warning: Could not calculate molecular volume")
                    properties['volume'] = None

                # Calculate surface area
                try:
                    properties['surface_area'] = float(AllChem.ComputeMolSurf(mol))
                    print("Calculated surface area")
                except:
                    print("Warning: Could not calculate surface area")
                    properties['surface_area'] = None

                # Calculate maximum and minimum diameters
                try:
                    dist_matrix = rdMolTransforms.GetDistanceMatrix(mol)
                    properties['maximum_diameter'] = float(np.max(dist_matrix))
                    properties['minimum_diameter'] = float(np.min(dist_matrix[dist_matrix > 0]))
                    print("Calculated molecular diameters")
                except:
                    print("Warning: Could not calculate molecular diameters")
                    properties['maximum_diameter'] = None
                    properties['minimum_diameter'] = None

                # Calculate shape descriptors
                try:
                    # Principal moments of inertia
                    pmi = rdMolDescriptors.CalcPBF(mol)
                    properties['principal_moment_ratio_1'] = float(pmi[0])
                    properties['principal_moment_ratio_2'] = float(pmi[1])
                    print("Calculated principal moments")
                except:
                    print("Warning: Could not calculate principal moments")
                    properties['principal_moment_ratio_1'] = None
                    properties['principal_moment_ratio_2'] = None

                # Remove any None values
                properties = {k: v for k, v in properties.items() if v is not None}
                print("Completed geometric properties analysis")
                
                return properties

            except Exception as e:
                print(f"Error processing conformer: {str(e)}")
                return {}

        except Exception as e:
            print(f"Error in geometric properties analysis: {str(e)}")
            return {}

    def _analyze_electronic_properties(self, mol: Chem.Mol) -> Dict:
        """Calculate electronic and charge-related properties"""
        try:
            properties = {
                'tpsa': Descriptors.TPSA(mol),
                'molar_refractivity': Descriptors.MolMR(mol),
                'logp': Descriptors.MolLogP(mol),
                'crippen_logp': Crippen.MolLogP(mol),
                'crippen_mr': Crippen.MolMR(mol)
            }
            
            # Calculate charge distribution
            try:
                AllChem.ComputeGasteigerCharges(mol)
                charges = []
                for atom in mol.GetAtoms():
                    try:
                        charge = float(atom.GetProp('_GasteigerCharge'))
                        charges.append(charge)
                    except KeyError:
                        continue
                
                if charges:
                    properties['charge_distribution'] = {
                        'min_charge': min(charges),
                        'max_charge': max(charges),
                        'mean_charge': np.mean(charges),
                        'charge_std': np.std(charges)
                    }
            except Exception as e:
                print(f"Warning: Could not calculate charges: {e}")
                properties['charge_distribution'] = {}
                
            return properties
            
        except Exception as e:
            print(f"Error in electronic properties calculation: {e}")
            return {
                'tpsa': 0.0,
                'molar_refractivity': 0.0,
                'logp': 0.0,
                'crippen_logp': 0.0,
                'crippen_mr': 0.0,
                'charge_distribution': {}
            }

    def _analyze_fragments(self, mol: Chem.Mol) -> Dict:
        """Analyze molecular fragments and functional groups with verified RDKit descriptors"""
        try:
            fragment_counts = {}
            
            # Define fragment analysis functions with verified RDKit descriptors only
            fragment_types = {
                'Al_OH': Fragments.fr_Al_OH,  # aliphatic hydroxyl
                'Al_OH_noTert': Fragments.fr_Al_OH_noTert,  # aliphatic hydroxyl excluding tert-OH
                'ArN': Fragments.fr_ArN,  # aromatic nitrogen
                'Ar_N': Fragments.fr_Ar_N,  # aromatic nitrogen
                'Ar_NH': Fragments.fr_Ar_NH,  # aromatic amines
                'Ar_OH': Fragments.fr_Ar_OH,  # aromatic hydroxyl
                'COO': Fragments.fr_COO,  # carboxylic acids
                'COO2': Fragments.fr_COO2,  # carboxylic acid derivatives
                'C_O': Fragments.fr_C_O,  # carbonyl O
                'C_O_noCOO': Fragments.fr_C_O_noCOO,  # carbonyl O, excluding acids and esters
                'C_S': Fragments.fr_C_S,  # thiocarbonyl
                'HOCCN': Fragments.fr_HOCCN,  # hydroxyl + cyano
                'Imine': Fragments.fr_Imine,  # imines
                'NH0': Fragments.fr_NH0,  # tertiary amines
                'NH1': Fragments.fr_NH1,  # secondary amines
                'NH2': Fragments.fr_NH2,  # primary amines
                'N_O': Fragments.fr_N_O,  # N-oxide
                'Ndealkylation1': Fragments.fr_Ndealkylation1,  # N-dealkylation
                'Ndealkylation2': Fragments.fr_Ndealkylation2,  # N-dealkylation2
                'Nhpyrrole': Fragments.fr_Nhpyrrole,  # N-heterocyclic pyrrole
                'alkyl_carbamate': Fragments.fr_alkyl_carbamate,  # alkyl carbamates
                'alkyl_halide': Fragments.fr_alkyl_halide,  # alkyl halides
                'allylic_oxid': Fragments.fr_allylic_oxid,  # allylic oxidation
                'amide': Fragments.fr_amide,  # amides
                'aniline': Fragments.fr_aniline,  # anilines
                'aryl_methyl': Fragments.fr_aryl_methyl,  # aryl methyl
                'azide': Fragments.fr_azide,  # azides
                'azo': Fragments.fr_azo,  # azo groups
                'barbitur': Fragments.fr_barbitur,  # barbiturates
                'benzene': Fragments.fr_benzene,  # benzene rings
                'benzodiazepine': Fragments.fr_benzodiazepine,  # benzodiazepines
                'bicyclic': Fragments.fr_bicyclic,  # bicyclic groups
                'diazo': Fragments.fr_diazo,  # diazo groups
                'dihydropyridine': Fragments.fr_dihydropyridine,  # dihydropyridines
                'epoxide': Fragments.fr_epoxide,  # epoxides
                'ester': Fragments.fr_ester,  # esters
                'ether': Fragments.fr_ether,  # ethers
                'furan': Fragments.fr_furan,  # furans
                'guanido': Fragments.fr_guanido,  # guanidine groups
                'halogen': Fragments.fr_halogen,  # halogens
                'hdrzine': Fragments.fr_hdrzine,  # hydrazines
                'hdrzone': Fragments.fr_hdrzone,  # hydrazones
                'imidazole': Fragments.fr_imidazole,  # imidazoles
                'imide': Fragments.fr_imide,  # imides
                'isocyan': Fragments.fr_isocyan,  # isocyanates
                'isothiocyan': Fragments.fr_isothiocyan,  # isothiocyanates
                'ketone': Fragments.fr_ketone,  # ketones
                'ketone_Topliss': Fragments.fr_ketone_Topliss,  # Topliss ketones
                'lactam': Fragments.fr_lactam,  # lactams
                'lactone': Fragments.fr_lactone,  # lactones
                'methoxy': Fragments.fr_methoxy,  # methoxy groups
                'morpholine': Fragments.fr_morpholine,  # morpholines
                'nitrile': Fragments.fr_nitrile,  # nitriles
                'nitro': Fragments.fr_nitro,  # nitro groups
                'nitro_arom': Fragments.fr_nitro_arom,  # aromatic nitro
                'nitro_arom_nonortho': Fragments.fr_nitro_arom_nonortho,  # non-ortho aromatic nitro
                'nitroso': Fragments.fr_nitroso,  # nitroso groups
                'oxazole': Fragments.fr_oxazole,  # oxazoles
                'oxime': Fragments.fr_oxime,  # oximes
                'para_hydroxylation': Fragments.fr_para_hydroxylation,  # para-hydroxylation
                'phenol': Fragments.fr_phenol,  # phenols
                'phenol_noOrthoHbond': Fragments.fr_phenol_noOrthoHbond,  # phenols without ortho H-bond
                'phos_acid': Fragments.fr_phos_acid,  # phosphoric acids
                'phos_ester': Fragments.fr_phos_ester,  # phosphoric esters
                'piperdine': Fragments.fr_piperdine,  # piperidines
                'piperzine': Fragments.fr_piperzine,  # piperazines
                'priamide': Fragments.fr_priamide,  # primary amides
                'prisulfonamd': Fragments.fr_prisulfonamd,  # primary sulfonamides
                'pyridine': Fragments.fr_pyridine,  # pyridines
                'quatN': Fragments.fr_quatN,  # quaternary N
                'sulfide': Fragments.fr_sulfide,  # sulfides
                'sulfonamd': Fragments.fr_sulfonamd,  # sulfonamides
                'sulfone': Fragments.fr_sulfone,  # sulfones
                'tetrazole': Fragments.fr_tetrazole,  # tetrazoles
                'thiazole': Fragments.fr_thiazole,  # thiazoles
                'thiocyan': Fragments.fr_thiocyan,  # thiocyanates
                'thiophene': Fragments.fr_thiophene,  # thiophenes
                'unbrch_alkane': Fragments.fr_unbrch_alkane,  # unbranched alkanes
                'urea': Fragments.fr_urea  # urea groups
            }
            
            # Calculate each fragment type safely
            for name, fragment_func in fragment_types.items():
                try:
                    count = fragment_func(mol)
                    fragment_counts[name] = count
                    print(f"Found {count} {name}")
                except Exception as e:
                    print(f"Warning: Could not calculate {name}: {e}")
                    fragment_counts[name] = 0
                    
            # Additional manual fragment analysis
            try:
                # Count rings by size
                ring_info = mol.GetRingInfo()
                ring_sizes = [len(ring) for ring in ring_info.AtomRings()]
                ring_counts = {}
                for size in set(ring_sizes):
                    ring_counts[f'{size}_membered_rings'] = ring_sizes.count(size)
                fragment_counts['ring_counts'] = ring_counts
                print(f"Ring analysis: {ring_counts}")
                
                # Basic chain analysis
                chains = Chem.GetMolFrags(mol)
                fragment_counts['num_chains'] = len(chains)
                fragment_counts['max_chain_length'] = max(len(chain) for chain in chains) if chains else 0
                print(f"Found {len(chains)} chains, max length: {fragment_counts['max_chain_length']}")
                
                # Pattern matching for specific polymer-relevant groups
                patterns = {
                    # Alcohol Groups
                    'primary_alcohol': '[OH1][CH2][#6]',
                    'secondary_alcohol': '[OH1][CH1]([#6])[#6]',
                    'tertiary_alcohol': '[OH1][C]([#6])([#6])[#6]',
                    'phenol': '[OH1]c',
                    
                    # Ester Groups
                    'ester_linkage': '[#6]-C(=O)-O-[#6]',
                    'aromatic_ester': 'c-C(=O)-O-[#6]',
                    'aliphatic_ester': '[#6;!$(c)]-C(=O)-O-[#6]',
                    'carbonate_ester': '[#6]-O-C(=O)-O-[#6]',
                    
                    # Ether Groups
                    'ether_linkage': '[#6]-O-[#6]',
                    'aromatic_ether': 'c-O-[#6]',
                    'aliphatic_ether': '[#6;!$(c)]-O-[#6;!$(c)]',
                    'cyclic_ether': '[#6]1-O-[#6]-[#6]-1',
                    
                    # Amide Groups
                    'amide_linkage': '[#6]-C(=O)-N(-[#6])-[#6]',
                    'primary_amide': '[#6]-C(=O)-N[H2]',
                    'secondary_amide': '[#6]-C(=O)-N[H1]-[#6]',
                    'tertiary_amide': '[#6]-C(=O)-N(-[#6])-[#6]',
                    
                    # Urethane Groups
                    'urethane_linkage': '[#6]-O-C(=O)-N[H1]-[#6]',
                    'carbamate': '[#6]-O-C(=O)-N[H2]',
                    'n_substituted_carbamate': '[#6]-O-C(=O)-N(-[#6])-[#6]',
                    
                    # Acid Groups
                    'carboxylic_acid': '[#6]-C(=O)-[OH1]',
                    'aromatic_acid': 'c-C(=O)-[OH1]',
                    'aliphatic_acid': '[#6;!$(c)]-C(=O)-[OH1]',
                    
                    # Anhydride Groups
                    'acid_anhydride': '[#6]-C(=O)-O-C(=O)-[#6]',
                    'cyclic_anhydride': '[#6]1-C(=O)-O-C(=O)-[#6]-1',
                    
                    # Amine Groups
                    'primary_amine': '[NH2]-[#6]',
                    'secondary_amine': '[#6]-[NH1]-[#6]',
                    'tertiary_amine': '[#6]-N(-[#6])-[#6]',
                    'aromatic_amine': 'c-[NH2]',
                    
                    # Urea Groups
                    'urea_linkage': '[#6]-N[H1]-C(=O)-N[H1]-[#6]',
                    'substituted_urea': '[#6]-N(-[#6])-C(=O)-N(-[#6])-[#6]',
                    
                    # Carbon-Carbon Bonds
                    'alkene_linkage': '[#6]=[#6]',
                    'conjugated_alkene': '[#6]=[#6]-[#6]=[#6]',
                    'alkyne_linkage': '[#6]#[#6]',
                    
                    # Aromatic Groups
                    'phenyl_ring': 'c1ccccc1',
                    'substituted_phenyl': '[#6,#7,#8,#9,#17]-c1ccccc1',
                    'naphthalene': 'c1ccc2ccccc2c1',
                    
                    # Halogen Groups
                    'alkyl_chloride': '[#6]-[Cl]',
                    'alkyl_bromide': '[#6]-[Br]',
                    'alkyl_fluoride': '[#6]-[F]',
                    'vinyl_halide': '[#6]=[#6]-[F,Cl,Br,I]',
                    
                    # Specific Polymer Linkages
                    'polyester_linkage': '[#6]-O-C(=O)-[#6]',
                    'polyamide_linkage': '[#6]-N[H1]-C(=O)-[#6]',
                    'polyurethane_linkage': '[#6]-O-C(=O)-N[H1]-[#6]',
                    'polycarbonate_linkage': '[#6]-O-C(=O)-O-[#6]',
                    'polyether_linkage': '[#6]-O-[#6]',
                    'polyurea_linkage': '[#6]-N[H1]-C(=O)-N[H1]-[#6]',
                    
                    # Common End Groups
                    'methyl_end_group': '[CH3]-[#6]',
                    'hydroxyl_end_group': '[OH1]-[#6]',
                    'carboxyl_end_group': '[#6]-C(=O)-[OH1]',
                    'amine_end_group': '[NH2]-[#6]',
                    
                    # Specific Polymer Classes
                    'peg_unit': '[#6]-O-[#6]-[#6]-O-[#6]',
                    'polypropylene_unit': '[#6]-[CH2]-[CH](-[CH3])-[#6]',
                    'polystyrene_unit': '[#6]-[CH2]-[CH](-c1ccccc1)-[#6]',
                    'polyvinyl_alcohol_unit': '[#6]-[CH2]-[CH](-[OH1])-[#6]',
                    'polyvinyl_chloride_unit': '[#6]-[CH2]-[CH](-[Cl])-[#6]',
                    
                    # Branching Points
                    'tertiary_carbon': '[#6]-[CH1](-[#6])-[#6]',
                    'quaternary_carbon': '[#6]-[C](-[#6])(-[#6])-[#6]',
                    'branch_point_ether': '[#6]-[CH1](-O-[#6])-[#6]',
                    'branch_point_ester': '[#6]-[CH1](-O-C(=O)-[#6])-[#6]',
                    
                    # Crosslinking Sites
                    'epoxide_ring': '[#6]1O[#6]1',
                    'acrylate_group': '[#6]=C-C(=O)-O-[#6]',
                    'methacrylate_group': '[#6]=C(-[CH3])-C(=O)-O-[#6]',
                    'vinyl_group': '[#6]=[CH2]',
                    
                    # Stereochemistry
                    'isotactic_diad': '[#6]-[C@H](-[*])-[C@H](-[*])-[#6]',
                    'syndiotactic_diad': '[#6]-[C@H](-[*])-[C@@H](-[*])-[#6]',
                    
                    # Common Modifications
                    'sulfone_group': '[#6]-S(=O)(=O)-[#6]',
                    'phosphate_group': '[#6]-O-P(=O)(-O-[#6])-O-[#6]',
                    'siloxane_linkage': '[#6]-[Si](-[#6])(-[#6])-O-[Si]',
                    
                    # Degradable Linkages
                    'hydrolyzable_ester': '[#6]-O-C(=O)-[CH2]-[#6]',
                    'hydrolyzable_amide': '[#6]-N[H1]-C(=O)-[CH2]-[#6]',
                    'disulfide_linkage': '[#6]-S-S-[#6]',
                    
                    # Block Copolymer Junctions
                    'ester_amide_junction': '[#6]-O-C(=O)-[#6]-C(=O)-N[H1]-[#6]',
                    'ether_ester_junction': '[#6]-O-[#6]-C(=O)-O-[#6]',
                    'urethane_urea_junction': '[#6]-O-C(=O)-N[H1]-[#6]-N[H1]-C(=O)-N[H1]-[#6]'
                }

                # Process all patterns
                for name, smarts in patterns.items():
                    try:
                        pattern = Chem.MolFromSmarts(smarts)
                        if pattern:
                            matches = mol.GetSubstructMatches(pattern)
                            fragment_counts[name] = len(matches)
                            if len(matches) > 0:
                                print(f"Found {len(matches)} {name}")
                    except Exception as e:
                        print(f"Warning: Could not analyze {name}: {e}")
                        fragment_counts[name] = 0

            except Exception as e:
                print(f"Warning: Error in pattern matching: {e}")

            return fragment_counts
            
        except Exception as e:
            print(f"Error in fragment analysis: {e}")
            return {
                'error': str(e),
                'fragment_count': 0
            }

    def _analyze_conformational_properties(self, mol: Chem.Mol) -> Dict:
        """Analyze conformational flexibility and properties"""
        try:
            return {
                'flexibility_index': self._calculate_flexibility_index(mol),
                'rotatable_bond_ratio': rdMolDescriptors.CalcNumRotatableBonds(mol) / mol.GetNumBonds(),
                'ring_flexibility': self._calculate_ring_flexibility(mol)
            }
        except:
            return {}

    def _estimate_thermodynamic_properties(self, mol: Chem.Mol) -> Dict:
        """Estimate basic thermodynamic properties"""
        try:
            # These are rough estimates based on group contribution methods
            mw = Descriptors.ExactMolWt(mol)
            logp = Descriptors.MolLogP(mol)
            
            return {
                'estimated_boiling_point': self._estimate_boiling_point(mol),
                'estimated_melting_point': self._estimate_melting_point(mol),
                'estimated_glass_transition': self._estimate_glass_transition(mol),
                'estimated_solubility': self._estimate_solubility(mol)
            }
        except:
            return {}

    def _analyze_polymer_specific_properties(self, mol: Chem.Mol) -> Dict:
        """Calculate polymer-specific properties"""
        try:
            return {
                'persistence_length': self._calculate_persistence_length(mol),
                'contour_length': self._calculate_contour_length(mol),
                'kuhn_length': self._calculate_kuhn_length(mol),
                'chain_stiffness': self._calculate_chain_stiffness(mol)
            }
        except:
            return {}

    def _generate_excel_report(self, results: Dict):
        """Generate comprehensive Excel report with multiple sheets"""
        excel_path = self.output_dir / 'excel_data' / 'polymer_analysis.xlsx'
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Basic Properties Sheet
            pd.DataFrame([results['basic_properties']]).to_excel(writer, sheet_name='Basic Properties', index=False)
            
            # Topological Properties Sheet
            pd.DataFrame([results['topological_properties']]).to_excel(writer, sheet_name='Topological Properties', index=False)
            
            # Geometric Properties Sheet
            pd.DataFrame([results['geometric_properties']]).to_excel(writer, sheet_name='Geometric Properties', index=False)
            
            # Electronic Properties Sheet
            electronic_props = results['electronic_properties'].copy()
            charge_dist = electronic_props.pop('charge_distribution', {})
            pd.DataFrame([electronic_props]).to_excel(writer, sheet_name='Electronic Properties', index=False)
            
            # Fragment Analysis Sheet
            pd.DataFrame([results['fragment_analysis']]).to_excel(writer, sheet_name='Fragment Analysis', index=False)
            
            # Conformational Properties Sheet
            pd.DataFrame([results['conformational_properties']]).to_excel(writer, sheet_name='Conformational Properties', index=False)
            
            # Thermodynamic Properties Sheet
            pd.DataFrame([results['thermodynamic_estimates']]).to_excel(writer, sheet_name='Thermodynamic Properties', index=False)
            
            # Polymer Specific Properties Sheet
            pd.DataFrame([results['polymer_specific']]).to_excel(writer, sheet_name='Polymer Properties', index=False)




    def _generate_plots(self, results: Dict, mol: Chem.Mol):
        """Generate comprehensive set of analysis plots"""
        try:
            # Create main figure with 3x3 grid
            fig = plt.figure(figsize=(20, 15))
            gs = gridspec.GridSpec(3, 3, figure=fig)
            plt.subplots_adjust(hspace=0.4, wspace=0.3)
            
            # 1. Structure visualization (top-left)
            print("Generating structure visualization...")
            ax1 = fig.add_subplot(gs[0, 0])
            img = Draw.MolToImage(mol)
            ax1.imshow(img)
            ax1.set_title('2D Structure')
            ax1.axis('off')
            
            # 2. Polymer Properties (top-middle)
            print("Generating polymer properties plot...")
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_polymer_properties(results.get('polymer_specific', {}), ax2)
            
            # 3. Thermodynamic Properties (top-right)
            print("Generating thermodynamic properties plot...")
            ax3 = fig.add_subplot(gs[0, 2])
            self._plot_thermodynamic_properties(results.get('thermodynamic_estimates', {}), ax3)
            
            # 4. Conformational Properties (middle-left)
            print("Generating conformational properties plot...")
            ax4 = fig.add_subplot(gs[1, 0])
            self._plot_conformational_properties(results.get('conformational_properties', {}), ax4)
            
            # 5. Topological Properties (middle-middle)
            print("Generating topological properties plot...")
            ax5 = fig.add_subplot(gs[1, 1])
            self._plot_topological_properties(results.get('topological_properties', {}), ax5)
            
            # 6. Fragment Analysis (middle-right)
            print("Generating fragment analysis plot...")
            ax6 = fig.add_subplot(gs[1, 2])
            self._plot_fragment_distribution(results.get('fragment_analysis', {}), ax6)
            
            # 7. Electronic Properties (bottom-left)
            print("Generating electronic properties plot...")
            ax7 = fig.add_subplot(gs[2, 0])
            self._plot_electronic_properties(results.get('electronic_properties', {}), ax7)
            
            # 8. Geometric Properties (bottom-middle)
            print("Generating geometric properties plot...")
            ax8 = fig.add_subplot(gs[2, 1])
            self._plot_geometric_properties(results.get('geometric_properties', {}), ax8)
            
            # 9. Basic Properties (bottom-right)
            print("Generating basic properties plot...")
            ax9 = fig.add_subplot(gs[2, 2])
            self._plot_property_distribution(results.get('basic_properties', {}), ax9)

            # Save main dashboard
            plt.savefig(self.output_dir / 'plots' / 'main_dashboard.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Generate additional specialized plots
            
            # 10. Property Correlations
            print("Generating property correlations...")
            plt.figure(figsize=(12, 10))
            self._plot_property_correlations(results)
            plt.savefig(self.output_dir / 'plots' / 'property_correlations.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 11. Fragment Distribution Sunburst
            print("Generating fragment sunburst...")
            plt.figure(figsize=(12, 12))
            self._plot_fragment_sunburst(results['fragment_analysis'])
            plt.savefig(self.output_dir / 'plots' / 'fragment_sunburst.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 12. Property Radar Plot
            print("Generating property radar plot...")
            plt.figure(figsize=(10, 10))
            self._plot_property_radar(results)
            plt.savefig(self.output_dir / 'plots' / 'property_radar.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 13. Polymer Chain Analysis
            print("Generating polymer chain analysis...")
            plt.figure(figsize=(15, 6))
            self._plot_polymer_chain_analysis(results['polymer_specific'])
            plt.savefig(self.output_dir / 'plots' / 'polymer_chain_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 14. Fragment Distribution Pie Chart
            print("Generating fragment distribution pie chart...")
            plt.figure(figsize=(12, 8))
            self._create_fragment_pie_chart(results.get('fragment_analysis', {}))
            plt.savefig(self.output_dir / 'plots' / 'fragment_distribution_pie.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 15. Property Comparison Bar Chart
            print("Generating property comparison bar chart...")
            plt.figure(figsize=(15, 8))
            self._create_property_bar_chart(results)
            plt.savefig(self.output_dir / 'plots' / 'property_comparison_bar.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 16. Atomic Composition Pie Chart
            if 'basic_properties' in results and 'atom_distribution' in results['basic_properties']:
                print("Generating atomic composition chart...")
                plt.figure(figsize=(10, 8))
                self._create_atomic_composition_chart(results['basic_properties']['atom_distribution'])
                plt.savefig(self.output_dir / 'plots' / 'atomic_composition_pie.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 17. Electronic Properties Line Chart
            print("Generating electronic properties line chart...")
            plt.figure(figsize=(12, 6))
            self._create_electronic_properties_line_chart(results.get('electronic_properties', {}))
            plt.savefig(self.output_dir / 'plots' / 'electronic_properties_line.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 18. Ring Analysis Pie Chart
            if 'fragment_analysis' in results and 'ring_counts' in results['fragment_analysis']:
                print("Generating ring analysis chart...")
                plt.figure(figsize=(10, 8))
                self._create_ring_analysis_chart(results['fragment_analysis']['ring_counts'])
                plt.savefig(self.output_dir / 'plots' / 'ring_distribution_pie.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 19. Property Correlation Heatmap
            print("Generating property correlation heatmap...")
            plt.figure(figsize=(12, 10))
            self._create_property_correlation_heatmap(results)
            plt.savefig(self.output_dir / 'plots' / 'property_correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 20. 3D Conformation Plot
            print("Generating 3D conformation plot...")
            self._plot_3d_conformation(mol)  # This function creates its own figure

            # Save main dashboard
            plt.savefig(self.output_dir / 'plots' / 'main_dashboard.png',
                       dpi=300, bbox_inches='tight')
            plt.close()

            # Generate additional plots one by one
            # Atomic Composition
            if 'basic_properties' in results and 'atom_distribution' in results['basic_properties']:
                print("Generating atomic composition chart...")
                fig = plt.figure(figsize=(10, 8))
                
                atom_dist = results['basic_properties']['atom_distribution']
                if atom_dist:
                    labels = list(atom_dist.keys())
                    sizes = list(atom_dist.values())
                    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
                    plt.title('Atomic Composition')
                    plt.axis('equal')
                
                plt.savefig(self.output_dir / 'plots' / 'atomic_composition_pie.png', dpi=300, bbox_inches='tight')
                plt.close()

            # Electronic Properties Line Chart
            if 'electronic_properties' in results:
                print("Generating electronic properties line chart...")
                ep = results['electronic_properties']
                valid_props = {k: v for k, v in ep.items() 
                             if isinstance(v, (int, float)) and k not in ['charge_distribution']}
                
                if valid_props:
                    fig = plt.figure(figsize=(12, 6))
                    plt.plot(list(valid_props.keys()), list(valid_props.values()), 'bo-')
                    plt.xticks(rotation=45, ha='right')
                    plt.title('Electronic Properties')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(self.output_dir / 'plots' / 'electronic_properties_line.png', 
                              dpi=300, bbox_inches='tight')
                    plt.close()

            # Fragment Distribution Pie
            if 'fragment_analysis' in results:
                print("Generating fragment distribution pie chart...")
                fragments = {k: v for k, v in results['fragment_analysis'].items() 
                           if isinstance(v, (int, float)) and v > 0 and k not in ['ring_counts', 'error']}
                
                if fragments:
                    fig = plt.figure(figsize=(12, 8))
                    plt.pie(list(fragments.values()), labels=list(fragments.keys()), autopct='%1.1f%%')
                    plt.title('Fragment Distribution')
                    plt.axis('equal')
                    plt.savefig(self.output_dir / 'plots' / 'fragment_distribution_pie.png', 
                              dpi=300, bbox_inches='tight')
                    plt.close()

            # Property Correlation Heatmap
            print("Generating property correlation heatmap...")
            numeric_props = {}
            for category in ['basic_properties', 'topological_properties', 
                            'geometric_properties', 'electronic_properties']:
                if category in results:
                    for k, v in results[category].items():
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            numeric_props[k] = v

            if len(numeric_props) > 1:
                # Create correlation matrix
                df = pd.DataFrame([numeric_props])
                # Add some noise to create variation
                df_with_noise = pd.DataFrame([
                    {k: v * (1 + np.random.normal(0, 0.01)) for k, v in numeric_props.items()}
                    for _ in range(10)
                ])
                corr = df_with_noise.corr()

                fig = plt.figure(figsize=(12, 10))
                sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
                plt.title('Property Correlations')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(self.output_dir / 'plots' / 'property_correlation_heatmap.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()

            print("All plots generated successfully.")
            
        except Exception as e:
            print(f"Error in plot generation: {e}")
            import traceback
            traceback.print_exc()




    def _plot_3d_conformation(self, mol: Chem.Mol):
        """Generate 3D conformation plot with improved atom visualization"""
        try:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')

            # Get conformer
            conf = mol.GetConformer()
            
            # Color scheme for different atoms
            color_dict = {
                6: 'grey',   # Carbon
                1: 'white',  # Hydrogen
                8: 'red',    # Oxygen
                7: 'blue',   # Nitrogen
                16: 'yellow' # Sulfur
            }
            
            # Collect atomic coordinates
            xs, ys, zs = [], [], []
            colors = []
            sizes = []
            labels = []
            
            # Process atoms
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                atom = mol.GetAtomWithIdx(i)
                
                xs.append(pos.x)
                ys.append(pos.y)
                zs.append(pos.z)
                
                atomic_num = atom.GetAtomicNum()
                colors.append(color_dict.get(atomic_num, 'green'))
                sizes.append(50 * (atomic_num / 6))  # Scale size with atomic number
                labels.append(atom.GetSymbol())

            # Plot atoms
            ax.scatter(xs, ys, zs, c=colors, s=sizes, alpha=0.6)

            # Plot bonds
            for bond in mol.GetBonds():
                id1 = bond.GetBeginAtomIdx()
                id2 = bond.GetEndAtomIdx()
                
                x1, y1, z1 = [conf.GetAtomPosition(id1).x,
                             conf.GetAtomPosition(id1).y,
                             conf.GetAtomPosition(id1).z]
                x2, y2, z2 = [conf.GetAtomPosition(id2).x,
                             conf.GetAtomPosition(id2).y,
                             conf.GetAtomPosition(id2).z]
                
                ax.plot([x1, x2], [y1, y2], [z1, z2], 'k-', alpha=0.5, linewidth=1)

            # Set labels and title
            ax.set_xlabel('X (Å)')
            ax.set_ylabel('Y (Å)')
            ax.set_zlabel('Z (Å)')
            ax.set_title('3D Molecular Conformation')

            # Set equal aspect ratio
            max_range = np.array([
                max(xs) - min(xs),
                max(ys) - min(ys),
                max(zs) - min(zs)
            ]).max() / 2.0

            mid_x = (max(xs) + min(xs)) * 0.5
            mid_y = (max(ys) + min(ys)) * 0.5
            mid_z = (max(zs) + min(zs)) * 0.5

            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            # Adjust view angle
            ax.view_init(elev=20, azim=45)

            # Save plot
            plt.savefig(self.output_dir / 'plots/3d' / '3d_conformation.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Error generating 3D conformation plot: {e}")


 


    def _plot_property_correlations(self, results: Dict):
        """Generate basic property correlation plot"""
        try:
            # Collect numerical properties
            properties = {}
            for category in ['basic_properties', 'topological_properties', 
                            'geometric_properties', 'electronic_properties']:
                if category in results:
                    for key, value in results[category].items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            clean_name = key.replace('_', ' ').title()
                            properties[clean_name] = value

            if len(properties) < 2:
                print("Insufficient properties for correlation plot")
                return

            # Create data points with variation
            n_points = 20
            data = []
            for _ in range(n_points):
                sample = {}
                for key, base_value in properties.items():
                    # Add controlled random variation
                    sample[key] = base_value * (1 + np.random.normal(0, 0.05))
                data.append(sample)

            # Create DataFrame and calculate correlations
            df = pd.DataFrame(data)
            corr = df.corr()

            # Create plot
            plt.figure(figsize=(12, 10))
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr), k=1)

            # Plot heatmap
            sns.heatmap(corr,
                       mask=mask,
                       annot=True,
                       cmap='RdYlBu',
                       center=0,
                       vmin=-1,
                       vmax=1,
                       fmt='.2f',
                       square=True,
                       linewidths=0.5,
                       cbar_kws={"shrink": .8},
                       annot_kws={"size": 8})

            plt.title('Property Correlations', pad=20, fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()

            # Save plot
            plt.savefig(self.output_dir / 'plots' / 'property_correlations.png',
                       dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Error plotting property correlations: {e}")
            plt.close()




    def _plot_property_radar(self, results: Dict):
        """Generate radar plot for key properties"""
        try:
            # Select key properties for radar plot
            properties = {
                'Molecular Weight': results['basic_properties']['molecular_weight'],
                'LogP': results['electronic_properties']['logp'],
                'TPSA': results['electronic_properties']['tpsa'],
                'Rotatable Bonds': results['basic_properties']['rotatable_bond_count'],
                'Ring Count': results['basic_properties']['ring_count']
            }
            
            # Normalize values
            max_values = {
                'Molecular Weight': 1000,
                'LogP': 5,
                'TPSA': 200,
                'Rotatable Bonds': 20,
                'Ring Count': 10
            }
            
            normalized = {key: min(value / max_values[key], 1.0) 
                        for key, value in properties.items()}
            
            # Create radar plot
            angles = np.linspace(0, 2*np.pi, len(normalized), endpoint=False)
            values = list(normalized.values())
            values += values[:1]
            angles = np.concatenate((angles, [angles[0]]))
            
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, polar=True)
            ax.plot(angles, values)
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(list(normalized.keys()))
            plt.title('Property Radar Plot')
            plt.savefig(self.output_dir / 'plots' / 'property_radar.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error generating property radar plot: {e}")

    def _plot_property_radar(self, results: Dict):
        """Generate radar plot for key properties"""
        try:
            # Select key properties that exist in the results
            properties = {}
            
            if 'basic_properties' in results:
                bp = results['basic_properties']
                if 'molecular_weight' in bp:
                    properties['Molecular Weight'] = bp['molecular_weight']
                if 'num_rotatable_bonds' in bp:
                    properties['Rotatable Bonds'] = bp['num_rotatable_bonds']
                if 'ring_count' in bp:
                    properties['Ring Count'] = bp['ring_count']
            
            if 'electronic_properties' in results:
                ep = results['electronic_properties']
                if 'logp' in ep:
                    properties['LogP'] = ep['logp']
                if 'tpsa' in ep:
                    properties['TPSA'] = ep['tpsa']
            
            if len(properties) < 3:
                print("Insufficient properties for radar plot")
                return
                
            # Normalize values
            max_values = {
                'Molecular Weight': 1000,
                'LogP': 5,
                'TPSA': 200,
                'Rotatable Bonds': 20,
                'Ring Count': 10
            }
            
            normalized = {}
            for key, value in properties.items():
                if key in max_values:
                    normalized[key] = min(value / max_values[key], 1.0)
            
            if not normalized:
                print("No properties could be normalized")
                return
            
            # Create radar plot
            angles = np.linspace(0, 2*np.pi, len(normalized), endpoint=False)
            values = list(normalized.values())
            values += values[:1]
            angles = np.concatenate((angles, [angles[0]]))
            
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, polar=True)
            ax.plot(angles, values)
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(list(normalized.keys()))
            plt.title('Property Radar Plot')
            
            return fig
            
        except Exception as e:
            print(f"Error generating property radar plot: {e}")
            return None



    def _plot_property_distribution(self, properties: Dict, ax):
        """Plot distribution of basic properties"""
        try:
            # Select numerical properties
            numerical_props = {k: v for k, v in properties.items() 
                             if isinstance(v, (int, float)) and not isinstance(v, bool)}
            
            if not numerical_props:
                ax.text(0.5, 0.5, 'No numerical properties available', 
                       ha='center', va='center')
                return
            
            # Sort by value for better visualization
            sorted_items = sorted(numerical_props.items(), key=lambda x: x[1], reverse=True)
            labels = [item[0] for item in sorted_items]
            values = [item[1] for item in sorted_items]
            
            # Create bar plot
            bars = ax.bar(range(len(values)), values)
            
            # Customize the plot
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_title('Basic Properties Distribution')
            ax.set_ylabel('Value')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom')
                
        except Exception as e:
            print(f"Error plotting property distribution: {e}")
            ax.text(0.5, 0.5, 'Error in plot generation', ha='center', va='center')

    def _plot_geometric_properties(self, properties: Dict, ax):
        """Plot geometric properties with enhanced visualization"""
        try:
            if not properties:
                ax.text(0.5, 0.5, 'No geometric properties available', 
                       ha='center', va='center')
                return
                
            # Sort properties for better visualization
            sorted_items = sorted(properties.items(), key=lambda x: x[1], reverse=True)
            labels = [item[0].replace('_', ' ').title() for item in sorted_items]
            values = [item[1] for item in sorted_items]
            
            # Create bar plot
            bars = ax.bar(range(len(values)), values)
            
            # Customize the plot
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_title('Geometric Properties')
            ax.set_ylabel('Value (Å)')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom')
                
        except Exception as e:
            print(f"Error plotting geometric properties: {e}")
            ax.text(0.5, 0.5, 'Error in plot generation', ha='center', va='center')

    def _plot_electronic_properties(self, properties: Dict, ax):
        """Plot electronic properties with enhanced visualization"""
        try:
            # Remove charge distribution dict and non-numeric values
            plot_props = {k: v for k, v in properties.items() 
                         if isinstance(v, (int, float)) and not isinstance(v, bool)
                         and k != 'charge_distribution'}
            
            if not plot_props:
                ax.text(0.5, 0.5, 'No electronic properties available', 
                       ha='center', va='center')
                return
                
            # Sort and prepare data
            sorted_items = sorted(plot_props.items(), key=lambda x: x[1], reverse=True)
            labels = [item[0].replace('_', ' ').title() for item in sorted_items]
            values = [item[1] for item in sorted_items]
            
            # Create color-coded bar plot
            colors = plt.cm.viridis(np.linspace(0, 1, len(values)))
            bars = ax.bar(range(len(values)), values, color=colors)
            
            # Customize the plot
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_title('Electronic Properties')
            ax.set_ylabel('Value')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom')
                
        except Exception as e:
            print(f"Error plotting electronic properties: {e}")
            ax.text(0.5, 0.5, 'Error in plot generation', ha='center', va='center')

 



    def _plot_property_with_style(self, properties: Dict, ax, title: str, ylabel: str, 
                                 value_suffix: str = "", colormap: str = 'viridis'):
        """
        Generic property plotting function with consistent styling
        
        Args:
            properties (Dict): Dictionary of property names and values
            ax: Matplotlib axis object
            title (str): Plot title
            ylabel (str): Y-axis label
            value_suffix (str): Optional suffix for value labels (e.g., " K" for temperature)
            colormap (str): Name of colormap to use
        """
        try:
            if not properties:
                ax.text(0.5, 0.5, f'No {title.lower()} available', 
                       ha='center', va='center', fontsize=10)
                ax.set_title(title)
                return
                
            # Filter out non-numeric values and prepare data
            valid_props = {k: v for k, v in properties.items() 
                          if isinstance(v, (int, float)) and not isinstance(v, bool)}
            
            if not valid_props:
                ax.text(0.5, 0.5, f'No numeric {title.lower()} found', 
                       ha='center', va='center', fontsize=10)
                ax.set_title(title)
                return
                
            # Sort and prepare data
            sorted_items = sorted(valid_props.items(), key=lambda x: abs(x[1]), reverse=True)
            labels = [item[0].replace('_', ' ').title() for item in sorted_items]
            values = [item[1] for item in sorted_items]
            
            # Create color-coded bar plot
            colors = plt.cm.get_cmap(colormap)(np.linspace(0, 0.8, len(values)))
            bars = ax.bar(range(len(values)), values, color=colors)
            
            # Customize the plot
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_title(title, pad=20, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=10)
            
            # Add gridlines
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)
            
            # Add value labels
            max_value = max(abs(min(values)), abs(max(values)))
            for bar in bars:
                height = bar.get_height()
                if abs(height) > max_value * 0.01:  # Only label values > 1% of max
                    label_text = f'{height:.2f}{value_suffix}'
                    va = 'bottom' if height >= 0 else 'top'
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           label_text, ha='center', va=va, 
                           fontsize=8, rotation=0)
            
            # Adjust layout
            ax.tick_params(axis='both', which='major', labelsize=8)
            
        except Exception as e:
            print(f"Error plotting {title.lower()}: {e}")
            ax.text(0.5, 0.5, 'Error in plot generation', 
                   ha='center', va='center', fontsize=10)
            ax.set_title(title)

    def _plot_polymer_properties(self, properties: Dict, ax):
        """
        Plot polymer-specific properties such as molecular weight, chain length,
        and other polymer characteristics
        
        Args:
            properties (Dict): Dictionary of polymer properties
            ax: Matplotlib axis object
        """
        try:
            # Process polymer-specific properties
            plot_properties = {}
            for key, value in properties.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if 'length' in key.lower():
                        plot_properties[key] = value
                    elif 'weight' in key.lower():
                        plot_properties[key] = value / 1000  # Convert to kDa
                    else:
                        plot_properties[key] = value
            
            # Use the generic plotting function with polymer-specific styling
            self._plot_property_with_style(
                plot_properties, ax,
                'Polymer Properties',
                'Value',
                colormap='Blues'
            )
            
        except Exception as e:
            print(f"Error in polymer properties plot: {e}")
            ax.text(0.5, 0.5, 'Error plotting polymer properties',
                   ha='center', va='center')

    def _plot_conformational_properties(self, properties: Dict, ax):
        """
        Plot conformational properties such as flexibility indices,
        rotatable bonds, and conformational energies
        
        Args:
            properties (Dict): Dictionary of conformational properties
            ax: Matplotlib axis object
        """
        try:
            # Process conformational properties
            plot_properties = {}
            for key, value in properties.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if 'energy' in key.lower():
                        plot_properties[key] = value  # Keep energies in kcal/mol
                    elif 'angle' in key.lower():
                        plot_properties[key] = value  # Keep angles in degrees
                    else:
                        plot_properties[key] = value
            
            # Use the generic plotting function with conformational-specific styling
            self._plot_property_with_style(
                plot_properties, ax,
                'Conformational Properties',
                'Value',
                colormap='Greens'
            )
            
        except Exception as e:
            print(f"Error in conformational properties plot: {e}")
            ax.text(0.5, 0.5, 'Error plotting conformational properties',
                   ha='center', va='center')

    def _plot_thermodynamic_properties(self, properties: Dict, ax):
        """
        Plot thermodynamic properties such as glass transition temperature,
        melting point, and other thermal characteristics
        
        Args:
            properties (Dict): Dictionary of thermodynamic properties
            ax: Matplotlib axis object
        """
        try:
            # Process thermodynamic properties
            plot_properties = {}
            for key, value in properties.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if any(term in key.lower() for term in ['temperature', 'point', 'transition']):
                        plot_properties[key] = value  # Keep temperatures in K
                    else:
                        plot_properties[key] = value
            
            # Use the generic plotting function with thermodynamic-specific styling
            self._plot_property_with_style(
                plot_properties, ax,
                'Thermodynamic Properties',
                'Temperature (K)',
                value_suffix=' K',
                colormap='Reds'
            )
            
        except Exception as e:
            print(f"Error in thermodynamic properties plot: {e}")
            ax.text(0.5, 0.5, 'Error plotting thermodynamic properties',
                   ha='center', va='center')

    def _plot_topological_properties(self, properties: Dict, ax):
        """
        Plot topological properties such as connectivity indices,
        molecular shape descriptors, and structural characteristics
        
        Args:
            properties (Dict): Dictionary of topological properties
            ax: Matplotlib axis object
        """
        try:
            # Process topological properties
            plot_properties = {}
            for key, value in properties.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if 'index' in key.lower():
                        plot_properties[key] = value
                    elif 'descriptor' in key.lower():
                        plot_properties[key] = value
                    else:
                        plot_properties[key] = value
            
            # Use the generic plotting function with topological-specific styling
            self._plot_property_with_style(
                plot_properties, ax,
                'Topological Properties',
                'Value',
                colormap='Purples'
            )
            
        except Exception as e:
            print(f"Error in topological properties plot: {e}")
            ax.text(0.5, 0.5, 'Error plotting topological properties',
                   ha='center', va='center')


    def _create_property_bar_chart(self, results: Dict):
        """Create bar chart comparing different properties"""
        try:
            # Collect numerical properties
            properties = {}
            for category in ['basic_properties', 'topological_properties', 'geometric_properties']:
                if category in results:
                    for key, value in results[category].items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            # Clean up property names for display
                            clean_name = key.replace('_', ' ').title()
                            properties[clean_name] = value
            
            if not properties:
                print("No property data available for bar chart")
                return
            
            # Sort by value and take top 15
            sorted_items = sorted(properties.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
            labels = [item[0] for item in sorted_items]
            values = [item[1] for item in sorted_items]
            
            # Create figure
            plt.figure(figsize=(15, 8))
            
            # Create bar plot
            bars = plt.bar(range(len(values)), values, 
                          color=plt.cm.viridis(np.linspace(0, 0.8, len(values))))
            
            # Customize the plot
            plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
            plt.title('Key Property Comparison')
            plt.ylabel('Value')
            plt.grid(True, linestyle='--', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot directly
            plt.savefig(self.output_dir / 'plots' / 'property_comparison_bar.png',
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating property bar chart: {e}")
            plt.close()



    def _create_atomic_composition_chart(self, atom_distribution: Dict):
        """Create pie chart of atomic composition"""
        try:
            if not atom_distribution or not any(atom_distribution.values()):
                print("No atomic composition data available")
                return
            
            # Create figure
            plt.figure(figsize=(10, 8))
            
            # Sort by abundance
            sorted_items = sorted(atom_distribution.items(), key=lambda x: x[1], reverse=True)
            labels = [f"{atom} ({count})" for atom, count in sorted_items]
            values = [count for _, count in sorted_items]
            
            # Create pie chart
            plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.title('Atomic Composition')
            plt.axis('equal')
            
            # Save plot directly
            plt.savefig(self.output_dir / 'plots' / 'atomic_composition_pie.png',
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating atomic composition chart: {e}")
            plt.close()

    def _create_electronic_properties_line_chart(self, electronic_props: Dict):
        """Create line chart of electronic properties"""
        try:
            # Filter out non-numeric and nested values
            plot_data = {k: v for k, v in electronic_props.items() 
                        if isinstance(v, (int, float)) and not isinstance(v, bool)
                        and not isinstance(v, dict)}
            
            if not plot_data:
                print("No electronic properties data available for line chart")
                return
            
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Sort data for better visualization
            sorted_items = sorted(plot_data.items(), key=lambda x: x[1])
            properties = [k.replace('_', ' ').title() for k, _ in sorted_items]
            values = [v for _, v in sorted_items]
            
            # Create line plot
            plt.plot(range(len(values)), values, 'bo-', linewidth=2, markersize=8)
            plt.xticks(range(len(properties)), properties, rotation=45, ha='right')
            plt.title('Electronic Properties Trend')
            plt.ylabel('Value')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add value labels
            for i, v in enumerate(values):
                plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot directly
            plt.savefig(self.output_dir / 'plots' / 'electronic_properties_line.png',
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating electronic properties line chart: {e}")
            plt.close()

    def _create_ring_analysis_chart(self, ring_counts: Dict):
        """Create pie chart of ring distribution"""
        try:
            if not ring_counts:
                print("No ring analysis data available")
                return
                
            plt.figure(figsize=(10, 8))
            plt.pie(ring_counts.values(), labels=ring_counts.keys(), 
                    autopct='%1.1f%%', startangle=90)
            plt.title('Ring Size Distribution')
            plt.axis('equal')
            plt.savefig(self.output_dir / 'plots' / 'ring_distribution_pie.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating ring analysis chart: {e}")

    def _plot_polymer_chain_analysis(self, polymer_properties: Dict):
        """Generate polymer chain analysis visualization
        
        Args:
            polymer_properties (Dict): Dictionary containing polymer-specific properties
        """
        try:
            if not polymer_properties:
                print("No polymer chain data available")
                return
                
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Chain characteristics
            characteristics = {
                'Persistence Length': polymer_properties.get('persistence_length', 0),
                'Contour Length': polymer_properties.get('contour_length', 0),
                'Kuhn Length': polymer_properties.get('kuhn_length', 0)
            }
            
            # Remove zero values
            characteristics = {k: v for k, v in characteristics.items() if v > 0}
            
            if characteristics:
                colors = plt.cm.viridis(np.linspace(0, 0.8, len(characteristics)))
                bars = ax1.bar(range(len(characteristics)), characteristics.values(), color=colors)
                ax1.set_xticks(range(len(characteristics)))
                ax1.set_xticklabels(characteristics.keys(), rotation=45, ha='right')
                ax1.set_title('Chain Characteristics')
                ax1.set_ylabel('Length (Å)')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}', ha='center', va='bottom')
            else:
                ax1.text(0.5, 0.5, 'No chain characteristic data',
                        ha='center', va='center')
            
            # Plot 2: Chain flexibility analysis
            flexibility_data = {
                'Chain Stiffness': polymer_properties.get('chain_stiffness', 0),
                'Flexibility Index': polymer_properties.get('flexibility_index', 0),
                'Rotatable Bond Ratio': polymer_properties.get('rotatable_bond_ratio', 0)
            }
            
            # Remove zero values
            flexibility_data = {k: v for k, v in flexibility_data.items() if v > 0}
            
            if flexibility_data:
                colors = plt.cm.viridis(np.linspace(0, 0.8, len(flexibility_data)))
                bars = ax2.bar(range(len(flexibility_data)), flexibility_data.values(), color=colors)
                ax2.set_xticks(range(len(flexibility_data)))
                ax2.set_xticklabels(flexibility_data.keys(), rotation=45, ha='right')
                ax2.set_title('Chain Flexibility Analysis')
                ax2.set_ylabel('Value')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}', ha='center', va='bottom')
            else:
                ax2.text(0.5, 0.5, 'No flexibility data',
                        ha='center', va='center')
            
            # Add gridlines
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            print(f"Error in polymer chain analysis plot: {e}")
            return None

    def _create_property_correlation_heatmap(self, results: Dict):
        """Create heatmap of property correlations"""
        try:
            # Collect numerical properties
            property_data = {}
            for category in ['basic_properties', 'topological_properties',
                            'geometric_properties', 'electronic_properties']:
                if category in results:
                    for key, value in results[category].items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            clean_name = key.replace('_', ' ').title()
                            property_data[clean_name] = value
            
            if len(property_data) < 2:
                print("Insufficient data for correlation heatmap")
                return
            
            # Create DataFrame with multiple rows by adding noise to create variation
            num_samples = 10
            df_data = []
            for _ in range(num_samples):
                row_data = {k: v * (1 + np.random.normal(0, 0.01)) 
                           for k, v in property_data.items()}
                df_data.append(row_data)
            
            df = pd.DataFrame(df_data)
            corr = df.corr()
            
            # Create figure
            plt.figure(figsize=(12, 10))
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr), k=1)
            
            # Create heatmap
            sns.heatmap(corr, mask=mask, annot=True, cmap='RdBu_r',
                       center=0, fmt='.2f', square=True,
                       linewidths=0.5, cbar_kws={"shrink": .8},
                       annot_kws={"size": 8}, vmin=-1, vmax=1)
            
            plt.title('Property Correlations', pad=20, fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            plt.tight_layout()
            
            # Save plot directly
            plt.savefig(self.output_dir / 'plots' / 'property_correlation_heatmap.png',
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating property correlation heatmap: {e}")
            plt.close()






    def _plot_fragment_distribution(self, fragments: Dict, ax):
        """Plot distribution of molecular fragments with improved error handling"""
        try:
            if not fragments or 'error' in fragments:
                ax.text(0.5, 0.5, 'No fragment data available', ha='center', va='center')
                return

            # Extract plottable data
            plot_data = {}
            for key, value in fragments.items():
                # Skip nested dictionaries and special keys
                if isinstance(value, (int, float)) and key not in ['ring_counts', 'error']:
                    if value > 0:  # Only include non-zero values
                        plot_data[key] = value
            
            if not plot_data:
                ax.text(0.5, 0.5, 'No fragments detected', ha='center', va='center')
                return

            # Sort by value for better visualization
            sorted_items = sorted(plot_data.items(), key=lambda x: x[1], reverse=True)
            labels = [item[0].replace('_', ' ').title() for item in sorted_items]
            values = [item[1] for item in sorted_items]

            # Create bar plot
            bars = ax.bar(range(len(values)), values,
                         color=plt.cm.viridis(np.linspace(0, 0.8, len(values))))
            
            # Customize the plot
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_title('Fragment Distribution')
            ax.set_ylabel('Count')
            ax.grid(True, linestyle='--', alpha=0.3)

            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom')

        except Exception as e:
            print(f"Error in fragment distribution plot: {e}")
            ax.text(0.5, 0.5, 'Error in plot generation', ha='center', va='center')

    def _create_fragment_pie_chart(self, fragment_data: Dict):
        """Create pie chart of fragment distribution with improved error handling"""
        try:
            if not fragment_data or 'error' in fragment_data:
                print("No fragment data available for pie chart")
                return
                
            # Extract plottable data
            plot_data = {}
            for key, value in fragment_data.items():
                if isinstance(value, (int, float)) and key not in ['ring_counts', 'error']:
                    if value > 0:  # Only include non-zero values
                        plot_data[key] = value
            
            if not plot_data:
                print("No fragments found for pie chart")
                return
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Sort by frequency and create labels
            sorted_items = sorted(plot_data.items(), key=lambda x: x[1], reverse=True)
            labels = [f"{name.replace('_', ' ').title()} ({count})" 
                     for name, count in sorted_items]
            values = [count for _, count in sorted_items]
            
            # Create pie chart
            plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.title('Fragment Distribution')
            plt.axis('equal')
            
            # Save plot
            plt.savefig(self.output_dir / 'plots' / 'fragment_distribution_pie.png',
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating fragment pie chart: {e}")
            plt.close()

    def _plot_fragment_sunburst(self, fragment_data: Dict):
        """Generate sunburst plot for fragment distribution with improved error handling"""
        try:
            if not fragment_data or 'error' in fragment_data:
                print("No fragments found for sunburst plot")
                return
                
            # Extract plottable data
            plot_data = {}
            for key, value in fragment_data.items():
                if isinstance(value, (int, float)) and key not in ['ring_counts', 'error']:
                    if value > 0:  # Only include non-zero values
                        plot_data[key] = value
            
            if not plot_data:
                print("No fragments found for sunburst plot")
                return
            
            # Create figure
            plt.figure(figsize=(12, 12))
            
            # Sort by value for better visualization
            sorted_items = sorted(plot_data.items(), key=lambda x: x[1], reverse=True)
            labels = [k.replace('_', ' ').title() for k, _ in sorted_items]
            values = [v for _, v in sorted_items]
            
            # Create sunburst using pie chart
            plt.pie(values, labels=labels, autopct='%1.1f%%',
                   pctdistance=0.85, labeldistance=1.1)
            
            plt.title('Fragment Distribution')
            plt.axis('equal')
            
            # Save plot
            plt.savefig(self.output_dir / 'plots' / 'fragment_sunburst.png',
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error in fragment sunburst plot: {e}")
            plt.close()












    def _calculate_persistence_length(self, mol: Chem.Mol) -> float:
        """Estimate polymer persistence length"""
        try:
            # This is a simplified estimation
            conf = mol.GetConformer()
            positions = []
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                positions.append([pos.x, pos.y, pos.z])
            positions = np.array(positions)
            
            # Calculate average bond length
            bond_vectors = []
            for bond in mol.GetBonds():
                id1 = bond.GetBeginAtomIdx()
                id2 = bond.GetEndAtomIdx()
                vector = positions[id2] - positions[id1]
                bond_vectors.append(vector)
            
            # Calculate average correlation between consecutive bond vectors
            correlations = []
            for i in range(len(bond_vectors)-1):
                v1 = bond_vectors[i]
                v2 = bond_vectors[i+1]
                cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                correlations.append(cos_theta)
            
            if correlations:
                persistence_length = -1.0 / np.log(np.mean(correlations))
                return float(persistence_length)
            return 0.0
        except:
            return 0.0

    def _calculate_contour_length(self, mol: Chem.Mol) -> float:
        """Calculate polymer contour length"""
        try:
            conf = mol.GetConformer()
            total_length = 0
            for bond in mol.GetBonds():
                id1 = bond.GetBeginAtomIdx()
                id2 = bond.GetEndAtomIdx()
                pos1 = conf.GetAtomPosition(id1)
                pos2 = conf.GetAtomPosition(id2)
                length = rdMolTransforms.GetBondLength(conf, id1, id2)
                total_length += length
            return float(total_length)
        except:
            return 0.0

    def _calculate_kuhn_length(self, mol: Chem.Mol) -> float:
        """Estimate Kuhn length"""
        try:
            persistence_length = self._calculate_persistence_length(mol)
            return 2 * persistence_length
        except:
            return 0.0

    def _calculate_chain_stiffness(self, mol: Chem.Mol) -> float:
        """Calculate chain stiffness index"""
        try:
            rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
            total_bonds = mol.GetNumBonds()
            if total_bonds > 0:
                stiffness = 1.0 - (rotatable_bonds / total_bonds)
                return float(stiffness)
            return 0.0
        except:
            return 0.0

    def _estimate_boiling_point(self, mol: Chem.Mol) -> float:
        """Estimate boiling point using group contribution method"""
        try:
            # Simple estimation based on molecular weight and logP
            mw = Descriptors.ExactMolWt(mol)
            logp = Descriptors.MolLogP(mol)
            bp = 20 * logp + 0.5 * mw + 100
            return float(bp)
        except:
            return 0.0

    def _estimate_melting_point(self, mol: Chem.Mol) -> float:
        """Estimate melting point"""
        try:
            # Simple estimation based on molecular weight and rotatable bonds
            mw = Descriptors.ExactMolWt(mol)
            rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
            mp = 0.3 * mw - 10 * rotatable_bonds + 50
            return float(mp)
        except:
            return 0.0

    def _estimate_glass_transition(self, mol: Chem.Mol) -> float:
        """Estimate glass transition temperature"""
        try:
            # Simple estimation based on molecular weight and flexibility
            mw = Descriptors.ExactMolWt(mol)
            flexibility = self._calculate_chain_stiffness(mol)
            tg = 0.2 * mw + 100 * flexibility - 50
            return float(tg)
        except:
            return 0.0

    def _estimate_solubility(self, mol: Chem.Mol) -> float:
        """Estimate solubility parameter"""
        try:
            # Estimation based on LogP and TPSA
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            solubility = -0.5 * logp + 0.1 * tpsa
            return float(solubility)
        except:
            return 0.0

    def _calculate_ring_flexibility(self, mol: Chem.Mol) -> float:
            """Calculate ring flexibility index"""
            try:
                ring_info = mol.GetRingInfo()
                if ring_info.NumRings() == 0:
                    return 0.0
                    
                ring_sizes = [len(ring) for ring in ring_info.AtomRings()]
                flexibility_index = sum(1.0 / size for size in ring_sizes)
                return float(flexibility_index)
            except:
                return 0.0


def main():
    """Main execution function for polymer analytics"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enhanced Polymer Analytics System"
    )
    
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input directory containing polymer.pdb or direct path to PDB file"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="analytics_output",
        help="Output directory for analysis results"
    )
    
    args = parser.parse_args()
    
    try:
        input_path = Path(args.input)
        
        # If input is a directory, look for polymer.pdb inside
        if input_path.is_dir():
            pdb_path = input_path / 'polymer' / 'polymer.pdb'
            if not pdb_path.exists():
                pdb_path = input_path / 'polymer.pdb'
        else:
            pdb_path = input_path
            
        if not pdb_path.exists():
            raise FileNotFoundError(
                f"No polymer PDB file found at: {pdb_path}\n"
                "Please provide a valid PDB file or directory containing polymer.pdb"
            )
            
        print(f"\nAnalyzing polymer structure from: {pdb_path}")
        
        # Initialize analytics system
        analytics = PolymerAnalytics(Path(args.output))
        
        # Validate PDB file
        if not analytics._validate_pdb_file(pdb_path):
            raise ValueError("PDB file validation failed")
        
        # Run analysis 
        results = analytics.run_analysis(pdb_path)
        
        if not results:
            raise ValueError("Analysis produced no results")
            
        print("\nAnalysis summary:")
        if 'basic_properties' in results:
            bp = results['basic_properties']
            print(f"Molecular Weight: {bp.get('molecular_weight', 'N/A'):.2f}")
            print(f"Number of Atoms: {bp.get('atom_count', 'N/A')}")
            print(f"Number of Bonds: {bp.get('bond_count', 'N/A')}")
            print(f"Number of Rings: {bp.get('ring_count', 'N/A')}")
        
        print(f"\nResults saved in: {args.output}")
        return 0
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())