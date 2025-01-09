"""
Streamlined Integrated Polymer System
-----------------------------------
Integrates unified polymer builder with analysis and monitoring capabilities.


python streamlined-integrated-system.py -i monomer_data.txt -o polymer_output -v

python streamlined-integrated-system.py -i monomer_data.txt -o polymer_output

python streamlined-integrated-system.py -i existing_polymer_directory --analysis-only -o polymer_output
python streamlined-integrated-system.py -i ./polymer_output --analysis-only -o polymer_output




Interactive mode: python streamlined-integrated-system.py

Command-line mode: python streamlined-integrated-system.py -i path/to/input.pdb -o output_dir -m 2 -v

Get help: python streamlined-integrated-system.py --help

"""

from pathlib import Path
import logging
import shutil
from datetime import datetime
import json
import sys
import argparse

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, rdMolDescriptors
import numpy as np

from unified_polymer_builder import EnhancedPolymerBuilder
from polymer_analytics import PolymerAnalytics
from polymer_monitor import PolymerMonitor
from enhanced_validation import validate_polymer_structure
from collections import Counter


class IntegratedPolymerSystem:
    """Enhanced integration system for polymer building and analysis"""
    
    def __init__(self, output_dir: str = "polymer_output"):
        self.output_dir = Path(output_dir)
        self.setup_system()
        
    def setup_system(self):
        """Set up integrated system components"""
        self.dirs = {
            'root': self.output_dir,
            'analytics': self.output_dir / 'analytics',
            'monitoring': self.output_dir / 'monitoring',
            'logs': self.output_dir / 'logs'
        }
        
        for directory in self.dirs.values():
            directory.mkdir(parents=True, exist_ok=True)
            
        # Initialize components with proper connections
        self.builder = EnhancedPolymerBuilder(self.output_dir)
        self.analytics = PolymerAnalytics(self.dirs['analytics'])
        self.monitor = PolymerMonitor(self.dirs['monitoring'])
        self.builder.set_monitor(self.monitor)
        
        # Set up shared data store
        self.shared_data = {
            'monomers': {},
            'sequence': [],
            'polymer': None,
            'analysis_results': None
        }



    def run_analysis(self, input_dir: Path) -> bool:
        """Run standalone analysis by properly invoking the analytics module"""
        try:
            print("\nStarting polymer structure analysis...")
            
            # Verify input polymer structure exists
            pdb_path = input_dir / 'polymer' / 'polymer.pdb'
            if not pdb_path.exists():
                raise FileNotFoundError(f"PDB file not found at: {pdb_path}")
                
            print(f"Loading polymer structure from: {pdb_path}")
            
            # Prepare analysis data for the analytics module
            analysis_data = {
                'polymer_path': pdb_path,
                'output_dir': self.dirs['analytics']
            }
            
            # Run the analysis using PolymerAnalytics
            print("\nRunning comprehensive analysis...")
            results = self.analytics.run_analysis(analysis_data)
            
            if results:
                print("\nAnalysis completed successfully!")
                print(f"Results saved in: {self.dirs['analytics']}")
                return True
            else:
                print("\nAnalysis failed to generate results")
                return False
                
        except Exception as e:
            print(f"\nError during analysis: {str(e)}")
            logging.error(f"Analysis error: {e}")
            return False



    def _save_structure_info(self, analysis_data):
        """Save structure information to file"""
        info_file = self.dirs['analytics'] / 'structure_info.json'
        info = {
            'timestamp': datetime.now().isoformat(),
            'total_atoms': analysis_data['structure_info']['total_atoms'],
            'residue_types': analysis_data['structure_info']['residue_types'],
            'chains': len(set(info['chain'] for info in analysis_data['atomic_info'])),
            'residue_composition': dict(Counter(info['residue'] for info in analysis_data['atomic_info']))
        }
        
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\nStructure Information:")
        print(f"Total atoms: {info['total_atoms']}")
        print(f"Residue types: {info['residue_types']}")
        print(f"Number of chains: {info['chains']}")
        print("\nResidue composition:")
        for residue, count in info['residue_composition'].items():
            print(f"  {residue}: {count} residues")



    def save_analysis_results(self, results):
        """Save comprehensive analysis results"""
        try:
            results_dir = self.dirs['analytics']
            
            # Save main analysis results
            results_file = results_dir / 'analysis_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
                
            # Save monitoring data
            if hasattr(self, 'monitor'):
                monitoring_file = results_dir / 'monitoring_results.json'
                monitoring_data = self.monitor.get_monitoring_data()
                with open(monitoring_file, 'w') as f:
                    json.dump(monitoring_data, f, indent=2)
                    
            # Save validation report
            validation_file = results_dir / 'validation_report.json'
            validation_results = {
                'timestamp': datetime.now().isoformat(),
                'validation_checks': results.get('validation_checks', {})
            }
            with open(validation_file, 'w') as f:
                json.dump(validation_results, f, indent=2)
                
            print(f"\nDetailed results saved in: {results_dir}")
            
        except Exception as e:
            logging.error(f"Error saving results: {e}")

    def print_analysis_summary(self, results):
        """Print a summary of the analysis results"""
        print("\nAnalysis Results Summary:")
        if isinstance(results, dict):
            if 'molecular_weight' in results:
                print(f"- Molecular Weight: {results['molecular_weight']:.2f} g/mol")
            if 'total_atoms' in results:
                print(f"- Total Atoms: {results['total_atoms']}")
            if 'ring_count' in results:
                print(f"- Number of Rings: {results['ring_count']}")
            if 'end_to_end_distance' in results:
                print(f"- End-to-End Distance: {results['end_to_end_distance']:.2f} Å")
            if 'radius_of_gyration' in results:
                print(f"- Radius of Gyration: {results['radius_of_gyration']:.2f} Å")
        else:
            print("No detailed results available")


    def process_input(self, input_file: str) -> bool:
            """Process input file with proper data sharing"""
            try:
                monomers = {}
                sequence = []
                current_section = None
                
                with open(input_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                            
                        if line.startswith('[') and line.endswith(']'):
                            current_section = line[1:-1].upper()
                            continue
                            
                        if current_section != 'SEQUENCE' and ':' in line:
                            name, smiles = [x.strip() for x in line.split(':')]
                            if name and smiles:
                                monomers[name] = smiles
                                
                        elif current_section == 'SEQUENCE':
                            parts = [x.strip() for x in line.split(',')]
                            if len(parts) == 2:
                                count = int(parts[0])
                                name = parts[1]
                                sequence.append((count, name))
                
                self.shared_data['monomers'] = monomers
                self.shared_data['sequence'] = sequence
                
                for name, smiles in monomers.items():
                    success = self.builder.process_monomer(name, smiles)
                    if not success:
                        raise ValueError(f"Failed to process monomer: {name}")
                        
                return True
                
            except Exception as e:
                logging.error(f"Input processing error: {e}")
                return False

    def build_polymer(self) -> bool:
        """Build polymer with integrated monitoring"""
        try:
            polymer = self.builder.build_polymer(self.shared_data['sequence'])
            if polymer is None:
                raise ValueError("Polymer build failed")
                
            self.shared_data['polymer'] = polymer
            return True
            
        except Exception as e:
            logging.error(f"Build error: {e}")
            return False

    def run_complete_workflow(self, input_file: str) -> bool:
        """Run complete workflow from input to analysis"""
        try:
            if not self.process_input(input_file):
                raise ValueError("Input processing failed")
                
            if not self.build_polymer():
                raise ValueError("Polymer build failed")
                
            if not self.run_analysis(self.output_dir):
                raise ValueError("Analysis failed")
                
            return True
            
        except Exception as e:
            logging.error(f"Workflow error: {e}")
            return False

    def generate_reports(self) -> bool:
        """Generate reports from analysis results"""
        try:
            reports_dir = self.dirs['root'] / 'reports'
            reports_dir.mkdir(exist_ok=True)
            
            if not hasattr(self, 'analytics'):
                raise ValueError("Analytics system not initialized")
                
            # Run analysis and get results
            results = self.analytics.run_analysis()
            if not results:
                raise ValueError("Analysis failed to produce results")
                
            # Save analysis results
            self.analytics.save_analysis_results(results)
            
            print(f"\nAnalysis completed successfully")
            print(f"Results saved in: {self.analytics.analytics_dir}")
            
            return True
            
        except Exception as e:
            logging.error(f"Report generation error: {e}")
            return False








def main():
    """Enhanced interactive main function with proper logging and file handling"""
    # Configure default logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        print("\nWelcome to the Polymer Processing System!")
        print("----------------------------------------")
        
        # First ask about verbose mode
        print("\nEnable verbose output (debug mode)?")
        verbose = input("Enter choice (y/n, default: n): ").strip().lower()
        if verbose == 'y':
            logging.getLogger().setLevel(logging.DEBUG)
            print("\nVerbose output enabled.")
        
        # Then ask for processing mode
        print("\nPlease select processing mode:")
        print("1. Build and Analyze Polymer")
        print("2. Analysis Only")
        
        while True:
            mode = input("\nEnter your choice (1 or 2): ").strip()
            if mode in ['1', '2']:
                break
            print("Invalid choice. Please enter 1 or 2.")

        # Get input path
        if mode == '1':
            print("\nFor polymer building, you need a monomer definition file (e.g., monomer_data.txt)")
            print("The file should contain monomer SMILES strings and sequence information.")
            input_path = input("\nEnter path to your monomer definition file: ").strip()
            input_path = Path(input_path)
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
        else:
            print("\nFor analysis, you need a polymer.pdb file.")
            print("Please enter the complete path to your polymer.pdb file.")
            print("Example: C:/path/to/your/polymer.pdb")
            
            while True:
                input_path = input("\nEnter the full path to polymer.pdb: ").strip()
                input_path = Path(input_path)
                
                # Validate the input path
                if not input_path.exists():
                    print(f"Error: File not found at {input_path}")
                    retry = input("Would you like to try another path? (y/n): ").strip().lower()
                    if retry != 'y':
                        raise FileNotFoundError("No valid PDB file provided")
                    continue
                
                if not input_path.is_file():
                    print(f"Error: {input_path} is not a file")
                    retry = input("Would you like to try another path? (y/n): ").strip().lower()
                    if retry != 'y':
                        raise FileNotFoundError("No valid PDB file provided")
                    continue
                
                if input_path.suffix.lower() != '.pdb':
                    print(f"Error: {input_path} is not a PDB file")
                    retry = input("Would you like to try another path? (y/n): ").strip().lower()
                    if retry != 'y':
                        raise FileNotFoundError("No valid PDB file provided")
                    continue
                
                # Check if file is accessible
                try:
                    with open(input_path, 'rb') as f:
                        # Just test if we can read from it
                        f.read(1)
                except PermissionError:
                    print(f"Error: Cannot access file {input_path}. File may be in use.")
                    retry = input("Would you like to try another path? (y/n): ").strip().lower()
                    if retry != 'y':
                        raise PermissionError("Cannot access PDB file")
                    continue
                
                # If we get here, we have a valid and accessible PDB file
                print(f"\nFound valid PDB file at: {input_path}")
                break

        # Get output directory
        print("\nSpecify where to save the results")
        output_dir = input("Enter output directory path (default: polymer_output): ").strip()
        if not output_dir:
            output_dir = "polymer_output"
        output_dir = Path(output_dir)

        # Ensure output directory is accessible
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            test_file = output_dir / '.test'
            test_file.touch()
            test_file.unlink()
        except PermissionError:
            raise PermissionError(f"Cannot write to output directory: {output_dir}")

        # Process with selected options
        print(f"\nProcessing with following parameters:")
        print(f"Mode: {'Build and Analyze' if mode == '1' else 'Analysis Only'}")
        print(f"Verbose output: {'Yes' if verbose == 'y' else 'No'}")
        print(f"Input: {input_path}")
        print(f"Output: {output_dir}")
        
        proceed = input("\nProceed with processing? (y/n): ").strip().lower()
        if proceed != 'y':
            print("\nProcessing cancelled.")
            return 1

        # Initialize system and run processing
        system = IntegratedPolymerSystem(str(output_dir))
        
        if mode == '1':
            print("\nRunning complete polymer building and analysis workflow...")
            success = system.run_complete_workflow(str(input_path))
        else:
            print("\nRunning standalone polymer analysis...")
            # For analysis mode, we need to modify the parent directory structure
            polymer_dir = output_dir / 'polymer'
            polymer_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy the input PDB file with proper error handling
            target_pdb = polymer_dir / 'polymer.pdb'
            try:
                # First try to read the source file
                with open(input_path, 'rb') as source:
                    content = source.read()
                
                # Then write to the target file
                with open(target_pdb, 'wb') as target:
                    target.write(content)
                    
                success = system.run_analysis(output_dir)
                
            except PermissionError:
                print(f"\nError: Cannot access one of the files. Please ensure no other programs are using them.")
                return 1
            except Exception as e:
                print(f"\nError copying PDB file: {str(e)}")
                return 1

        if success:
            print(f"\nProcessing completed successfully!")
            print(f"Results have been saved in: {output_dir}")
            print("\nOutput directories:")
            print(f"- Analytics: {output_dir}/analytics")
            print(f"- Plots: {output_dir}/analytics/plots")
            print(f"- Reports: {output_dir}/analytics/excel_data")
            if mode == '1':
                print(f"- Polymer structure: {output_dir}/polymer")
                print(f"- Monomer files: {output_dir}/monomers")
            return 0
        
        print("\nProcessing failed. Check the error messages above.")
        return 1

    except KeyboardInterrupt:
        print("\n\nProcessing cancelled by user.")
        return 1
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        logging.error(f"Error: {e}")
        return 1
    finally:
        # Cleanup any temporary resources
        try:
            if 'system' in locals():
                del system
        except Exception:
            pass

            

def run_cli():
    """Command-line interface entry point"""
    parser = argparse.ArgumentParser(description="Polymer Processing System")
    parser.add_argument("-i", "--input", help="Input file (monomer data or PDB)")
    parser.add_argument("-o", "--output", default="polymer_output", help="Output directory")
    parser.add_argument("-m", "--mode", choices=['1', '2'], help="Processing mode (1=Build, 2=Analysis)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.input:
        # Run in non-interactive mode
        if not args.mode:
            print("Error: --mode is required when using command line arguments")
            return 1
            
        system = IntegratedPolymerSystem(args.output)
        if args.mode == '1':
            return system.run_complete_workflow(args.input)
        else:
            return system.run_analysis(Path(args.input))
    else:
        # Run in interactive mode
        return main()

if __name__ == "__main__":
    sys.exit(run_cli())