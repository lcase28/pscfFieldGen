# Library imports
from pscfFieldGen.generation import (
    UniformParticleField, 
    generate_field_file,
    read_input_file )

# Standard Library Imports
import argparse
from pathlib import Path

if __name__=="__main__":
    POSITION_TOLERANCE = 0.001
    # Get command file from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--file","-f", type=str, required=True)
    parser.add_argument("--trace","-t", action='store_true')
    parser.add_argument("--nowarn",action='store_false')
    args = parser.parse_args()
    filepath = Path(args.file)
    tflag = args.trace
    wflag = args.nowarn
    param, calc, out, cmon = read_input_file(filepath, trace=tflag, omissionWarnings=wflag)
    
    # Generate File
    if tflag:
        print("\nGenerating Field File")
    generate_field_file(param, calc, out, core=cmon)
    if tflag:
        print("\nField Generation Complete")
            
            
            
