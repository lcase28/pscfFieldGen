# Library imports
from fieldGeneration.crystal_structs.crystalStructs.lattice import Lattice
from fieldGeneration.fieldGenerators import FieldCalculator
from fieldGeneration.pscfFileManagers.paramfile import expandLatticeParameters, getInterfaceWidth, getMonomerFractions, ParamFile
from fieldGeneration.pscfFileManagers.fieldfile import WaveVectFieldFile
from fieldGeneration.stringTools import str_to_num, wordsGenerator

# Standard Library Imports
import argparse
from copy import deepcopy
import numpy as np
from pathlib import Path


def generate_field_file(param, calculator, kgridFileName, kgrid=None):
    """
    From the given ParamFile (param), and FieldCalculator (calculator),
    generate an initial guess field file at kgridFileName.
    
    No check is done to verify compatibility between calculator and param. These checks
    are the caller's responsibility.
    
    Parameters
    ----------
    param : pscfFileManagers.paramfile.ParamFile
        The param file being used with the field
    calculator : fieldGenerators.FieldCalculator
        The FieldCalculator used to do the field calculation.
    kgridFileName : pathlib.Path or string
        The path and file name to which to write the resulting field file.
    kgrid : pscfFileManagers.fieldfile.WaveVectFieldFile (optional)
        If given, assumed to match param and calculator, and is updated to hold
        the resultant field. 
        If None, a new WaveVectFieldFile is instantiated to match
        the param file.
    """
    monFrac = getMonomerFractions(param)
    interface = getInterfaceWidth(param)
    ngrid = param.ngrid
    newField = calculator.to_kgrid(monFrac, ngrid, interfaceWidth=interface)
    # Create clean field file if needed.
    if kgrid is None:
        kgrid = WaveVectFieldFile()
        kgrid.dim = param.dim
        kgrid.crystal_system = param.crystal_system
        kgrid.N_cell_param = param.N_cell_param
        kgrid.cell_param = param.cell_param
        kgrid.group_name = param.group_name
        kgrid.N_monomer = param.N_monomer
        kgrid.ngrid = ngrid
    kgrid.fields = newField
    kgrid.write(kgridFileName.open(mode='x'))
    

if __name__=="__main__":
    # Get command file from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--file","-f", type=str, required=True)
    args = parser.parse_args()
    filepath = Path(args.file)
    # Set initial flags
    hasParam = False
    hasStyle = False
    nparticle = -1
    hasOutFile = False
    hasPositions = False
    # Set default values
    input_style = 'motif'
    outfilestring = 'rho_kgrid'
    # Parse input file
    with filepath.open(mode='r') as cmdFile:
        words = wordsGenerator(cmdFile)
        for word in words:
            if word == 'parameter_file':
                param = ParamFile(next(words))
                hasParam = True
            elif word == 'coord_input_style':
                input_style = next(words)
                if input_style == 'motif' or input_style == 'basis':
                    hasStyle = True
                else:
                    raise(ValueError("Invalid option, {}, given for coord_input_style".format(input_style)))
            elif word == 'N_particles':
                nparticle = str_to_num(next(words))
            elif word == 'particle_positions':
                if nparticle <= 0:
                    raise(ValueError("N_particles must be specified before particle_positions"))
                elif not hasParam:
                    raise(ValueError("parameter_file must be specified before particle_positions"))
                else:
                    numData = param.dim * nparticle
                    positionList = np.array( [str_to_num(next(words)) for i in range(numData)] )
                    partPositions = np.reshape(positionList, (nparticle, param.dim))
                    hasPositions = True
            elif word == 'output_file':
                outfilestring = next(words)
                outFile = Path(outfilestring)
                hasOutFile = True
            elif word == 'finish':
                #do nothing
                doneFlag = True
            else:
                raise(NotImplementedError("No operation has been set for keyword {}.".format(word)))
    # Check for presence of required data
    if not hasParam:
        raise(ValueError("Input keyword 'parameter_file' must be specified"))
    if nparticle <= 0:
        raise(ValueError("Input keyword 'N_particles' must be specified"))
    if not hasPositions:
        raise(ValueError("Particle coordinates must be specified with keyword 'particle_positions'."))
    if not hasStyle:
        raise(RuntimeWarning("coord_input_style not specified. 'motif' assumed."))
    if not hasOutFile:
        raise(RuntimeWarning("Output file name not specified with keyword 'output_file'. Using 'rho_kgrid'."))
    # Create Lattice Object
    latticeParams = expandLatticeParameters(param)
    dim = param.dim
    lattice = Lattice.latticeFromParameters(dim, **latticeParams)
    # Create Calculator Object
    group_name = param.group_name
    crystal_system = param.crystal_system
    calculator = FieldCalculator(dim = dim, \
                                lattice = lattice, \
                                N_particles = nparticle, \
                                particlePositions = partPositions, \
                                coord_input_style = input_style, \
                                systemName = crystal_system, \
                                groupName = group_name)
    # Generate File
    generate_field_file(param, calculator, outFile)
            
            
            
