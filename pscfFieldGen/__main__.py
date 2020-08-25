# Library imports
from pscfFieldGen.structure import ( Lattice,
                        ParticleBase,
                        ScatteringParticle,
                        POSITION_TOLERANCE,
                        buildCrystal )
from pscfFieldGen.fieldGenerators import FieldCalculator
from pscfFieldGen.filemanagers import PscfParam, PscfppParam
from pscfFieldGen.util.stringTools import str_to_num, wordsGenerator

# Standard Library Imports
import argparse
from copy import deepcopy
import numpy as np
from pathlib import Path
import warnings


def generate_field_file(param, calculator, kgridFileName, core=0, kgrid=None):
    """
    From the given ParamFile (param), and FieldCalculator (calculator),
    generate an initial guess field file at kgridFileName.
    
    No check is done to verify compatibility between calculator and param. These checks
    are the caller's responsibility.
    
    Parameters
    ----------
    param : pscfFieldGen.filemanagers.ParamFile
        The param file being used with the field
    calculator : fieldGenerators.FieldCalculator
        The FieldCalculator used to do the field calculation.
    kgridFileName : pathlib.Path or string
        The path and file name to which to write the resulting field file.
    core : integer
        The index identifying the monomer to be placed in the core of the particles.
    kgrid : pscfFileManagers.fieldfile.WaveVectFieldFile (optional)
        If given, assumed to match param and calculator, and is updated to hold
        the resultant field. 
        If None, a new WaveVectFieldFile is instantiated to match
        the param file.
    """
    monFrac = param.getMonomerFractions()
    interface = param.getInterfaceWidth(core)
    ngrid = param.ngrid
    newField = calculator.to_kgrid(monFrac, ngrid, interfaceWidth=interface, coreindex=core)
    # Create clean field file if needed.
    if kgrid is None:
        kgrid = param.cleanFieldFile()
    kgrid.fields = newField
    kgrid.write(kgridFileName.open(mode='x'))

SOFTWARE_MAP = { "pscf" : PscfParam, "pscfpp" : PscfppParam }

if __name__=="__main__":
    POSITION_TOLERANCE = 0.001
    # Get command file from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--file","-f", type=str, required=True)
    parser.add_argument("--trace","-t", action='store_true')
    args = parser.parse_args()
    filepath = Path(args.file)
    
    # Set initial flags
    hasSoftware = False
    hasParam = False
    hasStyle = False
    hasCore = False
    nparticle = -1
    hasOutFile = False
    hasPositions = False
    
    # Set default values
    ParamFile = None # class of parameter file can be set based on flag
    input_style = 'motif'
    outfilestring = 'rho_kgrid'
    core_monomer = 0
    
    # Parse input file
    with filepath.open(mode='r') as cmdFile:
        words = wordsGenerator(cmdFile)
        for word in words:
            if word == 'software':
                software = next(words)
                ParamFile = SOFTWARE_MAP.get(software,None)
                if ParamFile is None:
                    raise(ValueError("Invalid software ({}) given.".format(software)))
                hasSoftware = True
                data = software
            elif word == 'parameter_file':
                if not hasSoftware:
                    raise(ValueError("Keyword 'software' must appear before 'parameter_file'"))
                filename = next(words)
                param = ParamFile.fromFileName(filename)
                hasParam = True
                data = filename
            elif word == 'coord_input_style':
                input_style = next(words)
                if input_style == 'motif' or input_style == 'basis':
                    hasStyle = True
                    data = input_style
                else:
                    raise(ValueError("Invalid option, {}, given for coord_input_style".format(input_style)))
            elif word == 'core_monomer':
                core_monomer = int(next(words))
                if core_monomer >= 0:
                    hasCore = True
                    data = core_monomer
                else:
                    raise(ValueError("core_monomer must be a non-negative integer. Given {}.".format(core_monomer)))
            elif word == 'N_particles':
                nparticle = str_to_num(next(words))
                if nparticle <= 0:
                    raise(ValueError("Invalid N_particles given ({}). Must be >= 1.".format(nparticle)))
                else:
                    data = nparticle
            elif word == 'particle_positions':
                if nparticle <= 0:
                    raise(ValueError("N_particles must be specified before particle_positions"))
                elif not hasParam:
                    raise(ValueError("parameter_file must be specified before particle_positions"))
                else:
                    numData = param.dim * nparticle
                    positionList = np.array( [str_to_num(next(words)) for i in range(numData)] )
                    partPositions = np.reshape(positionList, (nparticle, param.dim))
                    data = partPositions
                    hasPositions = True
            elif word == 'output_file':
                outfilestring = next(words)
                outFile = Path(outfilestring)
                data = outFile
                hasOutFile = True
            elif word == 'finish':
                #do nothing
                data = ''
                doneFlag = True
            else:
                raise(NotImplementedError("No operation has been set for keyword {}.".format(word)))
            # if trace requested, echo input file as read
            if args.trace:
                print('{}\n\t\t{}'.format(word, data))
    
    # Check for presence of required data
    if not hasSoftware:
        raise(ValueError("Input keyword 'software' must be specified"))
    if not hasParam:
        raise(ValueError("Input keyword 'parameter_file' must be specified"))
    if nparticle <= 0:
        raise(ValueError("Input keyword 'N_particles' must be specified"))
    if not hasPositions:
        raise(ValueError("Particle coordinates must be specified with keyword 'particle_positions'."))
    
    # Warn of absence of optional data and state assumptions.
    if not hasStyle:
        warnings.warn(RuntimeWarning("coord_input_style not specified. 'motif' assumed."))
    if not hasOutFile:
        warnings.warn(RuntimeWarning("Output file name not specified with keyword 'output_file'. Using 'rho_kgrid'."))
    if not hasCore:
        warnings.warn(RuntimeWarning("core_monomer not specified. Assuming monomer 0."))
    
    # Create Lattice Object
    if args.trace:
        print("\nCreating System Lattice")
    latticeParams = param.latticeParameters
    dim = param.dim
    lattice = Lattice.latticeFromParameters(dim, **latticeParams)
    if args.trace:
        print("\t\t{}".format(lattice))
    
    # Create Crystal Object
    if args.trace:
        print("\nCreating Crystal\n")
    groupname = param.group_name
    crystalsystem = param.crystal_system
    crystal = buildCrystal( input_style, 
                            nparticle, 
                            partPositions, 
                            lattice, 
                            group_name=groupname,
                            crystal_system=crystalsystem )
    if args.trace:
        print("Crystal being generated:")
        print(crystal.longString)
    # Create Calculator Object
    if args.trace:
        print("\nSetting Up Calculator")
    calculator = FieldCalculator(crystal)
    # Generate File
    if args.trace:
        print("\nGenerating Field File")
    generate_field_file(param, calculator, outFile, core=core_monomer)
    if args.trace:
        print("\nField Generation Complete")
            
            
            
