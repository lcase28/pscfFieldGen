# Project imports
from pscfFieldGen.structure import ( 
    Lattice,
    BasisCrystal,
    MotifCrystal,
    SpaceGroup,
    ParticleBase,
    ScatteringParticle,
    buildCrystal )
from pscfFieldGen.structure.grids import(
    getKgridCount,
    IterableWaveVector )
from pscfFieldGen.structure.network import NetworkFieldBasis
from pscfFieldGen.filemanagers import PscfParam, PscfppParam
from pscfFieldGen.util.stringTools import str_to_num, wordsGenerator
import pscfFieldGen.util.contexttools as contexttools
import pscfFieldGen.filemanagers.pscf as pscf

# Standard Library Imports
import argparse
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
import itertools
import numba
import numpy as np
import scipy as sp
import subprocess
import pathlib
import re
import time
import warnings

def seed_calculator(calculator,paramWrap, cache=True):
    if isinstance(calculator, ParticleFieldBase):
        _seed_particle_calculator(calculator, paramWrap, cache)

def _seed_particle_calculator(calculator, paramWrap,cache):
    ngrid = paramWrap.ngrid
    lat = paramWrap.getLattice()
    calculator.seedCalculator(ngrid,lat)

def generate_field_file(param, calculator, kgridFileName, core=0):
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
    kgridFileName : pathlib.Path
        The path and file name to which to write the resulting field file.
    core : integer
        The index identifying the monomer to be placed in the core of the particles.
    """
    kgridFileName = kgridFileName.resolve()
    monFrac = param.getMonomerFractions()
    if isinstance(calculator, ParticleFieldBase):
        interface = param.getInterfaceWidth(core)
        ngrid = param.ngrid
        calculator.seedCalculator(ngrid)
        newField = calculator.to_kgrid(monFrac, ngrid, interfaceWidth=interface, coreindex=core)
        # Create clean field file if needed.
        kgrid = param.cleanWaveFieldFile()
        kgrid.fields = newField
        kgrid.write(kgridFileName.open(mode='w'))
    elif isinstance(calculator, LamellarFieldGen):
        newField = calculator.to_field(monFrac)
        star = param.cleanStarFieldFile(2) # This option only available for lamellar.
        star.fields = newField
        star.write(kgridFileName.open(mode='w'))
    elif isinstance(calculator,NetworkFieldGen):
        root = kgridFileName.parent
        newField = calculator.to_rgrid(monFrac,param.file,core,root)
        rgrid = param.cleanCoordFieldFile()
        rgrid.fields = newField
        rgrid.write(kgridFileName.open('w'))

def read_input_file(filepath, trace=False, omissionWarnings=False):
    """
    Read an input file for pscfFieldGen and return data for field generation.
    
    Parameters
    ----------
    filepath : pathlib.Path
        The path to the input file. File will be opened and closed during call.
    trace : Boolean (optional, default False)
        If True, a detailed trace of the read will be printed to standard output.
        If False, the call will run silently except for errors and warnings.
    omissionWarnings : Boolean (optional, default False)
        If True, warn the caller about optional data omitted from the file.
        If False, omitted data will be silently set to a default.
    
    Returns
    -------
    param : pscfFieldGen.filemanagers.ParamFile derivative
        The parameter file specified in the file. Exact class is chosen
        based on software specification in the file.
    calculator : pscfFieldGen.generation.UniformParticleField
        The calculator object seeded with necessary structural information.
    outFile : pathlib.Path
        The filepath specified in the input file to output field data.
    core_monomer : int
        The monomer id of the monomer intended to go in the particle cores.
    
    Raises
    ------
    ValueError : 
        When a required input is omitted from the input file.
        Specifically the software, parameter_file, N_particles, 
        and particle_positions fields
    """
    SOFTWARE_MAP = { "pscf" : PscfParam, "pscfpp" : PscfppParam }
    
    # Parse input file
    with filepath.open(mode='r') as cmdFile:
        # Parse file into "words"
        words = wordsGenerator(cmdFile)
        
        # Software
        word = next(words)
        if word == 'software':
            software = next(words)
            ParamFile = SOFTWARE_MAP.get(software,None)
            if ParamFile is None:
                raise(ValueError("Invalid software ({}) given.".format(software)))
            data = software
            if trace:
                print('{}\n\t{}'.format(word, data))
        else:
            raise(ValueError("Input keyword 'software' must be specified first."))
        
        # Parameter File
        word = next(words)
        if word == 'parameter_file':
            filename = next(words)
            param = ParamFile.fromFileName(filename)
            data = filename
            if trace:
                print('{}\n\t{}'.format(word, data))
        else:
            raise(ValueError("Input keyword 'parameter_file' must be specified after software."))
        
        # Output File
        word = next(words)
        if word == 'output_file':
            outfilestring = next(words)
            outFile = pathlib.Path(outfilestring)
            data = outFile
            if trace:
                print('{}\n\t{}'.format(word, data))
        else:
            raise(ValueError("Input keyword 'output_file' must be specified after 'parameter_file'."))
        
        # Structure Type
        word = next(words)
        if word == 'structure_type':
            struct_type = next(words)
            data = struct_type
            if trace:
                print('{}\n\t{}'.format(word, data))
            if struct_type == 'particle':
                calculator, core_mon = _read_particle_input(words,param,trace,omissionWarnings)
            elif struct_type == 'network':
                calculator, core_mon = _read_network_input(words,param,trace,omissionWarnings)
            elif struct_type == 'lamellar':
                calculator = _read_lamellar_input(words, param, trace, omissionWarnings)
                core_mon = 0
            else:
                raise(ValueError("Unrecognized structure_type = {}".format(struct_type)))
        else:
            raise(ValueError("Keyword 'structure_type' must be specified before structure data."))
            
    return param, calculator, outFile, core_mon

def _read_particle_input(words, param, trace=False, omissionWarnings=False):
    """
    Read an input file for pscfFieldGen and return data for field generation.
    
    Parameters
    ----------
    words : WordGenerator
        A stream outputting the string components of the input file.
    trace : Boolean (optional, default False)
        If True, a detailed trace of the read will be printed to standard output.
        If False, the call will run silently except for errors and warnings.
    omissionWarnings : Boolean (optional, default False)
        If True, warn the caller about optional data omitted from the file.
        If False, omitted data will be silently set to a default.
    
    Returns
    -------
    calculator : pscfFieldGen.generation.UniformParticleField
        The calculator object seeded with necessary structural information.
    core_monomer : int
        The monomer id of the monomer intended to go in the particle cores.
    
    Raises
    ------
    ValueError : 
        When a required input is omitted from the input file.
        Specifically the software, parameter_file, N_particles, 
        and particle_positions fields
    """
    # Set initial flags
    hasStyle = False
    hasCore = False
    nparticle = -1
    hasPositions = False
    
    # Set default values
    input_style = 'motif'
    core_monomer = 0
    
    # Parse remainder of input file
    for word in words:
        if word == 'coord_input_style':
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
            else:
                numData = param.dim * nparticle
                positionList = np.array( [str_to_num(next(words)) for i in range(numData)] )
                partPositions = np.reshape(positionList, (nparticle, param.dim))
                data = partPositions
                hasPositions = True
        elif word == 'finish':
            #do nothing
            data = ''
            doneFlag = True
        else:
            raise(NotImplementedError("No operation has been set for keyword {}.".format(word)))
        # if trace requested, echo input file as read
        if trace:
            print('{}\n\t{}'.format(word, data))
    
    # Check for presence of required data
    if nparticle <= 0:
        raise(ValueError("Input keyword 'N_particles' must be specified"))
    if not hasPositions:
        raise(ValueError("Particle coordinates must be specified with keyword 'particle_positions'."))
    
    # Warn of absence of optional data and state assumptions.
    if omissionWarnings:
        if not hasStyle:
            warnings.warn(RuntimeWarning("coord_input_style not specified. 'motif' assumed."))
        if not hasCore:
            warnings.warn(RuntimeWarning("core_monomer not specified. Assuming monomer 0."))
    
    # Create Lattice Object
    if trace:
        print("\nCreating System Lattice")
    latticeParams = param.latticeParameters
    dim = param.dim
    lattice = Lattice.latticeFromParameters(dim, **latticeParams)
    if trace:
        print("\t\t{}".format(lattice))
    
    # Create Crystal Object
    if trace:
        print("\nCreating Crystal\n")
    groupname = param.group_name
    crystalsystem = param.crystal_system
    crystal = buildCrystal( input_style, 
                            nparticle, 
                            partPositions, 
                            lattice, 
                            group_name=groupname,
                            crystal_system=crystalsystem )
    if trace:
        print("Crystal being generated:")
        print(crystal.longString)
    
    # Create Calculator Object
    if trace:
        print("\nSetting Up Calculator")
    calculator = UniformParticleField(crystal)
    
    return calculator, core_monomer

def _read_lamellar_input(words, param, trace, omissionWarnings):
    """
    Read an input file for pscfFieldGen and return data for field generation.
    
    Parameters
    ----------
    words : Stream of single-word streams
        The stream of input file data.
    trace : Boolean (optional, default False)
        If True, a detailed trace of the read will be printed to standard output.
        If False, the call will run silently except for errors and warnings.
    omissionWarnings : Boolean (optional, default False)
        If True, warn the caller about optional data omitted from the file.
        If False, omitted data will be silently set to a default.
    
    Returns
    -------
    calculator : pscfFieldGen.generation.UniformParticleField
        The calculator object seeded with necessary structural information.
    
    Raises
    ------
    ValueError : 
        When Parameter File dimensionality > 1
    """
    word = next(words)
    if not word == 'finish':
        raise(ValueError("Expected 'finish' flag"))
    if not param.dim == 1:
        raise(ValueError("Lamellar Field requires 1-dimensional parameter file"))
    return LamellarFieldGen()

def _read_network_input(words, param, trace, omissionWarnings):
    """
    Read an input file for pscfFieldGen and return data for field generation.
    
    Parameters
    ----------
    filepath : pathlib.Path
        The path to the input file. File will be opened and closed during call.
    trace : Boolean (optional, default False)
        If True, a detailed trace of the read will be printed to standard output.
        If False, the call will run silently except for errors and warnings.
    omissionWarnings : Boolean (optional, default False)
        If True, warn the caller about optional data omitted from the file.
        If False, omitted data will be silently set to a default.
    
    Returns
    -------
    param : pscfFieldGen.filemanagers.ParamFile derivative
        The parameter file specified in the file. Exact class is chosen
        based on software specification in the file.
    calculator : pscfFieldGen.generation.UniformParticleField
        The calculator object seeded with necessary structural information.
    outFile : pathlib.Path
        The filepath specified in the input file to output field data.
    core_monomer : int
        The monomer id of the monomer intended to go in the particle cores.
    
    Raises
    ------
    ValueError : 
        When a required input is omitted from the input file.
        Specifically the software, parameter_file, N_particles, 
        and particle_positions fields
    """
    hasParam = False
    hasField = False
    hasCore = False
    
    core_monomer = 0
    
    # Parse input file
    for word in words:
        if word == 'network_parameter_file':
            filename = next(words)
            fieldParam = pscf.ParamFile(filename)
            hasParam = True
            data = filename
        elif word == 'network_star_file':
            filename = next(words)
            fieldFile = pscf.SymFieldFile(filename)
            hasField = True
            data = filename
        elif word == 'core_monomer':
            core_monomer = int(next(words))
            if core_monomer >= 0:
                hasCore = True
                data = core_monomer
            else:
                raise(ValueError("core_monomer must be a non-negative integer. Given {}.".format(core_monomer)))
        elif word == 'finish':
            #do nothing
            data = ''
            doneFlag = True
        else:
            raise(ValueError("Unrecognized keyword {}.".format(word)))
        if trace:
            print('{}\n\t{}'.format(word, data))
    
    if not hasParam:
        raise(ValueError("Missing Required input 'network_parameter_file'."))
    if not hasField:
        raise(ValueError("Missing Required input 'network_star_file'"))
    
    # Warn of absence of optional data and state assumptions.
    if omissionWarnings:
        if not hasCore:
            warnings.warn(RuntimeWarning("core_monomer not specified. Assuming monomer 0."))
    
    calculator = NetworkFieldGen(fieldParam,fieldFile)
    
    return calculator, core_monomer

class ParticleGenerator:
    """ Field Generator Implementing the Form-Factor Method
    """
    def __init__(self, crystal):
        """ Initialize a new FieldCalculator.
        
        Parameters
        ----------
        crystal : pscfFieldGen.structure.BasisCrystal or MotifCrystal
            The crystal structure the calculator is supposed to produce
            fields for.
        """
        self._crystal = crystal
    
    @classmethod
    def fromFile(cls, wordstream, entrykey, param):
        """ Initialize and return a new ParticleGenerator from a file.
        
        Parameters
        ----------
        wordstream : util.stringTools.FileParser
            The data stream for the file.
        entrykey : string
            The key marking entry into this block.
            Should be "ParticleGenerator{"
        param : paramfile object
            The param file being used as a basis.
        """
        if not entrykey == "ParticleGenerator{":
            msg = "ParticleGenerator expected key 'ParticleGenerator{{'; got '{}'"
            raise(ValueError(msg.format(entrykey)))
        word = next(wordstream)
        if not isCrystalKey(word):
            msg = "Unrecognized keyword '{}' in ParticleGenerator{{...}} block."
            raise(ValueError(msg.format(word)))
        crystal = readCrystalFromFile(wordstream, entrykey, param)
        word = next(wordstream)
        if not word == "}":
            msg = "Expected '}}' to close ParticleGenerator{{ block; got {}."
            raise(ValueError(msg.format(word)))
        return cls(crystal)
    
    def to_kgrid(self, frac, ngrid, lattice=None, interfaceWidth=0.0):
        """
        Return the reciprocal space grid of densities.
        
        Parameters
        ----------
        frac : numerical, array-like
            volume fractions of all monomer types. Sum of all values = 1.
            Value at index 0 represents the "core" or particle-forming monomer.
            And must also be monomer 1 by PSCF indications.
        ngrid : int, array-like
            The number of grid points in each (real-space) direction.
        lattice : pscfFieldGen.structure.Lattice, optional
            The lattice on which to generate the field.
            If not specified, the most recent lattice will be reused.
        interfaceWidth : float
            The interface width to be used in smearing interfaces.
        
        Returns
        -------
        kgrid : numpy.ndarray
            Full array of all Fourier amplitudes for each wavevector.
            Shape is (n_vector, n_monomer).
        """
        twopi = 2.0 * np.pi
        imag = 1.0j # store reference to unit complex value
        
        frac = np.array(frac)
        nspecies = len(frac) # number of monomers
        ngrid = np.array(ngrid)
        nbrill =  getKgridCount(ngrid) # number of wavevectors
        if lattice is not None:
            self._crystal.lattice = lattice # update lattice
        lat = self._crystal.lattice # store local reference to lattice
        vol = lat.volume # unit cell volume
        coreindex, refVolume = self._crystal.chooseCore(frac) # determine core
        
        rho = np.zeros((nbrill,nspecies),dtype=np.complex128) # pre-allocate
        for (t, wave) in enumerate(IterableWavevector(ngrid, lat)):
            if t == 0: 
                # 0-th wave-vector -- corresponds to volume fractions
                rho[t,:] = frac[:] 
            else:
                total = rho[t,coreindex] # local reference to zero-initialized amplitude
                q_norm = twopi * wave.magnitude
                fsmear = np.exp( -( (interfaceWidth**2) * q_norm**2) / 2.0 )
                for particle in self._crystal:
                    eiqR = np.exp( imag * twopi * wave * particle.position )
                    ff = particle.formFactorAmplitude( wave, refVol )
                    total += eiqR * ff
                rho[t, coreindex] = total * (1/vol) * fsmear
                rhoTemp = -rho[t, coreindex] / (1 - frac[coreindex])
                for j in range(nspecies):
                    if not j == coreindex:
                        rho[t, j] = rhoTemp * frac[j]
                for (j,r) in enumerate(rho[t,:]):
                    if r == -0.0:
                        rho[t,j] = 0.0
        return rho

class LamellarGenerator(object):
    def __init__(self):
        pass
    
    @classmethod
    def fromFile(cls, wordstream, entrykey, param):
        altForms = [ "LamellarGenerator{}", "LamellarGenerator"]
        if entrykey == "LamellarGenerator{":
            word = next(wordstream)
            if not word == "}":
                msg = "Expected '}}' to close LamellarGenerator{{ block; got {}."
                raise(ValueError(msg.format(word)))
        elif entrykey in altForms:
            pass # because no input is required, shorthand formats are accepted.
        else:
            msg = "Unrecognized key ({}) for LamellarGenerator."
            raise(ValueError(msg.format(entrykey)))
        return cls()
    
    def to_field(self,monfrac):
        """ 
        A Field Generation Method for the lamellar phase.
        
        Returns a symmetrized field array with 2 stars for
        each monomer.
        """
        nmon = len(monfrac)
        monfrac = np.array(monfrac)
        minind = np.argmin(monfrac)
        minfrac = monfrac[minind]
        out = np.zeros((2,nmon))
        out[0,:] = monfrac
        out[1,minind] = minfrac / np.sqrt(2)
        rhotemp = -out[1,minind] / (1 - monfrac[minind])
        for i in range(nmon):
            if not i == minind:
                out[1,i] = rhotemp * monfrac[i]
        return out

class NetworkGenerator(object):
    """ A generator for network phase initial guesses. """
    def __init__(self, pfile, crystal):
        """ 
        Initialize the NetworkFieldGen.
        
        Presently, Network field generation supports only Fortran version of PSCF.
        
        Parameters
        ----------
        pfile : pscfFieldGen.filemanagers.pscf.ParamFile
            A PSCF (Fortran) parameter file containing the
            heading, followed by a FIELD_TO_RGRID command
            to be used for generating the level-set determining
            rgrid field.
        crystal : pscfFieldGen.structure.network.NetworkCrystal
            A NetworkCrystal Field-file wrapper.
        """
        self._param = pfile
        self._crystal = crystal
        inputFname = self._param.fieldTransforms[0][1]
        inputFname += "_internal"
        self._param.fieldTransforms[0][1] = inputFname
        self._sym_name = pfile.fieldTransforms[0][1]
    
    @classmethod
    def fromFile(cls, wordstream, entrykey, param):
        """ Initialize and return a new NetworkGenerator from a file.
        
        Parameters
        ----------
        wordstream : util.stringTools.FileParser
            The data stream for the file.
        entrykey : string
            The key marking entry into this block.
            Should be "NetworkGenerator{".
        param : pscfFieldGen.filemanagers.pscf.ParamFile object
            The param file being used as a basis.
        """
        if not entrykey == "NetworkGenerator{":
            msg = "NetworkGenerator expected key 'NetworkGenerator{{'; got {}"
            raise(ValueError(msg.format(entrykey)))
        word = next(wordstream)
        if word == 'network_param':
            filename = next(wordstream)
            netparam = pscf.ParamFile(filename)
        else:
            msg = "Expected keyword 'network_param': got {}."
            raise(ValueError(msg.format(word)))
        word = next(wordstream)
        if word == "NetworkCrystal{":
            crystal = NetworkCrystal.fromFile(wordstream, word, netparam)
        elif word.lower() == 'star_file':
            val = next(wordstream)
            crystal = NetworkCrystal(val)
        else:
            msg = "Expected block 'NetworkStar{{...}}' or keyword 'star_file': got {}."
            raise(ValueError(msg.format(word)))
        word = next(wordstream)
        if not word == "}":
            msg = "Expected '}}' to close NetworkGenerator{{ block; got {}."
            raise(ValueError(msg.format(word)))
        return cls(netparam, crystal)
    
    def _to_raw_rgrid(self):
        outfile = "networkgenlog"
        pfile = "param_network_internal"
        with open(pfile,'w') as f:
            self._param.write(f)
        with open(self._sym_name,'w') as f:
            self._crystal.write(f)
        with open(pfile) as fin:
            with open(outfile,'w') as fout:
                lastLaunch = subprocess.run("pscf",stdin=fin,stdout=fout)
        lastLaunch.check_returncode()
        outfield = self._param.fieldTransforms[0][2]
        rgrid = pscf.CoordFieldFile(outfield)
        return rgrid.fields
    
    def to_rgrid(self, frac, param, root):
        """
        Use the level-set method to generate an initial guess field.
        
        Parameters
        ----------
        frac : list-like
            Overall volume fractions of each monomer.
        param : filemanagers.[pscf.ParamFile or pscfpp.ParamFile]
            Parameter file being used in the final calculation.
        root : pathlib.Path
            The Root directory to perform the level-set operations.
            The method will generate its own sub-directory within this
            root in order to isolate its internal files.
        """
        self._param.cell_param = deepcopy(param.cell_param)
        coreMon = self._crystal.update(frac, self._param.cell_param)
        
        # Convert symmetrized rho to coordinate grid using PSCF
        internalroot = root/"_network_generator_internal_"
        internalroot.mkdir()
        with contexttools.cd(internalroot):
            rgrid_raw = self._to_raw_rgrid()
        rgrid = np.zeros_like(rgrid_raw)
        
        # Find levelset threshold
        hist, edges = np.histogram(rgrid_raw[:,coreMon],bins=1000)
        npoints = np.sum(hist)
        fnorm = hist / npoints
        fcum = np.cumsum(fnorm)
        setbin = np.argmin(np.abs(fcum - fcore))
        setval = edges[setbin]
        
        # Search for points within the level set
        for i in range(npoints):
            if rgrid_raw[i,coreMon] < setval:
                rgrid[i,coreMon] = 1.0
            else:
                for j in range(nmon):
                    if not j == coreMon:
                        rgrid[i,j] = frac[j]/fnonCore
        return rgrid
        
