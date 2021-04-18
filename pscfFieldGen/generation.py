# Project imports
from pscfFieldGen.structure import ( 
    Lattice,
    BasisCrystal,
    MotifCrystal,
    SpaceGroup )
from pscfFieldGen.structure.grids import(
    getKgridCount,
    IterableWavevector )
from pscfFieldGen.structure.crystals import (
    isCrystalKey,
    readCrystalFromFile )
from pscfFieldGen.structure.network import NetworkCrystal
from pscfFieldGen.filemanagers import PscfParam, PscfppParam
from pscfFieldGen.util.stringTools import str_to_num, wordsGenerator, FileParser
import pscfFieldGen.util.contexttools as contexttools
from pscfFieldGen.util.tracing import TraceLevel, TRACER, debug
import pscfFieldGen.filemanagers.pscf as pscf

# Standard Library Imports
import argparse
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
import itertools
import numpy as np
import scipy as sp
import subprocess
import pathlib
import re
import time
import warnings

def generate_field_file(param, calculator, kgridFileName):
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
    """
    kgridFileName = kgridFileName.resolve()
    monFrac = param.getMonomerFractions()
    if isinstance(calculator, ParticleGenerator):
        newField = calculator.to_kgrid(param)
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
        newField = calculator.to_rgrid(monFrac,param.file,root)
        rgrid = param.cleanCoordFieldFile()
        rgrid.fields = newField
        rgrid.write(kgridFileName.open('w'))

def read_input_file(filepath, trace=TraceLevel.NONE):
    """
    Read an input file for pscfFieldGen and return data for field generation.
    
    Parameters
    ----------
    filepath : pathlib.Path
        The path to the input file. File will be opened and closed during call.
    trace : TraceLevel
        The TraceLevel to use as a filter in output.
    
    Returns
    -------
    param : pscfFieldGen.filemanagers.ParamFile derivative
        The parameter file specified in the file. Exact class is chosen
        based on software specification in the file.
    calculator : ParticleGenerator, LamellarGenerator, NetworkGenerator
        The calculator object seeded with necessary structural information.
    outFile : pathlib.Path
        The filepath specified in the input file to output field data.
    
    Raises
    ------
    ValueError : 
        When a required input is omitted from the input file.
        Specifically the software, parameter_file, N_particles, 
        and particle_positions fields
    """
    SOFTWARE_MAP = { "pscf" : PscfParam, "pscfpp" : PscfppParam }
    
    TRACER.filterLevel = trace
    
    # Parse input file
    with FileParser(filepath) as words:
        # Block opener
        word = next(words)
        if not word == "PscfFieldGen{":
            msg = "Expected key 'PscfFieldGen{{'; got '{}'"
            raise(ValueError(msg.format(word)))
        TRACER.trace("Reading pscfFieldGen input file.",TraceLevel.EVENT)
        # Software
        word = next(words)
        if word == 'software':
            software = next(words)
            ParamFile = SOFTWARE_MAP.get(software,None)
            if ParamFile is None:
                raise(ValueError("Invalid software ({}) given.".format(software)))
            data = software
            TRACER.trace("Using software {}.".format(software),TraceLevel.ALL)
        else:
            raise(ValueError("Input keyword 'software' must be specified first."))
        # Parameter File
        word = next(words)
        if word == 'parameter_file':
            filename = next(words)
            param = ParamFile.fromFileName(filename)
            data = filename
            TRACER.trace("Using Parameter File {}.".format(data),TraceLevel.ALL)
        else:
            raise(ValueError("Input keyword 'parameter_file' must be specified after software."))
        # Output File
        word = next(words)
        if word == 'output_file':
            outfilestring = next(words)
            outFile = pathlib.Path(outfilestring)
            data = outFile
            TRACER.trace("Will write field to file {}.".format(data),TraceLevel.ALL)
        else:
            raise(ValueError("Keyword 'output_file' must be specified after 'parameter_file'."))
        # Structure Type
        word = next(words)
        if isGeneratorKey(word):
            calculator = readGeneratorFromFile(words,word,param)
        else:
            raise(ValueError("Unrecognized Keyword {}."))
        # End of Block
        word = next(words)
        if not word == "}":
            msg = "Expected '}}' to close PscfFieldGen{{ block; got {}."
            raise(ValueError(msg.format(word)))
    return param, calculator, outFile

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
        TRACER.trace("Reading ParticleGenerator from File.",TraceLevel.EVENT)
        word = next(wordstream)
        if not isCrystalKey(word):
            msg = "Unrecognized keyword '{}' in ParticleGenerator{{...}} block."
            raise(ValueError(msg.format(word)))
        crystal = readCrystalFromFile(wordstream, word, param)
        word = next(wordstream)
        if not word == "}":
            msg = "Expected '}}' to close ParticleGenerator{{ block; got {}."
            raise(ValueError(msg.format(word)))
        TRACER.trace("Done reading ParticleGenerator.",TraceLevel.EVENT)
        return cls(crystal)
    
    def to_kgrid(self, param, frac=None, ngrid=None, lattice=None, interfaceWidth=None):
        """
        Return the reciprocal space grid of densities.
        
        Parameters
        ----------
        param : pscfFieldGen.filemanagers.ParamFile
            The parameter file on which to base the generated field.
        frac : numerical, array-like (optional)
            volume fractions of all monomer types. Sum of all values = 1.
            If included, values will override those determined from param.
        ngrid : int, array-like (optional)
            The number of grid points in each (real-space) direction.
            If included, will override value determined from param.
        lattice : pscfFieldGen.structure.Lattice, (optional)
            The lattice on which to generate the field.
            If included, will override values determined from param.
        interfaceWidth : float, (optional)
            The interface width to be used in smearing interfaces.
            If included, will override values determined from param.
        
        Returns
        -------
        kgrid : numpy.ndarray
            Full array of all Fourier amplitudes for each wavevector.
            Shape is (n_vector, n_monomer).
        """
        _fn_ = "ParticleGenerator.to_kgrid"
        # Check inputs
        if frac is None:
            frac = param.getMonomerFractions()
        if ngrid is None:
            ngrid = param.ngrid
        if lattice is None:
            lattice = param.getLattice()
        # Constants
        twopi = 2.0 * np.pi
        imag = 1.0j # store reference to unit complex value
        # Calculate stable values before loop
        frac = np.array(frac)
        debug(_fn_,"monomer fractions = {}",frac)
        debug(_fn_,"generating on {}",lattice)
        nspecies = len(frac) # number of monomers
        ngrid = np.array(ngrid)
        debug(_fn_,"ngrid = {}",ngrid)
        nbrill =  getKgridCount(ngrid) # number of wavevectors
        self._crystal.lattice = lattice # update lattice
        lat = self._crystal.lattice # store local reference to lattice
        vol = lat.volume # unit cell volume
        debug(_fn_,"unit cell volume = {}",vol)
        coreindex, refVolume = self._crystal.chooseCore(frac) # determine core
        debug(_fn_,"core index = {}",coreindex)
        debug(_fn_,"particle volume = {}",refVolume)
        if interfaceWidth is None:
            interfaceWidth = param.getInterfaceWidth(coreindex)
        # Generate rho field
        rho = np.zeros((nbrill,nspecies),dtype=np.complex128) # pre-allocate
        # Iterate over wavevectors
        for (t, wave) in enumerate(IterableWavevector(ngrid, lat)):
            debug(_fn_,"wavevector {}",t)
            debug(_fn_,"q = {}",wave)
            if t == 0: 
                # 0-th wave-vector -- corresponds to volume fractions
                rho[t,:] = frac[:] 
            else:
                total = rho[t,coreindex] # local reference to zero-initialized amplitude
                q_norm = twopi * wave.magnitude
                debug(_fn_,"q_norm = |q| = {}",q_norm)
                fsmear = np.exp( -( (interfaceWidth**2) * q_norm**2) / 2.0 )
                debug(_fn_,"f_smear = {}",fsmear)
                # iterate over particles
                for particle in self._crystal:
                    debug(_fn_,"particle {}",particle)
                    qR = wave * particle.position 
                    debug(_fn_,"q*R = {}",qR)
                    eiqR = np.exp( imag * twopi * qR )
                    debug(_fn_,"exp(i*q*R) = {}",eiqR)
                    ff = particle.formFactorAmplitude( wave, refVolume )
                    debug(_fn_,"f_j(q) = {}",ff)
                    total += eiqR * ff
                    debug(_fn_,"running total f_j(q)*exp(i*q*R) = {}",total)
                rho[t, coreindex] = total * (1/vol) * fsmear
                rhoTemp = -rho[t, coreindex] / (1 - frac[coreindex])
                for j in range(nspecies):
                    if not j == coreindex:
                        rho[t, j] = rhoTemp * frac[j]
                for (j,r) in enumerate(rho[t,:]):
                    if r == -0.0:
                        rho[t,j] = 0.0
            debug("ParticleGenerator.to_kgrid","rho[{},:] = {}",t,rho[t,:])
        return rho

class LamellarGenerator:
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
        TRACER.trace("Read LamellarGenerator from file.",TraceLevel.EVENT)
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

class NetworkGenerator:
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
        TRACER.trace("Reading NetworkGenerator from File.",TraceLevel.EVENT)
        word = next(wordstream)
        if word == 'network_param':
            filename = next(wordstream)
            netparam = pscf.ParamFile(filename)
        else:
            msg = "Expected keyword 'network_param': got {}."
            raise(ValueError(msg.format(word)))
        TRACER.trace("Read network param file.",TraceLevel.EVENT)
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
        TRACER.trace("Finished NetworkGenerator.",TraceLevel.EVENT)
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

_entry_key_map = {  "ParticleGenerator{" :   ParticleGenerator, \
                    "LamellarGenerator{" :   LamellarGenerator, \
                    "NetworkGenerator{" :   NetworkGenerator  }

def isGeneratorKey(entryKey):
    """ Return True if valid entryKey is given. """
    return entryKey in _entry_key_map

def readGeneratorFromFile(wordstream, entrykey, param):
    """ Return Crystal object read from file.
    
    Type of crystal is chosen based on entrykey.
    
    Parameters
    ----------
    wordstream : util.stringTools.FileParser
        The data stream from the input file.
    entryKey : string
        The entry key triggering the call.
    param : ParamFile
        The parameter file on which the structure
        is based.
    """
    if not isGeneratorKey(entrykey):
        msg = "No Generator Type associated with key {}."
        raise(ValueError(msg.format(entrykey)))
    cls = _entry_key_map.get(entrykey)
    return cls.fromFile(wordstream,entrykey,param)

