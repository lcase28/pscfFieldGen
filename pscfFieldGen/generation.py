# Project imports
from pscfFieldGen.structure import ( 
    Lattice,
    CrystalBase,
    CrystalMotif,
    SpaceGroup,
    ParticleBase,
    ScatteringParticle,
    buildCrystal )

from pscfFieldGen.filemanagers import PscfParam, PscfppParam
from pscfFieldGen.util.stringTools import str_to_num, wordsGenerator

# Standard Library Imports
import argparse
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
import itertools
import numba
import numpy as np
import scipy as sp
import pathlib
import re
import time
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
    kgridFileName : pathlib.Path
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
    startTime = time.time()
    calculator.seedCalculator(ngrid)
    newField = calculator.to_kgrid(monFrac, ngrid, interfaceWidth=interface, coreindex=core)
    endTime = time.time()
    rtime = endTime - startTime
    print("\nCalculated field in {} seconds.\n".format(rtime))
    #dataset = {"name":kgridFileName.parent.name,"param":param,"calc":calculator,"field":newField}
    #with open("numbaData",'wb') as f:
    #    pickle.dump(dataset,f)
    # Create clean field file if needed.
    if kgrid is None:
        kgrid = param.cleanFieldFile()
    kgrid.fields = newField
    kgrid.write(kgridFileName.open(mode='x'))

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
                outFile = pathlib.Path(outfilestring)
                data = outFile
                hasOutFile = True
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
    if not hasSoftware:
        raise(ValueError("Input keyword 'software' must be specified"))
    if not hasParam:
        raise(ValueError("Input keyword 'parameter_file' must be specified"))
    if nparticle <= 0:
        raise(ValueError("Input keyword 'N_particles' must be specified"))
    if not hasPositions:
        raise(ValueError("Particle coordinates must be specified with keyword 'particle_positions'."))
    
    # Warn of absence of optional data and state assumptions.
    if omissionWarnings:
        if not hasStyle:
            warnings.warn(RuntimeWarning("coord_input_style not specified. 'motif' assumed."))
        if not hasOutFile:
            warnings.warn(RuntimeWarning("Output file name not specified with keyword 'output_file'. Using 'rho_kgrid'."))
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
    
    return param, calculator, outFile, core_monomer

class FieldCalculatorBase(ABC):
    """ Base class for varieties of FieldCalculator Styles.
    
    """
    def __init__(self, crystal):
        """ Initialize a new FieldCalculator.
        
        The crystal structure is input as a derivative of CrystalBase.
        All deriving classes should plan to work within the CrystalBase
        structure framework, but can have extensions to the crystal to 
        enable any special behavior.
        
        Parameters
        ----------
        crystal : pscfFieldGen.structure.CrystalBase or CrystalMotif
            The crystal structure the calculator is supposed to produce
            fields for.
        """
        self.crystal = crystal
        self.dim = self.crystal.dim
        self.nparticles = self.crystal.n_particles
        self.lattice = self.crystal.lattice
        self.reciprocal_lattice = self.lattice.reciprocal
        self.__brillouin_cache = dict()
        self.__kgrid_cache = dict()
        self.__nstep_cache = dict()
        super().__init__()
    
    def seedCalculator(self,ngrid,lattice=None):
        """ Verify that cached values are available for given ngrid and lattice.
        
        Field Calculators will cache reusable values from the calculation.
        Since each calculator has a fixed crystal structure, these cached values
        are determined by ngrid and the lattice. This method will check for
        cached results relevant to the given ngrid and lattice parameters.
        If they are available, no more work will be done. If they are not, 
        results will be calculated and cached.
        
        Derived classes should override and include a call to super().seedCalculator
        
        When overriding, values should be cached according to the minimum
        parameters making them unique. For example, Brillouin zone wave-vectors
        depend only on the ngrid, not on lattice; thus only ngrid should be
        used to check for their value.
        
        Parameters
        ----------
        ngrid : array-like
            The number of grid points along each lattice dimension.
        lattice : pscfFieldGen.structure.Lattice
            The lattice representation.
        
        Returns
        -------
        ngrid : array-like
            Echo's the ngrid used for seeding.
        lattice : pscfFieldGen.structure.lattice
            Returns the lattice that was actually used for the seeding.
        """
        lattice = self.chooseLattice(lattice)
        self.getBrillouinArray(ngrid)
        return ngrid, lattice
    
    def chooseLattice(self,lat):
        """ Choose either default or override lattice and reciprocal lattice. """
        if lat is None:
            return self.lattice
        else:
            raise(NotImplementedError("Lattice Override checks have not been implemented"))
        
    def getKgrid(self,ngrid):
        key = str(tuple(ngrid))
        if key in self.__kgrid_cache:
            kgrid = self.__kgrid_cache.get(key)
        else:
            ngrid = np.array(ngrid)
            # Shift grid for k-grid
            kgrid = np.zeros_like(ngrid)
            for (i,x) in enumerate(ngrid):
                if i == 0:
                    kgrid[i] = (x/2) + 1
                else:
                    kgrid[i] = x
            self.__kgrid_cache.update({key:kgrid})
        return kgrid
    
    def getNumKgridPoints(self,ngrid):
        """ Calculate the total number of points in reciprocal grid. """
        key = str(tuple(ngrid))
        if key in self.__nstep_cache:
            record = self.__nstep_cache.get(key)
        else:
            kgrid = self.getKgrid(ngrid)
            record = np.prod(kgrid)
            self.__nstep_cache.update({key:record})
        return record
    
    def getBrillouinArray(self,ngrid):
        """ Determine the wavevectors in the first brillouin zone.
        
        From the ngrid real-space discretization, determine the full
        set of wave-vectors shifted to the first brillouin zone.
        Return the result as a numpy array where each row is a wave-vector.
        
        Brillouin vector sets are stored after initial generation to be
        available for repeat calculations. If prior results are available,
        the same array is returned.
        
        Parameters
        ----------
        ngrid : array-like
            The number of grid points along each lattice vector in the
            real-space discretization.
        
        Returns
        -------
        brillouin : numpy.ndarray (Treat as Read-only)
            The array of wave-vectors.
        """
        key = str(tuple(ngrid))
        if key in self.__brillouin_cache:
            record = self.__brillouin_cache.get(key)
        else:
            kgrid = self.getKgrid(ngrid)
            nvect = self.getNumKgridPoints(ngrid)
            ngrid = np.array(ngrid)
            record = FieldCalculatorBase.__generate_brillouin(ngrid,kgrid,nvect)
            self.__brillouin_cache.update({key:record})
        return record
    
    @staticmethod
    @numba.jit(nopython=True,cache=True)
    def __generate_brillouin(ngrid,kgrid,nvect):
        """ Generate brillouin zone wave-vectors """
        dim = len(ngrid)
        dshift = dim - 1  #used in aliasing
        # Define aliasing method
        def miller_to_brillouin(G):
            out = np.zeros_like(G,dtype=np.float64)
            out[0] = G[0]
            for i in [1,2]:
                if dshift >= i:
                    if G[i] > ngrid[i]/2:
                        out[i] = G[i] - ngrid[i]
                    else:
                        out[i] = G[i]
            return out
        # Iterate
        brillouin = np.zeros((nvect,dim),dtype=np.float64)
        G = np.zeros_like(kgrid,dtype=np.float64)
        n = 0
        if dim == 3:
            for i in range(kgrid[0]):
                for j in range(kgrid[1]):
                    for k in range(kgrid[2]):
                        G[0] = i
                        G[1] = j
                        G[2] = k
                        brillouin[n,:] = miller_to_brillouin(G)
                        n = n + 1
        elif dim == 2:
            for i in range(kgrid[0]):
                for j in range(kgrid[1]):
                    G[0] = i
                    G[1] = j
                    brillouin[n,:] = miller_to_brillouin(G)
                    n = n + 1
        else:
            raise(ValueError("ngrid must match 2 or 3 dimensional system"))
        return brillouin
    
    @abstractmethod
    def to_kgrid(self, frac, ngrid, interfaceWidth, coreindex=0, lattice=None):
        """
        Return the reciprocal space grid of monomer volume fractions
        
        Parameters
        ----------
        frac : numerical, array-like
            volume fractions of all monomer types. Sum of all values = 1.
            Value at index 0 represents the "core" or particle-forming monomer.
            And must also be monomer 1 by PSCF indications.
        ngrid : int, array-like
            The number of grid points in each (real-space) direction.
        interfaceWidth : float
            An estimated width of the interface for smearing particle edges.
        coreindex : int (optional)
            The monomer id number for the monomer taken to be in the core of the 
            particles. Default is 0.
        lattice : pscfFieldGen.structure.Lattice (optional)
            An overloading Lattice to use in place of the default. Must match the type
            of the default lattice.
        
        Returns
        -------
        rho : numpy.ndarray (complex-valued)
            An array of fourier coefficients for the monomer volume fraction field.
            Should contain a row for each fourier space grid point, and a column for
            each monomer.
        """
        pass
    
class UniformParticleField(FieldCalculatorBase):
    """ Calculates density fields assuming all particles are equal size. """
    
    def __init__(self, crystal):
        """
        Initialize a new FieldGenerator.
        
        Parameters
        ----------
        crystal : pscfFieldGen.structure.CrystalBase or CrystalMotif
            The crystal structure the calculator is supposed to produce
            fields for.
        """
        super().__init__(crystal)
        self.partForm = self.crystal.particles[0].formFactor
        self.__complexSum_cache = dict()
        self.__qNorm_cache = dict()
    
    def seedCalculator(self,ngrid,lattice=None):
        """ Verify that cached values are available for given ngrid and lattice.
        
        Field Calculators will cache reusable values from the calculation.
        Since each calculator has a fixed crystal structure, these cached values
        are determined by ngrid and the lattice. This method will check for
        cached results relevant to the given ngrid and lattice parameters.
        If they are available, no more work will be done. If they are not, 
        results will be calculated and cached.
        
        Derived classes should override and include a call to super().seedCalculator
        
        When overriding, values should be cached according to the minimum
        parameters making them unique. For example, Brillouin zone wave-vectors
        depend only on the ngrid, not on lattice; thus only ngrid should be
        used to check for their value.
        
        Parameters
        ----------
        ngrid : array-like
            The number of grid points along each lattice dimension.
        lattice : pscfFieldGen.structure.Lattice
            The lattice representation.
        
        Returns
        -------
        ngrid : array-like
            Echo's the ngrid used for seeding.
        lattice : pscfFieldGen.structure.lattice
            Returns the lattice that was actually used for the seeding.
        """
        ngrid, lattice = super().seedCalculator(ngrid,lattice)
        self.getComplexSum(ngrid)
        self.getQNorm(ngrid,self.lattice,True)
        return ngrid, self.lattice
    
    def getComplexSum(self,ngrid):
        """ Determine the complexSum values.
        
        The values returned represent sum_{particles}(exp(iq*R_{j})) for
        each q.
        
        Parameters
        ----------
        ngrid : array-like
            The number of grid points along each lattice vector in the
            real-space discretization.
        
        Returns
        -------
        complexSum : numpy.ndarray (Treat as Read-only)
            The array of partial fourier coefficients.
        """
        key = str(tuple(ngrid))
        if key in self.__complexSum_cache:
            record = self.__complexSum_cache.get(key)
        else:
            nbrill = self.getNumKgridPoints(ngrid)
            #brill = np.asarray(self.getBrillouinArray(ngrid), dtype=np.float64)
            brill = self.getBrillouinArray(ngrid)
            npos = self.nparticles
            pos = self.getPositionArray()
            record = UniformParticleField.__calculate_sums(nbrill,brill,npos,pos)
            self.__complexSum_cache.update({key:record})
        return record
    
    def getQNorm(self,ngrid,lat,holdValue=False):
        """ Determine the magnitude of wave-vectors.
        
        Parameters
        ----------
        ngrid : array-like
            The number of grid points along each lattice vector in the
            real-space discretization.
        lattice : pscfFieldGen.structure.Lattice
            The lattice to be used to calculate the magnitude.
        
        Returns
        -------
        qNorm : numpy.ndarray (Treat as Read-only)
            Wave-Vector Magnitudes.
        """
        key = str(tuple(ngrid))
        key += str(tuple(lat.latticeParameters))
        if key in self.__complexSum_cache:
            record = self.__complexSum_cache.get(key)
        else:
            nbrill = self.getNumKgridPoints(ngrid)
            brill = np.asarray(self.getBrillouinArray(ngrid), dtype=np.float64)
            #record = UniformParticleField.__calculate_qnorm(nbrill,brill,lattice.reciprocal)
            record = UniformParticleField.__calculate_qnorm_metTen(nbrill,brill,lat.reciprocal.metricTensor)
            if holdValue:
                self.__complexSum_cache.update({key:record})
        return record
    
    def getPositionArray(self):
        pos = np.zeros((self.nparticles,self.dim))
        n = 0
        for r in self.crystal.particlePositions():
            pos[n,:] = r
            n += 1
        return pos
    
    @staticmethod
    @numba.jit(nopython=True,cache=True)
    def __calculate_sums(nbrill,brill,npos,pos):
        out = np.zeros(nbrill,dtype=np.complex128)
        for b in range(nbrill):
            #R = 0.0
            #I = 0.0
            q = brill[b,:]
            for p in range(npos):
                r = pos[p,:]
                qr = 2.0 * np.pi * np.dot(q,r)
                out[b] += np.exp( 1.0j * qr )
                #R = R + np.cos(qr)
                #I = I + np.sin(qr)
            #out[b] = R + 1.0j*I
        return out
    
    ## TODO: Make Lattice Class compatible with numba.jit
    @staticmethod
    @numba.jit(nopython=True,cache=True)
    def __calculate_qnorm(nbrill, brill, lattice):
        out = np.zeros(nbrill)
        for i in range(nbrill):
            q = brill[i,:]
            out[i] = lattice.vectorNorm(q)
        return out
    
    @staticmethod
    @numba.jit(nopython=True,cache=True)
    def __calculate_qnorm_metTen(nbrill, brill, metTen):
        # Temporary workaround until lattice can be directly interfaced from jit
        out = np.zeros(nbrill)
        dim = len(brill[0,:])
        for i in range(nbrill):
            q = brill[i,:]
            out[i] = 2.0 * np.pi * np.sqrt( np.dot( q, np.dot( metTen, q ) ) )
        return out
    
    def to_kgrid(self, frac, ngrid,interfaceWidth=None, coreindex=0,lattice=None):
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
        """
        frac = np.array(frac)
        nspecies = frac.size
        vol = self.lattice.volume
        particleVol = frac[coreindex] * vol / self.nparticles
        
        nbrill = self.getNumKgridPoints(ngrid)
        brill = self.getBrillouinArray(ngrid)
        compSums = self.getComplexSum(ngrid)
        lattice = self.chooseLattice(lattice)
        qnorms = self.getQNorm(ngrid,lattice)
        form = self.partForm
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            formFact = self.__getFormFactors(qnorms,particleVol,form)
        
        args = [nspecies, frac, nbrill,brill, compSums, qnorms, vol, formFact, interfaceWidth, coreindex]
        kgrid = self.__kgrid_calc(*args)
        return kgrid
    
    @staticmethod
    @numba.jit(forceobj=True)
    def __getFormFactors(qnorms,particleVol,formFact):
        out = np.zeros_like(qnorms)
        n = len(qnorms)
        for t in range(n):
            if t == 0:
                out[t] = 0.0
            else:
                out[t] = formFact(qnorms[t], particleVol)
        return out
        
    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def __kgrid_calc(   nspecies, 
                        frac, 
                        nbrill, 
                        brill,
                        compSums, 
                        qnorms, 
                        vol, 
                        formFact,
                        interfaceWidth,
                        coreindex ):
        rho = np.zeros((nbrill,nspecies),dtype=np.complex128)
        for t in range(nbrill):
            brillouin = brill[t,:]
            compSum = compSums[t]
            q_norm = qnorms[t]
            if t == 0: 
                # 0-th wave-vector -- corresponds to volume fractions
                rho[t,:] = frac[:] 
            else:
                ff = formFact[t] #particleForm(q_norm, particleVol)
                fsmear = np.exp(-( (interfaceWidth**2) * q_norm**2) / 2.0)
                rho[t, coreindex] = compSum * (1/vol) * ff * fsmear
                rhoTemp = -rho[t, coreindex] / (1 - frac[coreindex]) #np.sum(frac[1:])
                for i in range(nspecies):
                    if not i == coreindex:
                        rho[t, i] = rhoTemp * frac[i]
                for (i,r) in enumerate(rho[t,:]):
                    if r == -0.0:
                        rho[t,i] = 0.0
        return rho
