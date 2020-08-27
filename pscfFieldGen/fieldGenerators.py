# Imports
from pscfFieldGen.structure import ( Lattice, 
                                    CrystalBase, 
                                    CrystalMotif, 
                                    SpaceGroup )
from pscfFieldGen.util.stringTools import str_to_num, wordsGenerator

from abc import ABC, abstractmethod
import numba
import numpy as np
import scipy as sp
import pathlib
import itertools
import re
from enum import Enum

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
    @numba.njit
    def __generate_brillouin(ngrid,kgrid,nvect):
        """ Generate brillouin zone wave-vectors """
        dim = len(ngrid)
        dshift = dim - 1  #used in aliasing
        # Define aliasing method
        def miller_to_brillouin(G):
            out = np.zeros_like(G,dtype=np.int64)
            out[0] = G[0]
            for i in [1,2]:
                if dshift >= i:
                    if G[i] > ngrid[i]/2:
                        out[i] = G[i] - ngrid[i]
                    else:
                        out[i] = G[i]
            return out
        # Iterate
        brillouin = np.zeros((nvect,dim),dtype=np.int64)
        G = np.zeros_like(kgrid,dtype=np.int64)
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
            brill = np.asarray(self.getBrillouinArray(ngrid), dtype=np.float64)
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
    @numba.njit
    def __calculate_sums(nbrill,brill,npos,pos):
        out = 1.0j*np.zeros(nbrill,dtype=np.float64)
        for b in range(nbrill):
            R = 0.0
            I = 0.0
            q = brill[b,:]
            for p in range(npos):
                r = pos[p,:]
                qr = 2.0 * np.pi * np.dot(q,r)
                R = R + np.cos(qr)
                I = I + np.sin(qr)
            out[b] = R + 1.0j*I
        return out
    
    ## TODO: Make Lattice Class compatible with numba.jit
    @staticmethod
    @numba.njit
    def __calculate_qnorm(nbrill, brill, lattice):
        out = np.zeros(nbrill)
        for i in range(nbrill):
            q = brill[i,:]
            out[i] = lattice.vectorNorm(q)
        return out
    
    @staticmethod
    @numba.njit
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
        form = self.partForm.formFactorAmplitude
        formFact = self.__getFormFactors(qnorms,particleVol,form)
        
        args = [nspecies, frac, nbrill,brill, compSums, qnorms, vol, formFact, interfaceWidth, coreindex]
        kgrid = self.__kgrid_calc(*args)
        return kgrid
    
    @staticmethod
    @numba.jit(forceobj=True)
    def __getFormFactors(qnorms,particleVol,form):
        out = np.zeros_like(qnorms)
        n = len(qnorms)
        for t in range(n):
            if t == 0:
                out[t] = 0.0
            else:
                out[t] = form(qnorms[t], particleVol)
        return out
        
    @staticmethod
    @numba.njit
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
        rho = 1j*np.zeros((nbrill,nspecies))
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

