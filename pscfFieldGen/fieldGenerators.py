# Imports
from pscfFieldGen.structure import ( Lattice, 
                                    CrystalBase, 
                                    CrystalMotif, 
                                    SpaceGroup )
from pscfFieldGen.util.stringTools import str_to_num, wordsGenerator

import numpy as np
import scipy as sp
import pathlib
import itertools
import re
from enum import Enum

class FieldCalculator(object):
    """ Generator class for 3D k-grid density fields of n-monomer systems """
    
    # sentinel of -1 indicates value must be dynamic
    readCounts =   {"dim" : 1,
                    "lattice" : -1,
                    "coord_input_style" : 1,
                    "N_particles" : 1,
                    "particlePositions" : -1,
                    "sigma_smear" : 1}
    
    __defaultParams = { "a" : 1, "b" : 1, "c" : 1, \
                        "alpha" : 90, "beta" : 90, "gamma" : 90 }
    
    class FieldRecord(object):
        """ 
            Helper class to cache reusable portions of previous 
            calculations in the instance of same ngrid.
        """
        def __init__(self, **kwargs):
            """ Generate an empty record. """
            self.brillouin = []
            self.complex_sums = []
            self.qNorm = []
            self.nEntry = 0
        
        def add(self, brillouin, complex_sum, qNorm):
            self.brillouin.append(brillouin)
            self.complex_sums.append(complex_sum)
            self.qNorm.append(qNorm)
            self.nEntry += 1
            
        def __len__(self):
            return self.nEntry
            
        def records(self):
            if self.nEntry == 0:
                pass
            else:
                for n in range(self.nEntry):
                    b = self.brillouin[n]
                    c = self.complex_sums[n]
                    q = self.qNorm[n]
                    yield (n, b, c, q)
        
    def __init__(self, crystal, **kwargs):
        """
        Initialize a new FieldGenerator.
        
        Parameters
        ----------
        crystal : pscfFieldGen.structure.CrystalBase or CrystalMotif
            The crystal structure the calculator is supposed to produce
            fields for.
        """
        self.crystal = crystal
        self.dim = self.crystal.dim
        self.lattice = self.crystal.lattice
        self.reciprocal_lattice = self.lattice.reciprocal
        self.partForm = self.crystal.particles[0].formFactor
        self.nparticles = self.crystal.n_particles
        self.smear = kwargs.get("sigma_smear", 0.0)
        # Cache pre-calculated results which can be recycled whenever ngrid is same
        self._cached_results = dict()
        super().__init__()
    
    @classmethod
    def from_file(cls, fname):
        """
            Return a FieldCalculator instance  based on the file "fname"
            
            Parameters
            ----------
            fname : pathlib.Path
                Path to file being used to instantiate.
        """
        #print("Reading Input File")
        with fname.open(mode='r') as f:
            kwargs = {}
            words = wordsGenerator(f)
            for word in words:
                key = word # Next word should be acceptable keyword
                readCount = cls.readCounts.get(key, None)
                if readCount is not None:
                    if readCount == 1:
                        data = next(words) #.next()
                        try:
                            data = str_to_num(data)
                        except(ValueError, TypeError):
                            pass
                    elif readCount == -1:
                        # sentinel indicates special case
                        if key == "lattice":
                            dim = kwargs.get("dim")
                            if dim is not None:
                                if dim == 1:
                                    nconst = 1
                                    raise(NotImplementedError("1-dimensional case not implemented"))
                                elif dim == 2:
                                    nconst = 3
                                    constNames = ["a", "b", "gamma"]
                                    #raise(NotImplementedError("2-dimensional case not implemented"))
                                elif dim == 3:
                                    nconst = 6
                                    constNames = ["a","b","c","alpha","beta","gamma"]
                                else:
                                    raise(ValueError("dim may not exceed 3"))
                                constants = [str_to_num(next(words)) for i in range(nconst)]
                                const = dict(zip(constNames,constants))
                                #print("Constants: ",constants)
                                data = Lattice.latticeFromParameters(dim, **const)
                                #print("Lattice: ",data)
                            else:
                                raise(IOError("Dim must be specified before lattice constants"))
                        elif key == "particlePositions":
                            dim = kwargs.get("dim")
                            if dim is None:
                                raise(IOError("dim must be specified before particle positions"))
                            nparticles = kwargs.get("N_particles")
                            if nparticles is None:
                                raise(IOError("N_particles must be specified before particles positions"))
                            data = np.array([str_to_num(next(words)) for i in range(dim * nparticles)])
                            data = np.reshape(data, (nparticles, dim))
                        else:
                            raise(NotImplementedError("{} has not been fully implemented as a dynamic read variable".format(key)))
                    else:
                        # implies either invalid number readCount
                        # or readCount = 0 ==> ignore entry
                        pass
                else:
                    raise(ValueError("{} is not a valid keyword for FieldCalculator".format(key)))
                kwargs.update([(key, data)])
            return cls(**kwargs)
    
    def to_kgrid(self, frac, ngrid,interfaceWidth=None, coreindex=0):
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
        key = str(tuple(ngrid))
        if key in self._cached_results:
            record = self._cached_results.get(key)
        else:
            record = self._generateFieldRecord(ngrid)
            self._cached_results.update({key: record})
        
        frac = np.array(frac)
        nspecies = frac.size
        nwaves = len(record)
        rho = 1j*np.zeros((nwaves, nspecies))
        vol = self.lattice.volume
        particleVol = frac[coreindex] * vol / self.nparticles
        
        # primary loop for n-dimensional generation
        for (t, brillouin, compSum, q_norm) in record.records():
            if t == 0: 
                # 0-th wave-vector -- corresponds to volume fractions
                rho[t,:] = frac[:] 
            else:
                ff, fsmear = self.partForm.formFactorAmplitude(q_norm, particleVol, smear = self.smear)
                if interfaceWidth is not None:
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
    
    def _generateFieldRecord(self, ngrid):
        """ Populate and return a FieldRecord object """
        f = self.__class__.FieldRecord()
        ngrid = np.array(ngrid)
        # Shift grid for k-grid
        kgrid = np.zeros_like(ngrid)
        for (i,x) in enumerate(ngrid):
            if i == 0:
                kgrid[i] = x/2
            else:
                kgrid[i] = x - 1
        for G in itertools.product(*[range(x+1) for x in kgrid]):
            # G is wave-vector in n-dimensions.
            G = np.array(G) #convert tuple to array
            brillouin = self.miller_to_brillouin(G, ngrid)
            if np.array_equiv(brillouin, np.zeros_like(brillouin)):
                # 0-th wave-vector -- corresponds to volume fractions
                # set all but brillouin to None to ensure that, in event of
                # logical errors regarding use of FieldRecord, run will be
                # terminated with runtime error.
                # TODO: clean up handling of 0th wave-vector.
                f.add(brillouin,None,None)
            else:
                # sum of wave-vector dot particle positions
                R, I = self.sum_ff(brillouin)
                compsum = R + 1j*I
                q_norm = 2 * np.pi * self.reciprocal_lattice.vectorNorm(brillouin)
                f.add(brillouin,compsum,q_norm)
        return f
                
    def sum_ff(self, q):
        """
        Returns real and imaginary components of 
        
        .. math::
        
            $$\sum_{n=1}^{N_particles} exp{i\mathbf{q}\dot\mathbf{r}_{n}}$$
        
        Where :math:$\mathbf{r}_{n}$ is the position of particle :math:$n$
        
        Parameters
        ----------
        q : array-like
            Reciprocal space indices (first brillouin zone wave vector).
        
        Returns
        -------
        R : real, floating point
            Real component of sum(exp(i * (q dot r)))
        I : real, floating point
            Imaginary component of sum(exp(i * (q dot r)))
        """
        R = 0
        I = 0
        for r in self.crystal.particlePositions():
            # By definition of reciprocal space lattice,
            #   dot product of q (recip) and r (real) 
            #   calculated same as normal (b/c a_i dot a*_j = delta_ij )
            qR = 2 * np.pi * np.dot(q, r) #self.particles[i,:])
            R = R + np.cos(qR)
            I = I + np.sin(qR)
        
        return R, I
    
    def miller_to_brillouin(self, G, grid):
        """
        Convert miller indices to first brillouin zone (Aliasing)
        """
        out = np.zeros_like(G, dtype=int)
        out[0] = G[0]
        dim = self.dim-1
        for i in [1,2]:
            if dim >= i:
                if G[i] > grid[i]/2:
                    out[i] = G[i] - grid[i]
                else:
                    out[i] = G[i]
        return out
    
