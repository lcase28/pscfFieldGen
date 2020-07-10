# Imports
from crystal_structs.crystalStructs.lattice import Lattice
from .particleForms import ParticleForm, SphereForm, Circle2DForm
import numpy as np
import scipy as sp
from .stringTools import str_to_num, wordsGenerator
import pathlib
import itertools
import re
from enum import Enum

class FieldCalculator(object):
    """ Generator class for 3D k-grid density fields of n-monomer systems """
    
    # sentinel of -1 indicates value must be dynamic
    readCounts =   {"dim" : 1,
                    "lattice" : -1,
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
            self.real = []
            self.imag = []
            self.qNorm = []
            self.nEntry = 0
        
        def add(self, brillouin, real, imag, qNorm):
            self.brillouin.append(brillouin)
            self.real.append(real)
            self.imag.append(imag)
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
                    r = self.real[n]
                    i = self.imag[n]
                    q = self.qNorm[n]
                    yield (n, b, r, i, q)
        
    def __init__(self, **kwargs):
        """
        Initialize a new FieldGenerator.
        
        Keyword Parameters
        ------------------
        dim : integer
            Dimensionality of the system (1-3)
        formfactor : ParticleForm
            Object capable of returning the particle form factor.
        lattice : Lattice object
            Object representing the basis vectors of the lattice
        N_particle : integer
            Number of particles in the system
        particlePositions : array-like, N_particles by dim
            Positions of particles in coordinates of the basis vectors.
        """
        self.dim = kwargs.get("dim", 3)
        self.lattice = kwargs.get("lattice", \
            Lattice.latticeFromParameters(dim = self.dim, **self.__defaultParams))
        self.reciprocal_lattice = self.lattice.reciprocal
        self.particles = kwargs.get("particlePositions",None)
        self.nparticles = kwargs.get("N_particles")
        if self.dim == 3:
            defPartForm = SphereForm
        elif self.dim == 2:
            defPartForm = Circle2DForm
        self.partForm = kwargs.get("formfactor", defPartForm)
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
    
    # TODO: Figure out how to generate 2D, 1D initial guesses
    def to_kgrid(self, frac, ngrid,interfaceWidth=None):
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
        coreindex = 0
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
        for (t, brillouin, R, I, q_norm) in record.records():
            if t == 0: 
                # 0-th wave-vector -- corresponds to volume fractions
                rho[t,:] = frac[:] 
            else:
                compSum = R + 1j*I
                ff, fsmear = self.partForm.formFactorAmplitude(q_norm, particleVol, smear = self.smear)
                if interfaceWidth is not None:
                    fsmear = np.exp(-( (interfaceWidth**2) * q_norm**2) / 2.0)
                rho[t, coreindex] = compSum * (1/vol) * ff * fsmear
                rhoTemp = -rho[t, coreindex] / np.sum(frac[1:])
                for i in range(nspecies-1):
                    rho[t, i+1] = rhoTemp * frac[i+1]
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
                f.add(brillouin,None,None,None)
            else:
                # sum of wave-vector dot particle positions
                R, I = self.sum_ff(brillouin)
                q_norm = 2 * np.pi * self.reciprocal_lattice.vectorNorm(brillouin)
                f.add(brillouin,R,I,q_norm)
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
        for i in range(self.nparticles):
            # By definition of reciprocal space lattice,
            #   dot product of q (recip) and r (real) 
            #   calculated same as normal (b/c a_i dot a*_j = delta_ij )
            qR = 2 * np.pi * np.dot(q, self.particles[i,:])
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
    
#class FieldGenerator(ABC):
#    """ Abstract base class for field generators """
#    
#    def __init__(**kwargs):
#        lat = kwargs.get("lattice", None)
#        if lat is not None:
#            self.lattice = 
#        super().__init(**kwargs)
#    
#    @abstractmethod
#    def WriteFile(**kwargs):
#        pass
#    
#    def DensityFields(self, **kwargs):

#class FieldGenerator(object):
#    """ Generator class for 3D k-grid density fields of n-monomer systems """
#    
#    # sentinel of -1 indicates value must be dynamic
#    readCounts =   {"dim" : 1,
#                    "N_monomer" : 1,
#                    "crystal_system" : 1,
#                    "N_cell_param" : 1,
#                    "cell_param" : -1,
#                    "lattice_const" : -1,
#                    "group_name" : 1,
#                    "N_particles" : 1,
#                    "particlePositions" : -1,
#                    "particleScale" : 1,
#                    "sigma_smear" : 1,
#                    "ngrid" : -1,
#                    "output_filename" : 1}
#    
#    __defaultParams = { "a" : 1, "b" : 1, "c" : 1 \
#                        "alpha" : 90, "beta" : 90, "gamma" : 90 }
#    
#    
#    def __init__(self, **kwargs):
#        """
#        Initialize a new FieldGenerator.
#        
#        Keyword Parameters
#        ------------------
#        dim : integer
#            Dimensionality of the system (1-3)
#        formfactor : ParticleForm
#            Object capable of returning the particle form factor.
#        lattice : Lattice object
#            Object representing the basis vectors of the lattice
#        N_particle : integer
#            Number of particles in the system
#        particlePositions : array-like, N_particles by dim
#            Positions of particles in coordinates of the basis vectors.
#        particleScale : float
#            Ratio of desired particle size to default particle size calculated
#            from volume fractions. Values < 1 will give smaller particles.
#            Scale applied to the defining dimension of the particle (radius of 
#            sphere or cylinder, for example)
#        N_monomer : integer
#            Number of monomer species present in the system
#        crystal_system : string
#            The crystal system being considered (cubic, tetragonal, etc).
#            Name must be enclosed in single quotes.
#        N_cell_param : int
#            Minimum number of parameters required to specify the lattice,
#            per PSCF specifications.
#        cell_param : array-like
#            The cell parameters, per PSCF specifications.
#        group_name : string
#            The name of the group as specified by PSCF
#        ngrid : array-like of int
#            Grid points to use in SCFT calculations in each dimension.
#        output_filename : string
#            Name to use when generating initial guess kgrid file.
#        """
#        #self.lattice = kwargs.get("lattice_const", Lattice.from_parameters(1,1,1,90,90,90))
#        #self.reciprocal_lattice = self.lattice.reciprocal
#        self.dim = kwargs.get("dim", 3)
#        self.lattice = kwargs.get("lattice_const", \
#            Lattice.latticeFromParameters(dim = self.dim, **self.__defaultParams)
#        self.reciprocal_lattice = self.lattice.reciprocal
#        self.particles = kwargs.get("particlePositions",None)
#        self.nparticles = kwargs.get("N_particles")
#        self.nspecies = kwargs.get("N_monomer")
#        self.smear = kwargs.get("sigma_smear")
#        self.crystal_system = kwargs.get("crystal_system")
#        self.nCellParam = kwargs.get("N_cell_param")
#        self.cellParam = kwargs.get("cell_param")
#        self.group_name = kwargs.get("group_name")
#        self.ngrid = kwargs.get("ngrid")
#        self.outfile = kwargs.get("output_filename", "rho_kgrid")
#        self.formfactor = kwargs.get("formfactor", sphere_form_factor)
#        self.particleScale = kwargs.get("particleScale", 1.0)
#        super().__init__()
#    
#    @classmethod
#    def from_file(cls, fname):
#        """
#        Return a FieldGenerator instance  based on the file "fname"
#        
#        Parameters
#        ----------
#        fname : pathlib.Path
#            Path to file being used to instantiate.
#        """
#        #print("Reading Input File")
#        with open(fname) as f:
#            kwargs = {}
#            words = wordsGenerator(f)
#            for word in words:
#                key = word # Next word should be acceptable keyword
#                readCount = cls.readCounts.get(key, None)
#                if readCount is not None:
#                    if readCount == 1:
#                        data = next(words) #.next()
#                        try:
#                            data = str_to_num(data)
#                        except(ValueError, TypeError):
#                            pass
#                    elif readCount == -1:
#                        # sentinel indicates special case
#                        if key == "ngrid":
#                            dim = kwargs.get("dim")
#                            if dim is not None:
#                                data = np.array([str_to_num(next(words)) for i in range(dim)])
#                            else:
#                                raise(IOError("dim must be specified before ngrid"))
#                        elif key == "cell_param":
#                            nparam = kwargs.get("N_cell_param")
#                            if nparam is not None:
#                                data = np.array([str_to_num(next(words)) for i in range(nparam)])
#                            else:
#                                raise(IOError("N_cell_param must be specified before cell_param"))
#                        elif key == "lattice_const":
#                            dim = kwargs.get("dim")
#                            if dim is not None:
#                                if dim == 1:
#                                    nconst = 1
#                                    raise(NotImplementedError("1-dimensional case not implemented"))
#                                elif dim == 2:
#                                    nconst = 3
#                                    raise(NotImplementedError("2-dimensional case not implemented"))
#                                elif dim == 3:
#                                    nconst = 6
#                                else:
#                                    raise(ValueError("dim may not exceed 3"))
#                                constants = [str_to_num(next(words)) for i in range(nconst)]
#                                #print("Constants: ",constants)
#                                data = Lattice.from_parameters(*constants)
#                                #print("Lattice: ",data)
#                            else:
#                                raise(IOError("Dim must be specified before lattice constants"))
#                        elif key == "particlePositions":
#                            dim = kwargs.get("dim")
#                            if dim is None:
#                                raise(IOError("dim must be specified before particle positions"))
#                            nparticles = kwargs.get("N_particles")
#                            if nparticles is None:
#                                raise(IOError("N_particles must be specified before particles positions"))
#                            data = np.array([str_to_num(next(words)) for i in range(dim * nparticles)])
#                            data = np.reshape(data, (nparticles, dim))
#                        else:
#                            raise(NotImplementedError("{} has not been fully implemented as a dynamic read variable".format(key)))
#                    elif readCount > 1:
#                        # Simply generate a list of data values 
#                        data = [next(words) for i in range(readCount)]
#                        for (i, w) in enumerate(data):
#                            try:
#                                data[i] = str_to_num(w)
#                            except(ValueError):
#                                data[i] = w
#                    else:
#                        # implies either invalid number readCount
#                        # or readCount = 0 ==> ignore entry
#                        pass
#                else:
#                    raise(ValueError("{} is not a valid keyword for FieldGenerators".format(key)))
#                kwargs.update([(key, data)])
#            return cls(**kwargs)
#    
#    # TODO: Figure out how to generate 2D, 1D initial guesses
#    def to_kgrid(self, frac):
#        """
#        Return the reciprocal space grid of densities.
#        
#        Parameters
#        ----------
#        frac : numerical, array-like
#            volume fractions of all monomer types. Sum of all values = 1.
#            Value at index 0 represents the "core" or particle-forming monomer.
#            And must also be monomer 1 by PSCF indications.
#        """
#        coreindex = 0
#        ngrid = self.ngrid
#        print("ngrid = ", ngrid)
#        # Shift grid for k-grid
#        kgrid = np.zeros_like(ngrid)
#        for (i,x) in enumerate(ngrid):
#            if i == 0:
#                kgrid[i] = x/2
#            else:
#                kgrid[i] = x - 1
#        print("kgrid = ",kgrid)
#            
#        nwaves = str_to_num(np.prod(kgrid + np.ones_like(kgrid)))
#        print("nwaves = ", nwaves)
#        rho = np.zeros((nwaves, self.nspecies))
#        print("rho init size: ", rho.size)
#        vol = self.lattice.volume
#        print("volume = ", vol)
#        particleVol = frac[coreindex] * vol / self.nparticles
#        #const = frac[coreindex] / self.nparticles
#        #print("frac / nparticles = ", const)
#        #Rsph = ((3 * frac[coreindex] * vol) / (4 * np.pi * self.nparticles))**(1./3)
#        #Rsph = self.particleScale * Rsph
#        #print("Rsph = ", Rsph)
#        #a, b, c, alpha, beta, gamma = self.lattice.latticeParameters
#        #print("Lattice params: ", a, b, c, alpha, beta, gamma)
#        
#        # primary loop for n-dimensional generation
#        t = -1
#        for G in itertools.product(*[range(x+1) for x in kgrid]):
#            # G is wave-vector in n-dimensions.
#            G = np.array(G) #convert tuple to array
#            brillouin = self.miller_to_brillouin(G, ngrid)
#            t = t + 1
#            print("\nt = ", t)
#            print("miller: ", G)
#            print("brillouin: ", brillouin)
#            if np.array_equiv(brillouin, np.zeros_like(brillouin)):
#                # 0-th wave-vector -- corresponds to volume fractions
#                rho[t,:] = frac[:] 
#            else:
#                # sum of wave-vector dot particle positions
#                R, I = self.sum_ff(brillouin)
#                print("R = ", R)
#                ##   Here, [1/a, 1/b, 1/c] effectively acts as the reciprocal
#                ##     metric tensor for orthorhombic system.
#                ##   To generalize to non-orthogonal basis, replace this step
#                ##     and next with formal metric tensor calculation.
#                #GdotR = np.multiply(brillouin, [1/a, 1/b, 1/c])
#                #print("GdotR = ", GdotR)
#                ##   In following line, (2*np.pi) term is residual from reciprocal
#                ##     vectors being defined as b_i = 2*pi*(a_j x a_k)/V
#                ##   Formally, this 2*pi is not part of reciprocal space, and does
#                ##     not figure in to metric tensor. -- this is normalizing constant
#                ##   In effect, q = 2*pi * (g*), where g* is the direction-defining wave-vector
#                #qR = 2 * np.pi * Rsph * np.dot(GdotR, GdotR)**0.5
#                #print("qR = ", qR)
#                
#                #   Should return same thing as above
#                #recipBasis = np.asarray(self.reciprocal_lattice) # reciprocal basis vectors in [1 1 1 90 90 90] basis
#                #q_norm = np.dot(brillouin, recipBasis) # wave-vector in unit cartesian basis
#                #qR = Rsph * np.linalg.norm(q_norm) # 2*pi factor included by reciprocal_lattice
#                q_norm = 2 * pi * self.reciprocal_lattice.vectorNorm(brillouin)
#                #qR = Rsph * q_norm
#                #print("qR = ",qR)
#                ff, fsmear = self.partForm.formFactorAmplitude(q_norm, particleVol, smear = self.smear)
#                
#                #rho[t, coreindex] = const * R * self.formfactor(qR) * np.exp(-(self.smear**2 * qR**2 / 2))
#                rho[t, coreindex] = (1/vol) * R * ff * fsmear # * np.exp( -(sigma_smear**2 * qR**2 / 2) )
#                rhoTemp = -rho[t, coreindex] / np.sum(frac[1:])
#                for i in range(self.nspecies-1):
#                    rho[t, i+1] = rhoTemp * frac[i+1]
#                for (i,r) in enumerate(rho[t,:]):
#                    if r == -0.0:
#                        rho[t,i] = 0.0
#            print("rho[{}] = {}".format(t, rho[t,:]))
#        
#        print("rho final size: ", rho.size)
#        return rho
#                
#    def to_file(self, frac, fileRoot=None):
#        """
#        Write the reciprocal space density to file.
#        
#        COMPATIBILITY WARNING: function is written assuming linux, unix or MacOS.
#        
#        Parameters
#        ----------
#        frac : numerical, array-like
#            Volume fraction of each monomer type in the system.
#        fileRoot : string, optional
#            Root directory in which to write the file. Optionally terminated with "/"
#        """
#        if fileRoot is not None:
#            fileRoot = fileRoot.strip().rstrip("/")
#            fname = fileRoot + "/" + self.outfile.strip().strip("'")
#        else:
#            fname = self.outfile.strip().strip("'")
#        with open(fname, "w") as f:
#            f.write("    format 1 0")
#            f.write("\ndim\n\t{}".format(self.dim))
#            f.write("\ncrystal_system\n\t{}".format(self.crystal_system))
#            f.write("\nN_cell_param\n\t{}".format(self.nCellParam))
#            
#            cellParamString = "\ncell_param\n"
#            for i in range(self.nCellParam):
#                cellParamString += "\t{}"
#            try:
#                f.write(cellParamString.format(*self.cellParam))
#            except(TypeError):
#                f.write(cellParamString.format(self.cellParam))
#            
#            f.write("\ngroup_name\n\t{}".format(self.group_name))
#            f.write("\nN_monomer\n\t{}".format(self.nspecies))
#            
#            ngridString = "\nngrid\n"
#            for i in range(self.dim):
#                ngridString += "\t{}"
#            try:
#                f.write(ngridString.format(*self.ngrid))
#            except(TypeError):
#                f.write(ngridString.format(self.ngrid))
#            
#            rho = self.to_kgrid(frac)
#            baseString = "({:.4E},0.0000E+00)"
#            rowString = "\n" + baseString
#            for i in range(self.nspecies-1):
#                rowString += "  " + baseString
#            for i in range(len(rho[:,0])):
#                f.write(rowString.format(*rho[i,:]))
#        return 0
#        
#    def sum_ff(self, q):
#        """
#        Returns real and imaginary components of 
#        
#        .. math::
#        
#            $$\sum_{n=1}^{N_particles} exp{i\mathbf{q}\dot\mathbf{r}_{n}}$$
#        
#        Where :math:$\mathbf{r}_{n}$ is the position of particle :math:$n$
#        
#        Parameters
#        ----------
#        q : array-like
#            Reciprocal space indices (first brillouin zone wave vector).
#        
#        Returns
#        -------
#        R : real, floating point
#            Real component of sum(exp(i * (q dot r)))
#        I : real, floating point
#            Imaginary component of sum(exp(i * (q dot r)))
#        """
#        R = 0
#        I = 0
#        for i in range(self.nparticles):
#            # By definition of reciprocal space lattice,
#            #   dot product of q (recip) and r (real) 
#            #   calculated same as normal (b/c a_i dot a*_j = delta_ij )
#            qR = 2 * np.pi * np.dot(q, self.particles[i,:])
#            R = R + np.cos(qR)
#            I = I + np.sin(qR)
#        
#        return R, I
#    
#    def miller_to_brillouin(self, G, grid):
#        """
#        Convert miller indices to first brillouin zone (Aliasing)
#        """
#        out = np.zeros_like(G, dtype=int)
#        out[0] = G[0]
#        dim = self.dim-1
#        for i in [1,2]:
#            if dim >= i:
#                if G[i] > grid[i]/2:
#                    out[i] = G[i] - grid[i]
#                else:
#                    out[i] = G[i]
#        return out
#    
    
