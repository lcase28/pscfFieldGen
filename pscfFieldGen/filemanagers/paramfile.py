import pscfFieldGen.filemanagers.pscf as pscf
import pscfFieldGen.filemanagers.pscfpp as pscfpp
import pscfFieldGen.structure.lattice as lattice

from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np

class ParamFile(ABC):
    """ Abstract base class for field generation Parameter files """
    
    def __init__(self, p_file):
        self.file = p_file
    
    def _check_ParamFile_Type(self, given, needed):
        """
        Private method for use in constructors.
        
        Parameters
        ----------
        self : 
            The object instance.
        given : type
            The type of the parameter file given to the constructor.
        needed : type
            The type of parameter file necessary for the constructor.
        """
        errstr = "Improper Parameter File Type ( {} ) "
        errstr += "given to file wrapper of type {} "
        errstr += "which requires Parameter file type {}."
        if not given == needed:
            raise(TypeError(errstr.format(given, type(self), needed)))
        
    @classmethod
    @abstractmethod
    def fromFileName(cls, filename):
        """ Return a new instance of the class from a file called filename """
        pass
    
    @abstractmethod
    def cleanFieldFile(self):
        """
        Return a field file manager consistent with the parameter file.
        
        The field file manager can be of any type, but must meet two requirements:
        
            1.  Must have a mutable member or settable property called fieldFile.fields
                which accepts a concentration field as a numpy.ndarray consistent with 
                that returned by pscfFieldGen.fieldGenerators.FieldCalculator.to_kgrid().
            2.  Must have a method called fieldFile.write(filename) which accepts 
                a pathlib.Path object or string filename, opens the file, writes the
                field file contents to it, and closes the file.
        """
        pass
    
    @property
    @abstractmethod
    def dim(self):
        """ Dimensionality of the system in the parameter file (1, 2, or 3) """
        pass
    
    @property
    @abstractmethod
    def ngrid(self):
        """
        Number of grid points considered in the discretized real space.
        
        Method should return an array containing the number of real-space
        grid points used to discretize space in each dimension.
        
        Returns
        -------
        ngrid : numpy.ndarray
            Number of grid points in each dimension. len(self.ngrid) == self.dim
        """
        pass
    
    @property
    @abstractmethod
    def latticeParameters(self):
        """ 
        The full set of lattice parameters as a dictionary.
        
        From a parameter file that may reduce the number of lattice parameters
        based on the crystal system, calculate the full set of parameters.
        Return these parameters as a dictionary from their respective parameter
        keys, depending on dimensionality.
        
        dim  |  parameter keys
        -----------------------------------
        1    |  a
        -----------------------------------
        2    |  a, b, gamma
        -----------------------------------
        3    |  a, b, c, alpha, beta, gamma
        -----------------------------------
        """
        pass
    
    def getLattice(self):
        return lattice.latticeFromParameters(self.dim, **self.latticeParameters)
    
    @property
    @abstractmethod
    def nMonomer(self):
        """ The number of monomer (chemical) types in the system. """
        pass
    
    @abstractmethod
    def getMonomerFractions(self):
        """ 
        Calculate the Overall Monomer Volume Fractions in the system. 
        
        Default behavior returns an array of values.
        Each element of the array contains the overall volume fraction
        in the system of a correspondingly indexed monomer, accounting
        for all species in the system (solvent, polymer, or otherwise).
        The sum of all elements of monomerFractions is 1.0.
        Monomer indexing starts at 0.
        If parameter file does not have obvious indexing rules for
        monomers, implementing class should document their assumptions 
        in assigning indexes.
        
        Returns
        -------
        volumeFractions : numpy.ndarray
            The volume fraction of each monomer species in the system.
        """
        pass
    
    def monomerFraction(self, monomerID):
        """
        Return the overall volume fraction of specified monomer.
        
        Calculates the overall monomer fractions in the system and returns 
        the volume fraction of the monomer with index monomerID.
        
        Parameters
        ----------
        monomerID : int
            The index of the monomer of interest.
        
        Returns
        -------
        volumeFraction : float
            The volume fraction of the monomer in the system.
        
        Raises
        ------
        IndexError
            If monomerID is not a valid monomer index.
        """
        frac = self.getMonomerFractions()
        return frac[monomerID]
    
    @abstractmethod
    def getInterfaceWidth(self, core_monomer=0):
        """
        Calculate interfacial width from Helfand-Tagami Theory.
        
        For the system, calculate the 
        Helfand-Tagami interfacial width estimate, given by:
            
            w = 2 * sqrt(b^2 / (6 * chi))
            
        where b is the geometric mean of monomer statistical segment lengths
        and chi is the interaction parameter.
        
        For multiple-monomer systems (nMonomer > 2), implementing class can
        determine how to calculate chi value used in the above equation, and
        should document this choice.
        
        Parameters
        ----------
        core_monomer : int
            The integer index of the monomer to be treated as the core of the region
            enclosed by the interface. This selects which chi interactions will be
            considered.
        """
        pass
    
    @property
    @abstractmethod
    def crystal_system(self):
        """ Crystal system being described
        
        Values should be consistent with the crystal_system input in PSCF Fortran user guide.
        """
        pass
    
    @property
    @abstractmethod
    def group_name(self):
        """ The name of the space group.
        
        String should be consistent with the group_name entry described in PSCF Fortran user guide.
        """
        pass
    
class PscfParam(ParamFile):
    """ Parameter File Wrapper for PSCF (Fortran) ParamFile class. """
    
    def __init__(self, p_file):
        """
        Initialize a new PscfParam object.
        
        Parameters
        ----------
        p_file : pscfFieldGen.pscfFileManagers.ParamFile
            The ParamFile instance. (Treated as Read-Only by wrapper)
        """
        self._check_ParamFile_Type(type(p_file),pscf.ParamFile)
        super().__init__(p_file)
    
    @classmethod
    def fromFileName(cls, filename):
        """
        Return a new PscfParam object using the parameter file named `filename`. 
        
        Parameters
        ----------
        filename : str or pathlib.Path
            The name of the parameter file, as referenced from the current directory.
        """
        pfile = pscf.ParamFile(filename)
        return cls(pfile)
    
    def cleanWaveFieldFile(self):
        """
        Return a WaveVectFieldFile object consistent with the parameter file.
        """
        kgrid = pscf.WaveVectFieldFile()
        kgrid.dim = self.file.dim
        kgrid.crystal_system = self.file.crystal_system
        kgrid.N_cell_param = self.file.N_cell_param
        kgrid.cell_param = self.file.cell_param
        kgrid.group_name = self.file.group_name
        kgrid.N_monomer = self.file.N_monomer
        kgrid.ngrid = self.file.ngrid
        return kgrid
    
    def cleanCoordFieldFile(self):
        """
        Return a CoordFieldFile object consistent with the parameter file.
        """
        kgrid = pscf.CoordFieldFile()
        kgrid.dim = self.file.dim
        kgrid.crystal_system = self.file.crystal_system
        kgrid.N_cell_param = self.file.N_cell_param
        kgrid.cell_param = self.file.cell_param
        kgrid.group_name = self.file.group_name
        kgrid.N_monomer = self.file.N_monomer
        kgrid.ngrid = self.file.ngrid
        return kgrid
    
    def cleanWaveFieldFile(self, N_star=2):
        """
        Return a WaveVectFieldFile object consistent with the parameter file.
        """
        if not self.dim == 1 and not self.file.group_name == "-1":
            msg = "Method only implemented for lamellar '-1' Structures."
            raise(NotImplementedError(msg))
        kgrid = pscf.SymFieldFile()
        kgrid.dim = self.file.dim
        kgrid.crystal_system = self.file.crystal_system
        kgrid.N_cell_param = self.file.N_cell_param
        kgrid.cell_param = self.file.cell_param
        kgrid.group_name = self.file.group_name
        kgrid.N_monomer = self.file.N_monomer
        kgrid.N_star = N_star
        waves = [i for i in range(N_star)]
        kgrid.waves = np.array(waves)
        counts = [1]
        for i in range(N_star-1):
            counts.append(2)
        kgrid.counts = np.array(counts)
        return kgrid
    
    @property
    def dim(self):
        """ Dimensionality of the system in the parameter file (1, 2, or 3) """
        return self.file.dim
    
    @property
    def ngrid(self):
        """
        Number of grid points considered in the discretized real space.
        
        Method should return an array containing the number of real-space
        grid points used to discretize space in each dimension.
        
        Returns
        -------
        ngrid : numpy.ndarray
            Number of grid points in each dimension. len(self.ngrid) == self.dim
        """
        return np.array(self.file.ngrid)
    
    @property
    def latticeParameters(self):
        """ 
        The full set of lattice parameters as a dictionary.
        
        From a parameter file that may reduce the number of lattice parameters
        based on the crystal system, calculate the full set of parameters.
        Return these parameters as a dictionary from their respective parameter
        keys, depending on dimensionality.
        
        dim  |  parameter keys
        -----------------------------------
        1    |  a
        -----------------------------------
        2    |  a, b, gamma
        -----------------------------------
        3    |  a, b, c, alpha, beta, gamma
        -----------------------------------
        """
        # simplify syntax
        param = self.file
        # Determine lattice parameters
        if param.dim == 1:
            return {"a":param.cell_param[0]}
        if param.dim == 2:
            # 2D cases
            crystalSys = param.crystal_system.strip("'")
            if crystalSys == 'square':
                a = param.cell_param[0]
                return {"a":a, "b":a, "gamma":90}
            if crystalSys == 'hexagonal':
                a = param.cell_param[0]
                return {"a":a, "b":a, "gamma":120}
            if crystalSys == 'rectangular':
                a = param.cell_param[0]
                b = param.cell_param[1]
                return {"a":a, "b":b, "gamma":90}
            if crystalSys == 'oblique':
                return dict(zip(["a","b","gamma"],param.cell_param))
        if param.dim == 3:
            keys = ["a","b","c","alpha","beta","gamma"]
            crys = param.crystal_system.strip("'")
            cp = param.cell_param
            if crys == 'cubic':
                a = cp[0]
                vals = [a, a, a, 90, 90, 90]
                return dict(zip(keys,vals))
            if crys == 'tetragonal':
                a = cp[0]
                c = cp[1]
                vals = [a, a, c, 90, 90, 90]
                return dict(zip(keys,vals))
            if crys == 'orthorhombic':
                a = cp[0]
                b = cp[1]
                c = cp[2]
                vals = [a, b, c, 90, 90, 90]
                return dict(zip(keys,vals))
            if crys == 'monoclinic':
                a = cp[0]
                b = cp[1]
                c = cp[2]
                beta = cp[3]
                vals = [a, b, c, 90, beta, 90]
                return dict(zip(keys,vals))
            if crys == 'hexagonal':
                a = cp[0]
                c = cp[1]
                vals = [a, a, c, 90, 90, 120]
                return dict(zip(keys,vals))
            if crys == 'trigonal':
                a = cp[0]
                alpha = cp[1]
                vals = [a, a, a, alpha, alpha, alpha]
                return dict(zip(keys,vals))
            if crys == 'triclinic':
                vals = cp
                return dict(zip(keys,vals))
    
    @property
    def nMonomer(self):
        """ The number of monomer (chemical) types in the system. """
        return self.file.N_monomer
    
    def getMonomerFractions(self):
        """ 
        Calculate the Overall Monomer Volume Fractions in the system. 
        
        For single-component, single-chain systems the result is the sum of
        block fractions of all blocks with each monomer type.
        
        For multi-component systems in the Canonical Ensemble, the contribution
        of each block and solvent is the block fraction * blend fraction.
        
        For multi-component systems in the Grand-Canonical Ensemble, fractions
        are calculated assuming equi-molar mixtures of all components.
        
        If a parameter file specifies no polymers, an error is raised.
        
        Returns
        -------
        volumeFractions : numpy.ndarray
            The volume fraction of each monomer species in the system.
        """
        # simplify syntax
        param = self.file
        # Calculate monomer fractions
        nmonomer = param.N_monomer
        nchain = param.N_chain
        nsolvent = param.N_solvent
        ncomponent = nchain + nsolvent
        ensemble = param.ensemble # 0=Canonical, 1=Grand
        if nchain == 0:
            raise(ValueError("The given parameter file must contain at least one polymer chain"))
        # compositions will be stored here
        frac = np.zeros(nmonomer)
        if nchain == 1 and nsolvent == 0:
            Ntot = 0.0
            nblock = param.N_block[0]
            for b in range(nblock):
                mon = param.block_monomer[0][b] - 1
                Nb = param.block_length[0][b]
                Ntot += Nb
                frac[mon] += Nb
            for m in range(nmonomer):
                frac[m] = frac[m] / Ntot
        else:   # multi-component system
            if ensemble == 0:   # Canonical Ensemble
                # Add polymer contributions
                for c in range(nchain):
                    Ntot = 0.0
                    chain_frac = np.zeros(nmonomer)
                    nblock = param.N_block[c]
                    phi = param.phi_chain[c]
                    for b in range(nblock):
                        mon = param.block_monomer[c][b] - 1
                        Nb = param.block_length[c][b]
                        Ntot += Nb
                        chain_frac[mon] += Nb
                    for m in range(nmonomer):
                        frac[m] += phi * (chain_frac[m] / Ntot)
                if nsolvent > 0:
                    for s in range(nsolvent):
                        mon = param.solvent_monomer[s]
                        phi = param.phi_solvent[s]
                        frac[mon] += phi
            else:   # Grand Canonical Ensemble
                Ntot = 0.0
                for c in range(nchain):
                    nblock = param.N_block[c]
                    for b in range(nblock):
                        mon = param.block_monomer[c][b] - 1
                        Nb = param.block_length[c][b]
                        Ntot += Nb
                        frac[mon] += Nb
                if nsolvent > 0:
                    for s in range(nsolvent):
                        mon = param.solvent_monomer[s] - 1
                        Nb = param.solvent_size[s]
                        Ntot += Nb
                        frac[mon] += Nb
                for m in range(nmonomer):
                    frac[m] = frac[m] / Ntot
        return frac
    
    def getInterfaceWidth(self, core_monomer=0):
        """
        Calculate interfacial width from Helfand-Tagami Theory.
        
        For the system, calculate the 
        Helfand-Tagami interfacial width estimate, given by:
            
            w = 2 * sqrt(b^2 / (6 * chi))
            
        where b is the geometric mean of monomer statistical segment lengths
        and chi is the interaction parameter.
        
        
        For systems with multiple monomer types (nMonomer > 2),
        chi is taken to be the geometric
        mean of all chi parameters involving the core monomer.
        
        Parameters
        ----------
        core_monomer : int
            The integer index of the monomer to be treated as the core of the region
            enclosed by the interface. This selects which chi interactions will be
            considered.
        """
        param = self.file
        monomer_index = core_monomer
        nMon = param.N_monomer
        if monomer_index >= nMon or monomer_index < 0:
            errstr = "Invalid Monomer index ({}) given. Must be in range (0,{})."
            raise(ValueError(errstr.format(monomer_index, nMon-1)))
        segLen = np.array(param.kuhn)
        b = (1.0 * np.prod(segLen)) ** (1.0/len(segLen))
        if nMon == 2:
            chi = param.chi[1][0]
        else:
            chiprod = 1.0
            for i in range(nMon):
                if i < monomer_index:
                    chiprod *= param.chi[monomer_index][i]
                elif i > monomer_index:
                    chiprod *= param.chi[i][monomer_index]
                else:
                    chiprod *= 1.0
            chi = chiprod ** (1.0 / (nMon - 1))
        w = 2*b / np.sqrt(6.0 * chi)
        return w
    
    @property
    def crystal_system(self):
        """ Crystal system being described """
        return self.file.crystal_system
    
    @property
    def group_name(self):
        """ The name of the space group. """
        return self.file.group_name
    
class PscfppParam(ParamFile):
    """ Parameter File Wrapper for PSCF (C++/Cuda) ParamFile class. """
    
    def __init__(self, p_file):
        """
        Initialize a new PscfParam object.
        
        Parameters
        ----------
        p_file : pscfFieldGen.pscfppFileManagers.ParamFile
            The ParamFile instance. (Treated as Read-Only by wrapper)
        """
        self._check_ParamFile_Type(type(p_file),pscfpp.ParamFile)
        super().__init__(p_file)
        
    @classmethod
    def fromFileName(cls, filename):
        """ Return a new PscfppParam instance from a parameter file called filename """
        param = pscfpp.ParamFile(filename)
        return PscfppParam(param)
    
    def cleanFieldFile(self):
        """
        Return a pscfpp.WaveVectFieldFile instance consistent with the parameter file.
        
        NOTE: Currently requires that a groupName be present.
        """
        kgrid = pscfpp.WaveVectFieldFile()
        kgrid.dim = self.file.dim
        kgrid.crystal_system = self.file.unitCell.value(0)
        kgrid.N_cell_param, kgrid.cell_param = self._parse_cell_params()
        kgrid.group_name = self.file.groupName.value()
        kgrid.N_monomer = self.nMonomer
        kgrid.ngrid = self.ngrid
        return kgrid
    
    __nCellParamRef = { (1,'lamellar')      :   1, \
                        (2,'square')        :   1, \
                        (2,'hexagonal')     :   1, \
                        (2,'rectangular')   :   2, \
                        (2,'oblique')       :   3, \
                        (3,'cubic')         :   1, \
                        (3,'tetragonal')    :   2, \
                        (3,'orthorhombic')  :   3, \
                        (3,'monoclinic')    :   4, \
                        (3,'hexagonal')     :   2, \
                        (3,'trigonal')      :   2, \
                        (3,'triclinic')     :   6    }
    
    def _parse_cell_params(self):
        crys = self.file.unitCell.value(0)
        nparam = self.__class__.__nCellParamRef.get( (self.dim, crys) )
        param = [ self.file.unitCell.value(i+1) for i in range(nparam) ]
        return nparam, np.array(param)
    
    @property
    def dim(self):
        """ Dimensionality of the system in the parameter file (1, 2, or 3) """
        return self.file.dim
    
    @property
    def ngrid(self):
        """
        Number of grid points considered in the discretized real space.
        
        Returns
        -------
        ngrid : numpy.ndarray
            Number of grid points in each dimension. len(self.ngrid) == self.dim
        """
        mesh = np.zeros(self.dim,dtype=int)
        for i in range(self.dim):
            mesh[i] = self.file.mesh.value(i)
        return mesh
    
    @property
    def latticeParameters(self):
        """ 
        The full set of lattice parameters as a dictionary.
        
        dim  |  parameter keys
        -----------------------------------
        1    |  a
        -----------------------------------
        2    |  a, b, gamma
        -----------------------------------
        3    |  a, b, c, alpha, beta, gamma
        -----------------------------------
        """
        dim = self.dim
        param = self.file
        if dim == 1:
            return { "a":param.unitCell.value(2) }
        if dim == 2:
            # 2D cases
            keys = ["a","b","gamma"]
            vals = []
            crystalSys = param.unitCell.value(0)
            if crystalSys == 'square':
                a = param.unitCell.value(1)
                vals = [a, a, 90]
            if crystalSys == 'hexagonal':
                a = param.unitCell.value(1)
                vals = [a, a, 120]
            if crystalSys == 'rectangular':
                a = param.unitCell.value(1)
                b = param.unitCell.value(2)
                vals = [a, b, 90]
            if crystalSys == 'oblique':
                a = param.unitCell.value(1)
                b = param.unitCell.value(2)
                gamma = param.unitCell.value(3)
                vals = [a,b,gamma]
            return dict(zip(keys,vals))
        if dim == 3:
            keys = ["a","b","c","alpha","beta","gamma"]
            crys = param.unitCell.value(0)
            if crys == 'cubic':
                a = param.unitCell.value(1)
                vals = [a, a, a, 90, 90, 90]
                return dict(zip(keys,vals))
            if crys == 'tetragonal':
                a = param.unitCell.value(1)
                c = param.unitCell.value(2)
                vals = [a, a, c, 90, 90, 90]
                return dict(zip(keys,vals))
            if crys == 'orthorhombic':
                a = param.unitCell.value(1)
                b = param.unitCell.value(2)
                c = param.unitCell.value(3)
                vals = [a, b, c, 90, 90, 90]
                return dict(zip(keys,vals))
            if crys == 'monoclinic':
                a = param.unitCell.value(1)
                b = param.unitCell.value(2)
                c = param.unitCell.value(3)
                beta = param.unitCell.value(4)
                vals = [a, b, c, 90, beta, 90]
                return dict(zip(keys,vals))
            if crys == 'hexagonal':
                a = param.unitCell.value(1)
                c = param.unitCell.value(2)
                vals = [a, a, c, 90, 90, 120]
                return dict(zip(keys,vals))
            if crys == 'trigonal':
                a = param.unitCell.value(1)
                alpha = param.unitCell.value(2)
                vals = [a, a, a, alpha, alpha, alpha]
                return dict(zip(keys,vals))
            if crys == 'triclinic':
                a = param.unitCell.value(1)
                b = param.unitCell.value(2)
                c = param.unitCell.value(3)
                alpha = param.unitCell.value(4)
                beta = param.unitCell.value(5)
                gamma = param.unitCell.value(6)
                vals = [a,b,c,alpha,beta,gamma]
                return dict(zip(keys,vals))
    
    @property
    def nMonomer(self):
        """ The number of monomer (chemical) types in the system. """
        return self.file.Mixture.nMonomer.value()
    
    def getMonomerFractions(self):
        """ 
        Calculate the Overall Monomer Volume Fractions in the system. 
        
        The possibility for mixed specification of 'phi' and 'mu' among
        polymers in pscfpp is handled in two "steps"
        
            1.  Polymers specifying 'phi' are considered first, with their
                contributions to monomer fractions being weighted by their
                phi value. Thus, the contribution to total volume fraction
                for monomer m from polymer p is 
                f_(m|p) = phi_p * f_(m,p)
                where f_(m|p) is the contribution to overall volume fraction,
                f_(m,p) is the fraction of polymer p's volume filled by monomer
                m, and phi_p is the blend fraction of polymer p.
            2.  The remaining volume fraction (not accounted for by the
                phi-specified polymers) is split among the remaining polymers
                assuming an equimolar mixture of the mu-specified polymers.
        
        Returns
        -------
        volumeFractions : numpy.ndarray
            The volume fraction of each monomer species in the system.
        """
        nmonomer = self.nMonomer
        nchain = self.file.Mixture.nPolymer.value()
        nsolvent = 0    # Point solvents not yet implemented in pscfpp
        ncomponent = nchain + nsolvent
        if nchain == 0:
            raise(ValueError("The given parameter file must contain at least one polymer chain"))
        # Separate phi-specified and mu-specified polymers
        hasphi = []
        hasmu = []
        for p in self.file.Mixture.Polymer:
            if p.hasPhi:
                hasphi.append(p)
            elif p.hasMu:
                hasmu.append(p)
            else:
                raise(ValueError("Polymer missing phi or mu specification:\n{}".format(p)))
        # Initialize trackers
        phiUsed = 0.0
        frac = np.zeros(nmonomer)
        # Handle Phi-specified polymers
        if len(hasphi) > 0:
            for p in hasphi:
                comp = p.composition(nmonomer)
                phi = p.phi
                phiUsed += phi
                for m in range(nmonomer):
                    frac[m] += phi * comp[m]
        # Handle Mu-specified polymers
        if len(hasmu) > 0:
            phiLeft = 1.0 - phiUsed
            ntot = 0.0
            nmon = np.zeros(nmonomer)
            for p in hasmu:
                ntot += p.totalLength()
                nmon += p.monomerLengths(nmonomer)
            for m in range(nmonomer):
                frac[m] += phiLeft * nmon[m] / ntot
        # Return Result
        return frac
    
    def getInterfaceWidth(self, core_monomer=0):
        """
        Calculate interfacial width from Helfand-Tagami Theory.
        
        For the system, calculate the 
        Helfand-Tagami interfacial width estimate, given by:
            
            w = 2 * sqrt(b^2 / (6 * chi))
            
        where b is the geometric mean of monomer statistical segment lengths
        and chi is the interaction parameter.
        
        For systems with multiple monomer types (nMonomer > 2),
        chi is taken to be the geometric
        mean of all chi parameters involving the core monomer.
        
        Parameters
        ----------
        core_monomer : int
            The integer index of the monomer to be treated as the core of the region
            enclosed by the interface. This selects which chi interactions will be
            considered.
        """
        param = self.file
        # Extract data from parameter file
        nMon = self.nMonomer
        if core_monomer >= nMon or core_monomer < 0:
            estr = "Invalid Monomer ID ({}) given. Must be in range (0,{})."
            raise(ValueError(estr.format(core_monomer,nMon-1)))
        kuhn = self.kuhnArray
        chiVals = self.chiArray
        # Calculate geom. mean stat. segment length
        b = (1.0 * np.prod(kuhn)) ** (1.0/nMon)
        # Calculate geom. mean chi interaction
        if nMon == 2:
            chi = chiVals[1][0]
        else:
            chiprod = 1.0
            for i in range(nMon):
                if i < core_monomer:
                    chiprod *= chiVals[core_monomer][i]
                elif i > monomer_id:
                    chiprod *= chiVals[i][core_monomer]
                else:
                    chiprod *= 1.0
            chi = chiprod ** (1.0 / (nMon - 1))
        # Calculate and return Interface Width (w)
        w = 2*b / np.sqrt(6.0 * chi)
        return w
    
    @property
    def kuhnArray(self):
        """ Collect statistical segment lengths in numpy array. """
        param = self.file
        nMon = param.Mixture.nMonomer.value()
        kuhn = []
        for i in range(nMon):
            kuhn.append(param.Mixture.monomers.line(i).value(2))
        return np.array(kuhn)
    
    @property
    def chiArray(self):
        """ Collect chi interaction parameters in numpy array. """
        param = self.file
        nMon = param.Mixture.nMonomer.value()
        chi = np.zeros((nMon,nMon))
        # Calculate number of chi parameters expected
        nchi = 0
        for i in range(nMon+1):
            nchi += i
        # Collect chi parameters in array
        for i in range(nchi):
            entry = param.ChiInteraction.chi.line(i)
            row = entry.value(0)
            col = entry.value(1)
            val = entry.value(2)
            # Assume all monomer id's are valid
            if row >= col:
                chi[row][col] = val
            else:
                chi[col][row] = val
        return chi
    
    @property
    def crystal_system(self):
        """ Crystal system being described """
        return self.file.unitCell.value(0)
    
    @property
    def group_name(self):
        """ The name of the space group. """
        return self._change_groupName_format(self.file.groupName.value())
    
    @staticmethod
    def _change_groupName_format(group_name):
        """ Convert formatting between Fortran-style names and C++/Cuda style names """
        out = deepcopy(group_name)
        out = out.replace(r"_",r" ")
        out = out.replace(r"%",r"/")
        out = out.replace(r":",r" : ")
        return out
    
