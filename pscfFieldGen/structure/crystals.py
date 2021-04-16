""" Module containing definitions for building a crystal structure """

from pscfFieldGen.structure.lattice import Lattice
from pscfFieldGen.structure.symmetry import SpaceGroup
from pscfFieldGen.structure.particles import (
    ParticleBase, 
    ParticleSet, 
    isParticleKey, 
    readParticleFromFile )

from copy import deepcopy
import numpy as np

class BasisCrystal(object):
    """ Crystal class based on Lattice-Basis crystal definition """
    
    def __init__(self, particles, lattice=None, core_options=None):
        """ Initialize new CrystalBase object.
        
        Parameters
        ----------
        particles : ParticleSet, ParticleBase, or iterable of ParticleBase
            The particle(s) to be added initially to the crystal.
            All particles in the crystal must be defined on the same
            lattice.
        lattice : Lattice (optional)
            The lattice on which the crystal should be defined.
            If included, all particles must be defined on the same
            lattice. If not included, lattice will be inferred from
            the input particles.
        """
        testLattice = None
        if lattice is not None:
            if not isinstance(lattice,Lattice):
                msg = "Crystal lattice cannot be of type {}"
                raise(TypeError(msg.format(type(lattice).__name__)))
            testLattice = lattice
        if isinstance(particles, ParticleSet):
            if testLattice is None:
                testLattice = particles.lattice # infer lattice
            if not particles.lattice == testLattice:
                msg = "ParticleSet.lattice does not match Crystal lattice."
                raise(ValueError(msg.format(type(lattice).__name__)))
            self._basis = deepcopy(particles)
        elif isinstance(particles, ParticleBase):
            particles = [particles] # make iterable
            self._basis = ParticleSet(particles, lattice = testLattice)
        else:
            self._basis = ParticleSet(particles, lattice = testLattice)
        self._core_options = core_options
    
    @classmethod
    def fromFile(cls, wordstream, entrykey, param):
        """
        Return an instance of MotifCrystal from a file input.
        
        Parameters
        ----------
        wordstream : util.stringTools.FileParser
            The data stream for the file.
        entrykey : string
            The key marking entry into this block.
            Should be "MotifCrystal{"
        param : paramfile object
            The param file being used as a basis.
        """
        if not entrykey == "BasisCrystal{":
            raise(ValueError("Expected Key 'BasisCrystal{{'; got '{}'".format(entrykey)))
        lattice = param.getLattice()
        nmon = param.nMonomer
        
        core_options = [i for i in range(param.nMonomer)] # default
        core_option_set = False
        particles = []
        end_block = False
        while not end_block:
            word = next(wordstream)
            if word.lower() == "core_option":
                val = wordstream.next_int()
                if not val < nmon:
                    raise(ValueError("Core option {} exceeds nmonomer {}.".format(val,nmon)))
                if not core_option_set:
                    core_options = []
                    core_option_set = True
                core_options.append(val)
            elif isParticleKey(word):
                val = readParticleFromFile(wordstream, entrykey, lattice)
                particles.append(val)
            elif word == "}":
                end_block = True
            else:
                msg = "Unrecognized Key '{}' in MotifCrystal{{...}} block."
                raise(ValueError(msg.format(word)))
        return cls(particles, lattice, core_options)
    
    def particlePositions(self):
        """ Iterator over particle positions """
        for p in self._basis.particles:
            yield p.position
    
    @property
    def n_particles(self):
        return self._basis.nparticles
    
    @property
    def dim(self):
        return self._basis.dim
    
    @property
    def lattice(self):
        """ The lattice on which the crystal is defined. """
        return self._basis.lattice
    
    @lattice.setter
    def lattice(self, newLattice):
        """ Update the lattice for the crystal. """
        self._basis.lattice = newLattice
    
    def updateLattice(self, *args, **kwargs):
        """ 
        Change the lattice on which the crystal is defined.
        
        Wrapper for ParticleSet.updateLattice.
        
        The dimensionality of the crystal cannot be changed.
        
        Updated lattice data can be passed in four forms, with
        the expected format determined by the parameter names
        to which data is passed. Parameters below are listed in
        decreasing priority; parameter definitions also include
        priority numbers, with lower priority number indicating
        higher priority ( Priority 1 preferred over Priority 4 ).
        If multiple formats are included, the update is only 
        attempted with the highest-priority format included.
        
        Parameters
        ----------
        basis : array-like, optional, priority 1
            The new basis for the lattice. Must be able to be
            cast to a (dim-by-dim) numpy.ndarray. See Lattice
            for more detail.
        paramList : list-like, optional, priority 2
            A list-like iterable of lattice parameters. If a
            2D vector, should contain [a, b, gamma]. If a 3D
            vector, should contain [a, b, c, alpha, beta, gamma].
            See Lattice for more details.
        paramDict : dict, optional, priority 3
            A dict of "parameter name":value pairs.
            A 2D vector requires entries for "a", "b", "gamma".
            A 3D vector requires keys "a","b","c","alpha","beta","gamma".
        a : real, optional, priority 4
            The length of the first basis vector.
        b : real, optional, priority 4
            The length of the second basis vector.
        c : real, optional, priority 4
            The length of the third basis vector.
        alpha : real, optional, priority 4
            The angle (in degrees) between the basis vectors "b" and "c".
        beta : real, optional, priority 4
            The angle (in degrees) between the basis vectors "a" and "c".
        gamma : real, optional, priority 4
            The angle (in degrees) between the basis vectors "a" and "b".
        """
        return self._basis.updateLattice(*args, **kwargs)
    
    @property
    def particles(self):
        return self._basis.particles
    
    @property
    def coreOptions(self):
        """ List of possible core monomers. """
        return self._core_options
    
    @coreOptions.setter
    def coreOptions(self, new):
        self._core_options = new
    
    def addCoreOption(self, val):
        if val not in self._core_options:
            self._core_options.append(val)
    
    def chooseCore(self, monfrac):  
        """ Choose core monomer reference volumes for given Monomer Fractions.
        
        Parameters
        ----------
        monfrac : array-like of float
            Overall volume fraction of each monomer, with index
            corresponding to monomer id.
        
        Returns
        -------
        core_id : int
            ID of the core monomer.
        ref_vol : float
            Reference volume for the core monomer.
        """
        co = self._core_options
        if self._core_options is None:
            co = [i for i in range(len(monfrac))]
        # choose among all monomers
        coreid = 0
        minval = 10.0
        for i in co:
            if monfrac[i] < minval:
                coreid = i
                minval = monfrac[i]
        core_volume = self.lattice.volume * minval
        ref_vol = core_volume / self.n_particles
        return coreid, ref_vol
    
    def __iter__(self):
        """ Iterator over particles in crystal. """
        return iter(self._basis)
    
    def __str__(self):
        formstr = "< {} object with {} >"
        return formstr.format(type(self).__name__, self._output_data())
     
    def _output_data(self):
        formstr = "n_particles = {}, lattice = {}"
        return formstr.format(self.n_particles, self.lattice)
    
    @property
    def longString(self):
        """ Output string containing detailed data about crystal """
        out = "{}\nAll {}".format(self, self._basis.particleList())
        return out

class MotifCrystal(BasisCrystal):
    """ Crystal class based on SpaceGroup-Motif crystal definition """
    
    def __init__(self, space_group, motif, lattice=None, core_options=None):
        """
        Generator for MotifCrystal Class.
        
        Parameters
        ----------
        space_group : SpaceGroup object
            The set of symmetry operations for the crystal's space group.
        motif : ParticleSet, ParticleBase, iterable of ParticleBase
            The set of representative particles in the crystal's motif.
        lattice : Lattice (optional)
            The lattice of the crystal. If not specified, lattice is inferred.
        """
        self._space_group = space_group
        # Allow CrystalBase to verify lattice-motif agreement
        super().__init__(motif, lattice, core_options)
        # Motif Particles have been collected in self._basis ParticleSet object
        self._motif = self._basis
        basisParticles = self._space_group.applyToAll(self._motif, self._motif.particles)
        self._basis = ParticleSet(basisParticles)
    
    @classmethod
    def fromFile(cls, wordstream, entrykey, param):
        """
        Return an instance of MotifCrystal from a file input.
        
        Parameters
        ----------
        wordstream : util.stringTools.FileParser
            The data stream for the file.
        entrykey : string
            The key marking entry into this block.
            Should be "MotifCrystal{"
        param : paramfile object
            The param file being used as a basis.
        """
        if not entrykey == "MotifCrystal{":
            raise(ValueError("Expected Key 'MotifCrystal{{'; got '{}'".format(entrykey)))
        lattice = param.getLattice()
        group = param.group_name
        crystalsys = param.crystal_system
        spaceGroup = SpaceGroup(lattice.dim,group,crystalsys)
        nmon = param.nMonomer
        
        core_options = [i for i in range(param.nMonomer)] # default
        core_option_set = False
        particles = []
        end_block = False
        while not end_block:
            word = next(wordstream)
            if word.lower() == "core_option":
                val = wordstream.next_int()
                if not val < nmon:
                    raise(ValueError("Core option {} exceeds nmonomer {}.".format(val,nmon)))
                if not core_option_set:
                    core_options = []
                    core_option_set = True
                core_options.append(val)
            elif isParticleKey(word):
                val = readParticleFromFile(wordstream, entrykey, lattice)
                particles.append(val)
            elif word == "}":
                end_block = True
            else:
                msg = "Unrecognized Key '{}' in MotifCrystal{{...}} block."
                raise(ValueError(msg.format(word)))
        return cls(spaceGroup, particles, lattice, core_options)
    
    @BasisCrystal.lattice.setter
    def lattice(self, newLattice):
        """ Update the lattice for the crystal. """
        self._basis.lattice = newLattice
        self._motif.lattice = newLattice
    
    def updateLattice(self,*args,**kwargs):
        super().updateLattice(*args,**kwargs)
        self._motif.updateLattice(*args,**kwargs)
    
    def _output_data(self):
        formstr = super()._output_data()
        formstr += ", space group = {}".format(self._space_group)
        return formstr
    
    @property
    def longString(self):
        """ Output string containing detailed data about crystal """
        buildstr = super().longString
        buildstr += "\nFrom Motif {}".format(self._motif.particleList())
        return buildstr

_entry_key_map = {  "BasisCrystal{" :   BasisCrystal, \
                    "MotifCrystal{" :   MotifCrystal }

def isCrystalKey(entryKey):
    """ Return True if valid entryKey is given. """
    return entryKey in _entry_key_map

def readCrystalFromFile(wordstream, entrykey, param):
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
    if not isCrystalKey(entrykey):
        msg = "No Crystal Type associated with key {}."
        raise(ValueError(msg.format(entrykey)))
    cls = _entry_key_map.get(entrykey)
    return cls.fromFile(wordstream,entrykey,param)


