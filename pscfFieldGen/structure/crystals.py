""" Module containing definitions for building a crystal structure """

from pscfFieldGen.structure.lattice import Lattice
from pscfFieldGen.structure.symmetry import SpaceGroup
from pscfFieldGen.structure.particles import ScatteringParticle, ParticleBase, ParticleSet

from copy import deepcopy
import numpy as np

class CrystalBase(object):
    """ Crystal class based on Lattice-Basis crystal definition """
    
    def __init__(self, lattice, particles):
        self._lattice = lattice
        self._basis = deepcopy(particles)
    
    def particlePositions(self):
        """ Iterator over particle positions """
        for p in self._basis.particles:
            yield p.position
    
    @property
    def n_particles(self):
        return self._basis.nparticles
    
    @property
    def dim(self):
        return self._lattice.dim
    
    @property
    def lattice(self):
        return deepcopy(self._lattice)
    
    @property
    def particles(self):
        return deepcopy(self._basis.particles)
    
    def __str__(self):
        formstr = "< {} object with {} >"
        return formstr.format(type(self).__name__, self._output_data())
     
    def _output_data(self):
        formstr = "n_particles = {}, lattice = {}"
        return formstr.format(self.n_particles, self._lattice)
    
    @property
    def longString(self):
        """ Output string containing detailed data about crystal """
        out = "{}\nAll {}".format(self, self._basis.particleList())
        return out
    

class CrystalMotif(CrystalBase):
    """ Crystal class based on SpaceGroup-Motif crystal definition """
    
    def __init__(self, space_group, motif, lattice):
        """
        Generator for CrystalMotif Class.
        
        Parameters
        ----------
        space_group : SpaceGroup object
            The set of symmetry operations for the crystal's space group.
        motif : particles.ParticleSet
            The set of representative particles in the crystal's motif.
        lattice : Lattice object
            The lattice of the crystal.
        """
        self._space_group = space_group
        self._motif = deepcopy(motif)
        particles = self._build_basis_from_motif()
        super().__init__(lattice, particles)
        
    def _build_basis_from_motif(self):
        """ From the motif and space group of the crystal, build the basis. """
        buildingSet = ParticleSet()
        for motif_particle in self._motif.particles:
            name = motif_particle.typename
            posbase = motif_particle.position
            posList = self._space_group.evaluatePosition(posbase)
            for pos in posList:
                newParticle = motif_particle.replicate_at_position(pos)
                buildingSet.addParticle(newParticle)
        return buildingSet
    
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
        

def buildCrystal(style, N_particles, positions, lattice, **kwargs):
    """
    Initialize and return a crystal object.
    
    Parameters
    ----------
    style : string matching 'motif' or 'basis'
        A string flag identifying whether the particle set is complete ('basis')
        or needs to be built from symmetry ('motif')
    N_particles : int
        The number of particle positions in the position set.
    positions : numpy.ndarray
        The set of particle positions with each row representing the fractional coordinates
        of one particle. Array dimensions should be N_particle-by-2 for 2D systems,
        and N_particles-by-3 for 3D systems.
    lattice : structure.Lattice
        The lattice on which the crystal is defined.
    
    Keyword Parameters
    ------------------
    formFactor : Class exposing the ParticleForm interface.
        The scattering form factor for the particle. If omitted, a default is used.
    group_name : string
        The name of the space group.
    crystal_system : string
        The name of the crystal system the space group is in.
    space_group : structure.symmetry.SpaceGroup
        If style == 'motif', this input is required. It is the space group representing the
        symmetry of the crystal.
    """
    initSet = ParticleSet()
    formFactor = kwargs.get("formFactor", None)
    for i in range(N_particles):
        p = ScatteringParticle(positions[i,:], formFactor)
        initSet.addParticle(p)
    if style == "basis":
        return CrystalBase(lattice, initSet)
    else:
        groupName = kwargs.get("group_name",None)
        crystalSystem = kwargs.get("crystal_system",None)
        space_group = SpaceGroup(lattice.dim, crystalSystem, groupName)
        return CrystalMotif(space_group, initSet, lattice)
    

