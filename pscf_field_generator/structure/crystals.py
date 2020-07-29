""" Module containing definitions for building a crystal structure """

from .lattice import Lattice
from .symmetry import SpaceGroup
from .particles import ScatteringParticle, ParticleBase, ParticleSet

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
    

def buildCrystal(style, N_particles, positions, formFactor, lattice, space_group=None):
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
    formFactor : Object exposing the ParticleForm interface.
        The scattering form factor for the particle.
    lattice : structure.Lattice
        The lattice on which the crystal is defined.
    space_group : structure.symmetry.SpaceGroup
        If style == 'motif', this input is required. It is the space group representing the
        symmetry of the crystal.
    """
    initSet = ParticleSet()
    for i in range(N_particles):
        p = ScatteringParticle(positions[i,:], formFactor)
        initSet.addParticle(p)
    if style == "basis":
        return CrystalBase(lattice, initSet)
    else:
        return CrystalMotif(space_group, initSet, lattice)
    

