"""
Definition of Particle and Form-Factor classes
"""

from pscfFieldGen.structure.core import POSITION_TOLERANCE
from pscfFieldGen.structure.lattice import Lattice, Vector

from abc import ABC, abstractmethod
from copy import deepcopy
import numba
import numpy as np
from scipy.special import j1

@numba.jit("double(double,double)", nopython=True, cache=True)
def sphereFormFactorAmplitude(qNorm = 0.0, zero_q_magnitude = 1.0):
    """ 
    Returns the form factor amplitude for a spherical particle.
    
    Parameters
    ----------
    qNorm : real
        The magnitude of the wave-vector.
    zero_q_magnitude : real
        If R not specified, this is taken to be the volume
        of the sphere. Otherwise, it is simply treated as
        a scaling factor.
    
    Returns
    -------
    f_of_q : scalar
        The form factor at q.
    """
    R = ( ( 3 * zero_q_magnitude ) / (4 * np.pi) ) ** (1./3)
    qR = qNorm * R
    ff = zero_q_magnitude * 3 * (np.sin(qR) - qR * np.cos(qR)) / qR**3
    return ff

@numba.jit("double(double,double)",forceobj=True, cache=True)
def circleFormFactorAmplitude(qNorm = 0.0, zero_q_magnitude = 1.0):
    """ 
    Returns the form factor amplitude for a 2D circular particle.
    
    Parameters
    ----------
    qNorm : real
        The magnitude of the wave-vector.
    zero_q_magnitude : real
        The area of the circle.
    
    Returns
    -------
    f_of_q : scalar
        The form factor at q.
    """
    R = np.sqrt( zero_q_magnitude / np.pi )
    qR = qNorm * R
    bessarg = 2.0 * qR
    bess = j1(bessarg)
    ff = (2.0 / (qR**3)) * (qR - bess)
    ff = zero_q_magnitude*ff
    return ff

def defaultFormFactor(dim):
    """ Return default form factor for dimensionality dim """
    if dim == 2:
        return circleFormFactorAmplitude
    elif dim == 3:
        return sphereFormFactorAmplitude
    else:
        raise(ValueError("dim must be either 2 or 3. Gave {}".format(dim)))

class ParticlePosition(Vector):
    """ Sub-class of Vector, specialized for particle positions. """
    
    def __init__(self, components, lattice, keepInUnitCell=True):
        """ 
        Construct a new Vector.
        
        Parameters
        ----------
        components : 1D array-like
            The components of the vector, relative to lattice.
        lattice : Lattice
            The lattice on which the vector is defined.
        keepInUnitCell : bool (optional)
            If True (default), coordinates are restricted to 0 <= r < 1
        """
        self._keep_UC_flag = keepInUnitCell
        super().__init__(components,lattice)
    
    def __eq__(self, other):
        """ Return true if two particle positions are equal.
        
        Overrides the Vector equality method to use POSITION_TOLERANCE
        rather than more precise equality requirements of Vector.
        """
        if not isinstance(other,Vector):
            msg = "Operator == unavailable between ParticlePosition and {}"
            raise(TypeError(msg.format(type(other).__name__)))
        if not self._lattice == other._lattice:
            return False
        return np.allclose(self._components, other._components, rtol=1e-10, atol=POSITION_TOLERANCE)
    
    def _process_updated_components(self):
        self._enforce_UC()
        super()._process_updated_components()
    
    def _enforce_UC(self):
        """ Adjust current position for unit cell and tolerance constraints """
        onevec = np.ones(self._dim)
        testupper = np.isclose(onevec, self._components, atol=POSITION_TOLERANCE)
        atol = POSITION_TOLERANCE * onevec
        testlower = np.absolute(self._components) <= atol
        for i in range(self._dim):
            if self._keep_UC_flag:
                if self._components[i] >= 1 and not testupper[i]:
                    self._components[i] -= 1
                elif self._components[i] < 0 and not testlower[i]:
                    self._components[i] += 1
                if testlower[i] or testupper[i]:
                    self._components[i] = 0.0
            else:
                if testlower[i]:
                    self._components[i] = 0.0

class ParticleBase(ABC):
    """ Class representing a particle in a crystal stucture """
    
    def __init__(self, typename, position):
        """
        Parameters
        ----------
        typename : string
            A unique identifier for the particle type.
        position : ParticlePosition
            The location of the particle in the unit cell.
        """
        self._typename = typename
        if not isinstance(position, ParticlePosition):
            msg = "Position can not be given as type {}"
            raise(TypeError(msg.format(type(position).__name__)))
        self._position = deepcopy(position)
        self._dim = len(self._position)
    
    def replicate_at_position(self, newPosition):
        """
        Create and return a new ParticleBase at new position.
        
        The new instance returned from the method is identical to the
        current particle, but located at newposition.
        
        Parameters
        ----------
        position : array-like
            The position of the new particle.
        
        Returns
        -------
        new_particle : ParticleBase
            The new particle instance.
        
        Raises
        ------
        TypeError
            If template is not of type ParticleBase
        """
        new_particle = deepcopy(self)
        new_particle.position = newPosition
        return new_particle
    
    @property
    def typename(self):
        return self._typename
    
    @property
    def dim(self):
        return self._dim
    
    @property
    def position(self):
        return self._position
    
    @position.setter
    def position(self, newPosition):
        """
        Change the position of the particle.
        
        Parameters
        ----------
        newPosition : array-like or ParticlePosition
            The fractional coordinates of the new position.
        
        Returns
        -------
        None
        
        Raises
        ------
        ValueError
            If the new position does not match the dimensionality of the particle.
        """
        if isinstance(newPosition, Vector):
            if not newPosition.lattice == self._position.lattice:
                raise(ValueError("New Position Not on Correct Lattice"))
            self._position.components = newPosition.components
        else:
            npos = np.array(newPosition)
            if not len(npos) == self.dim:
                raise(ValueError("Dimension mismatch in Particle Position."))
            self._position.components = npos
    
    @abstractmethod
    def formFactorAmplitude(self, q, scaling_factor, *args, **kwargs):
        """ Return the Form Factor Amplitude of the particle.
        
        Calculate and Return the Form Factor Amplitude for this
        particle.
        
        Parameters
        ----------
        q : Vector
            The Wave-Vector
        scaling_factor: real
            A Scaling Factor on the Calculation. Typical Form Factors
            are scaled to approach 0 as abs(q)-->0, and this scaling
            factor is typically taken to be the particle volume.
            More Generally, the scaling factor is applied to the bare
            form factor for the particle shape.
        
        Returns
        -------
        formFactorAmplitude : real
            The form-factor amplitude at the given wave-vector.
        """
        pass
    
    def typeMatch(self, other):
        """ Return True if other is the same type. False otherwise. """
        return self.typename == other.typename
    
    def positionMatch(self,other):
        """ Return True if other is located at same position. False otherwise. """
        if not isinstance(other, ParticleBase):
            msg = "Can only compare Particle Types, not {}"
            raise(TypeError(msg.format(type(other).__name__)))
        return self._position == other._position
    
    def conflict(self, other):
        """ Return True if particles of different type are at same position. """
        if self.typeMatch(other):
            return False
        else:
            return self.positionMatch(other)
    
    def __eq__(self, other):
        return self.typeMatch(other) and self.positionMatch(other)
    
    def __str__(self):
        formstr = "< {} object with {} >"
        out = formstr.format(type(self).__name__, self._output_data())
        return out
        
    def _output_data(self):
        formstr = "Type = {}, Position = {}"
        return formstr.format(self.typename, self.position)

class ParticleSet(object):
    """ 
    A set of particles, 
    
    ParticleSet objects hold a collection of ParticleBase-like objects,
    subject to the requirement that each particle is unique and non-conflicting.
    This means that only one particle is at each position in space.
    """
    def __init__(self, startSet = None):
        self._particles = []
        if startSet is not None:
            for p in startSet:
                self.addParticle(p)
    
    def containsMatch(self, testParticle):
        """
        Return True if ParticleSet contains a particle equal to testParticle.
        Return False otherwise.
        """
        for p in self._particles:
            if p == testParticle:
                return True
        return False
    
    def containsConflict(self, testParticle):
        """
        Return True if ParticleSet contains a particle which conflicts with testParticle.
        Return False otherwise.
        """
        for p in self._particles:
            if p.conflict(testParticle):
                return True, p
        return False, None
    
    def addParticle(self, newParticle, replace_on_conflict = False):
        """
        Add newParticle to the particle set.
        
        If newParticle conflicts with a current particle,
        can either reject newParticle, or replace the original particle.
        """
        if self.containsMatch(newParticle):
            return False
        hasConflict, conflictParticle = self.containsConflict(newParticle)
        if hasConflict and replace_on_conflict:
            self._particles.remove(conflictParticle)
            self._particles.append(newParticle)
            return True
        if hasConflict and not replace_on_conflict:
            return False
        self._particles.append(newParticle)
    
    @property
    def nparticles(self):
        return len(self._particles)
    
    @property
    def particles(self):
        return deepcopy(self._particles)
    
    def __str__(self):
        return "< {} with {} particles >".format(type(self).__name__, self.nparticles)
    
    def particleList(self):
        """ Return a string with each particle separated by a newline """
        buildStr = "Particles:"
        for p in self._particles:
            buildStr += "\n{}".format(p)
        return buildStr
        
class ScatteringParticle(ParticleBase):
    """ Particle with associated Form Factor. """
    def __init__(self, position, formFactor=None):
        super().__init__("Micelle", position)
        if formFactor is None:
            self._formFactor = defaultFormFactor(self.dim)
        else:
            self._formFactor = formFactor
    
    def formFactorAmplitude(qnorm, vol, smear):
        return self._formFactor(qnorm, vol, smear)
    
    @property
    def formFactor(self):
        return self._formFactor
    
    def _output_data(self):
        out = "{}, form factor = {}".format(super()._output_data(), self._formFactor.__name__)
        return out

class Sphere(ParticleBase):
    
    def __init__(self, position):
        super().__init__("Sphere", position)
    
    def formFactorAmplitude(self, q, volume):
        qNorm = q.magnitude
        R = ( ( 3 * volume ) / (4 * np.pi) ) ** (1./3)
        qR = qNorm * R
        ff = volume * 3 * (np.sin(qR) - qR * np.cos(qR)) / qR**3
        return ff
        
class Cylinder2D(ParticleBase):
    
    def __init__(self, position):
        super().__init__("Cylinder2D", position)
    
    def formFactorAmplitude(self, q, volume):
        """ 
        Returns the form factor amplitude for a 2D circular particle.
        
        Parameters
        ----------
        q : Vector
            The wave-vector.
        area : real
            The area of the circle.
        
        Returns
        -------
        f_of_q : scalar
            The form factor at q.
        """
        qNorm = q.magnitude
        R = np.sqrt( area / np.pi )
        qR = qNorm * R
        bessarg = 2.0 * qR
        bess = j1(bessarg)
        ff = (2.0 / (qR**3)) * (qR - bess)
        ff = area * ff
        return ff
    
