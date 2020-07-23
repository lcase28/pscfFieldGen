from abc import ABC, abstractmethod
import numpy as np
import scipy as sp

class ParticleForm(ABC):
    """
        Abstract base for classes representing particle form factors
        in composition field initial guess generation.
    """
    
    @classmethod
    @abstractmethod
    def formFactorAmplitude(cls, qNorm = 0, zero_q_magnitude = 1, **kwargs):
        """ 
            Returns the form factor amplitude for the particle.
            
            Parameters
            ----------
            qNorm : real
                The magnitude of the wave-vector.
            zero_q_magnitude : real
                Scaling value for the form factor such that
                
                .. math::
                
                    \lim_{q \\to 0}f(q) = zero_q_magnitude
            R : real, scalar (optional, keyword)
                Characteristic size of the particle (override default)
            smear : real, scalar (optional, keyword)
                A value on [0,1] by which to smear the particle interfaces
                (normalized to the particle characteristic length)
            
            Returns
            -------
            f_of_q : scalar
                The form factor at q.
            f_smear : scalar
                The gaussian smearing form factor
        """
        return zero_q_magnitude
    
class SphereForm(ParticleForm):
    """ Sphere Form Factor """
    
    @classmethod
    def formFactorAmplitude(cls, qNorm = 0, zero_q_magnitude = 1, **kwargs):
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
            R : scalar (optional, keyworded)
                The radius of the sphere.
            smear : scalar (optional, keyworded)
                The Gaussian Smearing factor, normalized to sphere radius.
            
            Returns
            -------
            f_of_q : scalar
                The form factor at q.
            f_smear : scalar
                The gaussian smearing form factor
        """
        R_default = ( ( 3 * zero_q_magnitude ) / (4 * np.pi) ) ** (1./3)
        R = kwargs.get("R", R_default)
        
        smear = kwargs.get("smear", 0)
        
        qR = qNorm * R
        ff = 3 * (np.sin(qR) - qR * np.cos(qR)) / qR**3
        fsmear = np.exp( -(smear**2 * qR**2 / 2) )
        
        return ff, fsmear

class Circle2DForm(ParticleForm):
    """ Form factor for 2D circles """
    
    @classmethod
    def formFactorAmplitude(cls, qNorm = 0, zero_q_magnitude = 1, **kwargs):
        """ 
            Returns the form factor amplitude for a 2D circular particle.
            
            Parameters
            ----------
            qNorm : real
                The magnitude of the wave-vector.
            zero_q_magnitude : real
                If R not specified, this is taken to be the area
                of the circle. Otherwise, it is simply treated as
                a scaling factor.
            R : scalar (optional, keyworded)
                The radius of the circle
            smear : scalar (optional, keyworded)
                The gaussian smearing factor.
            
            Returns
            -------
            f_of_q : scalar
                The form factor at q.
            f_smear : scalar
                The gaussian smearing form factor
        """
        R_default = np.sqrt( zero_q_magnitude / np.pi )
        R = kwargs.get("R", R_default)
        
        smear = kwargs.get("smear",0)
        
        qR = qNorm * R
        #ff = ( 2.0 / (qR**2) ) * ( 1.0 - (sp.special.j1(2.0*qR) / qR))
        ff = (2.0 / (qR**3)) * (qR - sp.special.j1(2.0*qR))
        ff = zero_q_magnitude*ff
        f_smear = np.exp( -(smear**2 * qR**2 / 2.0) )
        return ff, f_smear

