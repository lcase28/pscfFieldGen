# Lattice Module. 
# Partially derived from crystals package available at:
#   https://github.com/LaurentRDC/crystals.git
# Generallized to handle 2-, or 3-dimensional systems

# Imports
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum, unique
import numpy as np

class Lattice(object):
    """ Object representing a crystallographic basis vector lattice"""
    
    def __init__(self, dim, basis, **kwargs):
        """
        Generate a lattice object.
        
        Params:
        -------
        dim: int, dim >= 2
            Number of dimensions in the lattice.
        basis: numpy.ndarray
            Matrix (size: dim-by-dim) representation of basis vectors.
            Coordinates of each vector are defined relative to the
            standard basis (mutually orthogonal unit vectors).
            Row i ( elements in basis[i,:] ) corresponds to lattice basis
            vector i.
        """
        self.dim = dim;
        self.basis = basis;
    
    @classmethod
    def latticeFromParameters(cls, dim, **kwargs):
        """
        Generate a lattice object for 2-D or 3-D lattice.
        
        Params:
        -------
        dim: int, in set {2, 3}
            The dimensionality of the resulting lattice
        
        Keyword Params:
        ---------------
        a:  float, required
            Magnitude of first basis vector.
        b:  float, required
            Magnitude of second basis vector.
        c:  float, (only if dim == 3)
            Magnitude of third basis vector.
        alpha:  float, only if dim == 3
            Angle (in degrees, range (0, 180) ) between vector b and c.
        beta:   float, only if dim == 3
            Angle (in degrees, range (0, 180) ) between vector a and c.
        gamma:  float, required
            Angle (in degrees, range (0, 180) ) between vectors a and b.
        
        Returns:
        --------
        lat: Lattice
            Lattice defined by given parameters.
            
        Raises:
        -------
        TypeError: If input arguments are non-numeric.
        ValueError: If dim not one of {2, 3}, or invalid lattice parameters.
        """
        basis = cls.basisFromParameters(dim, **kwargs)
        return cls(dim, basis)
            
    @classmethod
    def basisFromParameters(cls, dim, **kwargs):
        """
            Return a set of basis vectors.
            
            Params:
            -------
            dim: int, in set {2, 3}
                The dimensionality of the resulting lattice
            
            Keyword Params:
            ---------------
            a:  float, required
                Magnitude of first basis vector.
            b:  float, required
                Magnitude of second basis vector.
            c:  float, (only if dim == 3)
                Magnitude of third basis vector.
            alpha:  float, only if dim == 3
                Angle (in degrees, range (0, 180) ) between vector b and c.
            beta:   float, only if dim == 3
                Angle (in degrees, range (0, 180) ) between vector a and c.
            gamma:  float, required
                Angle (in degrees, range (0, 180) ) between vectors a and b.
            
            Returns:
            --------
            basis: numpy.ndarray, 'dim'-by-'dim'
                Lattice defined by given parameters. See 'Convention' below
                for details on orientation conventions used.
            
            Convention:
            -----------
            Notation:
                a, b, c - 1st, 2nd, 3rd lattice basis vectors
                x, y, z - 1st, 2nd, 3rd standard basis vectors
            a: First lattice vector
                Taken to lie on x such that its components are [a 0 0]
            b: Second Lattice vector
                Taken to lie in the x-y plane along with a.
                Its components then become: [b_x b_y, 0]
            c: Third lattice vector
                Taken to lie outside of x-y plane.
                Only (3D) basis vector with component in z-direction
                
            Raises:
            -------
            TypeError: If input arguments are non-numeric.
            ValueError: If dim not one of {2, 3}, or invalid lattice parameters.
        """
        # extract all parameters
        a = kwargs.get("a",None)
        b = kwargs.get("b",None)
        c = kwargs.get("c",None)
        alpha = kwargs.get("alpha",None)
        beta = kwargs.get("beta",None)
        gamma = kwargs.get("gamma",None)
        
        # check inputs
        if (a is None) or (b is None) or (gamma is None):
            raise TypeError("Required lattice parameter is missing.")
            
        if dim == 3 and ((c is None) or (alpha is None) or (beta is None)):
            raise TypeError("Missing lattice parameter for 3D lattice")
        
        # initialize basis
        basis = np.zeros((dim,dim))
        
        # Complete common calculations
        gammaRad = np.deg2rad(gamma)
        basis[0,0] = a
        basis[1,0] = b*np.cos(gammaRad)
        basis[1,1] = b*np.sin(gammaRad)
        
        # Additional 3D calculations
        if dim == 3:
            alphaRad = np.deg2rad(alpha)
            betaRad = np.deg2rad(beta)
            basis[2,0] = c*np.cos(betaRad)
            basis[2,1] = c*np.cos(alphaRad)*np.sin(gammaRad)
            basis[2,2] = np.sqrt( c**2 - basis[2,0]**2 - basis[2,1]**2)
        return basis
    
    ## Properties
    
    @property
    def latticeParameters(self):
        """
            Lattice parameters as list of lengths and angles.
            3D: [a, b, c, alpha, beta, gamma]
            2D: [a, b, gamma]
        """
        a = self.basis[0,:]
        b = self.basis[1,:]
        aMag = np.linalg.norm(a)
        bMag = np.linalg.norm(b)
        gamma = np.rad2deg( np.arccos( np.dot(a,b) / (aMag*bMag) ) )
        if self.dim == 2:
            return np.asarray([aMag, bMag, gamma])
        elif self.dim == 3:
            c = self.basis[2,:]
            cMag = np.linalg.norm(c)
            alpha = np.rad2deg( np.arccos( np.dot(b,c) / (bMag*cMag) ) )
            beta = np.rad2deg( np.arccos( np.dot(a,c) / (aMag*cMag) ) )
            return np.asarray([aMag, bMag, cMag, alpha, beta, gamma])
        else:
            return NotImplemented
    
    @property
    def latticeVectors(self):
        """ Return lattice vectors as numpy.ndarray """
        return deepcopy(self.basis)
    
    @property
    def volume(self):
        """ Area (2D) or Volume (3D) enclosed by basis vectors """
        return np.linalg.det(self.basis)
    
    @property
    def metricTensor(self):
        """ The real-space metric tensor """
        return np.matmul(self.basis, self.basis.T)
    
    @property
    def reciprocal(self):
        """ Lattice object for reciprocal lattice. """
        return self.__class__(self.dim, self.reciprocalVectors)
    
    @property
    def reciprocalVectors(self):
        """ Basis vectors of reciprocal lattice as numpy.ndarray,
            with rows corresponding to each vector on standard basis """
        reciprocalMetricTensor = np.linalg.inv(self.metricTensor)
        return np.matmul(reciprocalMetricTensor, self.basis)
    
    ## Instance Methods
    
    def toStandardBasis(self, vect):
        """
            Convert the given fractional coordinates to its
            coordinates in the standard basis.
            
            Params
            ------
            vect: array-like, list-like
                Fractional coordinates (in THIS basis) of the vector.
            
            Returns
            -------
            newVect : numpy array, (dim-by-1)
                Original vector, expressed in terms of the standard basis.
        """
        v = np.reshape( np.asarray(vect), (self.dim,1))
        newVect = np.matmul(self.basis.T, v)
        return newVect
    
    def fromStandardBasis(self, vect):
        """
            Convert the vector from the stanard basis to this lattice basis.
            
            Params
            ------
            vect: array-like, list-like
                Fractional coordinates (in THIS basis) of the vector.
            
            Returns
            -------
            newVect : numpy array, (dim-by-1)
                Original vector, expressed in terms of the standard basis.
        """
        v = np.reshape( np.asarray(vect), (self.dim,1))
        transformMatrix = np.linalg.inv(self.basis.T)
        newVect = np.matmul(transformMatrix, v)
        return newVect
        
    ## Implementation Note:
    ## The following methods are implemented based on
    ## the methods 'toStandardBasis' and 'fromStandardBasis'.
    ## Thus, they all assume that both lattices are defined 
    ## relative to the *same* "standard basis"
    def changeFromBasis(self, vect, reference):
        """
            Return vector coordinates relative to THIS lattice basis.
            
            Note: Method assumes current basis and reference are defined
            relative to the **same** standard basis. As such, if both
            were instantiated using the Lattice.latticeFromParameters method,
            the 'a' vectors in both lattices will be coincident, and will be
            coplanar with both Lattices' 'b' vectors.
            
            Parameters
            ----------
            vect : float, 1-D list-like
                The vector, in fractional coordinates relative to
                old basis.
            reference : Lattice
                The lattice whose vectors represent current basis
                of the given vector coordinates.
            
            Returns
            -------
            newVect : float, numpy.ndarray
                Coordinates of the vector relative to basis vectors
                of THIS lattice.
        """
        v = np.reshape( np.asarray(vect, dtype=np.float64), (self.dim, 1) )
        vectInStandardBasis = reference.toStandardBasis(vect)
        newVect = self.fromStandardBasis(vectInStandardBasis)
        return newVect
    
    def changeToBasis(self, vect, target):
        """
            Return vector coordinates relative to target lattice basis.
            
            Note: Method assumes current basis and reference are defined
            relative to the **same** standard basis. As such, if both
            were instantiated using the Lattice.latticeFromParameters method,
            the 'a' vectors in both lattices will be coincident, and will be
            coplanar with both Lattices' 'b' vectors.
            
            Parameters
            ----------
            vect : float, 1-D list-like
                The vector, in fractional coordinates relative to
                THIS basis.
            target : Lattice
                The lattice whose vectors represent the basis in which
                to cast the vector.
            
            Returns
            -------
            newVect : float, numpy.ndarray
                Coordinates of the vector relative to basis vectors
                of target lattice.
        """
        return target.changeFromBasis(vect, self)
        
    def toReciprocalBasis(self, vect):
        """
            Return the Reciprocal Basis coordinates of the vector
            currently expressed in Real Space coordinates
            
            Note: Method assumes current basis and reference are defined
            relative to the **same** standard basis. As such, if both
            were instantiated using the Lattice.latticeFromParameters method,
            the 'a' vectors in both lattices will be coincident, and will be
            coplanar with both Lattices' 'b' vectors.
            
            Params
            ------
            vect: array-like, list-like
                Fractional coordinates (in THIS basis) of the vector.
            
            Returns
            -------
            newVect : numpy array, (dim-by-1)
                Original vector, expressed in terms of the reciprocal basis.
        """
        newVect = self.changeToBasis(vect, self.reciprocal)
        return newVect
    
    def fromReciprocalBasis(self, vect):
        """
            Return the Real space coordinates of the vector
            currently expressed in Reciprocal Space coordinates
            
            Note: Method assumes current basis and reference are defined
            relative to the **same** standard basis. As such, if both
            were instantiated using the Lattice.latticeFromParameters method,
            the 'a' vectors in both lattices will be coincident, and will be
            coplanar with both Lattices' 'b' vectors.
            
            Params
            ------
            vect: array-like, list-like
                Fractional coordinates (in reciprocal basis) of the vector.
            
            Returns
            -------
            newVect : numpy array, (dim-by-1)
                Original vector, expressed in terms of the real space basis.
        """
        newVect = self.changeFromBasis(vect, self.reciprocal)
        return newVect
        
    def vectorDot(self, v1, v2):
        """
            Return the dot product of vectors, v1 and v2, both
            expressed in the basis of this lattice.
            
            Parameters
            ----------
            v1 : float, 1-D list-like
                The first vector, in fractional coordinates.
            v2 : float, 1-D list-like
                The second vector, in fractional coordinates.
            
            Returns
            -------
            v1_dot_v2 : float
                The dot product of v1 and v2
        """
        v1 = np.reshape( np.asarray(v1), (1, self.dim) )
        v2 = np.reshape( np.asarray(v2), (self.dim, 1) )
        return np.matmul( v1, np.matmul( self.metricTensor, v2 ) )
    
    def vectorNorm(self, vect):
        """ Returns the magnitude of a vector expressed in this basis """
        return np.sqrt( self.vectorDot( vect, vect ) )
    
    ## "Private" internal methods
    
    def __repr__(self):
        s = "< Lattice object with parameters {:.3f}A, {:.3f}, {:.3f}, {:.3f}, {:.3f} >"
        return s.format(*self.latticeParameters)
    
    def __hash__(self):
        return hash(self.latticeParameters) | super().hash()
        
    def __array__(self, *args, **kwargs):
        """ returns an ndarray. Each row is a lattice vector. """
        return np.array(self.latticeVectors, *args, *kwargs)
    
