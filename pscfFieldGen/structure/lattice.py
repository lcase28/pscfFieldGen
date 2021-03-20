# Imports
from copy import deepcopy
import numpy as np

class Lattice(object):
    """ 
    Object representing a crystallographic basis vector lattice.
    
    Lattice objects are designed to be immutable. 
    Modification of a Lattice's internal data can cause side-effects.
    """
    
    def __init__(self, dim, basis):
        """
        Generate a lattice object.
        
        Params:
        -------
        dim: int in (2,3)
            Number of dimensions in the lattice.
        basis: numpy.ndarray
            Matrix (size: dim-by-dim) representation of basis vectors.
            Coordinates of each vector are defined relative to the
            standard basis (mutually orthogonal unit vectors).
            Row i ( elements in basis[i,:] ) corresponds to lattice basis
            vector i.
        """
        if dim not in (2,3):
            raise(ValueError("Lattice must be 2D or 3D"))
        self._dim = dim;
        basis = np.array(basis)
        if not basis.shape == (dim, dim):
            raise(ValueError("Gave basis {} for dim {}".format(basis.shape,dim)))
        self._basis = basis;
        
        self._volume = np.linalg.det(self._basis)
        self._metric_tensor = np.matmul(self._basis, self._basis.T)
        
        reciprocalMetricTensor = np.linalg.inv(self._metric_tensor)
        self._reciprocal_basis =  np.matmul(reciprocalMetricTensor, self._basis)
    
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
    
    def copy(self):
        return Lattice(self._dim, self._basis)
    
    ## Properties
    
    @property
    def dim(self):
        return self._dim
    
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
        return np.array(self._basis)
    
    @property
    def basis(self):
        return np.array(self._basis)
    
    @property
    def volume(self):
        """ Area (2D) or Volume (3D) enclosed by basis vectors """
        return self._volume
    
    @property
    def metricTensor(self):
        """ The real-space metric tensor """
        return np.array(self._metric_tensor)
    
    def reciprocalLattice(self):
        """ Lattice object for reciprocal lattice. """
        return Lattice(self._dim, self._reciprocal_basis)
    
    def as3D(self):
        """
        Return a 3D analog lattice.
        
        If lattice is already 3D, returns a reference to itself.
        If lattice is 2D, the unit cartesian "z" basis vector is
        added as the third basis vector.
        """
        if self.dim == 3:
            return self
        btmp = self.basis
        ttmp = np.eye(3)
        ttmp[0,0] = btmp[0,0]
        ttmp[0,1] = btmp[0,1]
        ttmp[1,0] = btmp[1,0]
        ttmp[1,1] = btmp[1,1]
        return Lattice(3,ttmp)
    
    def isReciprocal(self,other):
        return self.reciprocalLattice() == other
    
    ## "Private" internal methods
    
    def __eq__(self, other):
        if not isinstance(other, Lattice):
            raise(TypeError("Cannot compare Lattice and {}".format(type(other).__name__)))
        if not self.dim == other.dim:
            return False
        return np.allclose(self._basis, other._basis)
            
    def __repr__(self):
        if self.dim == 3:
            s = "< 3D Lattice with parameters {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f} >"
        else:
            s = "< 2D Lattice with parameters {:.3f}, {:.3f}, {:.3f} >"
        return s.format(*self.latticeParameters)
    
    def __hash__(self):
        return hash(self.latticeParameters) | super().hash()
        
    def __array__(self, *args, **kwargs):
        """ returns an ndarray. Each row is a lattice vector. """
        return np.array(self.latticeVectors, *args, *kwargs)

def cartesianLattice(dim):
    if dim not in (2,3):
        raise(ValueError("Lattice must be 2D or 3D."))
    return Lattice(dim,np.eye(dim))

class Vector(object):
    """ A 2D or 3D Vector on a Defined lattice """
    def __init__(self, components, lattice):
        """ 
        Construct a new Vector.
        
        Parameters
        ----------
        components : 1D array-like
            The components of the vector, relative to lattice.
        lattice : Lattice
            The lattice on which the vector is defined.
        """
        # ensure types and compatibility of inputs
        components = np.array(components).flatten()
        if not isinstance(lattice,Lattice):
            raise(TypeError("A Vector's lattice must be an instance of Lattice"))
        if not len(components) == lattice.dim:
            raise(ValueError("Components must match lattice dimension."))
        
        # store input values
        self._components = components
        self._lattice = lattice
        self._dim = self._lattice.dim
        
        # Compute secondary properties
        self._cartesian_components = self._components @ self._lattice.basis
        self._magnitude = np.linalg.norm(self._cartesian_components)
        
        # Assign instance versions of static methods
        self.dot = self._instance_dot
        self.caseToLattice = self._instance_castToLattice
    
    ## Properties
    
    @property
    def dim(self):
        return self._dim
    
    @property
    def magnitude(self):
        return self._magnitude
    
    @property
    def components(self):
        return np.array(self._components)
    
    @components.setter
    def components(self, comp):
        """ 
        Set vector components.
        
        Parameters
        ----------
        comp : 1D array-like
            Must have the same number of elements as current components.
        """
        if not len(comp) == self._dim:
            raise(ValueError("Length of components {} exceeds dim of vector".format(comp)))
        self._components[:] = comp
        self._cartesian_components = self._components @ self._lattice.basis
        self._magnitude = np.linalg.norm(self._cartesian_components)
    
    @property
    def lattice(self):
        return self._lattice.copy()
    
    @property
    def cartesian(self):
        """ The vector components referenced to the standard unit cartesian lattice. """
        return np.array(self._cartesian_components)
    
    ## Operations
    
    def asCartesian(self):
        """ Return new Vector object representing this vector on cartesian lattice. """
        return Vector(self.cartesian, cartesianLattice(self.dim))
    
    def as3D(self): 
        """ 
        Cast the vector into a 3D lattice.
        
        If vector is 2D, returns a new vector with zero-component
        along third cartesian basis vector.
        If vector is 3D, returns the object itself.
        """
        if self.dim == 3:
            return self
        ctmp = np.array( [*self.components,0] )
        ltmp = self._lattice.as3D()
        return Vector(ctmp, ltmp)
    
    @staticmethod
    def dot(a,b):
        """ 
        Dot product between vectors a and b. 
        
        Method operates independent of the lattice in which the
        vectors are defined. Vectors on different lattices will
        be cast into the standard cartesian basis before performing
        the calculation.
        
        Parameters
        ----------
        a, b : Vector
            The vectors being dotted together.
        """
        if not (isinstance(a,Vector) and isinstance(b,Vector)):
            msg = "Vector.dot(a,b) not defined for arguments {} and {}"
            raise(TypeError(msg.format(type(a).__name__,type(b).__name__)))
        if a._lattice == b._lattice:
            atmp = a._cartesian_components
            btmp = b._cartesian_components
            return atmp @ btmp
        elif a._lattice.isReciprocal(b._lattice):
            atmp = a._components
            btmp = b._components
            return atmp @ btmp
        else:
            msg = "Incompatible lattice; unable to dot {} and {}"
            raise(ValueError(msg.format(a,b)))
    
    def _instance_dot(self,b):
        return Vector.dot(self,b)
    
    @staticmethod
    def castToLattice(a,lattice):
        """ 
        Return the coordinates of the vector on the given lattice.
        
        A result is only returned when a.dim <= lattice.dim;
        otherwise, a ValueError is raised.
        
        When a is a 2D vector, and lattice is 3D, the third
        component of a is taken to be 0, and the result is
        returned as a 3D vector.
        
        Parameters
        ----------
        a : Vector
            The vector to change the basis of.
        lattice : Lattice
            The new basis in which to express the vector.
        
        Returns
        -------
        b : Vector
            Vector a expressed on lattice.
        """
        # Screen input parameters
        typemsg = "Parameter {} of Vector.castToLattice(a,lattice) must be of type {}"
        if not isinstance(a,Vector):
            raise(TypeError(typemsg.format("a","Vector")))
        if not isinstance(lattice,Lattice):
            raise(TypeError(typemsg.format("lattice","Lattice")))
        if not a.dim <= lattice.dim:
            raise(ValueError("A vector can only be cast to a lattice of equal or larger dim."))
        assert( (a.dim in (2,3)) and (lattice.dim in (2,3)) )
        
        # obtain components of a on the standard cartesian lattice
        atmp = a._cartesian_components
        if a.dim == 2 and lattice.dim == 3:
            atmp = np.array([*atmp,0.0]) # add zero third component
        
        # When the rows of lattice.basis are components of the lattice
        # basis vectors on the standard cartesian basis, then the 
        # rows of the inverse of lattice.basis are components of the
        # standard basis vectors expressed on the lattice.
        tmatr = np.linalg.inv( lattice.basis )
        # Given this, components of the vector on the lattice are given
        # by the vector cartesian components right-multiplied by the 
        # inverse of lattice.basis.
        ctmp = atmp @ tmatr
        c = Vector(ctmp,lattice.copy())
        return c
    
    def _instance_castToLattice(self,lattice):
        """ 
        Return the coordinates of the vector on the given lattice. 
        
        Class instance version, allowing for both static and instance
        access to the method. Assigned to the name instance.castToLattice()
        on instantiation.
        """
        return Vector.castToLattice(self,lattice)
    
    ## Hooks
    
    def __abs__(self):
        """ Returns the length of the vector. """
        return self.magnitude
    
    def __add__(self,other):
        """ 
        Add two vectors.
        
        Vectors must be expressed on the same lattice.
        """
        if not isinstance(other,Vector):
            msg = "Unable to add objects of type 'Vector' and '{}'"
            raise(TypeError(msg.format(type(other).__name__)))
        if not self._lattice == other._lattice:
            msg = "Incompatible lattice; unable to add {} and {}"
            raise(ValueError(msg.format(self,other)))
        ctmp = self._components + other._components
        return Vector(ctmp,self.lattice)
    
    def __sub__(self,other):
        """ 
        Subtract two vectors.
        
        Vectors must be expressed on the same lattice.
        """
        if not isinstance(other,Vector):
            msg = "Unable to subtract objects of type 'Vector' and '{}'"
            raise(TypeError(msg.format(type(other).__name__)))
        if not self._lattice == other._lattice:
            msg = "Incompatible lattice; unable to subtract {} and {}"
            raise(ValueError(msg.format(self,other)))
        ctmp = self._components - other._components
        return Vector(ctmp,self.lattice)
    
    def __mul__(self,other):
        if isinstance(other,Vector):
            return self.dot(other)
        else:
            try:
                other = int(other)
            except:
                try:
                    other = float(other)
                except:
                    msg = "Operation * unavailable for types 'Vector' and '{}'."
                    raise(msg.format(type(other).__name__))
            ctmp = other * self._components
            ltmp = self._lattice.copy()
            return Vector(ctmp, ltmp)
    
    def __rmul__(self,other):
        return self * other
    
    def __len__(self):
        return self._dim
    
    def __repr__(self):
        s = "< Vector {} on {} >"
        compstr = str(self._components)
        latstr = self._lattice.__repr__()
        return s.format(compstr, latstr)
    
    def __array__(self, *args, **kwargs):
        """ When cast to an array, a vector returns its cartesian representation. """
        return self.cartesian

class VectorList(object):
    """ Class for storing large sets of Vectors on the same lattice. """
    
    def __init__(self, veclist, palloc=False, nvec=0, dim=1):
        """
        Initialize the vector list.
        
        Parameters
        ----------
        veclist : iterable of Vectors
            The vectors being combined in the VectorList.
            All Vectors in veclist must have the same lattice.
        palloc : Boolean, default=False
            True if nvec and dim are given for pre-allocation.
        nvec : int, required if palloc==True
            The number of vectors expected in the set. Allows for
            pre-allocation of memory when full vector set is known.
            Must be positive, non-zero.
        dim : int, optional, either (2,3)
            The dimensionality of the vectors. Allows for pre-allocation
            of memory.
        
        Raises
        ------
        ValueError : 
            If any vector in veclist does not match the lattice of
            the others in the list.
        ValueError : 
            If nvec or dim are not acceptable values.
        ValueError : 
            if given value for dim does not match the input vectors.
        """
        init_components = False
        self._nvec = 0
        set_dim = False
        self._dim = 1
        set_lat = False
        if palloc:
            nvec = int(nvec)
            if nvec <= 0:
                raise(ValueError("nvec must be positive, nonzero; gave {}".format(nvec)))
            dim = int(dim)
            if not (dim == 2 or dim == 3):
                raise(ValueError("dim must be either 2 or 3; gave {}".format(dim)))
            self._dim = dim
            set_dim = True
            self._components = np.zeros((nvec,dim))
            init_components = True
        msg = "Incompatible Vector {} {}"
        for v in veclist:
            # type check
            if not isinstance(v,Vector):
                raise(TypeError("Instance of {} input with veclist".format(type(v).__name__)))
            # dim check
            if not set_dim:
                self._dim == v.dim
                set_dim = True
            # lattice check
            if not set_lat:
                self._lattice = v.lattice
                set_lat = True
            # initialize components array (if not already)
            if not init_components:
                self._components = np.zeros((1,self._dim))
                init_components = True
            self.append(v)
    
    def append(v):
        """ Append vector at end of VectorList. """
        if not v.lattice == self._lattice:
            raise(ValueError(msg.format("with lattice",v.lattice)))
        if len(self._components) > self._nvec:
            self._components[self._nvec,:] = v.components
        else:
            newrow = np.array([[*v.components]])
            np.append( self._components, newrow, axis = 0 )
        self._nvec += 1
    
    
