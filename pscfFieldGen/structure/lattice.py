# Imports
from pscfFieldGen.util.tracing import debug
from copy import deepcopy
import numpy as np
import math
import weakref

## Debug Functions

## Helper Functions
_EXACT_COSD = {
        0.0 : +1.0,   60.0 : +0.5,   90.0 : 0.0,  120.0 : -0.5,
      180.0 : -1.0,  240.0 : -0.5,  270.0 : 0.0,  300.0 : +0.5
}
_EXACT_ACOSD = {
       +1.0 : 0.0, +0.5 : 60.0, 0.0 : 90.0, -0.5 : 120.0
}
def cosd(x):
    """Return the cosine of x (measured in degrees). """
    rv = _EXACT_COSD.get(x % 360.0, None)
    if rv is None:
        rv = math.cos(math.radians(x))
    return rv

def sind(x):
    """Return the sine of x (measured in degrees). """
    return math.sin(math.radians(x))

def acosd(x):
    """Return (in degrees) the arccosine of x. """
    rv = _EXACT_ACOSD.get(x, None)
    if rv is None:
        rv = math.degrees(math.acos(x))
    return rv
    
def asind(x):
    """Return (in degrees) the arcsine of x. """
    return math.degrees(math.asin(x))

class CoreLattice(object):
    """ 
    Object representing a crystallographic basis vector lattice.
    """
    
    def __init__(   self, 
                    dim, 
                    basis ):
        """
        Generate a lattice object.
        
        Params:
        -------
        dim: int in (2,3)
            Number of dimensions in the lattice.
        basis : array-likej
            Matrix (size: dim-by-dim) representation of basis vectors.
            Coordinates of each vector are defined relative to the
            standard basis (mutually orthogonal unit vectors).
            Row i ( elements in basis[i,:] ) corresponds to lattice basis
            vector i.
        """
        _fn_ = "lattice.__init__"
        if dim not in (2,3):
            raise(ValueError("Lattice must be 2D or 3D"))
        debug(_fn_,"received dim={}",dim)
        debug(_fn_,"received basis={}",basis)
        self._dim = dim;
        self._update_basis(basis)
        if self._dim == 2:
            self._param_keys = ["a","b","gamma"]
        else:
            self._param_keys = ["a","b","c","alpha","beta","gamma"]
        debug(_fn_,"set self._param_keys={}",self._param_keys)
    
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
        debug("lattice.latticeFromParameters","received dim={}, params={}",dim,kwargs)
        basis = cls.basisFromParameters(dim, **kwargs)
        debug("lattice.latticeFromParameters","received basis={}",basis)
        return cls(dim, basis)
            
    @classmethod
    def basisFromParameters(cls, dim, 
                            a=None, b=None, c=None,
                            alpha=None, beta=None, gamma=None):
        """
            Return a set of basis vectors in standard orientation.
            
            Params:
            -------
            dim: int, in set {2, 3}
                The dimensionality of the resulting lattice
            a:  float
                Magnitude of first basis vector.
            b:  float
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
        _fn_ = "lattice.basisFromParameters"
        debug(_fn_,"received dim={}",dim)
        debug(_fn_,"received a={}",a)
        debug(_fn_,"received b={}",b)
        debug(_fn_,"received c={}",c)
        debug(_fn_,"received alpha={}",alpha)
        debug(_fn_,"received beta={}",beta)
        debug(_fn_,"received gamma={}",gamma)
        # check inputs
        if (a is None) or (b is None) or (gamma is None):
            raise(ValueError("Required lattice parameter is missing."))
        a = float(a)
        b = float(b)
        gamma = float(gamma)
        
        # initialize basis
        basis = np.zeros((dim,dim))
        debug(_fn_,"Basis initialized to: {}",basis)
        
        # Complete common calculations
        cg = cosd(gamma)
        debug(_fn_,"cos(gamma)={}",cg)
        sg = sind(gamma)
        debug(_fn_,"sin(gamma)={}",sg)
        basis[0,0] = a
        basis[1,0] = b * cg
        basis[1,1] = b * sg
        debug(_fn_,"Basis after common setup: {}",basis)
        
        # Additional 3D calculations
        if dim == 3:
            if (c is None) or (alpha is None) or (beta is None):
                raise(ValueError("Required lattice parameter is missing."))
            c = float(c)
            alpha = float(alpha)
            beta = float(beta)
            
            # Trig Functions
            ca = cosd(alpha)
            debug(_fn_,"cos(alpha)={}",ca)
            cb = cosd(beta)
            debug(_fn_,"cos(beta)={}",cb)
            sb = sind(beta)
            debug(_fn_,"sin(beta)={}",sb)
            sg = sind(gamma)
            debug(_fn_,"sin(gamma)={}",sg)
            
            # Secondary Values
            unit_vol = np.sqrt( 1.0 + 2.0*ca*cb*cg - ca**2 - cb**2 - cg**2 )
            debug(_fn_,"unit_vol={}",unit_vol)
            cr = sg/(c*unit_vol) # length of reciprocal lattice vector *c*
            debug(_fn_,"c*={}",cr)
            car = (cb*cg - ca)/(sb*sg) # cosine of reciprocal angle *alpha*
            debug(_fn_,"cos(alpha*)={}",car)
            sar = math.sqrt(1.0 - car*car) # sine of reciprocal angle *alpha*
            debug(_fn_,"sin(alpha*)={}",sar)
            
            # Finish Updating Basis
            basis[2,0] = c * cb
            basis[2,1] = -car / sar / cr
            basis[2,2] = 1.0 / cr
            debug(_fn_,"Basis after 3D setup: {}",basis)
            
        return basis
    
    def copy(self):
        return deepcopy(self)
    
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
        return self.parameterList
    
    @property
    def parameterList(self):
        """
        Lattice parameters as list of lengths and angles.
        3D: [a, b, c, alpha, beta, gamma]
        2D: [a, b, gamma]
        """
        return np.array( self._param_list )
    
    @parameterList.setter
    def parameterList(self, newvals):
        """
        Lattice parameters as list of lengths and angles.
        2D: [a, b, gamma]
        3D: [a, b, c, alpha, beta, gamma]
        """
        newvals = np.array(newvals)
        msg = "Gave {} parameters for {}D lattice requiring {}"
        if self._dim == 2 and not len(newvals) == 3:
            raise( ValueError( msg.format( len(newVals), 2, 3 ) ) )
        if self._dim == 3 and not len(newvals) == 6:
            raise( ValueError( msg.format( len(newVals), 3, 6 ) ) )
        newParam = dict(zip(self._param_keys,newvals))
        newBasis = Lattice.basisFromParameters(self._dim, **newParam)
        self._update_basis(newBasis)
        
    @property
    def parameterDict(self):
        """
        Lattice parameters as dict of name:value mappings.
        2D keys: a, b, gamma
        3D keys: a, b, c, alpha, beta, gamma
        """
        return dict(zip(self._param_keys,self._param_list))
    
    @parameterDict.setter
    def parameterDict(self, newParams):
        try:
            newParams = dict(newParams)
        except Exception as err:
            msg = "Unable to cast {} to dict for parameterDict update."
            raise Exception(msg.format(newParams)) from err
        newBasis = Lattice.basisFromParameters(self._dim, **newParams)
        self._update_basis(newBasis)
    
    def updateParameters(self, **paramArgs):
        """ 
        Update specific lattice parameters.
        
        Any lattice parameters not specified will not be updated.
        
        Parameters
        ----------
        a : real, optional
            The length of the first basis vector.
        b : real, optional
            The length of the second basis vector.
        c : real, optional
            The length of the third basis vector.
        alpha : real, optional
            The angle (in degrees) between the basis vectors "b" and "c".
        beta : real, optional
            The angle (in degrees) between the basis vectors "a" and "c".
        gamma : real, optional
            The angle (in degrees) between the basis vectors "a" and "b".
        """
        d = self.parameterDict
        d.update(paramArgs)
        self.parameterDict = d
    
    @property
    def latticeVectors(self):
        """ Return lattice vectors as numpy.ndarray """
        return self.basis
    
    @property
    def basis(self):
        return np.array(self._basis)
    
    @basis.setter
    def basis(self, newBasis):
        self._update_basis(newBasis)
    
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
        if not self.dim == other.dim:
            return False
        return np.allclose(self._reciprocal_basis, other._basis)
    
    def isSimilar(self,other):
        """ Return True if Lattices are identical within a rotation. 
        
        'Similar' lattices are those which have the same lattice parameters,
        but are not necessarily in the standard orientation. In such a case,
        The basis vectors of the two lattices differ only by a rotation in
        space.
        """
        if not isinstance(other, Lattice):
            msg = "Cannot determine similarity of {} with Lattice"
            raise(TypeError(msg.format(type(other).__name__)))
        if not np.allclose(self._param_list, other._param_list):
            return False
        return np.allclose(self._metric_tensor, other._metric_tensor)
    
    ## "Private" internal methods
    
    def _update_basis(self, newBasis):
        """ 
        Internal method to update basis and pre-computed
        values.
        """
        _fn_ = "lattice._update_basis"
        dim = self._dim
        debug(_fn_,"stored local dim={}",dim)
        debug(_fn_,"received newBasis={}",newBasis)
        basis = np.array(newBasis)
        debug(_fn_,"stored local basis={}",basis)
        if not basis.shape == (dim, dim):
            raise(ValueError("Gave basis shape {} for dim {}".format(basis.shape,dim)))
        
        self._basis = basis;
        
        self._volume = np.linalg.det(self._basis)
        debug(_fn_,"calculated volume={}",self._volume)
        if abs(self._volume) < 1.0e-8:
            raise(ValueError("Basis vectors are degenerate."))
        if self._volume < 0.0:
            raise(ValueError("Basis is not right-handed."))
        
        self._metric_tensor = np.matmul(self._basis, self._basis.T)
        debug(_fn_,"calculated metric tensor={}",self._metric_tensor)
        self._update_parameters()
        
        reciprocalMetricTensor = np.linalg.inv(self._metric_tensor)
        debug(_fn_,"calculated reciprocal metric tensor={}",reciprocalMetricTensor)
        self._reciprocal_basis =  np.matmul(reciprocalMetricTensor, self._basis)
        debug(_fn_,"calculated reciprocal basis={}",self._reciprocal_basis)
        
    def _update_parameters(self):
        """ 
        Internal method to re-calculate lattice parameters
        after a basis update.
        """
        aMag = np.sqrt( self._metric_tensor[0,0] )
        bMag = np.sqrt( self._metric_tensor[1,1] )
        adotb = self._metric_tensor[0,1]
        gamma = acosd( adotb / (aMag*bMag) )
        if self.dim == 2:
            self._param_list =  np.array([aMag, bMag, gamma])
        elif self.dim == 3:
            cMag = np.sqrt( self._metric_tensor[2,2] )
            adotc = self._metric_tensor[0,2]
            bdotc = self._metric_tensor[1,2]
            alpha = acosd( bdotc / (bMag*cMag) )
            beta = acosd( adotc / (aMag*cMag) )
            self._param_list = np.array([aMag, bMag, cMag, alpha, beta, gamma])
    
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
    
    def __str__(self):
        return self.__repr__()
    
    def __array__(self, *args, **kwargs):
        """ returns an ndarray. Each row is a lattice vector. """
        return np.array(self.latticeVectors, *args, *kwargs)

class SharedLattice(CoreLattice):
    """ 
    Sub-class of lattice intended to force sharing of underlying lattice data
    """
    def __init__(self, dim, basis, _reciprocal=None):
        self._startup_flag = True
        super().__init__(dim,basis)
        if _reciprocal is None:
            self._recip = SharedLattice(dim, self._reciprocal_basis, self)
            self._origin = True
        else:
            self._recip = weakref.ref(_reciprocal)
            self._origin = False
        self._startup_flag = False
            
    def copy(self):
        return self
    
    def reciprocalLattice(self):
        if self._origin:
            return self._recip
        else:
            r = self._recip()
            if r is not None:
                return r
            else:
                # original origin was garbage collected. Make this new origin.
                self._origin = True
                self._recip = SharedLattice(dim, self._reciprocal_basis, self)
                return self._recip
    
    def isReciprocal(self,other):
        if self._origin:
            if self._recip is other:
                return True
            else:
                return super().isReciprocal(other)
        else:
            r = self._recip()
            if r is other:
                return True
            else:
                return super().isReciprocal(other)
    
    def _update_basis(self, newBasis, _secondCall=False):
        """ 
        Internal method to update basis and pre-computed values.
        
        Argument '_secondCall' is a private argument meant for
        use in tracking recursive calls for updating the referenced
        reciprocal lattices. Non-recursive calls (calls from outside
        this method) should not use this argument, and allow the default.
        """
        super()._update_basis(newBasis)
        if not self._startup_flag: # Do not cascade during instantiation
            if not _secondCall: # avoid infinite recursion
                recip = self.reciprocalLattice() # get strong reference if weakly referenced
                recip._update_basis(self._reciprocal_basis, _secondCall=True)

Lattice = SharedLattice # Make Shared Lattice the favored lattice implementation

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
        debug("Vector.__init__","received lattice {}",lattice)
        if not isinstance(lattice,Lattice):
            raise(TypeError("A Vector's lattice must be an instance of Lattice"))
        if not len(components) == lattice.dim:
            raise(ValueError("Components must match lattice dimension."))
        
        # store input values
        self._components = components
        self._lattice = lattice.copy()
        self._dim = self._lattice.dim
        
        # Compute secondary properties
        self._process_updated_components()
        
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
        comp = np.array(comp)
        if not len(comp) == self._dim:
            raise(ValueError("Length of components {} must match dim of vector".format(comp)))
        self._components = comp
        self._process_updated_components()
    
    @property
    def lattice(self):
        return self._lattice
    
    @lattice.setter
    def lattice(self, newLattice):
        if not isinstance(newLattice,Lattice):
            raise(TypeError("A Vector's lattice must be an instance of Lattice"))
        if not self._dim == newLattice.dim: 
            raise(ValueError("A Vector's new lattice must be same dimension."))
        self._lattice = newLattice.copy()
        self._process_updated_lattice()
    
    def updateLattice(  self,
                        basis = None,
                        paramList = None,
                        paramDict = None,
                        **paramArgs ):
        """ 
        Change the lattice on which the vector is defined.
        
        The dimensionality of the vector cannot be changed.
        
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
        if basis is not None:
            self._lattice.basis = basis
        elif paramList is not None:
            self._lattice.parameterList = paramList
        elif paramDict is not None:
            self._lattice.parameterDict = paramDict
        else:
            self._lattice.updateParameters(**paramArgs)
        self._process_updated_lattice()
    
    @property
    def cartesian(self):
        """ The vector components referenced to the standard unit cartesian lattice. """
        return np.array(self._cartesian_components)
    
    ## Operations
    
    def copy(self):
        """ A copy of the Vector instance. """
        return deepcopy(self)
    
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
        if a._lattice.isReciprocal(b._lattice):
            atmp = a._components
            btmp = b._components
            return atmp @ btmp
        elif a._lattice is b._lattice:
            atmp = a._cartesian_components
            btmp = b._cartesian_components
            return atmp @ btmp
        elif a._lattice.isSimilar(b._lattice):
            mt = a._lattice.metricTensor
            atmp = a._components
            btmp = b._components
            return atmp @ mt @ btmp
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
        c = Vector(ctmp,lattice)
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
            return NotImplemented
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
            return NotImplemented
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
                other = float(other)
            except:
                return NotImplemented
            ctmp = other * self._components
            ltmp = self._lattice
            return Vector(ctmp, ltmp)
    
    def __rmul__(self,other):
        return self.__mul__(other)
    
    def __len__(self):
        return self._dim
    
    def __getitem__(self, item):
        if isinstance(item, (int,slice)):
            return self._components[item]
        return [self._components[i] for i in item]
    
    def __setitem__(self, item, value):
        if isinstance(item,int):
            self._components[item] = value
        elif isinstance(item, slice):
            raise(ValueError("Cannot set slice for Vectors"))
        else:
            for i in item:
                if isinstance(i,slice):
                    raise(ValueError("Cannot set slice for Vectors"))
                self._components[i] = value
        self._process_updated_components()
        
    def __eq__(self,other):
        if not isinstance(other,Vector):
            return NotImplemented
        if not self._lattice == other._lattice:
            return False
        return np.allclose(self._components, other._components)
    
    def __repr__(self):
        s = "< Vector {} on {} >"
        compstr = str(self._components)
        latstr = self._lattice.__repr__()
        return s.format(compstr, latstr)
    
    def __array__(self, *args, **kwargs):
        """ When cast to an array, a vector returns its cartesian representation. """
        return self.cartesian
    
    ## Private methods
    def _process_updated_components(self):
        self._cartesian_components = self._components @ self._lattice.basis
        self._magnitude = np.linalg.norm(self._cartesian_components)
    
    def _process_updated_lattice(self):
        self._dim = self._lattice.dim
        self._cartesian_components = self._components @ self._lattice.basis
        self._magnitude = np.linalg.norm(self._cartesian_components)
        
