import math
import numpy as np

from pscfFieldGen.structure.lattice import Lattice, Vector
from pscfFieldGen.util.tracing import debug


def getKgrid(ngrid):
    """
    Determine the reciprocal space grid counts from a real-space grid.
    
    Parameters
    ----------
    ngrid : array-like
        Number of real-space grid points along each lattice basis vector.
    
    Returns
    -------
    kgrid : numpy.ndarray
        Number of grid points in each dimension for reciprocal wave-vector
        grid.
    """
    ngrid = np.array(ngrid)
    kgrid = np.array(ngrid)
    kgrid[0] = math.floor(ngrid[0]/2) + 1
    return kgrid

def getNgrid(kgrid):
    """
    Determine the real space grid counts from a reciprocal-space grid.
    
    Parameters
    ----------
    kgrid : numpy.ndarray
        Number of grid points in each dimension for reciprocal wave-vector
        grid.
    
    Returns
    -------
    ngrid : array-like
        Number of real-space grid points along each lattice basis vector.
    """
    kgrid = np.array(kgrid)
    ngrid = np.array(kgrid)
    ngrid[0] = (kgrid[0] - 1) * 2
    return kgrid

def rgridIterator(ngrid, fromKgrid=False):
    """
    Return a generator which outputs real-space grid indices.
    
    Returned values iterate in axis-order, with low-dimension
    axes (ngrid[0]) updating most rapidly, and high-dimension
    updating more slowly. 
    (Following convention of PSCF coordinate grid field file
    format).
    
    Parameters
    ----------
    ngrid : array-like
        Number of grid points along each axis.
    fromKgrid : boolean, optional
        If False (default), input ngrid indicates real-space grid points.
        If True, input ngrid represents the reciprocal-space grid points.
    
    Returns
    -------
    rgridIter : generator
        Generator returning grid point indices as one-dimensional 
        numpy.ndarray. Components are indices of grid points
        (integer < ngrid[i]).
        *** Note: if grid is 1-Dimensional, return values are integer
        indices, rather than arrays.
    """
    ngrid = np.array(ngrid)
    if fromKgrid:
        ngrid = getNgrid(ngrid)
    dim = len(ngrid)
    if dim == 1:
        for i in range(ngrid[0]):
            yield i
    elif dim == 2:
        for j in range(ngrid[1]):
            for i in range(ngrid[0]):
                yield np.array([i,j])
    elif dim == 3:
        for k in range(ngrid[2]):
            for j in range(ngrid[1]):
                for i in range(ngrid[0]):
                    yield np.array([i,j,k])
    else:
        raise(ValueError("rgridIterator only valid for ngrid of 1-,2-,and 3-dimensional grids"))

def positionIterator(ngrid, fromKgrid=False):
    """
    Return a generator which outputs real-space position vector components
    
    Parameters
    ----------
    ngrid : array-like
        Number of grid points along each axis.
    fromKgrid : boolean, optional
        If False (default), input ngrid indicates real-space grid points.
        If True, input ngrid represents the reciprocal-space grid points.
    
    Returns
    -------
    positionIter : generator
        Generator returning grid point indices as one-dimensional 
        numpy.ndarray. Components are fractional coordinates, with
        0.0 <= value < 1.0
        *** Note: if grid is 1-Dimensional, return values are integer
        indices, rather than arrays.
    """
    ngrid = np.array(ngrid)
    if fromKgrid:
        ngrid = getNgrid(ngrid)
    dim = len(ngrid)
    if dim == 1:
        for i in range(ngrid[0]):
            yield i/ngrid[0]
    elif dim == 2:
        for j in range(ngrid[1]):
            for i in range(ngrid[0]):
                yield np.array([i,j]) / ngrid
    elif dim == 3:
        for k in range(ngrid[2]):
            for j in range(ngrid[1]):
                for i in range(ngrid[0]):
                    yield np.array([i,j,k]) / ngrid
    else:
        raise(ValueError("positionIterator only valid for ngrid of 1-,2-,and 3-dimensional grids"))

def kgridIterator(ngrid, fromKgrid=False):
    """
    Return a generator which outputs wavevector components
    
    Parameters
    ----------
    ngrid : array-like
        Number of grid points along each axis.
    fromKgrid : boolean, optional
        If False (default), input ngrid indicates real-space grid points.
        If True, input ngrid represents the reciprocal-space grid points.
    
    Returns
    -------
    positionIter : generator
        Generator returning grid point indices as one-dimensional 
        numpy.ndarray.
        *** Note: if grid is 1-Dimensional, return values are integer
        indices, rather than arrays.
    """
    _fn_ = "structure.grids.kgridIterator"
    if fromKgrid:
        kgrid = np.array(ngrid)
        ngrid = getNgrid(kgrid)
    else:
        ngrid = np.array(ngrid)
        kgrid = getKgrid(ngrid)
    debug(_fn_,"ngrid = {}".format(ngrid))
    debug(_fn_,"kgrid = {}".format(kgrid))
    dim = len(ngrid)
    debug(_fn_,"dim = {}".format(dim))
    if dim == 1:
        for i in range(kgrid[0]):
            debug(_fn_,"i={}".format(i))
            yield i
    elif dim == 2:
        for i in range(kgrid[0]):
            for j in range(kgrid[1]):
                debug(_fn_,"i={}, j={}".format(i,j))
                yield miller_to_brillouin( np.array([i,j]), ngrid )
    elif dim == 3:
        for i in range(kgrid[0]):
            for j in range(kgrid[1]):
                for k in range(kgrid[2]):
                    debug(_fn_,"i={}, j={}, k={}".format(i,j,k))
                    yield miller_to_brillouin( np.array([i,j,k]), ngrid )
    else:
        raise(ValueError("positionIterator only valid for ngrid of 1-,2-,and 3-dimensional grids"))

def miller_to_brillouin(G, ngrid, fromKgrid=False):
    """
    Shift miller-indexed wavevector into first Brillouin zone.
    """
    _fn_ = "structure.grids.miller_to_brillouin"
    ngrid = np.array(ngrid)
    if fromKgrid:
        ngrid = getNgrid(ngrid)
    debug(_fn_, "ngrid = {}".format(ngrid))
    G = np.array(G)
    debug(_fn_, "miller = {}".format(G))
    out= np.zeros_like(G)
    dim = len(G)
    dshift = dim - 1
    out[0] = G[0]
    for i in [1,2]:
        if dshift >= i:
            if G[i] > ngrid[i]/2:
                out[i] = G[i] - ngrid[i]
            else:
                out[i] = G[i]
    debug(_fn_, "brillouin = {}".format(out))
    return out

class IterableWavevector(Vector):
    """
    Vector specialized to be iterable over wavevector grids.
    
    For a given grid and lattice, each instance will update
    in the style of an iterator. Unlike a typical iterator,
    IterableWavevector does not return separate instances of
    data with each call to next(instance). Rather, iteration
    requests trigger mutation of the instance itself, and a
    reference to the same instance is returned.
    """
    
    def __init__(self, ngrid, lattice):
        """
        Initialize an IterableWavevector.
        
        Parameters
        ----------
        ngrid : array-like
            The real-space grid discretization.
        lattice : Lattice
            The real-space lattice.
        """
        self._generator = kgridIterator(ngrid)
        if not isinstance(lattice, Lattice):
            msg = "Argument 'lattice' must be of type Lattice, not {}."
            raise(TypeError(msg.format(type(lattice).__name__)))
        lat = lattice.reciprocalLattice()
        super().__init__(np.zeros(lat.dim), lat)
    
    def __next__(self):
        self.components = next(self._generator)
        return self
    
    def __iter__(self):
        return self

def getKgridCount(ngrid, fromKgrid=False):
    """
    Return the total number of points on the wavevector grid.
    
    Parameters
    ----------
    ngrid : array-like
        The number of grid points along each axis.
    fromKgrid : boolean, optional
        If False (default), input ngrid indicates real-space grid points.
        If True, input ngrid represents the reciprocal-space grid points.
    
    Returns
    -------
    kgridCount : int
        The total number of kgrid points.
    """
    ngrid = np.array(ngrid)
    if fromKgrid:
        kgrid = ngrid
    else:
        kgrid = getKgrid(ngrid)
    return np.prod(kgrid)
    
    

