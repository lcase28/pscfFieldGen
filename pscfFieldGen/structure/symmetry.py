""" Module defining class to hold symmetry operation data """

from pscfFieldGen.structure.core import POSITION_TOLERANCE
from pscfFieldGen.util.stringTools import str_to_num, wordsGenerator

from copy import deepcopy
import enum
import numpy as np
import pathlib
import re
import sympy as sym

class GroupFileNotFoundError(Exception):
    def __init__(self,groupname, message=None):
        self.groupname = groupname
        if message is None:
            message = "No group with name {} found.".format(groupname)
        self.message = message
    
class GroupTooLargeError(Exception):
    """ Exception raised when a group exceeds its allowed size.
    
    Attributes
    ----------
    maxsize : int
        The maximum number of members in the group.
    source : 
        The object raising the exception.
    message : str
        Explanation of the error.
    """
    
    def __init__(self, maxsize, source, message=None):
        if message is None:
            message = "{} exceeded allowed size, {}".format(source, maxsize)
        self.maxsize = maxsize
        self.source = source
        self.message = message
        super().__init__(self.message)
            
class GeneralPosition(object):
    def __init__(self, dim, source=None, syms=None):
        self._dim = dim
        if source is None:
            x, y = sym.symbols('x y')
            symlist = [x, y]
            if dim == 3:
                symlist.append(sym.symbols('z'))
            self._variables = symlist
            symlist = np.array(symlist) * np.ones(dim)
            self._position = sym.Array([*symlist])
        else:
            self._variables = syms
            self._position = sym.simplify(source)
            if not len(syms) == dim or not len(source) == dim:
                raise(ValueError("Dimensionality mismatch"))
        self._evaluator = sym.lambdify(self._variables, self._position)
        self._capTranslation()
        self._evaluator = sym.lambdify(self._variables, self._position)
    
    def __array__(self):
        return np.array(self._position)
    
    def __eq__(self,other):
        if isinstance(other,GeneralPosition):
            symmap = dict(zip(other.symbol_list,self.symbol_list))
            temp = other._position.subs(symmap)
            return self._position == temp
        else:
            return NotImplemented
    
    @property
    def symbol_list(self):
        return self._variables
    
    @property
    def dim(self):
        return self._dim
    
    def evaluate(self, base):
        b = np.array(base)
        return np.array(self._evaluator(*b))
    
    def __str__(self):
        return str(self._position)
    
    def _capTranslation(self):
        trans = self.evaluate(np.zeros(self.dim))
        newPos = []
        for i in range(self.dim):
            orig = self._position[i]
            if trans[i] >= 1:
                newPos.append(orig - 1)
            elif trans[i] < 0:
                newPos.append(orig + 1)
            else:
                newPos.append(orig)
        self._position = sym.Array(newPos)
        
class SymmetryOperation(object):
    def __init__(self, dim, matrix):
        """
        Parameters
        ----------
        dim : int
            2 if the operations is meant for plane-groups.
            3 if the operation is meand for space-groups.
        matrix : numpy.ndarray or list or tuple
            If type numpy.ndarray, this should be a 4x4 matrix
            representing the symmetry operation.
            If a list or tuple, should contain two items:
                0. pointOperation : 2D array-like
                    The point symmetry operation in matrix form.
                    Should be shape (dim,dim)
                1. translation : 1D array-like
                    The translation component of the symmetry
                    operation. Should have len == dim.
        """
        self._dim = dim
        if isinstance(matrix, np.ndarray):
            if not matrix.shape == (4,4):
                msg = "Symmetry Operations matrices must have shape (4,4), not {}."
                raise(ValueError(msg.format(matrix.shape)))
            self._matrix = np.array(matrix)
            self._point = np.array(matrix[0:dim,0:dim])
            self._trans = np.array(matrix[0:dim,3])
        elif isinstance(matrix,(list,tuple)):
            pointOperation = matrix[0]
            translation = matrix[1]
            pointOp = np.array(pointOperation)
            if not pointOp.shape == (dim,dim):
                msg = "pointOperation {} has improper shape for {}D SymmetryOperation."
                raise(ValueError(msg.format(pointOp,dim)))
            self._point = pointOp
            trans = np.array(translation)
            if not ( len(trans) == dim and trans.shape == (dim,) ):
                msg = "Translation {} has improper shape for {}D SymmetryOperation."
                raise(ValueError(msg.format(trans,dim)))
            self._trans = trans
            self._matrix = np.zeros((4,4))
            self._matrix[0:dim,0:dim] = self._point
            self._matrix[0:dim,3] = self._trans
        self._capTranslation()
    
    @property
    def dim(self):
        return self._dim
    
    def asMatrix(self):
        """ The full 4x4 augmented matrix of the operation. """
        return np.array(self._matrix)
    
    @property
    def pointOperator(self):
        """ The {dim}x{dim} point operation component. """
        return np.array(self._point)
    
    @property
    def translationVector(self):
        """ The {dim}x1 translation vector of the operation. """
        return np.array(self._trans)
    
    def __mul__(self,other):
        return self.__matmul__(other)
    
    def __rmul__(self,other):
        return self.__rmatmul__(other)
        
    def __matmul__(self, other):
        if isinstance(other,SymmetryOperation):
            if other._dim == self._dim:
                pt = self._point @ other._point
                tr = (self._point @ other._trans) + self._trans
                return SymmetryOperation(self._dim, (pt,tr))
            else:
                raise(ValueError("Cannot multiply SymmetryOperations of differing dimensionality"))
        elif isinstance(other, GeneralPosition):
            if other.dim == self._dim:
                gep = np.array(other)
                result = (self._point @ gep) + self._trans
                return GeneralPosition(self._dim, result, other.symbol_list)
            else:
                raise(ValueError("Cannot multiply SymmetryOperation and GeneralPosition of different dimensions."))
        elif isinstance(other, np.ndarray):
            if len(other) == self._dim:
                pt = np.array(other)
                res = (self._point @ pt) + self._trans
                return res
            else:
                raise(ValueError("Dimension mismatch"))
        elif isinstance(other, Vector):
            comp = (self._point @ other.components) + self._trans
            newVec = other.copy()
            newVec.components = comp
            return newVec
        else:
            return NotImplemented
    
    def __rmatmul__(self, other):
        if isinstance(other,SymmetryOperation):
            return other.__matmul__(self)
        elif isinstance(other, np.ndarray):
            if len(other) == self._dim:
                pt = np.array(other)
                res = pt @ self._point
                return res
            else:
                raise(ValueError("Dimension mismatch"))
        elif isinstance(other, Vector):
            comp = other.components @ self._point
            newVec = other.copy()
            newVec.components = comp
            return newVec
        else:
            return NotImplemented
    
    def __rmod__(self,other):
        """ Phase shift generated by symmetry operation on reciprocal lattice vector. """
        if isinstance(other, np.ndarray):
            if len(other) == self._dim:
                pt = np.array(other)
                res = pt @ self._point
                return res
            else:
                raise(ValueError("Dimension mismatch"))
        elif isinstance(other, Vector):
            comp = other.components @ self._trans
            return comp
        else:
            return NotImplemented
    
    def __str__(self):
        return str(self._matrix)
    
    def __eq__(self, other):
        if isinstance(other, SymmetryOperation):
            return np.allclose(self._matrix, other._matrix)
    
    def _capTranslation(self):
        dim = self._dim
        testupper = np.isclose(np.ones(dim), self._trans)
        atol = 1E-8 * np.ones(dim)
        testlower = np.absolute(self._trans) <= atol
        for i in range(dim):
            if self._trans[i] >= 1 or testupper[i]:
                self._trans[i] -= 1
            elif self._trans[i] < 0 and not testlower[i]:
                self._trans[i] += 1
            elif self._trans[i] < 0 and testlower[i]:
                self._trans[i] = 0.0
        self._matrix[0:dim,3] = self._trans
    
    @property
    def reverse(self):
        return SymmetryOperation(self._dim, np.linalg.inv(self._matrix))
    
    def write(self, outstream):
        """ Write the operation to a filestream.
        
        The operation first prints the point symmetry matrix 
        (as a {dim}-by-{dim} matrix for {dim}=self.dim).
        In the line immediately following, the translation
        component is listed as a row vector. 
        Thus, if a 3D operation can be written as the augmented matrix
        
            m11  m12  m13  v1
            m21  m22  m23  v2
            m31  m32  m33  v3
            0.0  0.0  0.0  1.0
        
        it will be printed as
        
            m11  m12  m13
            m21  m22  m23
            m31  m32  m33
            v1   v2   v3
        
        Similarly, for a 2D operation that can be written as the matrix
        
            m11  m12  v1
            m21  m22  v2
            0.0  0.0  1.0
        
        it will be printed as
        
            m11  m12  
            m21  m22  
            v1   v2
        
        Parameters
        ----------
        outstream : File
            Writable open file object.
        """
        dim = self.dim
        gap = "  "
        outstream.write("\n")
        for i in range(dim):
            s = ""
            for j in range(dim):
                s += "{}{}".format(gap,self._matrix[i,j])
            s += "\n"
            outstream.write(s)
        s = ""
        for i in range(dim):
            s += "{}{}".format(gap,self._matrix[i,3])
        s += "\n"
        outstream.write(s)
        return outstream
    
    @classmethod
    def getUnitTranslations(cls, dim):
        oplist = []
        for i in range(dim):
            op = np.eye(4)
            op[i,3] = 1
            symm = cls(dim,op)
            oplist.append(symm)
        return oplist

class SymmetryGroup(object):
    """
    A set of symmetry operations, generally a closed set.
    
    The intention of the class is to contain a closed set of symmetry
    operations, however this is not strictly enforced. The constructor
    can be set to bypass the closed-group enforcement with an optional
    argument, and the class will not become unstable if the group is
    not closed.
    
    For efficiency purposes, the capacity of a SymmetryGroup can not 
    exceed 500. ( A finite value is required to avoid an infinite loop
    when ensuring the group is closed - 500 is guaranteed sufficient
    for crystallographic purposes, and was assumed generally sufficient
    for any extraneous purposes. )
    """
    
    ## Constructors
    
    def __init__(self, dim, ops, checkClosed=True, maxOperations=500):
        """
        Initialize a new SymmetryGroup instance.
        
        Parameters
        ----------
        dim : int
            2 if representing a set of 2D operations.
            3 if representing a set of 3D operations.
        ops : iterable set of SymmetryOperation objects
            The complete (or initial) set of symmetry operations in
            the group.
        checkClosed : bool (optional)
            When True (default) the constructor will ensure the group
            is closed by checking symmetry operation products and adding
            any new operations found to the group. This option allows 
            initialization of a closed group from just a generator set.
            When False, the constructor will accept the provided operations
            without checking or augmentation.
        maxOperations : int, value > 0 (optional)
            The maximum number of operations that the group is permitted to
            hold. This ensures that, when attempting to build a closed group,
            an error in the initial generator set does not lead to an infinite
            loop checking symmetry operations. Default value is 200, slightly 
            higher than the maximum size of any crystallographic space group.
        
        Raises
        ------
        TypeError :
            When dim or maxOperations can not be cast as an int.
            If any member of ops is not an instance of SymmetryOperation or 
            derived class.
        ValueError : 
            If dim is not a value of 2 or 3.
            If any member of ops has dimensionality different from that
            specified for the group.
            If maxOperations is not a positive, finite value.
        """
        # Check dim input value
        self._dim = int(dim)
        if self._dim not in [2,3]:
            raise(ValueError("SymmetryGroup only defined for 2D or 3D systems. Gave {}.".format(dim)))
        
        self.capacity = maxOperations
        
        # Collect initial symmetry operations
        self._ops = []
        for op in ops:
            self.addOperation(op)
        
        # Flag indicating confidence that the group is closed.
        # If true, the group was found closed, and has not been
        #   modified (using built-in accessors) since.
        # Used to save time in self.isClosed() checks when group
        #   is static.
        self._is_closed = False
        
        # Close group
        if checkClosed:
            self.makeClosed()
    
    @classmethod
    def fromFile(cls, filename, *args, **kwargs):
        """ Instantiate the group from a file.
        
        Parameters
        ----------
        filename : str or pathlib.Path
            The name of the file to read from.
        args, kwargs :
            All other arguments to the class constructor.
        """
        filename = pathlib.Path(filename)
        filename = filename.resolve()
        
        def read_operation(stream, dim):
            # Read operation matrix from stream
            pt = np.zeros((dim,dim))
            # read point operation matrix
            for i in range(dim):
                for j in range(dim):
                    val = next(stream)
                    pt[i,j] = str_to_num(val)
            tr = np.zeros(dim)
            # read translation component of operation
            for i in range(dim):
                val = next(stream)
                tr[i] = str_to_num(val)
            return (pt, tr)
        
        with open(filename) as f:
            words = wordsGenerator(f)
            # read dim
            key = next(words)
            if not key == "dim":
                msg = "Expected key 'dim', got '{}' in symmetry group file {}."
                raise(ValueError(msg.format(key,filename)))
            dim = str_to_num(next(words))
            # read size
            key = next(words)
            if not key == "size":
                msg = "Expected key 'size', got '{}' in symmetry group file {}."
                raise(ValueError(msg.format(key,filename)))
            size = str_to_num(next(words))
            # Read the set of operations
            ops = []
            for i in range(size):
                mat = read_operation(words,dim)
                ops.append(SymmetryOperation(dim,mat))
        
        return cls(dim, ops, *args, **kwargs)
    
    ## Accessors
    
    @property
    def dim(self):
        return self._dim
    
    @property
    def size(self):
        return len(self._ops)
    
    @property
    def capacity(self):
        return self._maxOperations
    
    @capacity.setter
    def capacity(self,maxOperations):
        # Check maxOperations input value
        self._maxOperations = int(maxOperations)
        if self._maxOperations <= 0:
            raise(ValueError("maxOperations must be a positive integer. Gave {}.".format(maxOperations)))
        if self._maxOperations > 500:
            raise(ValueError("maxOperations cannot exceed 500. Gave {}.".format(maxOperations)))
    
    def hasRoom(self):
        """ Return True if more operations can be added without exceeding maximum. """
        return self.size <= self._maxOperations
        
    @property
    def isClosed(self):
        if not self._is_closed:
            self._check_if_closed()
        return self._is_closed
    
    @property
    def operations(self):
        return [op for op in self._ops]
    
    def operation(self, index):
        return self[index]
    
    def write(self, outstream):
        """ Write the current group to a file.
        
        File format matches that read in the fromFile() method.
        
        The file starts with a header:
            
            dim     { 2 or 3 }
            size    { # of symmetry operations in group }
        
        Following this, each symmetry operation is printed according to 
        the specifications of its own write() method.
        
        This method can be used to save custom symmetry groups for later use.
        
        Parameters
        ----------
        outstream : stream
            The writable stream to which to save the symmetry group.
        """
        outstream.write("dim \t{}\n".format(self.dim))
        outstream.write("size\t{}\n".format(self.size))
        for op in self._ops:
            op.write(outstream)
        return outstream
    
    ## Mutators
     
    def addOperation(self, op, makeClosed=False):
        """
        Add the speficied symmetry operation to the group.
        
        Parameters
        ----------
        op : SymmetryOperation
            The SymmetryOperation instance to be added.
        makeClosed : bool (optional)
            If False (default), the new operation will be added to the
            group without any check of closed status.
            If True, self.makeClosed() will be called after the addition.
        
        Returns
        -------
        success : bool
            True if the operation was added.
            False if the operation was already present.
        
        Raises
        ------
        TypeError :
            If op is not an instance of SymmetryOperation.
        ValueError :
            If op does not match the dimensionality of the group.
        GroupTooLargeError : 
            If adding op would cause the group to exceed its maxOperations.
        """
        # Check input requirements
        if not isinstance(op,SymmetryOperation):
            raise(TypeError("op must be an instance of SymmetryOperation."))
        if not op.dim == self.dim:
            raise(ValueError("op must match the dimensionality of the group."))
        
        # Check if op already in the group
        if op in self:
            return False
        
        # Check if group has room to accept operation
        if not self.hasRoom():
            msg = "SymmetryGroup exceeded max size, {}, while adding operation {}."
            msg = msg.format(self._max_operations, op)
            raise(GroupTooLargeError(self._max_operations, self))
        
        # Add Operation
        self._ops.append(op)
        self._is_closed = False
        
        # Ensure group still closed
        if makeClosed:
            self.makeClosed()
        return True
    
    def makeClosed(self):
        """ Add symmetry operations to close the group.
        
        Iterate over all permuted pairs of symmetry operations,
        calculating their product and adding any unique results
        to the group.
        
        Raises
        ------
        GroupTooLargeError : 
            When the group reaches its maximum allowed size and
            still isn't closed.
        """
        # skip work if group was closed after last modification
        if self._is_closed:
            return None
        
        for op_one in self._ops:
            for op_two in self._ops:
                op_test = op_one @ op_two
                try:
                    success = self.addOperation(op_test)
                except GroupTooLargeError as err:
                    msg = "SymmetryGroup unable to close.\n{}".format(err.message)
                    raise(GroupTooLargeError(err.maxsize, self, msg))
                
        
        # Repeat search as many times as necessary to no longer find
        # any new operations, up to remaining_searches
        addedNew = True
        remaining_searches = 4 
        while addedNew:
            remaining_searches -= 1
            if remaining_searches <= 0:
                raise(RuntimeError("SymmetryGroup unable to close in allowed iterations."))
            addedNew = False
            i = 0
            while i < self.size:
                op_one = self.operation(i)
                j = 0
                while j < self.size:
                    op_two = self.operation(j)
                    op_test = op_one @ op_two
                    try:
                        success = self.addOperation(op_test)
                    except GroupTooLargeError as err:
                        msg = "SymmetryGroup unable to close.\n{}".format(err.message)
                        raise(GroupTooLargeError(err.maxsize, self, msg))
                    if success:
                        addedNew = True
                    j += 1
                i += 1
        # if program reaches this point, the group is closed.
        self._is_closed = True
        return None
    
    ## Operations
    
    def apply(self,other,target=[]):
        """ Return unique results of applying all symmetry operations.
        
        Apply all symmetry operations to other and return unique results.
        An object equal to Other may or may not be included in the results.
        If the group is closed, Other will be among the results.
        
        It is assumed that the operations and other are defined for the same
        lattice, such that left-multiplying other by the operation yields 
        the expected result as:
            result = Operation @ other
        
        Parameters
        ----------
        other : SymmetryOperation, GeneralPosition, Vector, numpy.ndarray
            The object to which to apply all operations.
            The objects must be of a types, <T> for which <T> @ SymmetryOperation
            is defined. SymmetryOperation defines this for itself, GeneralPosition,
            Vector, numpy.ndarray.
        target : list (optional)
            If included, results not already present in target will be
            added to it, and target will be returned.
        """
        if not isinstance(target,list):
            msg = "Target must be a list, not {}."
            raise(TypeError(msg.format(type(target).__name__)))
        for op in self._ops:
            res = op @ other
            if not res in target:
                target.append(res)
        return target
    
    def applyReciprocal(self,other, target=[]):
        """ Return unique results of applying all symmetry operations.
        
        Apply all symmetry operations to other and return unique results.
        An object equal to Other may or may not be included in the results.
        If the group is closed, Other will be among the results.
        
        It is assumed that the operations and other are defined on reciprocal
        lattices, such that right-multiplying other by the operation yields 
        the expected result as:
            result = other @ operation
        
        Parameters
        ----------
        other : SymmetryOperation, GeneralPosition, Vector, numpy.ndarray
            The object to which to apply all operations.
            The objects must be of a types, <T> for which SymmetryOperation @ <T>
            is defined. SymmetryOperation defines this for itself, GeneralPosition,
            Vector, numpy.ndarray.
        target : list (optional)
            If included, results not already present in target will be
            added to it, and target will be returned.
        """
        if not isinstance(target,list):
            msg = "Target must be a list, not {}."
            raise(TypeError(msg.format(type(target).__name__)))
        for op in self._ops:
            res = other @ op
            if not res in target:
                target.append(res)
        return target
    
    def applyToAll(self,others,target=[]):
        """ Return unique results of applying all symmetry operations to all.
        
        Apply all symmetry operations to others and return unique results.
        
        It is assumed that the operations and others are defined for the same
        lattice, such that left-multiplying other by the operation yields 
        the expected result as:
            result = Operation @ other
        
        Parameters
        ----------
        other : iterable of SymmetryOperation, GeneralPosition, Vector, numpy.ndarray
            The objects to which to apply all operations.
            The objects must be of a types, <T> for which <T> @ SymmetryOperation
            is defined. SymmetryOperation defines this for itself, GeneralPosition,
            Vector, numpy.ndarray.
        target : list (optional)
            If included, results not already present in target will be
            added to it, and target will be returned.
        """
        if not isinstance(target,list):
            msg = "Target must be a list, not {}."
            raise(TypeError(msg.format(type(target).__name__)))
        for obj in iter(others):
            target = self.apply(obj,target)
        return target
    
    def applyToAllReciprocal(self,others,target=[]):
        """ Return unique results of applying all symmetry operations to all.
        
        Apply all symmetry operations to others and return unique results.
        
        It is assumed that the operations and others are defined on reciprocal
        lattices, such that right-multiplying other by the operation yields 
        the expected result as:
            result = Operation @ other
        
        Parameters
        ----------
        other : iterable of SymmetryOperation, GeneralPosition, Vector, numpy.ndarray
            The objects to which to apply all operations.
            The objects must be of a types, <T> for which SymmetryOperation @ <T>
            is defined. SymmetryOperation defines this for itself, GeneralPosition,
            Vector, numpy.ndarray.
        target : list (optional)
            If included, results not already present in target will be
            added to it, and target will be returned.
        """
        if not isinstance(target,list):
            msg = "Target must be a list, not {}."
            raise(TypeError(msg.format(type(target).__name__)))
        for obj in iter(others):
            target = self.applyReciprocal(obj,target)
        return target
    
    def __getitem__(self,key):
        if isinstance(key,slice):
            indices = range(*key.indices(len(self._ops)))
            return [self._ops[i] for i in indices]
        return self._ops[i]
    
    def __contains__(self, obj):
        return obj in self._ops
    
    def __iter__(self):
        return iter(self._ops)
        
class CRYSTAL_SYSTEM(enum.Enum):
    OBLIQUE         = 1
    RECTANGULAR     = 2
    SQUARE          = 3
    HEXAGONAL       = 4
    TRICLINIC       = 5
    MONOCLINIC      = 6
    ORTHORHOMBIC    = 7
    TETRAGONAL      = 8
    TRIGONAL        = 9
    RHOMBOHEDRAL    = 9
    CUBIC           = 10

def getGroupSymmetryFileName(dim, crystal_system, group_name):
    """
    Get the group filename from given group name
    
    Parameters
    ----------
    dim : either 2 or 3
        Dimensionality of the unit cell.
        If 2, plane groups will be sought. If 3, space groups will be sought.
    crystal_system : group_data.CRYSTAL_SYSTEM
        The crystal system in which to look for the group.
        CRYSTAL_SYSTEM.HEXAGONAL applies to both 2D and 3D systems.
    group_name : string
        Name should match that expected as input into the PSCF software 
        (see: http://pscf.cems.umn.edu/) or its C++ version.
        Also accepts an integer or string equivallent of the group ID number.
    
    Returns
    -------
    filename : pathlib.Path
        Absolute path to the databased symmetry file.
    
    Raises
    ------
    ValueError if no group entry is found for the given parameter set.
    """
    # First check for ID number inputs
    group_id = str(group_name)
    key = (dim, crystal_system, group_id)
    out = GROUP_NAME_BY_ID.get( key, None )
    if out is not None:
        group_name = out
    # Check for filename by PSCF group name
    key = (dim, crystal_system, group_name)
    out = GROUP_FILE_BY_NAME.get( key, None )
    if out is None:
        # Check ' : 2' ending
        key = (dim, crystal_system, group_name+" : 2")
        out = GROUP_FILE_BY_NAME.get( key, None )
    if out is None:
        # Check ' : H' ending
        key = (dim, crystal_system, group_name+" : H")
        out = GROUP_FILE_BY_NAME.get( key, None )
    if out is None:
        # Assume group_name is already in pscfpp format (filename format)
        out = group_name
    # Build path to libraries
    fname = pathlib.Path(__file__).parent.absolute()
    fname = fname / "groups" / str(dim) / out
    if not fname.is_file():
        raise(GroupFileNotFoundError(group_name))
    return fname

class SpaceGroup(object):
    """ Class to generate symmetry operation group and GEPs for space groups """
    
    __max_ops = 195 # 192 operations + 3 unit translations
    __max_gep = 192
    
    def __init__(self, dim, crystal_system, group_name):
        """
        Initialize a SpaceGroup instance.
        
        Parameters
        ----------
        dim : int, either 2 or 3
            dimensionality of the plane (2D) or space (3D) group
        crystal_system : string
            The name of the crystal system.
            If dim = 2: oblique, rectangular, hexagonal, square.
            If dim = 3: triclinic, monoclinic, orthorhombic, tetragonal, trigonal, hexagonal, cubic
        group_name : string
            The name of the space group.
            See PSCF user manual (https://pscf.readthedocs.io/en/latest/#) for name formatting.
        """
        self._dim = dim
        self._crystal_system = crystal_system.strip("'")
        self._group_name = group_name.strip("'")
        sourcefile = getGroupSymmetryFileName(self.dim, self.crystalSystem, self.groupName)
        self._symmetry_group =  SymmetryGroup.fromFile( \
                                    sourcefile, \
                                    checkClosed = False, \
                                    maxOperations = SpaceGroup.__max_ops)
        self._general_positions = None
        self._build_GEPs()
    
    @property
    def dim(self):
        return self._dim
    
    @property
    def crystalSystem(self):
        return str(self._crystal_system)
    
    @property
    def groupName(self):
        return str(self._group_name)
    
    @property
    def symmetryCount(self):
        return self._symmetry_group.size
    
    @property
    def symmetryOperations(self):
        """ A list of symmetry operations in the space group. """
        return self._symmetry_group.operations
    
    @property
    def positionCount(self):
        """ The number of General Equivallent Positions (GEPs) in group.
        
        Class has been updated to bypass GEPs in position evaluations.
        As a result, GEPs are no longer generated on initialization.
        First call to an instance's positionCount property will trigger
        calculation of the GEPs, which will thereafter be stored.
        """
        if self._general_positions is None:
            self._build_GEPs()
        return len(self._general_positions)
    
    @property
    def generalPositions(self):
        """ A list of General Equivallent Positions (GEPs)
        
        Class has been updated to bypass GEPs in position evaluations.
        As a result, GEPs are no longer generated on initialization.
        First call to an instance's generalPositions property will trigger
        calculation of the GEPs, which will thereafter be stored.
        """
        if self._general_positions is None:
            self._build_GEPs()
        return [gep for gep in self._general_positions]
    
    def evaluatePosition(self, position, atol=POSITION_TOLERANCE):
        """ Apply each GEP to the given position and return the set of unique positions.
        'Uniqueness' determined by a separation of greater than atol. 
        """
        out = []
        for p in self._general_positions:
            nextPos = p.evaluate(position)
            matchFound = False
            for q in out:
                diff = np.absolute(nextPos - q)
                if np.all(diff < atol):
                    matchFound = True
                    break
            if not matchFound:
                out.append(nextPos)
        return out
    
    def evaluatePositions(self, positions, atol=POSITION_TOLERANCE):
        out = []
        for p in positions:
            out.append(self.evaluatePosition(p,atol))
    
    def apply(self,other,target=[]):
        """ Return unique results of applying all symmetry operations.
        
        Apply all symmetry operations to other and return unique results.
        An object equal to Other may or may not be included in the results.
        If the group is closed, Other will be among the results.
        
        It is assumed that the operations and other are defined for the same
        lattice, such that left-multiplying other by the operation yields 
        the expected result as:
            result = Operation @ other
        
        Parameters
        ----------
        other : Multiple
            The object to which to apply all operations.
            The object must be of a type, <T> for which <T> @ SymmetryOperation
            is defined. SymmetryOperation defines this for itself, GeneralPosition,
            Vector, numpy.ndarray.
        target : list (optional)
            If included, results not already present in target will be
            added to it, and target will be returned.
        """
        return self._symmetry_group.apply(other,target)
    
    def applyReciprocal(self,other, target=[]):
        """ Return unique results of applying all symmetry operations.
        
        Apply all symmetry operations to other and return unique results.
        An object equal to Other may or may not be included in the results.
        If the group is closed, Other will be among the results.
        
        It is assumed that the operations and other are defined on reciprocal
        lattices, such that right-multiplying other by the operation yields 
        the expected result as:
            result = other @ operation
        
        Parameters
        ----------
        other : SymmetryOperation, GeneralPosition, Vector, numpy.ndarray
            The object to which to apply all operations.
            The objects must be of a types, <T> for which SymmetryOperation @ <T>
            is defined. SymmetryOperation defines this for itself, GeneralPosition,
            Vector, numpy.ndarray.
        target : list (optional)
            If included, results not already present in target will be
            added to it, and target will be returned.
        """
        return self._symmetry_group.applyReciprocal(other,target)
    
    def applyToAll(self,others,target=[]):
        """ Return unique results of applying all symmetry operations to all.
        
        Apply all symmetry operations to others and return unique results.
        
        It is assumed that the operations and others are defined for the same
        lattice, such that left-multiplying other by the operation yields 
        the expected result as:
            result = Operation @ other
        
        Parameters
        ----------
        other : iterable of <Various Classes>
            The objects to which to apply all operations.
            The objects must be of a types, <T> for which <T> @ SymmetryOperation
            is defined. SymmetryOperation defines this for itself, GeneralPosition,
            Vector, numpy.ndarray.
        target : list (optional)
            If included, results not already present in target will be
            added to it, and target will be returned.
        """
        return self._symmetry_group.applyToAll(others,target)
    
    def applyToAllReciprocal(self,others,target=[]):
        """ Return unique results of applying all symmetry operations to all.
        
        Apply all symmetry operations to others and return unique results.
        
        It is assumed that the operations and others are defined on reciprocal
        lattices, such that right-multiplying other by the operation yields 
        the expected result as:
            result = Operation @ other
        
        Parameters
        ----------
        other : iterable of SymmetryOperation, GeneralPosition, Vector, numpy.ndarray
            The objects to which to apply all operations.
            The objects must be of a types, <T> for which SymmetryOperation @ <T>
            is defined. SymmetryOperation defines this for itself, GeneralPosition,
            Vector, numpy.ndarray.
        target : list (optional)
            If included, results not already present in target will be
            added to it, and target will be returned.
        """
        return self._symmetry_group.applyToAllReciprocal(others,target)
    
    def __str__(self):
        formstr = "< SpaceGroup object with dim = {}, system = {}, group name = {} >"
        return formstr.format(self.dim, self.crystalSystem, self.groupName)
    
    def _build_GEPs(self):
        pos = GeneralPosition(self._dim)
        geps = [pos]
        for symm in self.symmetryOperations:
            newPos = symm @ pos
            if not newPos in geps:
                geps.append(newPos)
        self._general_positions = geps
    
"""
Group identifying data
    Keys are tuples of (dimension, crystal_system, group_name)
    values are the ID number of the group.
"""
GROUP_FILE_BY_NAME = {  ( 2, 'oblique',      'p 1'            )  :  'p_1', \
                        ( 2, 'oblique',      'p 2'            )  :  'p_2', \
                        ( 2, 'rectangular',  'p m'            )  :  'p_m', \
                        ( 2, 'rectangular',  'p g'            )  :  'p_g', \
                        ( 2, 'rectangular',  'c m'            )  :  'c_m', \
                        ( 2, 'rectangular',  'p 2 m m'        )  :  'p_2_m_m', \
                        ( 2, 'rectangular',  'p 2 m g'        )  :  'p_2_m_g', \
                        ( 2, 'rectangular',  'p 2 g g'        )  :  'p_2_g_g', \
                        ( 2, 'rectangular',  'c 2 m m'        )  :  'c_2_m_m', \
                        ( 2, 'square',       'p 4'            )  :  'p_4', \
                        ( 2, 'square',       'p 4 m m'        )  :  'p_4_m_m', \
                        ( 2, 'square',       'p 4 g m'        )  :  'p_4_g_m', \
                        ( 2, 'hexagonal',    'p 3'            )  :  'p_3', \
                        ( 2, 'hexagonal',    'p 3 m 1'        )  :  'p_3_m_1', \
                        ( 2, 'hexagonal',    'p 3 1 m'        )  :  'p_3_1_m', \
                        ( 2, 'hexagonal',    'p 6'            )  :  'p_6', \
                        ( 2, 'hexagonal',    'p 6 m m'        )  :  'p_6_m_m', \
                        ( 3, 'triclinic',    'P 1'            )  :  'P_1', \
                        ( 3, 'triclinic',    'P -1'           )  :  'P_-1', \
                        ( 3, 'monoclinic',   'P 1 2 1'        )  :  'P_1_2_1', \
                        ( 3, 'monoclinic',   'P 1 21 1'       )  :  'P_1_21_1', \
                        ( 3, 'monoclinic',   'C 1 2 1'        )  :  'C_1_2_1', \
                        ( 3, 'monoclinic',   'P 1 m 1'        )  :  'P_1_m_1', \
                        ( 3, 'monoclinic',   'P 1 c 1'        )  :  'P_1_c_1', \
                        ( 3, 'monoclinic',   'C 1 m 1'        )  :  'C_1_m_1', \
                        ( 3, 'monoclinic',   'C 1 c 1'        )  :  'C_1_c_1', \
                        ( 3, 'monoclinic',   'P 1 2/m 1'      )  :  'P_1_2%m_1', \
                        ( 3, 'monoclinic',   'P 1 21/m 1'     )  :  'P_1_21%m_1', \
                        ( 3, 'monoclinic',   'C 1 2/m 1'      )  :  'C_1_2%m_1', \
                        ( 3, 'monoclinic',   'P 1 2/c 1'      )  :  'P_1_2%c_1', \
                        ( 3, 'monoclinic',   'P 1 21/c 1'     )  :  'P_1_21%c_1', \
                        ( 3, 'monoclinic',   'C 1 2/c 1'      )  :  'C_1_2%c_1', \
                        ( 3, 'orthorhombic', 'P 2 2 2'        )  :  'P_2_2_2', \
                        ( 3, 'orthorhombic', 'P 2 2 21'       )  :  'P_2_2_21', \
                        ( 3, 'orthorhombic', 'P 21 21 2'      )  :  'P_21_21_2', \
                        ( 3, 'orthorhombic', 'P 21 21 21'     )  :  'P_21_21_21', \
                        ( 3, 'orthorhombic', 'C 2 2 21'       )  :  'C_2_2_21', \
                        ( 3, 'orthorhombic', 'C 2 2 2'        )  :  'C_2_2_2', \
                        ( 3, 'orthorhombic', 'F 2 2 2'        )  :  'F_2_2_2', \
                        ( 3, 'orthorhombic', 'I 2 2 2'        )  :  'I_2_2_2', \
                        ( 3, 'orthorhombic', 'I 21 21 21'     )  :  'I_21_21_21', \
                        ( 3, 'orthorhombic', 'P m m 2'        )  :  'P_m_m_2', \
                        ( 3, 'orthorhombic', 'P m c 21'       )  :  'P_m_c_21', \
                        ( 3, 'orthorhombic', 'P c c 2'        )  :  'P_c_c_2', \
                        ( 3, 'orthorhombic', 'P m a 2'        )  :  'P_m_a_2', \
                        ( 3, 'orthorhombic', 'P c a 21'       )  :  'P_c_a_21', \
                        ( 3, 'orthorhombic', 'P n c 2'        )  :  'P_n_c_2', \
                        ( 3, 'orthorhombic', 'P m n 21'       )  :  'P_m_n_21', \
                        ( 3, 'orthorhombic', 'P b a 2'        )  :  'P_b_a_2', \
                        ( 3, 'orthorhombic', 'P n a 21'       )  :  'P_n_a_21', \
                        ( 3, 'orthorhombic', 'P n n 2'        )  :  'P_n_n_2', \
                        ( 3, 'orthorhombic', 'C m m 2'        )  :  'C_m_m_2', \
                        ( 3, 'orthorhombic', 'C m c 21'       )  :  'C_m_c_21', \
                        ( 3, 'orthorhombic', 'C c c 2'        )  :  'C_c_c_2', \
                        ( 3, 'orthorhombic', 'A m m 2'        )  :  'A_m_m_2', \
                        ( 3, 'orthorhombic', 'A b m 2'        )  :  'A_b_m_2', \
                        ( 3, 'orthorhombic', 'A m a 2'        )  :  'A_m_a_2', \
                        ( 3, 'orthorhombic', 'A b a 2'        )  :  'A_b_a_2', \
                        ( 3, 'orthorhombic', 'F m m 2'        )  :  'F_m_m_2', \
                        ( 3, 'orthorhombic', 'F d d 2'        )  :  'F_d_d_2', \
                        ( 3, 'orthorhombic', 'I m m 2'        )  :  'I_m_m_2', \
                        ( 3, 'orthorhombic', 'I b a 2'        )  :  'I_b_a_2', \
                        ( 3, 'orthorhombic', 'I m a 2'        )  :  'I_m_a_2', \
                        ( 3, 'orthorhombic', 'P m m m'        )  :  'P_m_m_m', \
                        ( 3, 'orthorhombic', 'P n n n : 2'    )  :  'P_n_n_n:2', \
                        ( 3, 'orthorhombic', 'P n n n : 1'    )  :  'P_n_n_n:1', \
                        ( 3, 'orthorhombic', 'P c c m'        )  :  'P_c_c_m', \
                        ( 3, 'orthorhombic', 'P b a n : 2'    )  :  'P_b_a_n:2', \
                        ( 3, 'orthorhombic', 'P b a n : 1'    )  :  'P_b_a_n:1', \
                        ( 3, 'orthorhombic', 'P m m a'        )  :  'P_m_m_a', \
                        ( 3, 'orthorhombic', 'P n n a'        )  :  'P_n_n_a', \
                        ( 3, 'orthorhombic', 'P m n a'        )  :  'P_m_n_a', \
                        ( 3, 'orthorhombic', 'P c c a'        )  :  'P_c_c_a', \
                        ( 3, 'orthorhombic', 'P b a m'        )  :  'P_b_a_m', \
                        ( 3, 'orthorhombic', 'P c c n'        )  :  'P_c_c_n', \
                        ( 3, 'orthorhombic', 'P b c m'        )  :  'P_b_c_m', \
                        ( 3, 'orthorhombic', 'P n n m'        )  :  'P_n_n_m', \
                        ( 3, 'orthorhombic', 'P m m n : 2'    )  :  'P_m_m_n:2', \
                        ( 3, 'orthorhombic', 'P m m n : 1'    )  :  'P_m_m_n:1', \
                        ( 3, 'orthorhombic', 'P b c n'        )  :  'P_b_c_n', \
                        ( 3, 'orthorhombic', 'P b c a'        )  :  'P_b_c_a', \
                        ( 3, 'orthorhombic', 'P n m a'        )  :  'P_n_m_a', \
                        ( 3, 'orthorhombic', 'C m c m'        )  :  'C_m_c_m', \
                        ( 3, 'orthorhombic', 'C m c a'        )  :  'C_m_c_a', \
                        ( 3, 'orthorhombic', 'C m m m'        )  :  'C_m_m_m', \
                        ( 3, 'orthorhombic', 'C c c m'        )  :  'C_c_c_m', \
                        ( 3, 'orthorhombic', 'C m m a'        )  :  'C_m_m_a', \
                        ( 3, 'orthorhombic', 'C c c a : 2'    )  :  'C_c_c_a:2', \
                        ( 3, 'orthorhombic', 'C c c a : 1'    )  :  'C_c_c_a:1', \
                        ( 3, 'orthorhombic', 'F m m m'        )  :  'F_m_m_m', \
                        ( 3, 'orthorhombic', 'F d d d : 2'    )  :  'F_d_d_d:2', \
                        ( 3, 'orthorhombic', 'F d d d : 1'    )  :  'F_d_d_d:1', \
                        ( 3, 'orthorhombic', 'I m m m'        )  :  'I_m_m_m', \
                        ( 3, 'orthorhombic', 'I b a m'        )  :  'I_b_a_m', \
                        ( 3, 'orthorhombic', 'I b c a'        )  :  'I_b_c_a', \
                        ( 3, 'orthorhombic', 'I m m a'        )  :  'I_m_m_a', \
                        ( 3, 'tetragonal',   'P 4'            )  :  'P_4', \
                        ( 3, 'tetragonal',   'P 41'           )  :  'P_41', \
                        ( 3, 'tetragonal',   'P 42'           )  :  'P_42', \
                        ( 3, 'tetragonal',   'P 43'           )  :  'P_43', \
                        ( 3, 'tetragonal',   'I 4'            )  :  'I_4', \
                        ( 3, 'tetragonal',   'I 41'           )  :  'I_41', \
                        ( 3, 'tetragonal',   'P -4'           )  :  'P_-4', \
                        ( 3, 'tetragonal',   'I -4'           )  :  'I_-4', \
                        ( 3, 'tetragonal',   'P 4/m'          )  :  'P_4%m', \
                        ( 3, 'tetragonal',   'P 42/m'         )  :  'P_42%m', \
                        ( 3, 'tetragonal',   'P 4/n : 2'      )  :  'P_4%n:2', \
                        ( 3, 'tetragonal',   'P 4/n : 1'      )  :  'P_4%n:1', \
                        ( 3, 'tetragonal',   'P 42/n : 2'     )  :  'P_42%n:2', \
                        ( 3, 'tetragonal',   'P 42/n : 1'     )  :  'P_42%n:1', \
                        ( 3, 'tetragonal',   'I 4/m'          )  :  'I_4%m', \
                        ( 3, 'tetragonal',   'I 41/a : 2'     )  :  'I_41%a:2', \
                        ( 3, 'tetragonal',   'I 41/a : 1'     )  :  'I_41%a:1', \
                        ( 3, 'tetragonal',   'P 4 2 2'        )  :  'P_4_2_2', \
                        ( 3, 'tetragonal',   'P 4 21 2'       )  :  'P_4_21_2', \
                        ( 3, 'tetragonal',   'P 41 2 2'       )  :  'P_41_2_2', \
                        ( 3, 'tetragonal',   'P 41 21 2'      )  :  'P_41_21_2', \
                        ( 3, 'tetragonal',   'P 42 2 2'       )  :  'P_42_2_2', \
                        ( 3, 'tetragonal',   'P 42 21 2'      )  :  'P_42_21_2', \
                        ( 3, 'tetragonal',   'P 43 2 2'       )  :  'P_43_2_2', \
                        ( 3, 'tetragonal',   'P 43 21 2'      )  :  'P_43_21_2', \
                        ( 3, 'tetragonal',   'I 4 2 2'        )  :  'I_4_2_2', \
                        ( 3, 'tetragonal',   'I 41 2 2'       )  :  'I_41_2_2', \
                        ( 3, 'tetragonal',   'P 4 m m'        )  :  'P_4_m_m', \
                        ( 3, 'tetragonal',   'P 4 b m'        )  :  'P_4_b_m', \
                        ( 3, 'tetragonal',   'P 42 c m'       )  :  'P_42_c_m', \
                        ( 3, 'tetragonal',   'P 42 n m'       )  :  'P_42_n_m', \
                        ( 3, 'tetragonal',   'P 4 c c'        )  :  'P_4_c_c', \
                        ( 3, 'tetragonal',   'P 4 n c'        )  :  'P_4_n_c', \
                        ( 3, 'tetragonal',   'P 42 m c'       )  :  'P_42_m_c', \
                        ( 3, 'tetragonal',   'P 42 b c'       )  :  'P_42_b_c', \
                        ( 3, 'tetragonal',   'I 4 m m'        )  :  'I_4_m_m', \
                        ( 3, 'tetragonal',   'I 4 c m'        )  :  'I_4_c_m', \
                        ( 3, 'tetragonal',   'I 41 m d'       )  :  'I_41_m_d', \
                        ( 3, 'tetragonal',   'I 41 c d'       )  :  'I_41_c_d', \
                        ( 3, 'tetragonal',   'P -4 2 m'       )  :  'P_-4_2_m', \
                        ( 3, 'tetragonal',   'P -4 2 c'       )  :  'P_-4_2_c', \
                        ( 3, 'tetragonal',   'P -4 21 m'      )  :  'P_-4_21_m', \
                        ( 3, 'tetragonal',   'P -4 21 c'      )  :  'P_-4_21_c', \
                        ( 3, 'tetragonal',   'P -4 m 2'       )  :  'P_-4_m_2', \
                        ( 3, 'tetragonal',   'P -4 c 2'       )  :  'P_-4_c_2', \
                        ( 3, 'tetragonal',   'P -4 b 2'       )  :  'P_-4_b_2', \
                        ( 3, 'tetragonal',   'P -4 n 2'       )  :  'P_-4_n_2', \
                        ( 3, 'tetragonal',   'I -4 m 2'       )  :  'I_-4_m_2', \
                        ( 3, 'tetragonal',   'I -4 c 2'       )  :  'I_-4_c_2', \
                        ( 3, 'tetragonal',   'I -4 2 m'       )  :  'I_-4_2_m', \
                        ( 3, 'tetragonal',   'I -4 2 d'       )  :  'I_-4_2_d', \
                        ( 3, 'tetragonal',   'P 4/m m m'      )  :  'P_4%m_m_m', \
                        ( 3, 'tetragonal',   'P 4/m c c'      )  :  'P_4%m_c_c', \
                        ( 3, 'tetragonal',   'P 4/n b m : 2'  )  :  'P_4%n_b_m:2', \
                        ( 3, 'tetragonal',   'P 4/n b m : 1'  )  :  'P_4%n_b_m:1', \
                        ( 3, 'tetragonal',   'P 4/n n c : 2'  )  :  'P_4%n_n_c:2', \
                        ( 3, 'tetragonal',   'P 4/n n c : 1'  )  :  'P_4%n_n_c:1', \
                        ( 3, 'tetragonal',   'P 4/m b m'      )  :  'P_4%m_b_m', \
                        ( 3, 'tetragonal',   'P 4/m n c'      )  :  'P_4%m_n_c', \
                        ( 3, 'tetragonal',   'P 4/n m m : 2'  )  :  'P_4%n_m_m:2', \
                        ( 3, 'tetragonal',   'P 4/n m m : 1'  )  :  'P_4%n_m_m:1', \
                        ( 3, 'tetragonal',   'P 4/n c c : 2'  )  :  'P_4%n_c_c:2', \
                        ( 3, 'tetragonal',   'P 4/n c c : 1'  )  :  'P_4%n_c_c:1', \
                        ( 3, 'tetragonal',   'P 42/m m c'     )  :  'P_42%m_m_c', \
                        ( 3, 'tetragonal',   'P 42/m c m'     )  :  'P_42%m_c_m', \
                        ( 3, 'tetragonal',   'P 42/n b c : 2' )  :  'P_42%n_b_c:2', \
                        ( 3, 'tetragonal',   'P 42/n b c : 1' )  :  'P_42%n_b_c:1', \
                        ( 3, 'tetragonal',   'P 42/n n m : 2' )  :  'P_42%n_n_m:2', \
                        ( 3, 'tetragonal',   'P 42/n n m : 1' )  :  'P_42%n_n_m:1', \
                        ( 3, 'tetragonal',   'P 42/m b c'     )  :  'P_42%m_b_c', \
                        ( 3, 'tetragonal',   'P 42/m n m'     )  :  'P_42%m_n_m', \
                        ( 3, 'tetragonal',   'P 42/n m c : 2' )  :  'P_42%n_m_c:2', \
                        ( 3, 'tetragonal',   'P 42/n m c : 1' )  :  'P_42%n_m_c:1', \
                        ( 3, 'tetragonal',   'P 42/n c m : 2' )  :  'P_42%n_c_m:2', \
                        ( 3, 'tetragonal',   'P 42/n c m : 1' )  :  'P_42%n_c_m:1', \
                        ( 3, 'tetragonal',   'I 4/m m m'      )  :  'I_4%m_m_m', \
                        ( 3, 'tetragonal',   'I 4/m c m'      )  :  'I_4%m_c_m', \
                        ( 3, 'tetragonal',   'I 41/a m d : 2' )  :  'I_41%a_m_d:2', \
                        ( 3, 'tetragonal',   'I 41/a m d : 1' )  :  'I_41%a_m_d:1', \
                        ( 3, 'tetragonal',   'I 41/a c d : 2' )  :  'I_41%a_c_d:2', \
                        ( 3, 'tetragonal',   'I 41/a c d : 1' )  :  'I_41%a_c_d:1', \
                        ( 3, 'trigonal',     'P 3'            )  :  'P_3', \
                        ( 3, 'trigonal',     'P 31'           )  :  'P_31', \
                        ( 3, 'trigonal',     'P 32'           )  :  'P_32', \
                        ( 3, 'trigonal',     'R 3 : H'        )  :  'R_3:H', \
                        ( 3, 'trigonal',     'R 3 : R'        )  :  'R_3:R', \
                        ( 3, 'trigonal',     'P -3'           )  :  'P_-3', \
                        ( 3, 'trigonal',     'R -3 : H'       )  :  'R_-3:H', \
                        ( 3, 'trigonal',     'R -3 : R'       )  :  'R_-3:R', \
                        ( 3, 'trigonal',     'P 3 1 2'        )  :  'P_3_1_2', \
                        ( 3, 'trigonal',     'P 3 2 1'        )  :  'P_3_2_1', \
                        ( 3, 'trigonal',     'P 31 1 2'       )  :  'P_31_1_2', \
                        ( 3, 'trigonal',     'P 31 2 1'       )  :  'P_31_2_1', \
                        ( 3, 'trigonal',     'P 32 1 2'       )  :  'P_32_1_2', \
                        ( 3, 'trigonal',     'P 32 2 1'       )  :  'P_32_2_1', \
                        ( 3, 'trigonal',     'R 3 2 : H'      )  :  'R_3_2:H', \
                        ( 3, 'trigonal',     'R 3 2 : R'      )  :  'R_3_2:R', \
                        ( 3, 'trigonal',     'P 3 m 1'        )  :  'P_3_m_1', \
                        ( 3, 'trigonal',     'P 3 1 m'        )  :  'P_3_1_m', \
                        ( 3, 'trigonal',     'P 3 c 1'        )  :  'P_3_c_1', \
                        ( 3, 'trigonal',     'P 3 1 c'        )  :  'P_3_1_c', \
                        ( 3, 'trigonal',     'R 3 m : H'      )  :  'R_3_m:H', \
                        ( 3, 'trigonal',     'R 3 m : R'      )  :  'R_3_m:R', \
                        ( 3, 'trigonal',     'R 3 c : H'      )  :  'R_3_c:H', \
                        ( 3, 'trigonal',     'R 3 c : R'      )  :  'R_3_c:R', \
                        ( 3, 'trigonal',     'P -3 1 m'       )  :  'P_-3_1_m', \
                        ( 3, 'trigonal',     'P -3 1 c'       )  :  'P_-3_1_c', \
                        ( 3, 'trigonal',     'P -3 m 1'       )  :  'P_-3_m_1', \
                        ( 3, 'trigonal',     'P -3 c 1'       )  :  'P_-3_c_1', \
                        ( 3, 'trigonal',     'R -3 m : H'     )  :  'R_-3_m:H', \
                        ( 3, 'trigonal',     'R -3 m : R'     )  :  'R_-3_m:R', \
                        ( 3, 'trigonal',     'R -3 c : H'     )  :  'R_-3_c:H', \
                        ( 3, 'trigonal',     'R -3 c : R'     )  :  'R_-3_c:R', \
                        ( 3, 'hexagonal',    'P 6'            )  :  'P_6', \
                        ( 3, 'hexagonal',    'P 61'           )  :  'P_61', \
                        ( 3, 'hexagonal',    'P 65'           )  :  'P_65', \
                        ( 3, 'hexagonal',    'P 62'           )  :  'P_62', \
                        ( 3, 'hexagonal',    'P 64'           )  :  'P_64', \
                        ( 3, 'hexagonal',    'P 63'           )  :  'P_63', \
                        ( 3, 'hexagonal',    'P -6'           )  :  'P_-6', \
                        ( 3, 'hexagonal',    'P 6/m'          )  :  'P_6%m', \
                        ( 3, 'hexagonal',    'P 63/m'         )  :  'P_63%m', \
                        ( 3, 'hexagonal',    'P 6 2 2'        )  :  'P_6_2_2', \
                        ( 3, 'hexagonal',    'P 61 2 2'       )  :  'P_61_2_2', \
                        ( 3, 'hexagonal',    'P 65 2 2'       )  :  'P_65_2_2', \
                        ( 3, 'hexagonal',    'P 62 2 2'       )  :  'P_62_2_2', \
                        ( 3, 'hexagonal',    'P 64 2 2'       )  :  'P_64_2_2', \
                        ( 3, 'hexagonal',    'P 63 2 2'       )  :  'P_63_2_2', \
                        ( 3, 'hexagonal',    'P 6 m m'        )  :  'P_6_m_m', \
                        ( 3, 'hexagonal',    'P 6 c c'        )  :  'P_6_c_c', \
                        ( 3, 'hexagonal',    'P 63 c m'       )  :  'P_63_c_m', \
                        ( 3, 'hexagonal',    'P 63 m c'       )  :  'P_63_m_c', \
                        ( 3, 'hexagonal',    'P -6 m 2'       )  :  'P_-6_m_2', \
                        ( 3, 'hexagonal',    'P -6 c 2'       )  :  'P_-6_c_2', \
                        ( 3, 'hexagonal',    'P -6 2 m'       )  :  'P_-6_2_m', \
                        ( 3, 'hexagonal',    'P -6 2 c'       )  :  'P_-6_2_c', \
                        ( 3, 'hexagonal',    'P 6/m m m'      )  :  'P_6%m_m_m', \
                        ( 3, 'hexagonal',    'P 6/m c c'      )  :  'P_6%m_c_c', \
                        ( 3, 'hexagonal',    'P 63/m c m'     )  :  'P_63%m_c_m', \
                        ( 3, 'hexagonal',    'P 63/m m c'     )  :  'P_63%m_m_c', \
                        ( 3, 'cubic',        'P 2 3'          )  :  'P_2_3', \
                        ( 3, 'cubic',        'F 2 3'          )  :  'F_2_3', \
                        ( 3, 'cubic',        'I 2 3'          )  :  'I_2_3', \
                        ( 3, 'cubic',        'P 21 3'         )  :  'P_21_3', \
                        ( 3, 'cubic',        'I 21 3'         )  :  'I_21_3', \
                        ( 3, 'cubic',        'P m -3'         )  :  'P_m_-3', \
                        ( 3, 'cubic',        'P n -3 : 2'     )  :  'P_n_-3:2', \
                        ( 3, 'cubic',        'P n -3 : 1'     )  :  'P_n_-3:1', \
                        ( 3, 'cubic',        'F m -3'         )  :  'F_m_-3', \
                        ( 3, 'cubic',        'F d -3 : 2'     )  :  'F_d_-3:2', \
                        ( 3, 'cubic',        'F d -3 : 1'     )  :  'F_d_-3:1', \
                        ( 3, 'cubic',        'I m -3'         )  :  'I_m_-3', \
                        ( 3, 'cubic',        'P a -3'         )  :  'P_a_-3', \
                        ( 3, 'cubic',        'I a -3'         )  :  'I_a_-3', \
                        ( 3, 'cubic',        'P 4 3 2'        )  :  'P_4_3_2', \
                        ( 3, 'cubic',        'P 42 3 2'       )  :  'P_42_3_2', \
                        ( 3, 'cubic',        'F 4 3 2'        )  :  'F_4_3_2', \
                        ( 3, 'cubic',        'F 41 3 2'       )  :  'F_41_3_2', \
                        ( 3, 'cubic',        'I 4 3 2'        )  :  'I_4_3_2', \
                        ( 3, 'cubic',        'P 43 3 2'       )  :  'P_43_3_2', \
                        ( 3, 'cubic',        'P 41 3 2'       )  :  'P_41_3_2', \
                        ( 3, 'cubic',        'I 41 3 2'       )  :  'I_41_3_2', \
                        ( 3, 'cubic',        'P -4 3 m'       )  :  'P_-4_3_m', \
                        ( 3, 'cubic',        'F -4 3 m'       )  :  'F_-4_3_m', \
                        ( 3, 'cubic',        'I -4 3 m'       )  :  'I_-4_3_m', \
                        ( 3, 'cubic',        'P -4 3 n'       )  :  'P_-4_3_n', \
                        ( 3, 'cubic',        'F -4 3 c'       )  :  'F_-4_3_c', \
                        ( 3, 'cubic',        'I -4 3 d'       )  :  'I_-4_3_d', \
                        ( 3, 'cubic',        'P m -3 m'       )  :  'P_m_-3_m', \
                        ( 3, 'cubic',        'P n -3 n : 2'   )  :  'P_n_-3_n:2', \
                        ( 3, 'cubic',        'P n -3 n : 1'   )  :  'P_n_-3_n:1', \
                        ( 3, 'cubic',        'P m -3 n'       )  :  'P_m_-3_n', \
                        ( 3, 'cubic',        'P n -3 m : 2'   )  :  'P_n_-3_m:2', \
                        ( 3, 'cubic',        'P n -3 m : 1'   )  :  'P_n_-3_m:1', \
                        ( 3, 'cubic',        'F m -3 m'       )  :  'F_m_-3_m', \
                        ( 3, 'cubic',        'F m -3 c'       )  :  'F_m_-3_c', \
                        ( 3, 'cubic',        'F d -3 m : 2'   )  :  'F_d_-3_m:2', \
                        ( 3, 'cubic',        'F d -3 m : 1'   )  :  'F_d_-3_m:1', \
                        ( 3, 'cubic',        'F d -3 c : 2'   )  :  'F_d_-3_c:2', \
                        ( 3, 'cubic',        'F d -3 c : 1'   )  :  'F_d_-3_c:1', \
                        ( 3, 'cubic',        'I m -3 m'       )  :  'I_m_-3_m', \
                        ( 3, 'cubic',        'I a -3 d'       )  :  'I_a_-3_d' }

"""
Group identifying data.
    Keys are tuples of (dimension, crystal_system, ID).
    Values are the group names (PSCF_Fortran format).
    For groups with multiple settings, selection follows
        PSCF_Fortran precedent in selecting ' : 2' and ' : R'
"""
GROUP_NAME_BY_ID = {   ( 2, 'oblique',       1   )  :  'p 1', \
                        ( 2, 'oblique',       2   )  :  'p 2', \
                        ( 2, 'rectangular',   3   )  :  'p m', \
                        ( 2, 'rectangular',   4   )  :  'p g', \
                        ( 2, 'rectangular',   5   )  :  'c m', \
                        ( 2, 'rectangular',   6   )  :  'p 2 m m', \
                        ( 2, 'rectangular',   7   )  :  'p 2 m g', \
                        ( 2, 'rectangular',   8   )  :  'p 2 g g', \
                        ( 2, 'rectangular',   9   )  :  'c 2 m m', \
                        ( 2, 'square',        10  )  :  'p 4', \
                        ( 2, 'square',        11  )  :  'p 4 m m', \
                        ( 2, 'square',        12  )  :  'p 4 g m', \
                        ( 2, 'hexagonal',     13  )  :  'p 3', \
                        ( 2, 'hexagonal',     14  )  :  'p 3 m 1', \
                        ( 2, 'hexagonal',     15  )  :  'p 3 1 m', \
                        ( 2, 'hexagonal',     16  )  :  'p 6', \
                        ( 2, 'hexagonal',     17  )  :  'p 6 m m', \
                        ( 3, 'triclinic',     1   )  :  'P 1', \
                        ( 3, 'triclinic',     2   )  :  'P -1', \
                        ( 3, 'monoclinic',    3   )  :  'P 1 2 1', \
                        ( 3, 'monoclinic',    4   )  :  'P 1 21 1', \
                        ( 3, 'monoclinic',    5   )  :  'C 1 2 1', \
                        ( 3, 'monoclinic',    6   )  :  'P 1 m 1', \
                        ( 3, 'monoclinic',    7   )  :  'P 1 c 1', \
                        ( 3, 'monoclinic',    8   )  :  'C 1 m 1', \
                        ( 3, 'monoclinic',    9   )  :  'C 1 c 1', \
                        ( 3, 'monoclinic',    10  )  :  'P 1 2/m 1', \
                        ( 3, 'monoclinic',    11  )  :  'P 1 21/m 1', \
                        ( 3, 'monoclinic',    12  )  :  'C 1 2/m 1', \
                        ( 3, 'monoclinic',    13  )  :  'P 1 2/c 1', \
                        ( 3, 'monoclinic',    14  )  :  'P 1 21/c 1', \
                        ( 3, 'monoclinic',    15  )  :  'C 1 2/c 1', \
                        ( 3, 'orthorhombic',  16  )  :  'P 2 2 2', \
                        ( 3, 'orthorhombic',  17  )  :  'P 2 2 21', \
                        ( 3, 'orthorhombic',  18  )  :  'P 21 21 2', \
                        ( 3, 'orthorhombic',  19  )  :  'P 21 21 21', \
                        ( 3, 'orthorhombic',  20  )  :  'C 2 2 21', \
                        ( 3, 'orthorhombic',  21  )  :  'C 2 2 2', \
                        ( 3, 'orthorhombic',  22  )  :  'F 2 2 2', \
                        ( 3, 'orthorhombic',  23  )  :  'I 2 2 2', \
                        ( 3, 'orthorhombic',  24  )  :  'I 21 21 21', \
                        ( 3, 'orthorhombic',  25  )  :  'P m m 2', \
                        ( 3, 'orthorhombic',  26  )  :  'P m c 21', \
                        ( 3, 'orthorhombic',  27  )  :  'P c c 2', \
                        ( 3, 'orthorhombic',  28  )  :  'P m a 2', \
                        ( 3, 'orthorhombic',  29  )  :  'P c a 21', \
                        ( 3, 'orthorhombic',  30  )  :  'P n c 2', \
                        ( 3, 'orthorhombic',  31  )  :  'P m n 21', \
                        ( 3, 'orthorhombic',  32  )  :  'P b a 2', \
                        ( 3, 'orthorhombic',  33  )  :  'P n a 21', \
                        ( 3, 'orthorhombic',  34  )  :  'P n n 2', \
                        ( 3, 'orthorhombic',  35  )  :  'C m m 2', \
                        ( 3, 'orthorhombic',  36  )  :  'C m c 21', \
                        ( 3, 'orthorhombic',  37  )  :  'C c c 2', \
                        ( 3, 'orthorhombic',  38  )  :  'A m m 2', \
                        ( 3, 'orthorhombic',  39  )  :  'A b m 2', \
                        ( 3, 'orthorhombic',  40  )  :  'A m a 2', \
                        ( 3, 'orthorhombic',  41  )  :  'A b a 2', \
                        ( 3, 'orthorhombic',  42  )  :  'F m m 2', \
                        ( 3, 'orthorhombic',  43  )  :  'F d d 2', \
                        ( 3, 'orthorhombic',  44  )  :  'I m m 2', \
                        ( 3, 'orthorhombic',  45  )  :  'I b a 2', \
                        ( 3, 'orthorhombic',  46  )  :  'I m a 2', \
                        ( 3, 'orthorhombic',  47  )  :  'P m m m', \
                        ( 3, 'orthorhombic',  48  )  :  'P n n n : 2', \
                        ( 3, 'orthorhombic',  49  )  :  'P c c m', \
                        ( 3, 'orthorhombic',  50  )  :  'P b a n : 2', \
                        ( 3, 'orthorhombic',  51  )  :  'P m m a', \
                        ( 3, 'orthorhombic',  52  )  :  'P n n a', \
                        ( 3, 'orthorhombic',  53  )  :  'P m n a', \
                        ( 3, 'orthorhombic',  54  )  :  'P c c a', \
                        ( 3, 'orthorhombic',  55  )  :  'P b a m', \
                        ( 3, 'orthorhombic',  56  )  :  'P c c n', \
                        ( 3, 'orthorhombic',  57  )  :  'P b c m', \
                        ( 3, 'orthorhombic',  58  )  :  'P n n m', \
                        ( 3, 'orthorhombic',  59  )  :  'P m m n : 2', \
                        ( 3, 'orthorhombic',  60  )  :  'P b c n', \
                        ( 3, 'orthorhombic',  61  )  :  'P b c a', \
                        ( 3, 'orthorhombic',  62  )  :  'P n m a', \
                        ( 3, 'orthorhombic',  63  )  :  'C m c m', \
                        ( 3, 'orthorhombic',  64  )  :  'C m c a', \
                        ( 3, 'orthorhombic',  65  )  :  'C m m m', \
                        ( 3, 'orthorhombic',  66  )  :  'C c c m', \
                        ( 3, 'orthorhombic',  67  )  :  'C m m a', \
                        ( 3, 'orthorhombic',  68  )  :  'C c c a : 2', \
                        ( 3, 'orthorhombic',  69  )  :  'F m m m', \
                        ( 3, 'orthorhombic',  70  )  :  'F d d d : 2', \
                        ( 3, 'orthorhombic',  71  )  :  'I m m m', \
                        ( 3, 'orthorhombic',  72  )  :  'I b a m', \
                        ( 3, 'orthorhombic',  73  )  :  'I b c a', \
                        ( 3, 'orthorhombic',  74  )  :  'I m m a', \
                        ( 3, 'tetragonal',    75  )  :  'P 4', \
                        ( 3, 'tetragonal',    76  )  :  'P 41', \
                        ( 3, 'tetragonal',    77  )  :  'P 42', \
                        ( 3, 'tetragonal',    78  )  :  'P 43', \
                        ( 3, 'tetragonal',    79  )  :  'I 4', \
                        ( 3, 'tetragonal',    80  )  :  'I 41', \
                        ( 3, 'tetragonal',    81  )  :  'P -4', \
                        ( 3, 'tetragonal',    82  )  :  'I -4', \
                        ( 3, 'tetragonal',    83  )  :  'P 4/m', \
                        ( 3, 'tetragonal',    84  )  :  'P 42/m', \
                        ( 3, 'tetragonal',    85  )  :  'P 4/n : 2', \
                        ( 3, 'tetragonal',    86  )  :  'P 42/n : 2', \
                        ( 3, 'tetragonal',    87  )  :  'I 4/m', \
                        ( 3, 'tetragonal',    88  )  :  'I 41/a : 2', \
                        ( 3, 'tetragonal',    89  )  :  'P 4 2 2', \
                        ( 3, 'tetragonal',    90  )  :  'P 4 21 2', \
                        ( 3, 'tetragonal',    91  )  :  'P 41 2 2', \
                        ( 3, 'tetragonal',    92  )  :  'P 41 21 2', \
                        ( 3, 'tetragonal',    93  )  :  'P 42 2 2', \
                        ( 3, 'tetragonal',    94  )  :  'P 42 21 2', \
                        ( 3, 'tetragonal',    95  )  :  'P 43 2 2', \
                        ( 3, 'tetragonal',    96  )  :  'P 43 21 2', \
                        ( 3, 'tetragonal',    97  )  :  'I 4 2 2', \
                        ( 3, 'tetragonal',    98  )  :  'I 41 2 2', \
                        ( 3, 'tetragonal',    99  )  :  'P 4 m m', \
                        ( 3, 'tetragonal',    100 )  :  'P 4 b m', \
                        ( 3, 'tetragonal',    101 )  :  'P 42 c m', \
                        ( 3, 'tetragonal',    102 )  :  'P 42 n m', \
                        ( 3, 'tetragonal',    103 )  :  'P 4 c c', \
                        ( 3, 'tetragonal',    104 )  :  'P 4 n c', \
                        ( 3, 'tetragonal',    105 )  :  'P 42 m c', \
                        ( 3, 'tetragonal',    106 )  :  'P 42 b c', \
                        ( 3, 'tetragonal',    107 )  :  'I 4 m m', \
                        ( 3, 'tetragonal',    108 )  :  'I 4 c m', \
                        ( 3, 'tetragonal',    109 )  :  'I 41 m d', \
                        ( 3, 'tetragonal',    110 )  :  'I 41 c d', \
                        ( 3, 'tetragonal',    111 )  :  'P -4 2 m', \
                        ( 3, 'tetragonal',    112 )  :  'P -4 2 c', \
                        ( 3, 'tetragonal',    113 )  :  'P -4 21 m', \
                        ( 3, 'tetragonal',    114 )  :  'P -4 21 c', \
                        ( 3, 'tetragonal',    115 )  :  'P -4 m 2', \
                        ( 3, 'tetragonal',    116 )  :  'P -4 c 2', \
                        ( 3, 'tetragonal',    117 )  :  'P -4 b 2', \
                        ( 3, 'tetragonal',    118 )  :  'P -4 n 2', \
                        ( 3, 'tetragonal',    119 )  :  'I -4 m 2', \
                        ( 3, 'tetragonal',    120 )  :  'I -4 c 2', \
                        ( 3, 'tetragonal',    121 )  :  'I -4 2 m', \
                        ( 3, 'tetragonal',    122 )  :  'I -4 2 d', \
                        ( 3, 'tetragonal',    123 )  :  'P 4/m m m', \
                        ( 3, 'tetragonal',    124 )  :  'P 4/m c c', \
                        ( 3, 'tetragonal',    125 )  :  'P 4/n b m : 2', \
                        ( 3, 'tetragonal',    126 )  :  'P 4/n n c : 2', \
                        ( 3, 'tetragonal',    127 )  :  'P 4/m b m', \
                        ( 3, 'tetragonal',    128 )  :  'P 4/m n c', \
                        ( 3, 'tetragonal',    129 )  :  'P 4/n m m : 2', \
                        ( 3, 'tetragonal',    130 )  :  'P 4/n c c : 2', \
                        ( 3, 'tetragonal',    131 )  :  'P 42/m m c', \
                        ( 3, 'tetragonal',    132 )  :  'P 42/m c m', \
                        ( 3, 'tetragonal',    133 )  :  'P 42/n b c : 2', \
                        ( 3, 'tetragonal',    134 )  :  'P 42/n n m : 2', \
                        ( 3, 'tetragonal',    135 )  :  'P 42/m b c', \
                        ( 3, 'tetragonal',    136 )  :  'P 42/m n m', \
                        ( 3, 'tetragonal',    137 )  :  'P 42/n m c : 2', \
                        ( 3, 'tetragonal',    138 )  :  'P 42/n c m : 2', \
                        ( 3, 'tetragonal',    139 )  :  'I 4/m m m', \
                        ( 3, 'tetragonal',    140 )  :  'I 4/m c m', \
                        ( 3, 'tetragonal',    141 )  :  'I 41/a m d : 2', \
                        ( 3, 'tetragonal',    142 )  :  'I 41/a c d : 2', \
                        ( 3, 'trigonal',      143 )  :  'P 3', \
                        ( 3, 'trigonal',      144 )  :  'P 31', \
                        ( 3, 'trigonal',      145 )  :  'P 32', \
                        ( 3, 'trigonal',      146 )  :  'R 3 : R', \
                        ( 3, 'trigonal',      147 )  :  'P -3', \
                        ( 3, 'trigonal',      148 )  :  'R -3 : R', \
                        ( 3, 'trigonal',      149 )  :  'P 3 1 2', \
                        ( 3, 'trigonal',      150 )  :  'P 3 2 1', \
                        ( 3, 'trigonal',      151 )  :  'P 31 1 2', \
                        ( 3, 'trigonal',      152 )  :  'P 31 2 1', \
                        ( 3, 'trigonal',      153 )  :  'P 32 1 2', \
                        ( 3, 'trigonal',      154 )  :  'P 32 2 1', \
                        ( 3, 'trigonal',      155 )  :  'R 3 2 : R', \
                        ( 3, 'trigonal',      156 )  :  'P 3 m 1', \
                        ( 3, 'trigonal',      157 )  :  'P 3 1 m', \
                        ( 3, 'trigonal',      158 )  :  'P 3 c 1', \
                        ( 3, 'trigonal',      159 )  :  'P 3 1 c', \
                        ( 3, 'trigonal',      160 )  :  'R 3 m : R', \
                        ( 3, 'trigonal',      161 )  :  'R 3 c : R', \
                        ( 3, 'trigonal',      162 )  :  'P -3 1 m', \
                        ( 3, 'trigonal',      163 )  :  'P -3 1 c', \
                        ( 3, 'trigonal',      164 )  :  'P -3 m 1', \
                        ( 3, 'trigonal',      165 )  :  'P -3 c 1', \
                        ( 3, 'trigonal',      166 )  :  'R -3 m : R', \
                        ( 3, 'trigonal',      167 )  :  'R -3 c : R', \
                        ( 3, 'hexagonal',     168 )  :  'P 6', \
                        ( 3, 'hexagonal',     169 )  :  'P 61', \
                        ( 3, 'hexagonal',     170 )  :  'P 65', \
                        ( 3, 'hexagonal',     171 )  :  'P 62', \
                        ( 3, 'hexagonal',     172 )  :  'P 64', \
                        ( 3, 'hexagonal',     173 )  :  'P 63', \
                        ( 3, 'hexagonal',     174 )  :  'P -6', \
                        ( 3, 'hexagonal',     175 )  :  'P 6/m', \
                        ( 3, 'hexagonal',     176 )  :  'P 63/m', \
                        ( 3, 'hexagonal',     177 )  :  'P 6 2 2', \
                        ( 3, 'hexagonal',     178 )  :  'P 61 2 2', \
                        ( 3, 'hexagonal',     179 )  :  'P 65 2 2', \
                        ( 3, 'hexagonal',     180 )  :  'P 62 2 2', \
                        ( 3, 'hexagonal',     181 )  :  'P 64 2 2', \
                        ( 3, 'hexagonal',     182 )  :  'P 63 2 2', \
                        ( 3, 'hexagonal',     183 )  :  'P 6 m m', \
                        ( 3, 'hexagonal',     184 )  :  'P 6 c c', \
                        ( 3, 'hexagonal',     185 )  :  'P 63 c m', \
                        ( 3, 'hexagonal',     186 )  :  'P 63 m c', \
                        ( 3, 'hexagonal',     187 )  :  'P -6 m 2', \
                        ( 3, 'hexagonal',     188 )  :  'P -6 c 2', \
                        ( 3, 'hexagonal',     189 )  :  'P -6 2 m', \
                        ( 3, 'hexagonal',     190 )  :  'P -6 2 c', \
                        ( 3, 'hexagonal',     191 )  :  'P 6/m m m', \
                        ( 3, 'hexagonal',     192 )  :  'P 6/m c c', \
                        ( 3, 'hexagonal',     193 )  :  'P 63/m c m', \
                        ( 3, 'hexagonal',     194 )  :  'P 63/m m c', \
                        ( 3, 'cubic',         195 )  :  'P 2 3', \
                        ( 3, 'cubic',         196 )  :  'F 2 3', \
                        ( 3, 'cubic',         197 )  :  'I 2 3', \
                        ( 3, 'cubic',         198 )  :  'P 21 3', \
                        ( 3, 'cubic',         199 )  :  'I 21 3', \
                        ( 3, 'cubic',         200 )  :  'P m -3', \
                        ( 3, 'cubic',         201 )  :  'P n -3 : 2', \
                        ( 3, 'cubic',         202 )  :  'F m -3', \
                        ( 3, 'cubic',         203 )  :  'F d -3 : 2', \
                        ( 3, 'cubic',         204 )  :  'I m -3', \
                        ( 3, 'cubic',         205 )  :  'P a -3', \
                        ( 3, 'cubic',         206 )  :  'I a -3', \
                        ( 3, 'cubic',         207 )  :  'P 4 3 2', \
                        ( 3, 'cubic',         208 )  :  'P 42 3 2', \
                        ( 3, 'cubic',         209 )  :  'F 4 3 2', \
                        ( 3, 'cubic',         210 )  :  'F 41 3 2', \
                        ( 3, 'cubic',         211 )  :  'I 4 3 2', \
                        ( 3, 'cubic',         212 )  :  'P 43 3 2', \
                        ( 3, 'cubic',         213 )  :  'P 41 3 2', \
                        ( 3, 'cubic',         214 )  :  'I 41 3 2', \
                        ( 3, 'cubic',         215 )  :  'P -4 3 m', \
                        ( 3, 'cubic',         216 )  :  'F -4 3 m', \
                        ( 3, 'cubic',         217 )  :  'I -4 3 m', \
                        ( 3, 'cubic',         218 )  :  'P -4 3 n', \
                        ( 3, 'cubic',         219 )  :  'F -4 3 c', \
                        ( 3, 'cubic',         220 )  :  'I -4 3 d', \
                        ( 3, 'cubic',         221 )  :  'P m -3 m', \
                        ( 3, 'cubic',         222 )  :  'P n -3 n : 2', \
                        ( 3, 'cubic',         223 )  :  'P m -3 n', \
                        ( 3, 'cubic',         224 )  :  'P n -3 m : 2', \
                        ( 3, 'cubic',         225 )  :  'F m -3 m', \
                        ( 3, 'cubic',         226 )  :  'F m -3 c', \
                        ( 3, 'cubic',         227 )  :  'F d -3 m : 2', \
                        ( 3, 'cubic',         228 )  :  'F d -3 c : 2', \
                        ( 3, 'cubic',         229 )  :  'I m -3 m', \
                        ( 3, 'cubic',         230 )  :  'I a -3 d'}

