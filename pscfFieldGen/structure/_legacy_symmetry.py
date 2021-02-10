""" Module defining class to hold symmetry operation data """

from pscfFieldGen.structure.core import POSITION_TOLERANCE

from copy import deepcopy
import enum
import numpy as np
import re
import sympy as sym

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
        matrix : 4x4 numpy array
            The symmetry operation in matrix representation.
        """
        self._dim = dim
        self._matrix = np.array(matrix)
        self._capTranslation()
    
    def __mul__(self,other):
        return self.__matmul__(other)
        
    def __matmul__(self, other):
        if type(other) == type(self):
            if other._dim == self._dim:
                matr = self._matrix @ other._matrix
                return SymmetryOperation(self._dim, matr)
            else:
                raise(ValueError("Cannot multiply SymmetryOperations of differing dimensionality"))
        elif isinstance(other, GeneralPosition):
            if other.dim == self._dim:
                gep = np.array(other)
                #print(gep, type(gep))
                if self._dim == 2:
                    gepAug = np.array([*gep, 0, 1])
                else:
                    gepAug = np.array([*gep, 1])
                #print(gepAug, type(gepAug))
                #print(self._matrix, type(self._matrix))
                result = self._matrix @ gepAug
                result = result[0:self._dim]
                return GeneralPosition(self._dim, result, other.symbol_list)
            else:
                raise(ValueError("Cannot multiply SymmetryOperation and GeneralPosition of different dimensions."))
        elif isinstance(other, np.ndarray):
            if len(other) == self._dim:
                pt = np.array(other)
                pt = pt.resize(4)
                pt[3] = 1
                res = self._matrix @ pt
                res = res[0:len(other)]
                return res
            else:
                raise(ValueError("Dimension mismatch"))
        else:
            return NotImplemented
    
    def __str__(self):
        return str(self._matrix)
    
    def __eq__(self, other):
        if isinstance(other, SymmetryOperation):
            return np.allclose(self._matrix, other._matrix)
    
    def _capTranslation(self):
        testupper = np.isclose(np.ones(3), self._matrix[0:3,3])
        atol = 1E-8 * np.ones(3)
        testlower = np.absolute(self._matrix[0:3,3]) <= atol
        for i in range(3):
            if self._matrix[i,3] >= 1 or testupper[i]:
                self._matrix[i,3] -= 1
            elif self._matrix[i,3] < 0 and not testlower[i]:
                self._matrix[i,3] += 1
            elif self._matrix[i,3] < 0 and testlower[i]:
                self._matrix[i,3] = 0.0
            #else:
                # empty case
                #print("component {} ok".format(i))
    
    @property
    def reverse(self):
        return SymmetryOperation(self._dim, np.linalg.inv(self._matrix))
    
    @classmethod
    def getUnitTranslations(cls, dim):
        op = np.eye(4)
        oplist = []
        for i in range(dim):
            symm = cls(dim,op)
            symm._matrix[i,3] = 1
            oplist.append(symm)
        return oplist

# testing for above classes
if __name__ == "__main__":
    print("Basic Testing Procedures:")
    print("Testing initialization of basic G.E.P.:")
    p = GeneralPosition(3)
    q = GeneralPosition(3)
    print("Testing that default initialization generates equvallent GEPs:", p==q)
    op = np.eye(4)
    print("Testing initialization of identity symmetry operation:")
    identity = SymmetryOperation(3,op)
    ip = identity * p
    print("Applying Identity operation to {}: {}".format(p, ip))
    op[0:3,3] = 0.5
    center = SymmetryOperation(3,op)
    print("Generated body-centering translation operation:\n{}".format(center))
    cp = center * p
    print("Applying centering operation to {}: {}".format(p,cp))
    print("Testing equality of {} and {}: {}".format(ip,cp,ip == cp))
    temp = identity * center
    print("Product of identity and centering operation\n{}\nEquals centering operation: {}".format(temp, temp == center))
    tp = temp * p
    print("Apply above operation to {}: {}; compare to I*p {}; compare to centered {}".format(p,tp, tp == ip, tp == cp))
    print("Comparing first symbols in 2 GEPs: {}".format(p.symbol_list[0] == q.symbol_list[0]))
    
    wrap = center * center
    print("Double application of centering operation gives identity:\n{} {}".format(wrap,wrap == identity))
    
    print("Testing negative translations and wrapping of GEPS:")
    op[0:3,3] = -0.25
    cent = SymmetryOperation(3,op)
    print("Operation",cent)
    wp = center * cp
    print("Apply to centered point: ", wp)
    wrp = cent * cent * p
    print("Apply negative translation twice to {}: {}".format(p, wrp))

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

"""
Group identifying data
    Keys are tuples of (dimension, crystal_system, group_name)
    values are the ID number of the group.
"""
GROUP_ID_BY_NAME = {    ( 2, 'oblique',      'p 1'           )   :  1, \
                        ( 2, 'oblique',      'p 2'           )   :  2, \
                        ( 2, 'rectangular',  'p m'           )   :  3, \
                        ( 2, 'rectangular',  'p g'           )   :  4, \
                        ( 2, 'rectangular',  'c m'           )   :  5, \
                        ( 2, 'rectangular',  'p 2 m m'       )   :  6, \
                        ( 2, 'rectangular',  'p 2 m g'       )   :  7, \
                        ( 2, 'rectangular',  'p 2 g g'       )   :  8, \
                        ( 2, 'rectangular',  'c 2 m m'       )   :  9, \
                        ( 2, 'square',       'p 4'           )   :  10, \
                        ( 2, 'square',       'p 4 m m'       )   :  11, \
                        ( 2, 'square',       'p 4 g m'       )   :  12, \
                        ( 2, 'hexagonal',    'p 3'           )   :  13, \
                        ( 2, 'hexagonal',    'p 3 m 1'       )   :  14, \
                        ( 2, 'hexagonal',    'p 3 1 m'       )   :  15, \
                        ( 2, 'hexagonal',    'p 6'           )   :  16, \
                        ( 2, 'hexagonal',    'p 6 m m'       )   :  17, \
                        ( 3, 'triclinic',    'P 1'           )   :  1, \
                        ( 3, 'triclinic',    'P -1'          )   :  2, \
                        ( 3, 'monoclinic',   'P 1 2 1'       )   :  3, \
                        ( 3, 'monoclinic',   'P 1 21 1'      )   :  4, \
                        ( 3, 'monoclinic',   'C 1 2 1'       )   :  5, \
                        ( 3, 'monoclinic',   'P 1 m 1'       )   :  6, \
                        ( 3, 'monoclinic',   'P 1 c 1'       )   :  7, \
                        ( 3, 'monoclinic',   'C 1 m 1'       )   :  8, \
                        ( 3, 'monoclinic',   'C 1 c 1'       )   :  9, \
                        ( 3, 'monoclinic',   'P 1 2/m 1'     )   :  10, \
                        ( 3, 'monoclinic',   'P 1 21/m 1'    )   :  11, \
                        ( 3, 'monoclinic',   'C 1 2/m 1'     )   :  12, \
                        ( 3, 'monoclinic',   'P 1 2/c 1'     )   :  13, \
                        ( 3, 'monoclinic',   'P 1 21/c 1'    )   :  14, \
                        ( 3, 'monoclinic',   'C 1 2/c 1'     )   :  15, \
                        ( 3, 'orthorhombic', 'P 2 2 2'       )   :  16, \
                        ( 3, 'orthorhombic', 'P 2 2 21'      )   :  17, \
                        ( 3, 'orthorhombic', 'P 21 21 2'     )   :  18, \
                        ( 3, 'orthorhombic', 'P 21 21 21'    )   :  19, \
                        ( 3, 'orthorhombic', 'C 2 2 21'      )   :  20, \
                        ( 3, 'orthorhombic', 'C 2 2 2'       )   :  21, \
                        ( 3, 'orthorhombic', 'F 2 2 2'       )   :  22, \
                        ( 3, 'orthorhombic', 'I 2 2 2'       )   :  23, \
                        ( 3, 'orthorhombic', 'I 21 21 21'    )   :  24, \
                        ( 3, 'orthorhombic', 'P m m 2'       )   :  25, \
                        ( 3, 'orthorhombic', 'P m c 21'      )   :  26, \
                        ( 3, 'orthorhombic', 'P c c 2'       )   :  27, \
                        ( 3, 'orthorhombic', 'P m a 2'       )   :  28, \
                        ( 3, 'orthorhombic', 'P c a 21'      )   :  29, \
                        ( 3, 'orthorhombic', 'P n c 2'       )   :  30, \
                        ( 3, 'orthorhombic', 'P m n 21'      )   :  31, \
                        ( 3, 'orthorhombic', 'P b a 2'       )   :  32, \
                        ( 3, 'orthorhombic', 'P n a 21'      )   :  33, \
                        ( 3, 'orthorhombic', 'P n n 2'       )   :  34, \
                        ( 3, 'orthorhombic', 'C m m 2'       )   :  35, \
                        ( 3, 'orthorhombic', 'C m c 21'      )   :  36, \
                        ( 3, 'orthorhombic', 'C c c 2'       )   :  37, \
                        ( 3, 'orthorhombic', 'A m m 2'       )   :  38, \
                        ( 3, 'orthorhombic', 'A b m 2'       )   :  39, \
                        ( 3, 'orthorhombic', 'A m a 2'       )   :  40, \
                        ( 3, 'orthorhombic', 'A b a 2'       )   :  41, \
                        ( 3, 'orthorhombic', 'F m m 2'       )   :  42, \
                        ( 3, 'orthorhombic', 'F d d 2'       )   :  43, \
                        ( 3, 'orthorhombic', 'I m m 2'       )   :  44, \
                        ( 3, 'orthorhombic', 'I b a 2'       )   :  45, \
                        ( 3, 'orthorhombic', 'I m a 2'       )   :  46, \
                        ( 3, 'orthorhombic', 'P m m m'       )   :  47, \
                        ( 3, 'orthorhombic', 'P n n n : 2'   )   :  48, \
                        ( 3, 'orthorhombic', 'P n n n : 1'   )   :  48, \
                        ( 3, 'orthorhombic', 'P c c m'       )   :  49, \
                        ( 3, 'orthorhombic', 'P b a n : 2'   )   :  50, \
                        ( 3, 'orthorhombic', 'P b a n : 1'   )   :  50, \
                        ( 3, 'orthorhombic', 'P m m a'       )   :  51, \
                        ( 3, 'orthorhombic', 'P n n a'       )   :  52, \
                        ( 3, 'orthorhombic', 'P m n a'       )   :  53, \
                        ( 3, 'orthorhombic', 'P c c a'       )   :  54, \
                        ( 3, 'orthorhombic', 'P b a m'       )   :  55, \
                        ( 3, 'orthorhombic', 'P c c n'       )   :  56, \
                        ( 3, 'orthorhombic', 'P b c m'       )   :  57, \
                        ( 3, 'orthorhombic', 'P n n m'       )   :  58, \
                        ( 3, 'orthorhombic', 'P m m n : 2'   )   :  59, \
                        ( 3, 'orthorhombic', 'P m m n : 1'   )   :  59, \
                        ( 3, 'orthorhombic', 'P b c n'       )   :  60, \
                        ( 3, 'orthorhombic', 'P b c a'       )   :  61, \
                        ( 3, 'orthorhombic', 'P n m a'       )   :  62, \
                        ( 3, 'orthorhombic', 'C m c m'       )   :  63, \
                        ( 3, 'orthorhombic', 'C m c a'       )   :  64, \
                        ( 3, 'orthorhombic', 'C m m m'       )   :  65, \
                        ( 3, 'orthorhombic', 'C c c m'       )   :  66, \
                        ( 3, 'orthorhombic', 'C m m a'       )   :  67, \
                        ( 3, 'orthorhombic', 'C c c a : 2'   )   :  68, \
                        ( 3, 'orthorhombic', 'C c c a : 1'   )   :  68, \
                        ( 3, 'orthorhombic', 'F m m m'       )   :  69, \
                        ( 3, 'orthorhombic', 'F d d d : 2'   )   :  70, \
                        ( 3, 'orthorhombic', 'F d d d : 1'   )   :  70, \
                        ( 3, 'orthorhombic', 'I m m m'       )   :  71, \
                        ( 3, 'orthorhombic', 'I b a m'       )   :  72, \
                        ( 3, 'orthorhombic', 'I b c a'       )   :  73, \
                        ( 3, 'orthorhombic', 'I m m a'       )   :  74, \
                        ( 3, 'tetragonal',   'P 4'           )   :  75, \
                        ( 3, 'tetragonal',   'P 41'          )   :  76, \
                        ( 3, 'tetragonal',   'P 42'          )   :  77, \
                        ( 3, 'tetragonal',   'P 43'          )   :  78, \
                        ( 3, 'tetragonal',   'I 4'           )   :  79, \
                        ( 3, 'tetragonal',   'I 41'          )   :  80, \
                        ( 3, 'tetragonal',   'P -4'          )   :  81, \
                        ( 3, 'tetragonal',   'I -4'          )   :  82, \
                        ( 3, 'tetragonal',   'P 4/m'         )   :  83, \
                        ( 3, 'tetragonal',   'P 42/m'        )   :  84, \
                        ( 3, 'tetragonal',   'P 4/n : 2'     )   :  85, \
                        ( 3, 'tetragonal',   'P 4/n : 1'     )   :  85, \
                        ( 3, 'tetragonal',   'P 42/n : 2'    )   :  86, \
                        ( 3, 'tetragonal',   'P 42/n : 1'    )   :  86, \
                        ( 3, 'tetragonal',   'I 4/m'         )   :  87, \
                        ( 3, 'tetragonal',   'I 41/a : 2'    )   :  88, \
                        ( 3, 'tetragonal',   'I 41/a : 1'    )   :  88, \
                        ( 3, 'tetragonal',   'P 4 2 2'       )   :  89, \
                        ( 3, 'tetragonal',   'P 4 21 2'      )   :  90, \
                        ( 3, 'tetragonal',   'P 41 2 2'      )   :  91, \
                        ( 3, 'tetragonal',   'P 41 21 2'     )   :  92, \
                        ( 3, 'tetragonal',   'P 42 2 2'      )   :  93, \
                        ( 3, 'tetragonal',   'P 42 21 2'     )   :  94, \
                        ( 3, 'tetragonal',   'P 43 2 2'      )   :  95, \
                        ( 3, 'tetragonal',   'P 43 21 2'     )   :  96, \
                        ( 3, 'tetragonal',   'I 4 2 2'       )   :  97, \
                        ( 3, 'tetragonal',   'I 41 2 2'      )   :  98, \
                        ( 3, 'tetragonal',   'P 4 m m'       )   :  99, \
                        ( 3, 'tetragonal',   'P 4 b m'       )   :  100, \
                        ( 3, 'tetragonal',   'P 42 c m'      )   :  101, \
                        ( 3, 'tetragonal',   'P 42 n m'      )   :  102, \
                        ( 3, 'tetragonal',   'P 4 c c'       )   :  103, \
                        ( 3, 'tetragonal',   'P 4 n c'       )   :  104, \
                        ( 3, 'tetragonal',   'P 42 m c'      )   :  105, \
                        ( 3, 'tetragonal',   'P 42 b c'      )   :  106, \
                        ( 3, 'tetragonal',   'I 4 m m'       )   :  107, \
                        ( 3, 'tetragonal',   'I 4 c m'       )   :  108, \
                        ( 3, 'tetragonal',   'I 41 m d'      )   :  109, \
                        ( 3, 'tetragonal',   'I 41 c d'      )   :  110, \
                        ( 3, 'tetragonal',   'P -4 2 m'      )   :  111, \
                        ( 3, 'tetragonal',   'P -4 2 c'      )   :  112, \
                        ( 3, 'tetragonal',   'P -4 21 m'     )   :  113, \
                        ( 3, 'tetragonal',   'P -4 21 c'     )   :  114, \
                        ( 3, 'tetragonal',   'P -4 m 2'      )   :  115, \
                        ( 3, 'tetragonal',   'P -4 c 2'      )   :  116, \
                        ( 3, 'tetragonal',   'P -4 b 2'      )   :  117, \
                        ( 3, 'tetragonal',   'P -4 n 2'      )   :  118, \
                        ( 3, 'tetragonal',   'I -4 m 2'      )   :  119, \
                        ( 3, 'tetragonal',   'I -4 c 2'      )   :  120, \
                        ( 3, 'tetragonal',   'I -4 2 m'      )   :  121, \
                        ( 3, 'tetragonal',   'I -4 2 d'      )   :  122, \
                        ( 3, 'tetragonal',   'P 4/m m m'     )   :  123, \
                        ( 3, 'tetragonal',   'P 4/m c c'     )   :  124, \
                        ( 3, 'tetragonal',   'P 4/n b m : 2' )   :  125, \
                        ( 3, 'tetragonal',   'P 4/n b m : 1' )   :  125, \
                        ( 3, 'tetragonal',   'P 4/n n c : 2' )   :  126, \
                        ( 3, 'tetragonal',   'P 4/n n c : 1' )   :  126, \
                        ( 3, 'tetragonal',   'P 4/m b m'     )   :  127, \
                        ( 3, 'tetragonal',   'P 4/m n c'     )   :  128, \
                        ( 3, 'tetragonal',   'P 4/n m m : 2' )   :  129, \
                        ( 3, 'tetragonal',   'P 4/n m m : 1' )   :  129, \
                        ( 3, 'tetragonal',   'P 4/n c c : 2' )   :  130, \
                        ( 3, 'tetragonal',   'P 4/n c c : 1' )   :  130, \
                        ( 3, 'tetragonal',   'P 42/m m c'    )   :  131, \
                        ( 3, 'tetragonal',   'P 42/m c m'    )   :  132, \
                        ( 3, 'tetragonal',   'P 42/n b c : 2')   :  133, \
                        ( 3, 'tetragonal',   'P 42/n b c : 1')   :  133, \
                        ( 3, 'tetragonal',   'P 42/n n m : 2')   :  134, \
                        ( 3, 'tetragonal',   'P 42/n n m : 1')   :  134, \
                        ( 3, 'tetragonal',   'P 42/m b c'    )   :  135, \
                        ( 3, 'tetragonal',   'P 42/m n m'    )   :  136, \
                        ( 3, 'tetragonal',   'P 42/n m c : 2')   :  137, \
                        ( 3, 'tetragonal',   'P 42/n m c : 1')   :  137, \
                        ( 3, 'tetragonal',   'P 42/n c m : 2')   :  138, \
                        ( 3, 'tetragonal',   'P 42/n c m : 1')   :  138, \
                        ( 3, 'tetragonal',   'I 4/m m m'     )   :  139, \
                        ( 3, 'tetragonal',   'I 4/m c m'     )   :  140, \
                        ( 3, 'tetragonal',   'I 41/a m d : 2')   :  141, \
                        ( 3, 'tetragonal',   'I 41/a m d : 1')   :  141, \
                        ( 3, 'tetragonal',   'I 41/a c d : 2')   :  142, \
                        ( 3, 'tetragonal',   'I 41/a c d : 1')   :  142, \
                        ( 3, 'trigonal',     'P 3'           )   :  143, \
                        ( 3, 'trigonal',     'P 31'          )   :  144, \
                        ( 3, 'trigonal',     'P 32'          )   :  145, \
                        ( 3, 'trigonal',     'R 3 : H'       )   :  146, \
                        ( 3, 'trigonal',     'R 3 : R'       )   :  146, \
                        ( 3, 'trigonal',     'P -3'          )   :  147, \
                        ( 3, 'trigonal',     'R -3 : H'      )   :  148, \
                        ( 3, 'trigonal',     'R -3 : R'      )   :  148, \
                        ( 3, 'trigonal',     'P 3 1 2'       )   :  149, \
                        ( 3, 'trigonal',     'P 3 2 1'       )   :  150, \
                        ( 3, 'trigonal',     'P 31 1 2'      )   :  151, \
                        ( 3, 'trigonal',     'P 31 2 1'      )   :  152, \
                        ( 3, 'trigonal',     'P 32 1 2'      )   :  153, \
                        ( 3, 'trigonal',     'P 32 2 1'      )   :  154, \
                        ( 3, 'trigonal',     'R 3 2 : H'     )   :  155, \
                        ( 3, 'trigonal',     'R 3 2 : R'     )   :  155, \
                        ( 3, 'trigonal',     'P 3 m 1'       )   :  156, \
                        ( 3, 'trigonal',     'P 3 1 m'       )   :  157, \
                        ( 3, 'trigonal',     'P 3 c 1'       )   :  158, \
                        ( 3, 'trigonal',     'P 3 1 c'       )   :  159, \
                        ( 3, 'trigonal',     'R 3 m : H'     )   :  160, \
                        ( 3, 'trigonal',     'R 3 m : R'     )   :  160, \
                        ( 3, 'trigonal',     'R 3 c : H'     )   :  161, \
                        ( 3, 'trigonal',     'R 3 c : R'     )   :  161, \
                        ( 3, 'trigonal',     'P -3 1 m'      )   :  162, \
                        ( 3, 'trigonal',     'P -3 1 c'      )   :  163, \
                        ( 3, 'trigonal',     'P -3 m 1'      )   :  164, \
                        ( 3, 'trigonal',     'P -3 c 1'      )   :  165, \
                        ( 3, 'trigonal',     'R -3 m : H'    )   :  166, \
                        ( 3, 'trigonal',     'R -3 m : R'    )   :  166, \
                        ( 3, 'trigonal',     'R -3 c : H'    )   :  167, \
                        ( 3, 'trigonal',     'R -3 c : R'    )   :  167, \
                        ( 3, 'hexagonal',    'P 6'           )   :  168, \
                        ( 3, 'hexagonal',    'P 61'          )   :  169, \
                        ( 3, 'hexagonal',    'P 65'          )   :  170, \
                        ( 3, 'hexagonal',    'P 62'          )   :  171, \
                        ( 3, 'hexagonal',    'P 64'          )   :  172, \
                        ( 3, 'hexagonal',    'P 63'          )   :  173, \
                        ( 3, 'hexagonal',    'P -6'          )   :  174, \
                        ( 3, 'hexagonal',    'P 6/m'         )   :  175, \
                        ( 3, 'hexagonal',    'P 63/m'        )   :  176, \
                        ( 3, 'hexagonal',    'P 6 2 2'       )   :  177, \
                        ( 3, 'hexagonal',    'P 61 2 2'      )   :  178, \
                        ( 3, 'hexagonal',    'P 65 2 2'      )   :  179, \
                        ( 3, 'hexagonal',    'P 62 2 2'      )   :  180, \
                        ( 3, 'hexagonal',    'P 64 2 2'      )   :  181, \
                        ( 3, 'hexagonal',    'P 63 2 2'      )   :  182, \
                        ( 3, 'hexagonal',    'P 6 m m'       )   :  183, \
                        ( 3, 'hexagonal',    'P 6 c c'       )   :  184, \
                        ( 3, 'hexagonal',    'P 63 c m'      )   :  185, \
                        ( 3, 'hexagonal',    'P 63 m c'      )   :  186, \
                        ( 3, 'hexagonal',    'P -6 m 2'      )   :  187, \
                        ( 3, 'hexagonal',    'P -6 c 2'      )   :  188, \
                        ( 3, 'hexagonal',    'P -6 2 m'      )   :  189, \
                        ( 3, 'hexagonal',    'P -6 2 c'      )   :  190, \
                        ( 3, 'hexagonal',    'P 6/m m m'     )   :  191, \
                        ( 3, 'hexagonal',    'P 6/m c c'     )   :  192, \
                        ( 3, 'hexagonal',    'P 63/m c m'    )   :  193, \
                        ( 3, 'hexagonal',    'P 63/m m c'    )   :  194, \
                        ( 3, 'cubic',        'P 2 3'         )   :  195, \
                        ( 3, 'cubic',        'F 2 3'         )   :  196, \
                        ( 3, 'cubic',        'I 2 3'         )   :  197, \
                        ( 3, 'cubic',        'P 21 3'        )   :  198, \
                        ( 3, 'cubic',        'I 21 3'        )   :  199, \
                        ( 3, 'cubic',        'P m -3'        )   :  200, \
                        ( 3, 'cubic',        'P n -3 : 2'    )   :  201, \
                        ( 3, 'cubic',        'P n -3 : 1'    )   :  201, \
                        ( 3, 'cubic',        'F m -3'        )   :  202, \
                        ( 3, 'cubic',        'F d -3 : 2'    )   :  203, \
                        ( 3, 'cubic',        'F d -3 : 1'    )   :  203, \
                        ( 3, 'cubic',        'I m -3'        )   :  204, \
                        ( 3, 'cubic',        'P a -3'        )   :  205, \
                        ( 3, 'cubic',        'I a -3'        )   :  206, \
                        ( 3, 'cubic',        'P 4 3 2'       )   :  207, \
                        ( 3, 'cubic',        'P 42 3 2'      )   :  208, \
                        ( 3, 'cubic',        'F 4 3 2'       )   :  209, \
                        ( 3, 'cubic',        'F 41 3 2'      )   :  210, \
                        ( 3, 'cubic',        'I 4 3 2'       )   :  211, \
                        ( 3, 'cubic',        'P 43 3 2'      )   :  212, \
                        ( 3, 'cubic',        'P 41 3 2'      )   :  213, \
                        ( 3, 'cubic',        'I 41 3 2'      )   :  214, \
                        ( 3, 'cubic',        'P -4 3 m'      )   :  215, \
                        ( 3, 'cubic',        'F -4 3 m'      )   :  216, \
                        ( 3, 'cubic',        'I -4 3 m'      )   :  217, \
                        ( 3, 'cubic',        'P -4 3 n'      )   :  218, \
                        ( 3, 'cubic',        'F -4 3 c'      )   :  219, \
                        ( 3, 'cubic',        'I -4 3 d'      )   :  220, \
                        ( 3, 'cubic',        'P m -3 m'      )   :  221, \
                        ( 3, 'cubic',        'P n -3 n : 2'  )   :  222, \
                        ( 3, 'cubic',        'P n -3 n : 1'  )   :  222, \
                        ( 3, 'cubic',        'P m -3 n'      )   :  223, \
                        ( 3, 'cubic',        'P n -3 m : 2'  )   :  224, \
                        ( 3, 'cubic',        'P n -3 m : 1'  )   :  224, \
                        ( 3, 'cubic',        'F m -3 m'      )   :  225, \
                        ( 3, 'cubic',        'F m -3 c'      )   :  226, \
                        ( 3, 'cubic',        'F d -3 m : 2'  )   :  227, \
                        ( 3, 'cubic',        'F d -3 m : 1'  )   :  227, \
                        ( 3, 'cubic',        'F d -3 c : 2'  )   :  228, \
                        ( 3, 'cubic',        'F d -3 c : 1'  )   :  228, \
                        ( 3, 'cubic',        'I m -3 m'      )   :  229, \
                        ( 3, 'cubic',        'I a -3 d'      )   :  230 }

"""
Group identifying data.
    Keys are tuples of (dimension, crystal_system, ID).
    Values are the group names.
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
                        ( 3, 'orthorhombic',  48  )  :  'P n n n : 1', \
                        ( 3, 'orthorhombic',  49  )  :  'P c c m', \
                        ( 3, 'orthorhombic',  50  )  :  'P b a n : 2', \
                        ( 3, 'orthorhombic',  50  )  :  'P b a n : 1', \
                        ( 3, 'orthorhombic',  51  )  :  'P m m a', \
                        ( 3, 'orthorhombic',  52  )  :  'P n n a', \
                        ( 3, 'orthorhombic',  53  )  :  'P m n a', \
                        ( 3, 'orthorhombic',  54  )  :  'P c c a', \
                        ( 3, 'orthorhombic',  55  )  :  'P b a m', \
                        ( 3, 'orthorhombic',  56  )  :  'P c c n', \
                        ( 3, 'orthorhombic',  57  )  :  'P b c m', \
                        ( 3, 'orthorhombic',  58  )  :  'P n n m', \
                        ( 3, 'orthorhombic',  59  )  :  'P m m n : 2', \
                        ( 3, 'orthorhombic',  59  )  :  'P m m n : 1', \
                        ( 3, 'orthorhombic',  60  )  :  'P b c n', \
                        ( 3, 'orthorhombic',  61  )  :  'P b c a', \
                        ( 3, 'orthorhombic',  62  )  :  'P n m a', \
                        ( 3, 'orthorhombic',  63  )  :  'C m c m', \
                        ( 3, 'orthorhombic',  64  )  :  'C m c a', \
                        ( 3, 'orthorhombic',  65  )  :  'C m m m', \
                        ( 3, 'orthorhombic',  66  )  :  'C c c m', \
                        ( 3, 'orthorhombic',  67  )  :  'C m m a', \
                        ( 3, 'orthorhombic',  68  )  :  'C c c a : 2', \
                        ( 3, 'orthorhombic',  68  )  :  'C c c a : 1', \
                        ( 3, 'orthorhombic',  69  )  :  'F m m m', \
                        ( 3, 'orthorhombic',  70  )  :  'F d d d : 2', \
                        ( 3, 'orthorhombic',  70  )  :  'F d d d : 1', \
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
                        ( 3, 'tetragonal',    85  )  :  'P 4/n : 1', \
                        ( 3, 'tetragonal',    86  )  :  'P 42/n : 2', \
                        ( 3, 'tetragonal',    86  )  :  'P 42/n : 1', \
                        ( 3, 'tetragonal',    87  )  :  'I 4/m', \
                        ( 3, 'tetragonal',    88  )  :  'I 41/a : 2', \
                        ( 3, 'tetragonal',    88  )  :  'I 41/a : 1', \
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
                        ( 3, 'tetragonal',    125 )  :  'P 4/n b m : 1', \
                        ( 3, 'tetragonal',    126 )  :  'P 4/n n c : 2', \
                        ( 3, 'tetragonal',    126 )  :  'P 4/n n c : 1', \
                        ( 3, 'tetragonal',    127 )  :  'P 4/m b m', \
                        ( 3, 'tetragonal',    128 )  :  'P 4/m n c', \
                        ( 3, 'tetragonal',    129 )  :  'P 4/n m m : 2', \
                        ( 3, 'tetragonal',    129 )  :  'P 4/n m m : 1', \
                        ( 3, 'tetragonal',    130 )  :  'P 4/n c c : 2', \
                        ( 3, 'tetragonal',    130 )  :  'P 4/n c c : 1', \
                        ( 3, 'tetragonal',    131 )  :  'P 42/m m c', \
                        ( 3, 'tetragonal',    132 )  :  'P 42/m c m', \
                        ( 3, 'tetragonal',    133 )  :  'P 42/n b c : 2', \
                        ( 3, 'tetragonal',    133 )  :  'P 42/n b c : 1', \
                        ( 3, 'tetragonal',    134 )  :  'P 42/n n m : 2', \
                        ( 3, 'tetragonal',    134 )  :  'P 42/n n m : 1', \
                        ( 3, 'tetragonal',    135 )  :  'P 42/m b c', \
                        ( 3, 'tetragonal',    136 )  :  'P 42/m n m', \
                        ( 3, 'tetragonal',    137 )  :  'P 42/n m c : 2', \
                        ( 3, 'tetragonal',    137 )  :  'P 42/n m c : 1', \
                        ( 3, 'tetragonal',    138 )  :  'P 42/n c m : 2', \
                        ( 3, 'tetragonal',    138 )  :  'P 42/n c m : 1', \
                        ( 3, 'tetragonal',    139 )  :  'I 4/m m m', \
                        ( 3, 'tetragonal',    140 )  :  'I 4/m c m', \
                        ( 3, 'tetragonal',    141 )  :  'I 41/a m d : 2', \
                        ( 3, 'tetragonal',    141 )  :  'I 41/a m d : 1', \
                        ( 3, 'tetragonal',    142 )  :  'I 41/a c d : 2', \
                        ( 3, 'tetragonal',    142 )  :  'I 41/a c d : 1', \
                        ( 3, 'trigonal',      143 )  :  'P 3', \
                        ( 3, 'trigonal',      144 )  :  'P 31', \
                        ( 3, 'trigonal',      145 )  :  'P 32', \
                        ( 3, 'trigonal',      146 )  :  'R 3 : H', \
                        ( 3, 'trigonal',      146 )  :  'R 3 : R', \
                        ( 3, 'trigonal',      147 )  :  'P -3', \
                        ( 3, 'trigonal',      148 )  :  'R -3 : H', \
                        ( 3, 'trigonal',      148 )  :  'R -3 : R', \
                        ( 3, 'trigonal',      149 )  :  'P 3 1 2', \
                        ( 3, 'trigonal',      150 )  :  'P 3 2 1', \
                        ( 3, 'trigonal',      151 )  :  'P 31 1 2', \
                        ( 3, 'trigonal',      152 )  :  'P 31 2 1', \
                        ( 3, 'trigonal',      153 )  :  'P 32 1 2', \
                        ( 3, 'trigonal',      154 )  :  'P 32 2 1', \
                        ( 3, 'trigonal',      155 )  :  'R 3 2 : H', \
                        ( 3, 'trigonal',      155 )  :  'R 3 2 : R', \
                        ( 3, 'trigonal',      156 )  :  'P 3 m 1', \
                        ( 3, 'trigonal',      157 )  :  'P 3 1 m', \
                        ( 3, 'trigonal',      158 )  :  'P 3 c 1', \
                        ( 3, 'trigonal',      159 )  :  'P 3 1 c', \
                        ( 3, 'trigonal',      160 )  :  'R 3 m : H', \
                        ( 3, 'trigonal',      160 )  :  'R 3 m : R', \
                        ( 3, 'trigonal',      161 )  :  'R 3 c : H', \
                        ( 3, 'trigonal',      161 )  :  'R 3 c : R', \
                        ( 3, 'trigonal',      162 )  :  'P -3 1 m', \
                        ( 3, 'trigonal',      163 )  :  'P -3 1 c', \
                        ( 3, 'trigonal',      164 )  :  'P -3 m 1', \
                        ( 3, 'trigonal',      165 )  :  'P -3 c 1', \
                        ( 3, 'trigonal',      166 )  :  'R -3 m : H', \
                        ( 3, 'trigonal',      166 )  :  'R -3 m : R', \
                        ( 3, 'trigonal',      167 )  :  'R -3 c : H', \
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
                        ( 3, 'cubic',         201 )  :  'P n -3 : 1', \
                        ( 3, 'cubic',         202 )  :  'F m -3', \
                        ( 3, 'cubic',         203 )  :  'F d -3 : 2', \
                        ( 3, 'cubic',         203 )  :  'F d -3 : 1', \
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
                        ( 3, 'cubic',         222 )  :  'P n -3 n : 1', \
                        ( 3, 'cubic',         223 )  :  'P m -3 n', \
                        ( 3, 'cubic',         224 )  :  'P n -3 m : 2', \
                        ( 3, 'cubic',         224 )  :  'P n -3 m : 1', \
                        ( 3, 'cubic',         225 )  :  'F m -3 m', \
                        ( 3, 'cubic',         226 )  :  'F m -3 c', \
                        ( 3, 'cubic',         227 )  :  'F d -3 m : 2', \
                        ( 3, 'cubic',         227 )  :  'F d -3 m : 1', \
                        ( 3, 'cubic',         228 )  :  'F d -3 c : 2', \
                        ( 3, 'cubic',         228 )  :  'F d -3 c : 1', \
                        ( 3, 'cubic',         229 )  :  'I m -3 m', \
                        ( 3, 'cubic',         230 )  :  'I a -3 d'}

## Internal method for creating the point symmetry generators. (4x4)
def _make_psm(members):
    """ Return a 4x4 point symmetry generator, with 0-translation component.
    
    Parameters
    ----------
    members : iterable set of 3-element tuples
        In each tuple, the first element is a value to place in the matrix.
        The last two elements in the tuple are the row and column indices
        at which the value is to be placed (each in range 0 < i,j < 2)
    
    Returns
    -------
    4x4 numpy array with values specified in members, and 0 elsewhere.
    """
    out = np.zeros((4,4))
    for (v,i,j) in members:
        out[i,j] = v
    out[:,3] = 0.
    out[3,:] = 0.
    out[3,3] = 1.
    return out

"""
Point Symmetry Generator Matrices.
    Generators a-n are from: de Graef and McHenry, Structure of Materials, 2nd edition
    
    Generator z was added for plane-groups based on definitions from the International Tables of Crystallography, Vol A.
    
    Plane groups are treated as space groups with unit normal as the third axis, and no 3rd axis translation.
    
    guide for generators
        a   Identity
        b   2-fold rotation, axis on *c*
        c   2-fold rotation, axis on *b*
        d   3-fold rotation, axis on [1,1,1] (determined from cubic)
        e   2-fold rotation, axis on [1,1,0] (determined from hexagonal trials)
        f
        g   4-fold rotation, axiz on *c* (at least in tetragonal)
        h   Inversion
        i   Mirror over *a*-*b* plane
        j   Mirror over *a*-*c* plane
        k   Mirror over (2x,x)-*c* plane (from hexagonal plane groups)
        l   Mirror over *b*-*c* plane (from hexagonal lattices)
        m
        n   3-fold rotation, axis on *c* (at least in hexagonal)
        inv Inversion operator
"""
POINT_SYMMETRY_GENERATORS = {   'a' : _make_psm( [ ( 1,0,0), ( 1,1,1), ( 1,2,2) ] ), \
                                'b' : _make_psm( [ (-1,0,0), (-1,1,1), ( 1,2,2) ] ), \
                                'c' : _make_psm( [ (-1,0,0), ( 1,1,1), (-1,2,2) ] ), \
                                'd' : _make_psm( [ ( 1,0,2), ( 1,1,0), ( 1,2,1) ] ), \
                                'e' : _make_psm( [ ( 1,0,1), ( 1,1,0), (-1,2,2) ] ), \
                                'f' : _make_psm( [ (-1,0,1), (-1,1,0), (-1,2,2) ] ), \
                                'g' : _make_psm( [ (-1,0,1), ( 1,1,0), ( 1,2,2) ] ), \
                                'h' : _make_psm( [ (-1,0,0), (-1,1,1), (-1,2,2) ] ), \
                                'i' : _make_psm( [ ( 1,0,0), ( 1,1,1), (-1,2,2) ] ), \
                                'j' : _make_psm( [ ( 1,0,0), (-1,1,1), ( 1,2,2) ] ), \
                                'k' : _make_psm( [ (-1,0,1), (-1,1,0), ( 1,2,2) ] ), \
                                'l' : _make_psm( [ ( 1,0,1), ( 1,1,0), ( 1,2,2) ] ), \
                                'm' : _make_psm( [ ( 1,0,1), (-1,1,0), (-1,2,2) ] ), \
                                'n' : _make_psm( [ (-1,0,1), (-1,1,1), ( 1,2,2), (1,1,0) ] ), \
                                'z' : _make_psm( [ (-1,0,0), ( 1,1,1), ( 1,2,2) ] ), \
                                'inv' : _make_psm( [ (-1,0,0), (-1,1,1), (-1,2,2) ] ) }

"""
Fractional coordinates of generator matrix translations.
Nomenclature and values from de Graef and McHenry, Structure of Materials, 2nd edition
"""
TRANSLATION_COMPONENTS = {  'A' : 1./6, \
                            'B' : 1./4, \
                            'C' : 1./3, \
                            'D' : 1./2, \
                            'E' : 2./3, \
                            'F' : 3./4, \
                            'G' : 5./6, \
                            'O' : 0.0, \
                            'X' : -3./8, \
                            'Y' : -1./4, \
                            'Z' : -1./8 }

"""
Dictionary of generator strings for each group.
    Keys are the tuple (dim, 'system', 'name') where:
        dim is the dimension of the group.
        'system' is a string giving the name of the crystal system (as the lattice used for the basis vectors)
        as defined by the PSCF documentation.
        'name' is the name of the space group, formatted according to specifications on PSCF website.
    
    Sources:
    Space group strings modified from the website for de Graef and McHenry, Structure of Materials, 2nd edition.
    Plane group strings were determined by the developer based on data in ITC-VolA.
"""
GENERATOR_STRINGS = {   \
    ( 2,  'oblique'       ,  'p 1'             )  :  '000'                                , \
    ( 2,  'oblique'       ,  'p 2'             )  :  '01bOOO0'                            , \
    ( 2,  'rectangular'   ,  'p m'             )  :  '01zOOO0'                            , \
    ( 2,  'rectangular'   ,  'p g'             )  :  '01zODO0'                            , \
    ( 2,  'rectangular'   ,  'c m'             )  :  '02aDDOzOOO0'                        , \
    ( 2,  'rectangular'   ,  'p 2 m m'         )  :  '02bOOOzOOO0'                        , \
    ( 2,  'rectangular'   ,  'p 2 m g'         )  :  '02bOOOzDOO0'                        , \
    ( 2,  'rectangular'   ,  'p 2 g g'         )  :  '02bOOOzDDO0'                        , \
    ( 2,  'rectangular'   ,  'c 2 m m'         )  :  '03aDDObOOOzOOO0'                    , \
    ( 2,  'square'        ,  'p 4'             )  :  '02bOOOgOOO0'                        , \
    ( 2,  'square'        ,  'p 4 m m'         )  :  '03bOOOgOOOzOOO0'                    , \
    ( 2,  'square'        ,  'p 4 g m'         )  :  '03bOOOgOOOzDDO0'                    , \
    ( 2,  'hexagonal'     ,  'p 3'             )  :  '01zOOO0'                            , \
    ( 2,  'hexagonal'     ,  'p 3 m 1'         )  :  '02nOOOkOOO0'                        , \
    ( 2,  'hexagonal'     ,  'p 3 1 m'         )  :  '02nOOOlOOO0'                        , \
    ( 2,  'hexagonal'     ,  'p 6'             )  :  '02nOOObOOO0'                        , \
    ( 2,  'hexagonal'     ,  'p 6 m m'         )  :  '03nOOObOOOlOOO0'                    , \
    ( 3,  'triclinic'     ,  'P 1'             )  :  '000'                                , \
    ( 3,  'triclinic'     ,  'P -1'            )  :  '100'                                , \
    ( 3,  'monoclinic'    ,  'P 1 2 1'         )  :  '01cOOO0'                            , \
    ( 3,  'monoclinic'    ,  'P 1 21 1'        )  :  '01cODO0'                            , \
    ( 3,  'monoclinic'    ,  'C 1 2 1'         )  :  '02aDDOcOOO0'                        , \
    ( 3,  'monoclinic'    ,  'P 1 m 1'         )  :  '01jOOO0'                            , \
    ( 3,  'monoclinic'    ,  'P 1 c 1'         )  :  '01jOOD0'                            , \
    ( 3,  'monoclinic'    ,  'C 1 m 1'         )  :  '02aDDOjOOO0'                        , \
    ( 3,  'monoclinic'    ,  'C 1 c 1'         )  :  '02aDDOjOOD0'                        , \
    ( 3,  'monoclinic'    ,  'P 1 2/m 1'       )  :  '11cOOO0'                            , \
    ( 3,  'monoclinic'    ,  'P 1 21/m 1'      )  :  '11cODO0'                            , \
    ( 3,  'monoclinic'    ,  'C 1 2/m 1'       )  :  '12aDDOcOOO0'                        , \
    ( 3,  'monoclinic'    ,  'P 1 2/c 1'       )  :  '11cOOD0'                            , \
    ( 3,  'monoclinic'    ,  'P 1 21/c 1'      )  :  '11cODD0'                            , \
    ( 3,  'monoclinic'    ,  'C 1 2/c 1'       )  :  '12aDDOcOOD0'                        , \
    ( 3,  'orthorhombic'  ,  'P 2 2 2'         )  :  '02bOOOcOOO0'                        , \
    ( 3,  'orthorhombic'  ,  'P 2 2 21'        )  :  '02bOODcOOD0'                        , \
    ( 3,  'orthorhombic'  ,  'P 21 21 2'       )  :  '02bOOOcDDO0'                        , \
    ( 3,  'orthorhombic'  ,  'P 21 21 21'      )  :  '02bDODcODD0'                        , \
    ( 3,  'orthorhombic'  ,  'C 2 2 21'        )  :  '03aDDObOODcOOD0'                    , \
    ( 3,  'orthorhombic'  ,  'C 2 2 2'         )  :  '03aDDObOOOcOOO0'                    , \
    ( 3,  'orthorhombic'  ,  'F 2 2 2'         )  :  '04aODDaDODbOOOcOOO0'                , \
    ( 3,  'orthorhombic'  ,  'I 2 2 2'         )  :  '03aDDDbOOOcOOO0'                    , \
    ( 3,  'orthorhombic'  ,  'I 21 21 21'      )  :  '03aDDDbDODcODD0'                    , \
    ( 3,  'orthorhombic'  ,  'P m m 2'         )  :  '02bOOOjOOO0'                        , \
    ( 3,  'orthorhombic'  ,  'P m c 21'        )  :  '02bOODjOOD0'                        , \
    ( 3,  'orthorhombic'  ,  'P c c 2'         )  :  '02bOOOjOOD0'                        , \
    ( 3,  'orthorhombic'  ,  'P m a 2'         )  :  '02bOOOjDOO0'                        , \
    ( 3,  'orthorhombic'  ,  'P c a 21'        )  :  '02bOODjDOO0'                        , \
    ( 3,  'orthorhombic'  ,  'P n c 2'         )  :  '02bOOOjODD0'                        , \
    ( 3,  'orthorhombic'  ,  'P m n 21'        )  :  '02bDODjDOD0'                        , \
    ( 3,  'orthorhombic'  ,  'P b a 2'         )  :  '02bOOOjDDO0'                        , \
    ( 3,  'orthorhombic'  ,  'P n a 21'        )  :  '02bOODjDDO0'                        , \
    ( 3,  'orthorhombic'  ,  'P n n 2'         )  :  '02bOOOjDDD0'                        , \
    ( 3,  'orthorhombic'  ,  'C m m 2'         )  :  '03aDDObOOOjOOO0'                    , \
    ( 3,  'orthorhombic'  ,  'C m c 21'        )  :  '03aDDObOODjOOD0'                    , \
    ( 3,  'orthorhombic'  ,  'C c c 2'         )  :  '03aDDObOOOjOOD0'                    , \
    ( 3,  'orthorhombic'  ,  'A m m 2'         )  :  '03aODDbOOOjOOO0'                    , \
    ( 3,  'orthorhombic'  ,  'A b m 2'         )  :  '03aODDbOOOcODO0'                    , \
    ( 3,  'orthorhombic'  ,  'A m a 2'         )  :  '03aODDbOOOjDOO0'                    , \
    ( 3,  'orthorhombic'  ,  'A b a 2'         )  :  '03aODDbOOOjDDO0'                    , \
    ( 3,  'orthorhombic'  ,  'F m m 2'         )  :  '04aODDaDODbOOOjOOO0'                , \
    ( 3,  'orthorhombic'  ,  'F d d 2'         )  :  '04aODDaDODbOOOjBBB0'                , \
    ( 3,  'orthorhombic'  ,  'I m m 2'         )  :  '03aDDDbOOOjOOO0'                    , \
    ( 3,  'orthorhombic'  ,  'I b a 2'         )  :  '03aDDDbOOOjDDO0'                    , \
    ( 3,  'orthorhombic'  ,  'I m a 2'         )  :  '03aDDDbOOOjDOO0'                    , \
    ( 3,  'orthorhombic'  ,  'P m m m'         )  :  '12bOOOcOOO0'                        , \
    ( 3,  'orthorhombic'  ,  'P n n n : 2'     )  :  '03bOOOcOOOhDDD1BBB2'                , \
    ( 3,  'orthorhombic'  ,  'P n n n : 1'     )  :  '03bOOOcOOOhDDD1BBB1'                , \
    ( 3,  'orthorhombic'  ,  'P c c m'         )  :  '12bOOOcOOD0'                        , \
    ( 3,  'orthorhombic'  ,  'P b a n : 2'     )  :  '03bOOOcOOOhDDO1BBO2'                , \
    ( 3,  'orthorhombic'  ,  'P b a n : 1'     )  :  '03bOOOcOOOhDDO1BBO1'                , \
    ( 3,  'orthorhombic'  ,  'P m m a'         )  :  '12bDOOcOOO0'                        , \
    ( 3,  'orthorhombic'  ,  'P n n a'         )  :  '12bDOOcDDD0'                        , \
    ( 3,  'orthorhombic'  ,  'P m n a'         )  :  '12bDODcDOD0'                        , \
    ( 3,  'orthorhombic'  ,  'P c c a'         )  :  '12bDOOcOOD0'                        , \
    ( 3,  'orthorhombic'  ,  'P b a m'         )  :  '12bOOOcDDO0'                        , \
    ( 3,  'orthorhombic'  ,  'P c c n'         )  :  '12bDDOcODD0'                        , \
    ( 3,  'orthorhombic'  ,  'P b c m'         )  :  '12bOODcODD0'                        , \
    ( 3,  'orthorhombic'  ,  'P n n m'         )  :  '12bOOOcDDD0'                        , \
    ( 3,  'orthorhombic'  ,  'P m m n : 2'     )  :  '03bOOOcDDOhDDO1BBO2'                , \
    ( 3,  'orthorhombic'  ,  'P m m n : 1'     )  :  '03bOOOcDDOhDDO1BBO1'                , \
    ( 3,  'orthorhombic'  ,  'P b c n'         )  :  '12bDDDcOOD0'                        , \
    ( 3,  'orthorhombic'  ,  'P b c a'         )  :  '12bDODcODD0'                        , \
    ( 3,  'orthorhombic'  ,  'P n m a'         )  :  '12bDODcODO0'                        , \
    ( 3,  'orthorhombic'  ,  'C m c m'         )  :  '13aDDObOODcOOD0'                    , \
    ( 3,  'orthorhombic'  ,  'C m c a'         )  :  '13aDDObODDcODD0'                    , \
    ( 3,  'orthorhombic'  ,  'C m m m'         )  :  '13aDDObOOOcOOO0'                    , \
    ( 3,  'orthorhombic'  ,  'C c c m'         )  :  '13aDDObOOOcOOD0'                    , \
    ( 3,  'orthorhombic'  ,  'C m m a'         )  :  '13aDDObODOcODO0'                    , \
    ( 3,  'orthorhombic'  ,  'C c c a : 2'     )  :  '04aDDObDDOcOOOhODD1OBB2'            , \
    ( 3,  'orthorhombic'  ,  'C c c a : 1'     )  :  '04aDDObDDOcOOOhODD1OBB1'            , \
    ( 3,  'orthorhombic'  ,  'F m m m'         )  :  '14aODDaDODbOOOcOOO0'                , \
    ( 3,  'orthorhombic'  ,  'F d d d : 2'     )  :  '05aODDaDODbOOOcOOOhBBB1ZZZ2'        , \
    ( 3,  'orthorhombic'  ,  'F d d d : 1'     )  :  '05aODDaDODbOOOcOOOhBBB1ZZZ1'        , \
    ( 3,  'orthorhombic'  ,  'I m m m'         )  :  '13aDDDbOOOcOOO0'                    , \
    ( 3,  'orthorhombic'  ,  'I b a m'         )  :  '13aDDDbOOOcDDO0'                    , \
    ( 3,  'orthorhombic'  ,  'I b c a'         )  :  '13aDDDbDODcODD0'                    , \
    ( 3,  'orthorhombic'  ,  'I m m a'         )  :  '13aDDDbODOcODO0'                    , \
    ( 3,  'tetragonal'    ,  'P 4'             )  :  '02bOOOgOOO0'                        , \
    ( 3,  'tetragonal'    ,  'P 41'            )  :  '02bOODgOOB0'                        , \
    ( 3,  'tetragonal'    ,  'P 42'            )  :  '02bOOOgOOD0'                        , \
    ( 3,  'tetragonal'    ,  'P 43'            )  :  '02bOODgOOF0'                        , \
    ( 3,  'tetragonal'    ,  'I 4'             )  :  '03aDDDbOOOgOOO0'                    , \
    ( 3,  'tetragonal'    ,  'I 41'            )  :  '03aDDDbDDDgODB0'                    , \
    ( 3,  'tetragonal'    ,  'P -4'            )  :  '02bOOOmOOO0'                        , \
    ( 3,  'tetragonal'    ,  'I -4'            )  :  '03aDDDbOOOmOOO0'                    , \
    ( 3,  'tetragonal'    ,  'P 4/m'           )  :  '12bOOOgOOO0'                        , \
    ( 3,  'tetragonal'    ,  'P 42/m'          )  :  '12bOOOgOOD0'                        , \
    ( 3,  'tetragonal'    ,  'P 4/n : 2'       )  :  '03bOOOgDDOhDDO1YBO2'                , \
    ( 3,  'tetragonal'    ,  'P 4/n : 1'       )  :  '03bOOOgDDOhDDO1YBO1'                , \
    ( 3,  'tetragonal'    ,  'P 42/n : 2'      )  :  '03bOOOgDDDhDDD1YYY2'                , \
    ( 3,  'tetragonal'    ,  'P 42/n : 1'      )  :  '03bOOOgDDDhDDD1YYY1'                , \
    ( 3,  'tetragonal'    ,  'I 4/m'           )  :  '13aDDDbOOOgOOO0'                    , \
    ( 3,  'tetragonal'    ,  'I 41/a : 2'      )  :  '04aDDDbDDDgODBhODB1OYZ2'            , \
    ( 3,  'tetragonal'    ,  'I 41/a : 1'      )  :  '04aDDDbDDDgODBhODB1OYZ1'            , \
    ( 3,  'tetragonal'    ,  'P 4 2 2'         )  :  '03bOOOgOOOcOOO0'                    , \
    ( 3,  'tetragonal'    ,  'P 4 21 2'        )  :  '03bOOOgDDOcDDO0'                    , \
    ( 3,  'tetragonal'    ,  'P 41 2 2'        )  :  '03bOODgOOBcOOO0'                    , \
    ( 3,  'tetragonal'    ,  'P 41 21 2'       )  :  '03bOODgDDBcDDB0'                    , \
    ( 3,  'tetragonal'    ,  'P 42 2 2'        )  :  '03bOOOgOODcOOO0'                    , \
    ( 3,  'tetragonal'    ,  'P 42 21 2'       )  :  '03bOOOgDDDcDDD0'                    , \
    ( 3,  'tetragonal'    ,  'P 43 2 2'        )  :  '03bOODgOOFcOOO0'                    , \
    ( 3,  'tetragonal'    ,  'P 43 21 2'       )  :  '03bOODgDDFcDDF0'                    , \
    ( 3,  'tetragonal'    ,  'I 4 2 2'         )  :  '04aDDDbOOOgOOOcOOO0'                , \
    ( 3,  'tetragonal'    ,  'I 41 2 2'        )  :  '04aDDDbDDDgODBcDOF0'                , \
    ( 3,  'tetragonal'    ,  'P 4 m m'         )  :  '03bOOOgOOOjOOO0'                    , \
    ( 3,  'tetragonal'    ,  'P 4 b m'         )  :  '03bOOOgOOOjDDO0'                    , \
    ( 3,  'tetragonal'    ,  'P 42 c m'        )  :  '03bOOOgOODjOOD0'                    , \
    ( 3,  'tetragonal'    ,  'P 42 n m'        )  :  '03bOOOgDDDjDDD0'                    , \
    ( 3,  'tetragonal'    ,  'P 4 c c'         )  :  '03bOOOgOOOjOOD0'                    , \
    ( 3,  'tetragonal'    ,  'P 4 n c'         )  :  '03bOOOgOOOjDDD0'                    , \
    ( 3,  'tetragonal'    ,  'P 42 m c'        )  :  '03bOOOgOODjOOO0'                    , \
    ( 3,  'tetragonal'    ,  'P 42 b c'        )  :  '03bOOOgOODjDDO0'                    , \
    ( 3,  'tetragonal'    ,  'I 4 m m'         )  :  '04aDDDbOOOgOOOjOOO0'                , \
    ( 3,  'tetragonal'    ,  'I 4 c m'         )  :  '04aDDDbOOOgOOOjOOD0'                , \
    ( 3,  'tetragonal'    ,  'I 41 m d'        )  :  '04aDDDbDDDgODBjOOO0'                , \
    ( 3,  'tetragonal'    ,  'I 41 c d'        )  :  '04aDDDbDDDgODBjOOD0'                , \
    ( 3,  'tetragonal'    ,  'P -4 2 m'        )  :  '03bOOOmOOOcOOO0'                    , \
    ( 3,  'tetragonal'    ,  'P -4 2 c'        )  :  '03bOOOmOOOcOOD0'                    , \
    ( 3,  'tetragonal'    ,  'P -4 21 m'       )  :  '03bOOOmOOOcDDO0'                    , \
    ( 3,  'tetragonal'    ,  'P -4 21 c'       )  :  '03bOOOmOOOcDDD0'                    , \
    ( 3,  'tetragonal'    ,  'P -4 m 2'        )  :  '03bOOOmOOOjOOO0'                    , \
    ( 3,  'tetragonal'    ,  'P -4 c 2'        )  :  '03bOOOmOOOjOOD0'                    , \
    ( 3,  'tetragonal'    ,  'P -4 b 2'        )  :  '03bOOOmOOOjDDO0'                    , \
    ( 3,  'tetragonal'    ,  'P -4 n 2'        )  :  '03bOOOmOOOjDDD0'                    , \
    ( 3,  'tetragonal'    ,  'I -4 m 2'        )  :  '04aDDDbOOOmOOOjOOO0'                , \
    ( 3,  'tetragonal'    ,  'I -4 c 2'        )  :  '04aDDDbOOOmOOOjOOD0'                , \
    ( 3,  'tetragonal'    ,  'I -4 2 m'        )  :  '04aDDDbOOOmOOOcOOO0'                , \
    ( 3,  'tetragonal'    ,  'I -4 2 d'        )  :  '04aDDDbOOOmOOOcDOF0'                , \
    ( 3,  'tetragonal'    ,  'P 4/m m m'       )  :  '13bOOOgOOOcOOO0'                    , \
    ( 3,  'tetragonal'    ,  'P 4/m c c'       )  :  '13bOOOgOOOcOOD0'                    , \
    ( 3,  'tetragonal'    ,  'P 4/n b m : 2'   )  :  '04bOOOgOOOcOOOhDDO1YYO2'            , \
    ( 3,  'tetragonal'    ,  'P 4/n b m : 1'   )  :  '04bOOOgOOOcOOOhDDO1YYO1'            , \
    ( 3,  'tetragonal'    ,  'P 4/n n c : 2'   )  :  '04bOOOgOOOcOOOhDDD1YYY2'            , \
    ( 3,  'tetragonal'    ,  'P 4/n n c : 1'   )  :  '04bOOOgOOOcOOOhDDD1YYY1'            , \
    ( 3,  'tetragonal'    ,  'P 4/m b m'       )  :  '13bOOOgOOOcDDO0'                    , \
    ( 3,  'tetragonal'    ,  'P 4/m n c'       )  :  '13bOOOgOOOcDDD0'                    , \
    ( 3,  'tetragonal'    ,  'P 4/n m m : 2'   )  :  '04bOOOgDDOcDDOhDDO1YBO2'            , \
    ( 3,  'tetragonal'    ,  'P 4/n m m : 1'   )  :  '04bOOOgDDOcDDOhDDO1YBO1'            , \
    ( 3,  'tetragonal'    ,  'P 4/n c c : 2'   )  :  '04bOOOgDDOcDDDhDDO1YBO2'            , \
    ( 3,  'tetragonal'    ,  'P 4/n c c : 1'   )  :  '04bOOOgDDOcDDDhDDO1YBO1'            , \
    ( 3,  'tetragonal'    ,  'P 42/m m c'      )  :  '13bDDOgDOOcODD0'                    , \
    ( 3,  'tetragonal'    ,  'P 42/m c m'      )  :  '13bOOOgOODcOOD0'                    , \
    ( 3,  'tetragonal'    ,  'P 42/n b c : 2'  )  :  '04bOOOgDDDcOODhDDD1YBY2'            , \
    ( 3,  'tetragonal'    ,  'P 42/n b c : 1'  )  :  '04bOOOgDDDcOODhDDD1YBY1'            , \
    ( 3,  'tetragonal'    ,  'P 42/n n m : 2'  )  :  '04bOOOgDDDcOOOhDDD1YBY2'            , \
    ( 3,  'tetragonal'    ,  'P 42/n n m : 1'  )  :  '04bOOOgDDDcOOOhDDD1YBY1'            , \
    ( 3,  'tetragonal'    ,  'P 42/m b c'      )  :  '13bOOOgOODcDDO0'                    , \
    ( 3,  'tetragonal'    ,  'P 42/m n m'      )  :  '13bOOOgDDDcDDD0'                    , \
    ( 3,  'tetragonal'    ,  'P 42/n m c : 2'  )  :  '04bOOOgDDDcDDDhDDD1YBY2'            , \
    ( 3,  'tetragonal'    ,  'P 42/n m c : 1'  )  :  '04bOOOgDDDcDDDhDDD1YBY1'            , \
    ( 3,  'tetragonal'    ,  'P 42/n c m : 2'  )  :  '04bOOOgDDDcDDOhDDD1YBY2'            , \
    ( 3,  'tetragonal'    ,  'P 42/n c m : 1'  )  :  '04bOOOgDDDcDDOhDDD1YBY1'            , \
    ( 3,  'tetragonal'    ,  'I 4/m m m'       )  :  '14aDDDbOOOgOOOcOOO0'                , \
    ( 3,  'tetragonal'    ,  'I 4/m c m'       )  :  '14aDDDbOOOgOOOcOOD0'                , \
    ( 3,  'tetragonal'    ,  'I 41/a m d : 2'  )  :  '05aDDDbDDDgODBcDOFhODB1OBZ2'        , \
    ( 3,  'tetragonal'    ,  'I 41/a m d : 1'  )  :  '05aDDDbDDDgODBcDOFhODB1OBZ1'        , \
    ( 3,  'tetragonal'    ,  'I 41/a c d : 2'  )  :  '05aDDDbDDDgODBcDOBhODB1OBZ2'        , \
    ( 3,  'tetragonal'    ,  'I 41/a c d : 1'  )  :  '05aDDDbDDDgODBcDOBhODB1OBZ1'        , \
    ( 3,  'trigonal'      ,  'P 3'             )  :  '01nOOO0'                            , \
    ( 3,  'trigonal'      ,  'P 31'            )  :  '01nOOC0'                            , \
    ( 3,  'trigonal'      ,  'P 32'            )  :  '01nOOE0'                            , \
    ( 3,  'hexagonal'     ,  'R 3 : H'         )  :  '02aECCnOOO0'                        , \
    ( 3,  'trigonal'      ,  'R 3 : R'         )  :  '01dOOO0'                            , \
    ( 3,  'trigonal'      ,  'P -3'            )  :  '11nOOO0'                            , \
    ( 3,  'hexagonal'     ,  'R -3 : H'        )  :  '12aECCnOOO0'                        , \
    ( 3,  'trigonal'      ,  'R -3 : R'        )  :  '11dOOO0'                            , \
    ( 3,  'trigonal'      ,  'P 3 1 2'         )  :  '02nOOOfOOO0'                        , \
    ( 3,  'trigonal'      ,  'P 3 2 1'         )  :  '02nOOOeOOO0'                        , \
    ( 3,  'trigonal'      ,  'P 31 1 2'        )  :  '02nOOCfOOE0'                        , \
    ( 3,  'trigonal'      ,  'P 31 2 1'        )  :  '02nOOCeOOO0'                        , \
    ( 3,  'trigonal'      ,  'P 32 1 2'        )  :  '02nOOEfOOC0'                        , \
    ( 3,  'trigonal'      ,  'P 32 2 1'        )  :  '02nOOEeOOO0'                        , \
    ( 3,  'hexagonal'     ,  'R 3 2 : H'       )  :  '03aECCnOOOeOOO0'                    , \
    ( 3,  'trigonal'      ,  'R 3 2 : R'       )  :  '02dOOOfOOO0'                        , \
    ( 3,  'trigonal'      ,  'P 3 m 1'         )  :  '02nOOOkOOO0'                        , \
    ( 3,  'trigonal'      ,  'P 3 1 m'         )  :  '02nOOOlOOO0'                        , \
    ( 3,  'trigonal'      ,  'P 3 c 1'         )  :  '02nOOOkOOD0'                        , \
    ( 3,  'trigonal'      ,  'P 3 1 c'         )  :  '02nOOOlOOD0'                        , \
    ( 3,  'hexagonal'     ,  'R 3 m : H'       )  :  '03aECCnOOOkOOO0'                    , \
    ( 3,  'trigonal'      ,  'R 3 m : R'       )  :  '02dOOOlOOO0'                        , \
    ( 3,  'hexagonal'     ,  'R 3 c : H'       )  :  '03aECCnOOOkOOD0'                    , \
    ( 3,  'trigonal'      ,  'R 3 c : R'       )  :  '02dOOOlDDD0'                        , \
    ( 3,  'trigonal'      ,  'P -3 1 m'        )  :  '12nOOOfOOO0'                        , \
    ( 3,  'trigonal'      ,  'P -3 1 c'        )  :  '12nOOOfOOD0'                        , \
    ( 3,  'trigonal'      ,  'P -3 m 1'        )  :  '12nOOOeOOO0'                        , \
    ( 3,  'trigonal'      ,  'P -3 c 1'        )  :  '12nOOOeOOD0'                        , \
    ( 3,  'hexagonal'     ,  'R -3 m : H'      )  :  '13aECCnOOOeOOO0'                    , \
    ( 3,  'trigonal'      ,  'R -3 m : R'      )  :  '12dOOOfOOO0'                        , \
    ( 3,  'hexagonal'     ,  'R -3 c : H'      )  :  '13aECCnOOOeOOD0'                    , \
    ( 3,  'trigonal'      ,  'R -3 c : R'      )  :  '12dOOOlDDD0'                        , \
    ( 3,  'hexagonal'     ,  'P 6'             )  :  '02nOOObOOO0'                        , \
    ( 3,  'hexagonal'     ,  'P 61'            )  :  '02nOOCbOOD0'                        , \
    ( 3,  'hexagonal'     ,  'P 65'            )  :  '02nOOEbOOD0'                        , \
    ( 3,  'hexagonal'     ,  'P 62'            )  :  '02nOOEbOOO0'                        , \
    ( 3,  'hexagonal'     ,  'P 64'            )  :  '02nOOCbOOO0'                        , \
    ( 3,  'hexagonal'     ,  'P 63'            )  :  '02nOOObOOD0'                        , \
    ( 3,  'hexagonal'     ,  'P -6'            )  :  '02nOOOiOOO0'                        , \
    ( 3,  'hexagonal'     ,  'P 6/m'           )  :  '12nOOObOOO0'                        , \
    ( 3,  'hexagonal'     ,  'P 63/m'          )  :  '12nOOObOOD0'                        , \
    ( 3,  'hexagonal'     ,  'P 6 2 2'         )  :  '03nOOObOOOeOOO0'                    , \
    ( 3,  'hexagonal'     ,  'P 61 2 2'        )  :  '03nOOCbOODeOOC0'                    , \
    ( 3,  'hexagonal'     ,  'P 65 2 2'        )  :  '03nOOEbOODeOOE0'                    , \
    ( 3,  'hexagonal'     ,  'P 62 2 2'        )  :  '03nOOEbOOOeOOE0'                    , \
    ( 3,  'hexagonal'     ,  'P 64 2 2'        )  :  '03nOOCbOOOeOOC0'                    , \
    ( 3,  'hexagonal'     ,  'P 63 2 2'        )  :  '03nOOObOODeOOO0'                    , \
    ( 3,  'hexagonal'     ,  'P 6 m m'         )  :  '03nOOObOOOkOOO0'                    , \
    ( 3,  'hexagonal'     ,  'P 6 c c'         )  :  '03nOOObOOOkOOD0'                    , \
    ( 3,  'hexagonal'     ,  'P 63 c m'        )  :  '03nOOObOODkOOD0'                    , \
    ( 3,  'hexagonal'     ,  'P 63 m c'        )  :  '03nOOObOODkOOO0'                    , \
    ( 3,  'hexagonal'     ,  'P -6 m 2'        )  :  '03nOOOiOOOkOOO0'                    , \
    ( 3,  'hexagonal'     ,  'P -6 c 2'        )  :  '03nOOOiOODkOOD0'                    , \
    ( 3,  'hexagonal'     ,  'P -6 2 m'        )  :  '03nOOOiOOOeOOO0'                    , \
    ( 3,  'hexagonal'     ,  'P -6 2 c'        )  :  '03nOOOiOODeOOO0'                    , \
    ( 3,  'hexagonal'     ,  'P 6/m m m'       )  :  '13nOOObOOOeOOO0'                    , \
    ( 3,  'hexagonal'     ,  'P 6/m c c'       )  :  '13nOOObOOOeOOD0'                    , \
    ( 3,  'hexagonal'     ,  'P 63/m c m'      )  :  '13nOOObOODeOOD0'                    , \
    ( 3,  'hexagonal'     ,  'P 63/m m c'      )  :  '13nOOObOODeOOO0'                    , \
    ( 3,  'cubic'         ,  'P 2 3'           )  :  '03bOOOcOOOdOOO0'                    , \
    ( 3,  'cubic'         ,  'F 2 3'           )  :  '05aODDaDODbOOOcOOOdOOO0'            , \
    ( 3,  'cubic'         ,  'I 2 3'           )  :  '04aDDDbOOOcOOOdOOO0'                , \
    ( 3,  'cubic'         ,  'P 21 3'          )  :  '03bDODcODDdOOO0'                    , \
    ( 3,  'cubic'         ,  'I 21 3'          )  :  '04aDDDbDODcODDdOOO0'                , \
    ( 3,  'cubic'         ,  'P m -3'          )  :  '13bOOOcOOOdOOO0'                    , \
    ( 3,  'cubic'         ,  'P n -3 : 2'      )  :  '04bOOOcOOOdOOOhDDD1YYY2'            , \
    ( 3,  'cubic'         ,  'P n -3 : 1'      )  :  '04bOOOcOOOdOOOhDDD1YYY1'            , \
    ( 3,  'cubic'         ,  'F m -3'          )  :  '15aODDaDODbOOOcOOOdOOO0'            , \
    ( 3,  'cubic'         ,  'F d -3 : 2'      )  :  '06aODDaDODbOOOcOOOdOOOhBBB1ZZZ2'    , \
    ( 3,  'cubic'         ,  'F d -3 : 1'      )  :  '06aODDaDODbOOOcOOOdOOOhBBB1ZZZ1'    , \
    ( 3,  'cubic'         ,  'I m -3'          )  :  '14aDDDbOOOcOOOdOOO0'                , \
    ( 3,  'cubic'         ,  'P a -3'          )  :  '13bDODcODDdOOO0'                    , \
    ( 3,  'cubic'         ,  'I a -3'          )  :  '14aDDDbDODcODDdOOO0'                , \
    ( 3,  'cubic'         ,  'P 4 3 2'         )  :  '04bOOOcOOOdOOOeOOO0'                , \
    ( 3,  'cubic'         ,  'P 42 3 2'        )  :  '04bOOOcOOOdOOOeDDD0'                , \
    ( 3,  'cubic'         ,  'F 4 3 2'         )  :  '06aODDaDODbOOOcOOOdOOOeOOO0'        , \
    ( 3,  'cubic'         ,  'F 41 3 2'        )  :  '06aODDaDODbODDcDDOdOOOeFBF0'        , \
    ( 3,  'cubic'         ,  'I 4 3 2'         )  :  '05aDDDbOOOcOOOdOOOeOOO0'            , \
    ( 3,  'cubic'         ,  'P 43 3 2'        )  :  '04bDODcODDdOOOeBFF0'                , \
    ( 3,  'cubic'         ,  'P 41 3 2'        )  :  '04bDODcODDdOOOeFBB0'                , \
    ( 3,  'cubic'         ,  'I 41 3 2'        )  :  '05aDDDbDODcODDdOOOeFBB0'            , \
    ( 3,  'cubic'         ,  'P -4 3 m'        )  :  '04bOOOcOOOdOOOlOOO0'                , \
    ( 3,  'cubic'         ,  'F -4 3 m'        )  :  '06aODDaDODbOOOcOOOdOOOlOOO0'        , \
    ( 3,  'cubic'         ,  'I -4 3 m'        )  :  '05aDDDbOOOcOOOdOOOlOOO0'            , \
    ( 3,  'cubic'         ,  'P -4 3 n'        )  :  '04bOOOcOOOdOOOlDDD0'                , \
    ( 3,  'cubic'         ,  'F -4 3 c'        )  :  '06aODDaDODbOOOcOOOdOOOlDDD0'        , \
    ( 3,  'cubic'         ,  'I -4 3 d'        )  :  '05aDDDbDODcODDdOOOlBBB0'            , \
    ( 3,  'cubic'         ,  'P m -3 m'        )  :  '14bOOOcOOOdOOOeOOO0'                , \
    ( 3,  'cubic'         ,  'P n -3 n : 2'    )  :  '14bDDOcDODdOOOeOOD1YYY2'            , \
    ( 3,  'cubic'         ,  'P n -3 n : 1'    )  :  '14bDDOcDODdOOOeOOD1YYY1'            , \
    ( 3,  'cubic'         ,  'P m -3 n'        )  :  '14bOOOcOOOdOOOeDDD0'                , \
    ( 3,  'cubic'         ,  'P n -3 m : 2'    )  :  '05bOOOcOOOdOOOeDDDhDDD1YYY2'        , \
    ( 3,  'cubic'         ,  'P n -3 m : 1'    )  :  '05bOOOcOOOdOOOeDDDhDDD1YYY1'        , \
    ( 3,  'cubic'         ,  'F m -3 m'        )  :  '16aODDaDODbOOOcOOOdOOOeOOO0'        , \
    ( 3,  'cubic'         ,  'F m -3 c'        )  :  '16aODDaDODbOOOcOOOdOOOeDDD0'        , \
    ( 3,  'cubic'         ,  'F d -3 m : 2'    )  :  '07aODDaDODbODDcDDOdOOOeFBFhBBB1ZZZ2', \
    ( 3,  'cubic'         ,  'F d -3 m : 1'    )  :  '07aODDaDODbODDcDDOdOOOeFBFhBBB1ZZZ1', \
    ( 3,  'cubic'         ,  'F d -3 c : 2'    )  :  '07aODDaDODbODDcDDOdOOOeFBFhFFF1XXX2', \
    ( 3,  'cubic'         ,  'F d -3 c : 1'    )  :  '07aODDaDODbODDcDDOdOOOeFBFhFFF1XXX1', \
    ( 3,  'cubic'         ,  'I m -3 m'        )  :  '15aDDDbOOOcOOOdOOOeOOO0'            , \
    ( 3,  'cubic'         ,  'I a -3 d'        )  :  '15aDDDbDODcODDdOOOeFBB0'             }

SYMMETRY_POSITION_COUNTS = {    \
    ( 2,  'oblique'       ,  'p 1'             )  :  (  3,   3,   1 ), \
    ( 2,  'oblique'       ,  'p 2'             )  :  (  4,   4,   2 ), \
    ( 2,  'rectangular'   ,  'p m'             )  :  (  4,   4,   2 ), \
    ( 2,  'rectangular'   ,  'p g'             )  :  (  4,   4,   2 ), \
    ( 2,  'rectangular'   ,  'c m'             )  :  (  5,   6,   4 ), \
    ( 2,  'rectangular'   ,  'p 2 m m'         )  :  (  5,   6,   4 ), \
    ( 2,  'rectangular'   ,  'p 2 m g'         )  :  (  5,   6,   4 ), \
    ( 2,  'rectangular'   ,  'p 2 g g'         )  :  (  5,   6,   4 ), \
    ( 2,  'rectangular'   ,  'c 2 m m'         )  :  (  6,  10,   8 ), \
    ( 2,  'square'        ,  'p 4'             )  :  (  5,   6,   4 ), \
    ( 2,  'square'        ,  'p 4 m m'         )  :  (  6,  10,   8 ), \
    ( 2,  'square'        ,  'p 4 g m'         )  :  (  6,  10,   8 ), \
    ( 2,  'hexagonal'     ,  'p 3'             )  :  (  4,   4,   2 ), \
    ( 2,  'hexagonal'     ,  'p 3 m 1'         )  :  (  5,   8,   6 ), \
    ( 2,  'hexagonal'     ,  'p 3 1 m'         )  :  (  5,   8,   6 ), \
    ( 2,  'hexagonal'     ,  'p 6'             )  :  (  5,   8,   6 ), \
    ( 2,  'hexagonal'     ,  'p 6 m m'         )  :  (  6,  14,  12 ), \
    ( 3,  'triclinic'     ,  'P 1'             )  :  (  4,   4,   1 ), \
    ( 3,  'triclinic'     ,  'P -1'            )  :  (  5,   5,   2 ), \
    ( 3,  'monoclinic'    ,  'P 1 2 1'         )  :  (  5,   5,   2 ), \
    ( 3,  'monoclinic'    ,  'P 1 21 1'        )  :  (  5,   5,   2 ), \
    ( 3,  'monoclinic'    ,  'C 1 2 1'         )  :  (  6,   7,   4 ), \
    ( 3,  'monoclinic'    ,  'P 1 m 1'         )  :  (  5,   5,   2 ), \
    ( 3,  'monoclinic'    ,  'P 1 c 1'         )  :  (  5,   5,   2 ), \
    ( 3,  'monoclinic'    ,  'C 1 m 1'         )  :  (  6,   7,   4 ), \
    ( 3,  'monoclinic'    ,  'C 1 c 1'         )  :  (  6,   7,   4 ), \
    ( 3,  'monoclinic'    ,  'P 1 2/m 1'       )  :  (  6,   7,   4 ), \
    ( 3,  'monoclinic'    ,  'P 1 21/m 1'      )  :  (  6,   7,   4 ), \
    ( 3,  'monoclinic'    ,  'C 1 2/m 1'       )  :  (  7,  11,   8 ), \
    ( 3,  'monoclinic'    ,  'P 1 2/c 1'       )  :  (  6,   7,   4 ), \
    ( 3,  'monoclinic'    ,  'P 1 21/c 1'      )  :  (  6,   7,   4 ), \
    ( 3,  'monoclinic'    ,  'C 1 2/c 1'       )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'P 2 2 2'         )  :  (  6,   7,   4 ), \
    ( 3,  'orthorhombic'  ,  'P 2 2 21'        )  :  (  6,   7,   4 ), \
    ( 3,  'orthorhombic'  ,  'P 21 21 2'       )  :  (  6,   7,   4 ), \
    ( 3,  'orthorhombic'  ,  'P 21 21 21'      )  :  (  6,   7,   4 ), \
    ( 3,  'orthorhombic'  ,  'C 2 2 21'        )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'C 2 2 2'         )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'F 2 2 2'         )  :  (  8,  19,  16 ), \
    ( 3,  'orthorhombic'  ,  'I 2 2 2'         )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'I 21 21 21'      )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'P m m 2'         )  :  (  6,   7,   4 ), \
    ( 3,  'orthorhombic'  ,  'P m c 21'        )  :  (  6,   7,   4 ), \
    ( 3,  'orthorhombic'  ,  'P c c 2'         )  :  (  6,   7,   4 ), \
    ( 3,  'orthorhombic'  ,  'P m a 2'         )  :  (  6,   7,   4 ), \
    ( 3,  'orthorhombic'  ,  'P c a 21'        )  :  (  6,   7,   4 ), \
    ( 3,  'orthorhombic'  ,  'P n c 2'         )  :  (  6,   7,   4 ), \
    ( 3,  'orthorhombic'  ,  'P m n 21'        )  :  (  6,   7,   4 ), \
    ( 3,  'orthorhombic'  ,  'P b a 2'         )  :  (  6,   7,   4 ), \
    ( 3,  'orthorhombic'  ,  'P n a 21'        )  :  (  6,   7,   4 ), \
    ( 3,  'orthorhombic'  ,  'P n n 2'         )  :  (  6,   7,   4 ), \
    ( 3,  'orthorhombic'  ,  'C m m 2'         )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'C m c 21'        )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'C c c 2'         )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'A m m 2'         )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'A b m 2'         )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'A m a 2'         )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'A b a 2'         )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'F m m 2'         )  :  (  8,  19,  16 ), \
    ( 3,  'orthorhombic'  ,  'F d d 2'         )  :  (  8,  19,  16 ), \
    ( 3,  'orthorhombic'  ,  'I m m 2'         )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'I b a 2'         )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'I m a 2'         )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'P m m m'         )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'P n n n : 2'     )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'P n n n : 1'     )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'P c c m'         )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'P b a n : 2'     )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'P b a n : 1'     )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'P m m a'         )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'P n n a'         )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'P m n a'         )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'P c c a'         )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'P b a m'         )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'P c c n'         )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'P b c m'         )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'P n n m'         )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'P m m n : 2'     )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'P m m n : 1'     )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'P b c n'         )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'P b c a'         )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'P n m a'         )  :  (  7,  11,   8 ), \
    ( 3,  'orthorhombic'  ,  'C m c m'         )  :  (  8,  19,  16 ), \
    ( 3,  'orthorhombic'  ,  'C m c a'         )  :  (  8,  19,  16 ), \
    ( 3,  'orthorhombic'  ,  'C m m m'         )  :  (  8,  19,  16 ), \
    ( 3,  'orthorhombic'  ,  'C c c m'         )  :  (  8,  19,  16 ), \
    ( 3,  'orthorhombic'  ,  'C m m a'         )  :  (  8,  19,  16 ), \
    ( 3,  'orthorhombic'  ,  'C c c a : 2'     )  :  (  8,  19,  16 ), \
    ( 3,  'orthorhombic'  ,  'C c c a : 1'     )  :  (  8,  19,  16 ), \
    ( 3,  'orthorhombic'  ,  'F m m m'         )  :  (  9,  35,  32 ), \
    ( 3,  'orthorhombic'  ,  'F d d d : 2'     )  :  (  9,  35,  32 ), \
    ( 3,  'orthorhombic'  ,  'F d d d : 1'     )  :  (  9,  35,  32 ), \
    ( 3,  'orthorhombic'  ,  'I m m m'         )  :  (  8,  19,  16 ), \
    ( 3,  'orthorhombic'  ,  'I b a m'         )  :  (  8,  19,  16 ), \
    ( 3,  'orthorhombic'  ,  'I b c a'         )  :  (  8,  19,  16 ), \
    ( 3,  'orthorhombic'  ,  'I m m a'         )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'P 4'             )  :  (  6,   7,   4 ), \
    ( 3,  'tetragonal'    ,  'P 41'            )  :  (  6,   7,   4 ), \
    ( 3,  'tetragonal'    ,  'P 42'            )  :  (  6,   7,   4 ), \
    ( 3,  'tetragonal'    ,  'P 43'            )  :  (  6,   7,   4 ), \
    ( 3,  'tetragonal'    ,  'I 4'             )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'I 41'            )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'P -4'            )  :  (  6,   7,   4 ), \
    ( 3,  'tetragonal'    ,  'I -4'            )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'P 4/m'           )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'P 42/m'          )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'P 4/n : 2'       )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'P 4/n : 1'       )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'P 42/n : 2'      )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'P 42/n : 1'      )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'I 4/m'           )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'I 41/a : 2'      )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'I 41/a : 1'      )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'P 4 2 2'         )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'P 4 21 2'        )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'P 41 2 2'        )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'P 41 21 2'       )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'P 42 2 2'        )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'P 42 21 2'       )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'P 43 2 2'        )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'P 43 21 2'       )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'I 4 2 2'         )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'I 41 2 2'        )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'P 4 m m'         )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'P 4 b m'         )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'P 42 c m'        )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'P 42 n m'        )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'P 4 c c'         )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'P 4 n c'         )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'P 42 m c'        )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'P 42 b c'        )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'I 4 m m'         )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'I 4 c m'         )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'I 41 m d'        )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'I 41 c d'        )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'P -4 2 m'        )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'P -4 2 c'        )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'P -4 21 m'       )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'P -4 21 c'       )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'P -4 m 2'        )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'P -4 c 2'        )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'P -4 b 2'        )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'P -4 n 2'        )  :  (  7,  11,   8 ), \
    ( 3,  'tetragonal'    ,  'I -4 m 2'        )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'I -4 c 2'        )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'I -4 2 m'        )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'I -4 2 d'        )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'P 4/m m m'       )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'P 4/m c c'       )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'P 4/n b m : 2'   )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'P 4/n b m : 1'   )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'P 4/n n c : 2'   )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'P 4/n n c : 1'   )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'P 4/m b m'       )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'P 4/m n c'       )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'P 4/n m m : 2'   )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'P 4/n m m : 1'   )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'P 4/n c c : 2'   )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'P 4/n c c : 1'   )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'P 42/m m c'      )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'P 42/m c m'      )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'P 42/n b c : 2'  )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'P 42/n b c : 1'  )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'P 42/n n m : 2'  )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'P 42/n n m : 1'  )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'P 42/m b c'      )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'P 42/m n m'      )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'P 42/n m c : 2'  )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'P 42/n m c : 1'  )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'P 42/n c m : 2'  )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'P 42/n c m : 1'  )  :  (  8,  19,  16 ), \
    ( 3,  'tetragonal'    ,  'I 4/m m m'       )  :  (  9,  35,  32 ), \
    ( 3,  'tetragonal'    ,  'I 4/m c m'       )  :  (  9,  35,  32 ), \
    ( 3,  'tetragonal'    ,  'I 41/a m d : 2'  )  :  (  9,  35,  32 ), \
    ( 3,  'tetragonal'    ,  'I 41/a m d : 1'  )  :  (  9,  35,  32 ), \
    ( 3,  'tetragonal'    ,  'I 41/a c d : 2'  )  :  (  9,  35,  32 ), \
    ( 3,  'tetragonal'    ,  'I 41/a c d : 1'  )  :  (  9,  35,  32 ), \
    ( 3,  'trigonal'      ,  'P 3'             )  :  (  5,   6,   3 ), \
    ( 3,  'trigonal'      ,  'P 31'            )  :  (  5,   6,   3 ), \
    ( 3,  'trigonal'      ,  'P 32'            )  :  (  5,   6,   3 ), \
    ( 3,  'hexagonal'     ,  'R 3 : H'         )  :  (  6,  12,   9 ), \
    ( 3,  'trigonal'      ,  'R 3 : R'         )  :  (  5,   6,   3 ), \
    ( 3,  'trigonal'      ,  'P -3'            )  :  (  6,   9,   6 ), \
    ( 3,  'hexagonal'     ,  'R -3 : H'        )  :  (  7,  21,  18 ), \
    ( 3,  'trigonal'      ,  'R -3 : R'        )  :  (  6,   9,   6 ), \
    ( 3,  'trigonal'      ,  'P 3 1 2'         )  :  (  6,   9,   6 ), \
    ( 3,  'trigonal'      ,  'P 3 2 1'         )  :  (  6,   9,   6 ), \
    ( 3,  'trigonal'      ,  'P 31 1 2'        )  :  (  6,   9,   6 ), \
    ( 3,  'trigonal'      ,  'P 31 2 1'        )  :  (  6,   9,   6 ), \
    ( 3,  'trigonal'      ,  'P 32 1 2'        )  :  (  6,   9,   6 ), \
    ( 3,  'trigonal'      ,  'P 32 2 1'        )  :  (  6,   9,   6 ), \
    ( 3,  'hexagonal'     ,  'R 3 2 : H'       )  :  (  7,  21,  18 ), \
    ( 3,  'trigonal'      ,  'R 3 2 : R'       )  :  (  6,   9,   6 ), \
    ( 3,  'trigonal'      ,  'P 3 m 1'         )  :  (  6,   9,   6 ), \
    ( 3,  'trigonal'      ,  'P 3 1 m'         )  :  (  6,   9,   6 ), \
    ( 3,  'trigonal'      ,  'P 3 c 1'         )  :  (  6,   9,   6 ), \
    ( 3,  'trigonal'      ,  'P 3 1 c'         )  :  (  6,   9,   6 ), \
    ( 3,  'hexagonal'     ,  'R 3 m : H'       )  :  (  7,  21,  18 ), \
    ( 3,  'trigonal'      ,  'R 3 m : R'       )  :  (  6,   9,   6 ), \
    ( 3,  'hexagonal'     ,  'R 3 c : H'       )  :  (  7,  21,  18 ), \
    ( 3,  'trigonal'      ,  'R 3 c : R'       )  :  (  6,   9,   6 ), \
    ( 3,  'trigonal'      ,  'P -3 1 m'        )  :  (  7,  15,  12 ), \
    ( 3,  'trigonal'      ,  'P -3 1 c'        )  :  (  7,  15,  12 ), \
    ( 3,  'trigonal'      ,  'P -3 m 1'        )  :  (  7,  15,  12 ), \
    ( 3,  'trigonal'      ,  'P -3 c 1'        )  :  (  7,  15,  12 ), \
    ( 3,  'hexagonal'     ,  'R -3 m : H'      )  :  (  8,  39,  36 ), \
    ( 3,  'trigonal'      ,  'R -3 m : R'      )  :  (  7,  15,  12 ), \
    ( 3,  'hexagonal'     ,  'R -3 c : H'      )  :  (  8,  39,  36 ), \
    ( 3,  'trigonal'      ,  'R -3 c : R'      )  :  (  7,  15,  12 ), \
    ( 3,  'hexagonal'     ,  'P 6'             )  :  (  6,   9,   6 ), \
    ( 3,  'hexagonal'     ,  'P 61'            )  :  (  6,   9,   6 ), \
    ( 3,  'hexagonal'     ,  'P 65'            )  :  (  6,   9,   6 ), \
    ( 3,  'hexagonal'     ,  'P 62'            )  :  (  6,   9,   6 ), \
    ( 3,  'hexagonal'     ,  'P 64'            )  :  (  6,   9,   6 ), \
    ( 3,  'hexagonal'     ,  'P 63'            )  :  (  6,   9,   6 ), \
    ( 3,  'hexagonal'     ,  'P -6'            )  :  (  6,   9,   6 ), \
    ( 3,  'hexagonal'     ,  'P 6/m'           )  :  (  7,  15,  12 ), \
    ( 3,  'hexagonal'     ,  'P 63/m'          )  :  (  7,  15,  12 ), \
    ( 3,  'hexagonal'     ,  'P 6 2 2'         )  :  (  7,  15,  12 ), \
    ( 3,  'hexagonal'     ,  'P 61 2 2'        )  :  (  7,  15,  12 ), \
    ( 3,  'hexagonal'     ,  'P 65 2 2'        )  :  (  7,  15,  12 ), \
    ( 3,  'hexagonal'     ,  'P 62 2 2'        )  :  (  7,  15,  12 ), \
    ( 3,  'hexagonal'     ,  'P 64 2 2'        )  :  (  7,  15,  12 ), \
    ( 3,  'hexagonal'     ,  'P 63 2 2'        )  :  (  7,  15,  12 ), \
    ( 3,  'hexagonal'     ,  'P 6 m m'         )  :  (  7,  15,  12 ), \
    ( 3,  'hexagonal'     ,  'P 6 c c'         )  :  (  7,  15,  12 ), \
    ( 3,  'hexagonal'     ,  'P 63 c m'        )  :  (  7,  15,  12 ), \
    ( 3,  'hexagonal'     ,  'P 63 m c'        )  :  (  7,  15,  12 ), \
    ( 3,  'hexagonal'     ,  'P -6 m 2'        )  :  (  7,  15,  12 ), \
    ( 3,  'hexagonal'     ,  'P -6 c 2'        )  :  (  7,  15,  12 ), \
    ( 3,  'hexagonal'     ,  'P -6 2 m'        )  :  (  7,  15,  12 ), \
    ( 3,  'hexagonal'     ,  'P -6 2 c'        )  :  (  7,  15,  12 ), \
    ( 3,  'hexagonal'     ,  'P 6/m m m'       )  :  (  8,  27,  24 ), \
    ( 3,  'hexagonal'     ,  'P 6/m c c'       )  :  (  8,  27,  24 ), \
    ( 3,  'hexagonal'     ,  'P 63/m c m'      )  :  (  8,  27,  24 ), \
    ( 3,  'hexagonal'     ,  'P 63/m m c'      )  :  (  8,  27,  24 ), \
    ( 3,  'cubic'         ,  'P 2 3'           )  :  (  7,  15,  12 ), \
    ( 3,  'cubic'         ,  'F 2 3'           )  :  (  9,  51,  48 ), \
    ( 3,  'cubic'         ,  'I 2 3'           )  :  (  8,  27,  24 ), \
    ( 3,  'cubic'         ,  'P 21 3'          )  :  (  7,  15,  12 ), \
    ( 3,  'cubic'         ,  'I 21 3'          )  :  (  8,  27,  24 ), \
    ( 3,  'cubic'         ,  'P m -3'          )  :  (  8,  27,  24 ), \
    ( 3,  'cubic'         ,  'P n -3 : 2'      )  :  (  8,  27,  24 ), \
    ( 3,  'cubic'         ,  'P n -3 : 1'      )  :  (  8,  27,  24 ), \
    ( 3,  'cubic'         ,  'F m -3'          )  :  ( 10,  99,  96 ), \
    ( 3,  'cubic'         ,  'F d -3 : 2'      )  :  ( 10,  99,  96 ), \
    ( 3,  'cubic'         ,  'F d -3 : 1'      )  :  ( 10,  99,  96 ), \
    ( 3,  'cubic'         ,  'I m -3'          )  :  (  9,  51,  48 ), \
    ( 3,  'cubic'         ,  'P a -3'          )  :  (  8,  27,  24 ), \
    ( 3,  'cubic'         ,  'I a -3'          )  :  (  9,  51,  48 ), \
    ( 3,  'cubic'         ,  'P 4 3 2'         )  :  (  8,  27,  24 ), \
    ( 3,  'cubic'         ,  'P 42 3 2'        )  :  (  8,  27,  24 ), \
    ( 3,  'cubic'         ,  'F 4 3 2'         )  :  ( 10,  99,  96 ), \
    ( 3,  'cubic'         ,  'F 41 3 2'        )  :  ( 10,  99,  96 ), \
    ( 3,  'cubic'         ,  'I 4 3 2'         )  :  (  9,  51,  48 ), \
    ( 3,  'cubic'         ,  'P 43 3 2'        )  :  (  8,  27,  24 ), \
    ( 3,  'cubic'         ,  'P 41 3 2'        )  :  (  8,  27,  24 ), \
    ( 3,  'cubic'         ,  'I 41 3 2'        )  :  (  9,  51,  48 ), \
    ( 3,  'cubic'         ,  'P -4 3 m'        )  :  (  8,  27,  24 ), \
    ( 3,  'cubic'         ,  'F -4 3 m'        )  :  ( 10,  99,  96 ), \
    ( 3,  'cubic'         ,  'I -4 3 m'        )  :  (  9,  51,  48 ), \
    ( 3,  'cubic'         ,  'P -4 3 n'        )  :  (  8,  27,  24 ), \
    ( 3,  'cubic'         ,  'F -4 3 c'        )  :  ( 10,  99,  96 ), \
    ( 3,  'cubic'         ,  'I -4 3 d'        )  :  (  9,  51,  48 ), \
    ( 3,  'cubic'         ,  'P m -3 m'        )  :  (  9,  51,  48 ), \
    ( 3,  'cubic'         ,  'P n -3 n : 2'    )  :  (  9,  51,  48 ), \
    ( 3,  'cubic'         ,  'P n -3 n : 1'    )  :  (  9,  51,  48 ), \
    ( 3,  'cubic'         ,  'P m -3 n'        )  :  (  9,  51,  48 ), \
    ( 3,  'cubic'         ,  'P n -3 m : 2'    )  :  (  9,  51,  48 ), \
    ( 3,  'cubic'         ,  'P n -3 m : 1'    )  :  (  9,  51,  48 ), \
    ( 3,  'cubic'         ,  'F m -3 m'        )  :  ( 11, 195, 192 ), \
    ( 3,  'cubic'         ,  'F m -3 c'        )  :  ( 11, 195, 192 ), \
    ( 3,  'cubic'         ,  'F d -3 m : 2'    )  :  ( 11, 195, 192 ), \
    ( 3,  'cubic'         ,  'F d -3 m : 1'    )  :  ( 11, 195, 192 ), \
    ( 3,  'cubic'         ,  'F d -3 c : 2'    )  :  ( 11, 195, 192 ), \
    ( 3,  'cubic'         ,  'F d -3 c : 1'    )  :  ( 11, 195, 192 ), \
    ( 3,  'cubic'         ,  'I m -3 m'        )  :  ( 10,  99,  96 ), \
    ( 3,  'cubic'         ,  'I a -3 d'        )  :  ( 10,  99,  96 ) }

""" Regex for parsing plane and space-group generator strings. """
GENERATOR_STRING_REGEX = re.compile(r"\A(?P<hasInv>[01])(?P<nGen>[0-7])(?P<genList>(?:[a-nz][A-GOXYZ]{3}){0,7})(?P<orig>0|1[A-GOXYZ]{3}[12])\Z")

""" Regex for splitting list of generators. """
GENERATOR_SET_REGEX = re.compile(r"[a-nz][A-GOXYZ]{3}")

""" Regex for parsing single generator specifiers. """
GENERATOR_COMPONENT_REGEX = re.compile(r"\A(?P<psm>[a-nz])(?P<acomp>[A-GOXYZ])(?P<bcomp>[A-GOXYZ])(?P<ccomp>[A-GOXYZ])\Z")

""" Regex for parsing alternate setting positions """
ALTERNATE_SETTING_REGEX = re.compile(r"\A1(?P<acomp>[A-GOXYZ])(?P<bcomp>[A-GOXYZ])(?P<ccomp>[A-GOXYZ])(?P<setnum>[12])\Z")

def getGroupID(dim, crystal_system, group_name):
    """
    Get the plane or space group number, according to the International Tables of Crystallography (2016).
    
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
        see: http://pscf.cems.umn.edu/
    
    Returns
    -------
    The Integer group number. 
        In 2D, these are in range [1,17]. In 3D, in range [1,230]
    
    Raises
    ------
    ValueError if no group entry is found for the given parameter set.
    """
    key = (dim, crystal_system, group_name)
    out = GROUP_NAME_BY_ID.get( key, None )
    if out is None:
        raise(ValueError("No Group ID entry found for input values {}.".format(key)))
    return out

def getGroupName(dim, crystal_system, group_id):
    """
    Get the plane or space group name, according to PSCF specifications ( http://pscf.cems.umn.edu/ ).
    
    Parameters
    ----------
    dim : either 2 or 3
        Dimensionality of the unit cell.
        If 2, plane groups will be sought. If 3, space groups will be sought.
    crystal_system : group_data.CRYSTAL_SYSTEM
        The crystal system in which to look for the group.
        CRYSTAL_SYSTEM.HEXAGONAL applies to both 2D and 3D systems.
    group_id : int
        The plane group or space group number according to the International Tables of Crystallography (2016).
    
    Returns
    -------
    String containing the group name.
    
    Raises
    ------
    ValueError if no group entry is found for the given parameter set.
    """
    key = (dim, crystal_system, group_id)
    out = GROUP_ID_BY_NAME.get( key, None )
    if out is None:
        raise(ValueError("No Group ID entry found for input values {}.".format(key)))
    return out

def _getGeneratorString(dim, crystal_system, group_name):
    out = GENERATOR_STRINGS.get((dim, crystal_system, group_name), None)
    #if out is None:
    #    # Try converting from C++/Cuda-style to Fortran-style before raising error
    #    newName = _change_groupName_format(group_name,toFortran=True)
    #    out = GENERATOR_STRINGS.get((dim, crystal_system, newName), None)
    #    if out is None:
    #        raise(ValueError("Given group is not available: (dim, crystal_system, group) = (,{!r},{!r})".format(dim,crystal_system, group_name)))
    return out

def _buildGenerator(dim, genstr):
    genmatch = GENERATOR_COMPONENT_REGEX.match(genstr)
    psmkey = genmatch.group('psm')
    akey = genmatch.group('acomp')
    bkey = genmatch.group('bcomp')
    ckey = genmatch.group('ccomp')
    try:
        psm = np.array(POINT_SYMMETRY_GENERATORS[psmkey])
        a = TRANSLATION_COMPONENTS[akey]
        b = TRANSLATION_COMPONENTS[bkey]
        c = TRANSLATION_COMPONENTS[ckey]
    except(KeyError):
        raise(ValueError("Invalid generator string {!r}".format(genstr)))
    psm[0,3] = a
    psm[1,3] = b
    psm[2,3] = c
    return SymmetryOperation(dim, psm)

def _originShift(dim, origstring):
    genmatch = ALTERNATE_SETTING_REGEX.match(origstring)
    if genmatch is not None:
        akey = genmatch.group('acomp')
        bkey = genmatch.group('bcomp')
        ckey = genmatch.group('ccomp')
        try:
            a = TRANSLATION_COMPONENTS[akey]
            b = TRANSLATION_COMPONENTS[bkey]
            c = TRANSLATION_COMPONENTS[ckey]
        except(KeyError):
            raise(ValueError("Invalid alternate setting string {!r}".format(origstring)))
        shift = np.eye(4)
        shift[0,3] = a
        shift[1,3] = b
        shift[2,3] = c
        shiftOp = SymmetryOperation(dim, shift)
        isdefault = bool(int(genmatch.group('setnum')) - 1) # Default is 2
        return { 'symmOp' : shiftOp, 'isDefault' : isdefault }
    return None

def _getGroupCounts(dim, crystal_system, group_name):
    key = ( dim, crystal_system, group_name )
    data = SYMMETRY_POSITION_COUNTS.get( key, None )
    #if data is None:
    #    # Try converting from C++/Cuda-style to Fortran-style before raising error
    #    newName = _change_groupName_format(group_name,toFortran=True)
    #    key = (dim, crystal_system, newName)
    #    data = SYMMETRY_POSITION_COUNTS.get( key, None )
    #    if data is None:
    #        raise(ValueError("No match found for space group {}".format(key)))
    out = { "num_generators" : data[0], \
            "num_operations" : data[1], \
            "num_positions" : data[2] }
    return out

def getGeneratorSet(dim, crystal_system, group_name):
    genstring = _getGeneratorString(dim, crystal_system, group_name)
    genstrmatch = GENERATOR_STRING_REGEX.match(genstring)
    hasInv = bool(int(genstrmatch.group('hasInv')))
    ngen = int(genstrmatch.group('nGen'))
    genliststr = genstrmatch.group('genList')
    orig = genstrmatch.group('orig')
    generators = []
    gen = SymmetryOperation(dim, POINT_SYMMETRY_GENERATORS['a'])
    generators.append(gen) # add the identity operation
    if hasInv:
        gen = SymmetryOperation(dim, POINT_SYMMETRY_GENERATORS['inv'])
        generators.append(gen)
    genlist = GENERATOR_SET_REGEX.findall(genliststr)
    #print(genlist)
    if not len(genlist) == ngen:
        raise(RuntimeError("Error with generator string {}: expected {} generators, got {}.".format(genstring, ngen, len(genlist))))
    for i in range(ngen):
        generators.append(_buildGenerator(dim, genlist[i]))
    orig_data = _originShift(dim, orig)
    counts_data = _getGroupCounts(dim, crystal_system, group_name)
    return generators, orig_data, counts_data
    
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
        try:
            generators, originData, countData = getGeneratorSet(dim, self._crystal_system, self._group_name)
        except(ValueError):
            raise(ValueError("Unable to find definition for dim={}, crystal_system={!r}, group_name={!r}".format(dim, self._crystal_system, self._group_name)))
        self._basic_generators = generators
        self._unit_translations = SymmetryOperation.getUnitTranslations(dim)
        self._generators = [*self._unit_translations, *self._basic_generators]
        if originData is None:
            self._multiple_settings_flag = False
        else:
            self._multiple_settings_flag = True
            self._setting_offset = originData['symmOp']
            self._is_default_setting = originData['isDefault']
        self._expected_symmetry_ops = countData['num_operations']
        self._expected_geps = countData['num_positions']
        self._build_group()
        if self._multiple_settings_flag:
            self._apply_setting()
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
        return len(self._symmetry_ops)
    
    @property
    def symmetryOperations(self):
        return deepcopy(self._symmetry_ops)
    
    @property
    def positionCount(self):
        return len(self._general_positions)
    
    @property
    def generalPositions(self):
        return deepcopy(self._general_positions)
    
    def evaluatePosition(self, position, atol=POSITION_TOLERANCE):
        """ Apply each GEP to the given position and return the set of unique positions. 'Uniqueness' determined by a separation of greater than atol. """
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
    
    def __str__(self):
        formstr = "< SpaceGroup object with dim = {}, system = {}, group name = {} >"
        return formstr.format(self.dim, self.crystalSystem, self.groupName)
    
    def _reached_expected_operations(self):
        if self.symmetryCount == self._expected_symmetry_ops:
            return True
        elif self.symmetryCount > self._expected_symmetry_ops:
            raise(RuntimeError("Allowed symmetry operations exceeded"))
        else:
            return False
    
    def _build_group(self):
        self._symmetry_ops = []
        for m in self._generators:
            self._symmetry_ops.append(m)
        #print(len(self._symmetry_ops))
        # to ensure full population of symmetries, perform search twice.
        # Expected number of operations previously determined. When this count
        #  is reached, the search is terminated.
        for k in range(2):
            i = 0
            while i < len(self._symmetry_ops):
                #print(len(self._symmetry_ops),i)
                op_one = self._symmetry_ops[i]
                j = 0
                while j < len(self._symmetry_ops):
                    #print(len(self._symmetry_ops),i,j)
                    op_two = self._symmetry_ops[j]
                    symm = op_one * op_two
                    if not symm in self._symmetry_ops:
                        self._symmetry_ops.append(symm)
                        if len(self._symmetry_ops) > self.__class__.__max_ops:
                            raise(RuntimeError("Exceeded maximum allowed symmetry operations"))
                    if self._reached_expected_operations():
                        break
                    j += 1
                    
                if self._reached_expected_operations():
                    break
                i += 1
                
            if self._reached_expected_operations():
                break
    
    def _apply_setting(self):
        if not self._is_default_setting:
            for (i,symm) in enumerate(self._symmetry_ops):
                self._symmetry_ops[i] = self._setting_offset @ symm @ self._setting_offset.reverse
    
    def _build_GEPs(self):
        pos = GeneralPosition(self._dim)
        geps = [pos]
        for symm in self._symmetry_ops:
            newPos = symm @ pos
            if not newPos in geps:
                geps.append(newPos)
        self._general_positions = geps
    
def __testGroup(dim, system, name):
    print("\nTest: {}D, {}, {}:".format(dim, system, name))
    sg = SpaceGroup(dim,system,name)
    print("gens:", len(sg._generators))
    print("ops:",len(sg._symmetry_ops))
    print("geps:",len(sg._general_positions))
    print("generators")
    for s in sg._generators:
        print(s)
    print("Operations:")
    for s in sg._symmetry_ops:
        print(s)
    print("GEPs")
    for s in sg._general_positions:
        print(s)

