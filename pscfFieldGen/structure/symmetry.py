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

