import numpy as np
import pytest

from pscfFieldGen.structure.lattice import Lattice, Vector, cartesianLattice

class Test_Lattice:
    """
    Unit Tests for Lattice Class.
    
    Tests:
    __init__ : 
        dim enforcement
        
        basis shape enforcement
        
    basisFromParameters : 
        enforce required parameters
        
        2D Accuracy
            square
            oblique
            
        3D Accuracy
            cubic
            triclinic
            
    Copy : 
        Ensure new instance created (not ... is ...)
        Ensure dim and basis match
        Ensure evaluate equal.
        
    __eq__ :
        Type Enforcement
        
        Dim Check
        
        Accuracy
        
    dim : 
        dim == lat.dim
        lat._dim == lat.dim
    
    basis : 
        Proper assignment
        
        Updating
        
    volume : 
        Calculation
        
        Updating
    
    metricTensor :
        Calculation
        
        Updating
    
    reciprocalLattice :
        Calculation
            test dim, basis
        
        Updating
            test dim, basis
        
    isReciprocal : 
        
    
    latticeParameters : 
        initialize from parameters -- keep list
        compare list to latticeParameters
    
    ParameterList : 
        Compare output to input parameters
        
        Test Updating
            Accuracy
            
            Enforcement
        
    """
    
    def test_init_dim_enforcement():
        """ Ensure dim enforced properly.
        
        Should only accept dim = 2 or dim = 3.
        """
        with pytest.raises(ValueError):
            bad_lattice = Lattice(1, np.array([1]))
        good_lattice_2 = Lattice(2, np.eye(2))
        good_lattice_3 = Lattice(3, np.eye(3))
        with pytest.raises(ValueError):
            bad_lattice = Lattice(4,np.eye(4))
        with pytest.raises(ValueError):
            bad_lattice = Lattice(10,np.eye(10))
    
    def test_init_basis_shape_enforcement():
        """ Ensure basis shape enforced. 
        
        A 2D lattice should only accept a 2x2 array-like
        A 3D lattice should only accept a 3x3 array-like
        """
        basis_2D = np.eye(2)
        basis_3D = np.eye(3)
        good_lattice_2 = Lattice(2,basis_2D)
        good_lattice_3 = Lattice(3, basis_3D)
        # Not matching dim
        with pytest.raises(ValueError):
            bad_lattice = Lattice(2, basis_3D)
        with pytest.raises(ValueError):
            bad_lattice = Lattice(3, basis_2D)
        # Incorrect shape by flattening
        with pytest.raises(ValueError):
            bad_lattice = Lattice(2, basis_2D.flatten())
        with pytest.raises(ValueError):
            bad_lattice = Lattice(3, basis_3D.flatten())
    
    def test_basisFromParameters_enforcement():
        paramlist = ["a","b","gamma","c","alpha","beta"]
        cartesian_ref = np.eye(2)
        cartesian_param = [1,1,90]
        args
        
    
    def test_basisFromParameters_2D():
        param_names = ["a","b","gamma"]
        # non-unit square
        square_ref = 2 * np.eye(2)
        cartesian_param = [2,2,90]
        # oblique
        oblique_ref = np.array( [ [ 2.0, 0.0 ], [ 1.0, 1.0 ] ] )
        oblique_param = [2, np.sqrt(2), 45.0]
    
    def test_basisFromParameters_3D():
        cartesian_ref = np.eye(3)
        cubic_ref = 2 * np.eye(3)
    
    
