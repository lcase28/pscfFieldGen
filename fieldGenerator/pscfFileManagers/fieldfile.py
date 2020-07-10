from abc import ABC, abstractmethod
from .iotools import IO, IOException
import numpy as np
from .version import Version
import string
import sys

class FieldFile(ABC):
    '''
        Hold data in a PSCF field file. 

        A FieldFile object contains the data in a field file in the PSCF
        symmetry-adapted Fourier expansion format (see web user manual).
        It can be used to represent either omega (chemical potential) or
        rho (volume fraction) fields. a

        The constructor reads a field file, creates a FieldFile object to store
        the data, and stores all of the parameters and field coefficients in 
        attributes of the object.

        Attributes:
        dim            -- [int] number of periodic spatial directions
        crystal_system -- [string] label of Bravais crystal system
        N_cell_param   -- [int] number of unit cell parameters
        cell_param     -- [list] list of real unit cell parameters
        group          -- [string] name of space group
        N_monomer      -- [int] number of monomer types
        N_star         -- [int] number of basis functions (or stars)
        fields         -- [list] list of list of coefficients

        The attribute field[i] is is a list (or array) of coefficients
        for a field associated with monomer type number i. The element
        field[i][j] is the coefficient of basis function j in the field
        associated with monomer type i. 
    '''

    # "Public" methods
    def __init__(self,filename):
        '''
        Read a PSCF symmetry-adapted field file, and create a new object.

        Argument:
        filename -- name of file

        The file named filename is opened and closed within this function.
        '''
        self.filename = filename
        self.file = open(filename, 'r')
        self._io   = IO()
        file = self.file

        # Read version line
        self.version = Version(self.file)

        self._input_unit_cell()
        self.group_name = self._input_var('char')
        self.N_monomer = self._input_var('int')
        self._readField();
        #self.N_star = self._input_var('int')

        ## Define empty lists
        #self.fields = []
        #self.waves = []
        #self.counts = []

        #for i in range(self.N_star):
        #    data = file.readline().split()
        #    if len(data) != self.N_monomer + self.dim + 1:
        #        raise IoException('Incorrect number of elements in field line')
        #    j = 0

        #    # Read field coefficients
        #    self.fields.append([])
        #    for k in range(self.N_monomer):
        #        value = float(data[j])
        #        self.fields[i].append(value)
        #        j += 1

        #    # Read field coefficients
        #    self.waves.append([])
        #    for k in range(self.dim):
        #        value = int(data[j])
        #        self.waves[i].append(value)
        #        j += 1

        #    # Read star_count
        #    self.counts.append(int(data[j]))

        self.file.close()
        self.file = None

    def write(self, file, major=1, minor=0):
        '''
        PURPOSE
           Write field to file in PSCF symmetry-adapted format.
        ARGUMENTS
           file  - file object or file name string
           major - major file format version number
           minor - minor file format version number
        COMMENT
           if file is a field object, it must be open for writing
        '''

        # If file argument is a string, open a file of that name
        if type(file) == type('thing'):
            temp = open(file,'w')
            file = temp
        self.file = file
           
        self.version.major = major
        self.version.minor = minor
        self.version.write(file)

        self._output_unit_cell()
        self._output_var('char', 'group_name')
        self._output_var( 'int', 'N_monomer')
        self._outputField()
        #self._output_var( 'int', 'N_star')

        #for i in range(self.N_star):
        #    for k in range(self.N_monomer):
        #        file.write('%20.12E' % self.fields[i][k])
        #    file.write('    ')
        #    for k in range(self.dim):
        #        file.write('%4d' % self.waves[i][k])
        #    file.write('%6d' % self.counts[i])
        #    file.write("\n")

        file.close()
        self.file = None
    
    # Abstract Methods
    @abstractmethod
    def addMonomer(self):
        ''' 
        PURPOSE
           Add a field with a coefficients set to common value
        ARGUMENTS
           value - value of all coefficients for new momoner
        COMMENT
            N_momomer is incremented by 1.
            Inheriting class must define update of the field variables
        '''
        self.N_monomer += 1
    
    @abstractmethod
    def duplicateMonomer(self, i):
        ''' 
        PURPOSE
           Add a field by duplicating field i
        ARGUMENTS
           i - index in range [0, N_monomer-1]
        COMMENT
            N_momomer is incremented.
            Inheriting class must define update of field variable.
        '''
        self.N_monomer += 1
    
    @abstractmethod
    def switchMonomers(self, i, j):
        '''
        PURPOSE
           Switch coefficients of fields i and j
        ARGUMENTS
           i - index in range [0, N_monomer-1]
           j - index in range [0, N_monomer-1]
        '''
        pass

    # "Private" methods
    
    # Abstract Methods
    @abstractmethod
    def _readField(self):
        pass
    
    @abstractmethod
    def _outputField(self):
        pass

    # Wrappers for input_... output_.. methods of IO)

    def _input_var(self, type, comment = None, f='A'):
        return self._io.input_var(self.file, type, comment, f)

    def _input_vec(self, type, n=None, comment=None, s='R',f='A'):
        return self._io.input_vec(self.file, type, n, comment, s, f)

    # Output methods (output by name)
    def _output_var(self, type, name, f='A'):
        if name in self.__dict__:#.has_key(name):
            data = self.__dict__[name]
            self._io.output_var(self.file, type, data, name, f)

    def _output_vec(self, type, name, n=None, s='R', f='A'):
        if name in self.__dict__:#.has_key(name):
            data = self.__dict__[name]
            self._io.output_vec(self.file, type, data, n, name, s, f)

    def _input_unit_cell(self):
        ''' Analog of subroutine _input_unit_cell in unit_cell_mod.f '''
        self.dim = self._input_var('int')
        self.crystal_system = self._input_var('char')
        self.N_cell_param = self._input_var('int')
        self.cell_param = self._input_vec('real',self.N_cell_param)

    def _output_unit_cell(self):
        ''' Analog of subroutine _output_unit_cell in unit_cell_mod.f '''
        self._output_var('int', 'dim')
        self._output_var('char', 'crystal_system')
        self._output_var('int', 'N_cell_param')
        self._output_vec('real', 'cell_param', self.N_cell_param)

class SymFieldFile(FieldFile):
    '''
        Hold data in a PSCF field file. 
    
        A FieldFile object contains the data in a field file in the PSCF
        symmetry-adapted Fourier expansion format (see web user manual).
        It can be used to represent either omega (chemical potential) or
        rho (volume fraction) fields. a
    
        The constructor reads a field file, creates a FieldFile object to store
        the data, and stores all of the parameters and field coefficients in 
        attributes of the object.
    
        Attributes:
        dim            -- [int] number of periodic spatial directions
        crystal_system -- [string] label of Bravais crystal system
        N_cell_param   -- [int] number of unit cell parameters
        cell_param     -- [list] list of real unit cell parameters
        group          -- [string] name of space group
        N_monomer      -- [int] number of monomer types
        N_star         -- [int] number of basis functions (or stars)
        fields         -- [list] list of list of coefficients
    
        The attribute field[i] is is a list (or array) of coefficients
        for a field associated with monomer type number i. The element
        field[i][j] is the coefficient of basis function j in the field
        associated with monomer type i. 
    '''

    # "Public" methods

    def __init__(self,filename):
        '''
        Read a PSCF symmetry-adapted field file, and create a new object.

        Argument:
        filename -- name of file

        The file named filename is opened and closed within this function.
        '''
        # Define field-specific data
        self.N_star = 1 # actual value read during _readField call
        # Define empty lists
        self.fields = []
        self.waves = []
        self.counts = []
        
        # Field read method called from super class
        super().__init__(filename)

    def write(self, file, major=1, minor=0):
        '''
        PURPOSE
           Write field to file in PSCF symmetry-adapted format.
        ARGUMENTS
           file  - file object or file name string
           major - major file format version number
           minor - minor file format version number
        COMMENT
           if file is a field object, it must be open for writing
        '''
        super().write(file, major, minor)

    def addMonomer(self, value = 0.0, **kwargs):
        ''' 
        PURPOSE
           Add a field with a coefficients set to common value
        ARGUMENTS
           value - value of all coefficients for new momoner
        COMMENT
            N_momomer is incremented by 1, and new field is last
        '''
        super().addMonomer(**kwargs)
        for k in range(self.N_star):
            self.fields[k].append(value)

    def duplicateMonomer(self, i):
        ''' 
        PURPOSE
           Add a field by duplicating field i
        ARGUMENTS
           i - index in range [0, N_monomer-1]
        COMMENT
            N_momomer is incremented, and duplicate field is last
        '''
        super().duplicateMonomer(i)
        for k in range(self.N_star):
            self.fields[k].append(self.fields[k][i])

    def switchMonomers(self, i, j):
        '''
        PURPOSE
           Switch coefficients of fields i and j
        ARGUMENTS
           i - index in range [0, N_monomer-1]
           j - index in range [0, N_monomer-1]
        '''
        for k in range(self.N_star):
            temp = self.fields[k][i]
            self.fields[k][i] = self.fields[k][j]
            self.fields[k][j] = temp

    # "Private" methods
    
    # Overriding inherited abstract methods
    def _readField(self):
        self.N_star = self._input_var('int')
        for i in range(self.N_star):
            data = self.file.readline().split()
            if len(data) != self.N_monomer + self.dim + 1:
                raise(IoException('Incorrect number of elements in field line'))
            j = 0

            # Read field coefficients
            self.fields.append([])
            for k in range(self.N_monomer):
                value = float(data[j])
                self.fields[i].append(value)
                j += 1

            # Read field coefficients
            self.waves.append([])
            for k in range(self.dim):
                value = int(data[j])
                self.waves[i].append(value)
                j += 1

            # Read star_count
            self.counts.append(int(data[j]))
    
    def _outputField(self):
        self._output_var( 'int', 'N_star')
        for i in range(self.N_star):
            for k in range(self.N_monomer):
                self.file.write('%20.12E' % self.fields[i][k])
            self.file.write('    ')
            for k in range(self.dim):
                self.file.write('%4d' % self.waves[i][k])
            self.file.write('%6d' % self.counts[i])
            self.file.write("\n")

class CoordFieldFile(FieldFile):
    '''
        Hold data in a PSCF field file. 
    
        A FieldFile object contains the data in a field file in the PSCF
        coordinate grid format (see web user manual).
        It can be used to represent either omega (chemical potential) or
        rho (volume fraction) fields. a
    
        The constructor reads a field file, creates a FieldFile object to store
        the data, and stores all of the parameters and field coefficients in 
        attributes of the object.
    
        Attributes:
        dim            -- [int] number of periodic spatial directions
        crystal_system -- [string] label of Bravais crystal system
        N_cell_param   -- [int] number of unit cell parameters
        cell_param     -- [list] list of real unit cell parameters
        group          -- [string] name of space group
        N_monomer      -- [int] number of monomer types
        ngrid         -- [int] list of grid point counts in each dimension
        fields         -- [list] numpy ndarray of spatial value
    
        The attribute field[:,i] is is a list (or array) of field values
        for a field associated with monomer type number i.
    '''

    # "Public" methods

    def __init__(self,filename):
        '''
        Read a PSCF symmetry-adapted field file, and create a new object.

        Argument:
        filename -- name of file

        The file named filename is opened and closed within this function.
        '''
        self.ngrid = [1] # Actual Value read during _readField call
        self.fields = []
        
        # _readField call made by super
        super().__init__(filename)

    def write(self, file, major=1, minor=0):
        '''
        PURPOSE
           Write field to file in PSCF coordinate-grid format.
        ARGUMENTS
           file  - file object or file name string
           major - major file format version number
           minor - minor file format version number
        COMMENT
           if file is a field object, it must be open for writing
        '''
        super().write(file, major, minor)

    # Overriding inherited abstract methods
    def addMonomer(self, value = 0.0):
        '''
        PURPOSE
           Add a field with a coefficients set to common value
        ARGUMENTS
           value - value of all coefficients for new momoner
        COMMENT
            N_momomer is incremented by 1, and new field is last
        '''
        self.N_monomer += 1
        newcol = value * np.ones((self.gridPoints,1))
        self.fields = np.append(self.fields, newcol, axis=1)

    def duplicateMonomer(self, i):
        ''' 
        PURPOSE
           Add a field by duplicating field i
        ARGUMENTS
           i - index in range [0, N_monomer-1]
        COMMENT
            N_momomer is incremented, and duplicate field is last
        '''
        self.N_monomer += 1
        self.fields = np.append(self.field, self.field[:,i])

    def switchMonomers(self, i, j):
        '''
        PURPOSE
           Switch coefficients of fields i and j
        ARGUMENTS
           i - index in range [0, N_monomer-1]
           j - index in range [0, N_monomer-1]
        '''
        temp = self.fields[:,i]
        self.fields[:,i] = self.fields[:,j]
        self.fields[:,j] = temp

    # "Private" methods
    
    # overriding inherited private abstract methods
    def _readField(self):
        self.ngrid = self._input_vec('int', n=self.dim, comment='ngrid')
        self.gridPoints = np.prod(self.ngrid)
        self.fields = np.zeros((self.gridPoints,self.N_monomer))
        for i in range(self.gridPoints):
            self.fields[i,:] = np.array(self._input_vec('real', n=self.dim, comment = None, s = 'R', f = 'N'))
    
    def _outputField(self):
        self._output_vec('int', 'ngrid', n=self.dim, s='R', f='A')
        self._nextFieldLine = []
        for i in range(self.gridPoints):
            self._nextFieldLine = self.fields[i,:]
            self._output_vec('real', '_nextFieldLine', n=self.N_monomer, s='R', f='N')
        delattr(self, '_nextFieldLine')
            
class WaveVectFieldFile(FieldFile):
    '''
        Hold data in a PSCF field file. 
    
        A FieldFile object contains the data in a field file in the PSCF
        wavevector grid Fourier expansion format (see web user manual).
        It can be used to represent either omega (chemical potential) or
        rho (volume fraction) fields. a
    
        The constructor reads a field file, creates a FieldFile object to store
        the data, and stores all of the parameters and field coefficients in 
        attributes of the object.
    
        Attributes:
        dim            -- [int] number of periodic spatial directions
        crystal_system -- [string] label of Bravais crystal system
        N_cell_param   -- [int] number of unit cell parameters
        cell_param     -- [list] list of real unit cell parameters
        group          -- [string] name of space group
        N_monomer      -- [int] number of monomer types
        ngrid          -- [int] list of real-space dimension-wise grid counts
        fields         -- [numpy.ndarray] numpy ndarray of fourier coefficients
    
        The attribute field[:,i] is is a list (or array) of coefficients
        for a field associated with monomer type number i. The element
        field[i][j] is the coefficient of basis function j in the field
        associated with monomer type i. 
    '''

    # "Public" methods

    def __init__(self,filename,skipField=False):
        '''
        Read a PSCF symmetry-adapted field file, and create a new object.

        Argument:
        filename -- name of file
        skipField -- when input file is being used for templating,
                    this will trigger initialization to not attempt to
                    read the field values. (this way, template for new phase
                    does not need to exactly meet grid counts)

        The file named filename is opened and closed within this function.
        '''
        self.ngrid = [1] # Actual Value read during _readField call
        self.fields = []
        
        self.skipFieldFlag = skipField
        
        # _readField call made by super
        super().__init__(filename)
        
        self.skipFieldFlag = False

    def write(self, file, major=1, minor=0):
        '''
        PURPOSE
           Write field to file in PSCF coordinate-grid format.
        ARGUMENTS
           file  - file object or file name string
           major - major file format version number
           minor - minor file format version number
        COMMENT
           if file is a field object, it must be open for writing
        '''
        super().write(file, major, minor)

    # Overriding inherited abstract methods
    def addMonomer(self, value = 0.0):
        '''
        PURPOSE
           Add a field with a coefficients set to common value
        ARGUMENTS
           value - value of all coefficients for new momoner
        COMMENT
            N_momomer is incremented by 1, and new field is last
        '''
        self.N_monomer += 1
        newcol = value * np.ones((self.gridPoints,1))
        self.fields = np.append(self.fields, newcol, axis=1)

    def duplicateMonomer(self, i):
        ''' 
        PURPOSE
           Add a field by duplicating field i
        ARGUMENTS
           i - index in range [0, N_monomer-1]
        COMMENT
            N_momomer is incremented, and duplicate field is last
        '''
        self.N_monomer += 1
        self.fields = np.append(self.field, self.field[:,i])

    def switchMonomers(self, i, j):
        '''
        PURPOSE
           Switch coefficients of fields i and j
        ARGUMENTS
           i - index in range [0, N_monomer-1]
           j - index in range [0, N_monomer-1]
        '''
        temp = self.fields[:,i]
        self.fields[:,i] = self.fields[:,j]
        self.fields[:,j] = temp
    
    def fieldSimilarity(self, target):
        """
        Return a scalar metric of field similarity for each monomer
        in the field files.
        
        Implemented as the inner product of normalized fourier
        space fields.
        
        PARAMETERS
        ----------
        target : WaveVectFieldFile
            The field file to be compared to this one
        
        RETURNS
        -------
        sim : 1xN_monomer numpy array
            Each element contains the similarity for the corresponding
            monomer. 1 indicates high similarity. 0 indicate orthogonality.
        
        THROWS
        ------
        TypeError
            If target is not a WaveVectFieldFile
        ValueError
            If target is not comparable to current
        """
        msgbase = "Field similarity check requires "
        if type(self) is not type(target):
            msg = msgbase + "type match.\n\tGiven: %s. Needs: %s"
            raise(TypeError(msg.format(str(type(target)),str(type(self)))))
        
        if not self.canCompare(target):
            msg = msgbase + "comparable fields. Mismatch in N_monomer, or n_grid."
            raise(ValueError(msg))
        
        field1 = np.conj(self.fields)
        field2 = target.fields
        
        norm1 = np.linalg.norm(field1,axis=0)
        norm2 = np.linalg.norm(field2,axis=0)
        normtot = np.multiply(norm1,norm2)
        
        inprod = np.diag(np.tensordot(field1,field2,axes=(0,0)))
        
        simComplex = np.divide(inprod,normtot) # This will return the complex similarity
        sim = np.sqrt(np.multiply(simComplex,np.conj(simComplex))) # take magnitude of each similarity
        
        return np.real(sim)
    
    def canCompare(self,target):
        if type(self) is not type(target):
            return False
        
        if not self.N_monomer == target.N_monomer:
            return False
        
        if not np.array_equiv(self.ngrid,target.ngrid):
            return False
            
        return True
    
    # "Private" methods
    
    # overriding inherited private abstract methods
    def _readField(self):
        self.ngrid = self._input_vec('int', n=self.dim, comment='ngrid')
        gp = int(self.ngrid[0] / 2) + 1
        for i in range(self.dim-1):
            gp = gp * self.ngrid[1+i]
        self.gridPoints = gp
        self.fields = 1j*np.zeros((self.gridPoints,self.N_monomer))
        if not self.skipFieldFlag:
            for i in range(self.gridPoints):
                try:
                    self.fields[i,:] = self._nextFieldLine()
                except ValueError as err:
                    msg = str(err) + "Line {} of {}".format(i,gp)
                    raise(ValueError(msg))
    
    def _nextFieldLine(self):
        s = self.file.readline() # get next line
        sMon = s.split(')')
        vals = np.zeros((2,self.N_monomer))
        for i in range(self.N_monomer):
            try:
                re, im = self._getComponents(sMon[i])
            except ValueError as err:
                msg = str(err) + "Monomer {}\n".format(i)
                raise(ValueError(msg))
            #vals[0,i], vals[1,i] = self._getComponents(sMon[i])
            #vals.append(self._getRealComponent(sMon[i]))
            vals[0,i] = re
            vals[1,i] = im
        retVal = vals[0,:] + 1j * vals[1,:]
        return retVal
        
    def _getComponents(self, sbase):
        s = sbase.strip()
        s = s.strip('()')
        slist = s.split(',')
        try:
            re = float(slist[0].strip())
            im = float(slist[1].strip())
        except(ValueError):
            raise(ValueError("Error Converting string {} to float.\n".format(sbase)))
        return re, im
        #return float(slist[0].strip()), float(slist[1].strip())
        
    
    def _outputField(self):
        self._output_vec('int', 'ngrid', n=self.dim, s='R', f='A')
        formstr = "    ({:.4E},{:.4E}) "
        for i in range(self.gridPoints):
            s = ""
            for j in range(self.N_monomer):
                re = np.real(self.fields[i,j])
                im = np.imag(self.fields[i,j])
                s += formstr.format(re,im)
            s += "\n"
            self.file.write(s)
            

