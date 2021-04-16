from .paramcomposite import ParamComposite
from .record import Record

import numpy as np

class Polymer():
    """ Wrapper for ParamComposite instances representing Polymer{...} blocks. """
    
    def __init__(self, source):
        """
        Setup new Polymer instance from existing ParamComposite.
        
        Parameters
        ----------
        source : ParamComposite
            The ParamComposite holding the data
        """
        if not source.label_ == "Polymer":
            raise(ValueError("Invalid source given for polymer:\n{}".format(source)))
        self.source = source
    
    # Mirroring behavior of ParamComposite source.
    
    @property
    def nBlock(self):
        return self.source.nBlock
    
    @property
    def nVertex(self):
        return self.source.nVertex
    
    @property
    def blocks(self):
        return self.source.blocks
    
    @property
    def phi(self):
        try:
            return self.source.phi.value()
        except(AttributeError):
            return None
    
    @property
    def mu(self):
        try:
            return self.source.mu.value()
        except(AttributeError):
            return None
    
    def write(self, filename):
        self.source.write(filename)
    
    def __str__(self):
        return str(self.source)
    
    # Specialized methods and properties
    
    @property
    def hasPhi(self):
        if self.phi == None:
            return False
        else:
            return True
    
    @property
    def hasMu(self):
        if self.mu == None:
            return False
        else:
            return True
    
    def totalLength(self,monomer=-1):
        """ 
        Return the combined length of all blocks of given monomer type.
        
        If no monomer is specified, the total length of all blocks is given.
        
        Parameters
        ----------
        monomer : int, optional
            The monomer index to be counted. If not specified, all blocks will
            be counted.
        """
        ntot = 0.0
        for i in range(self.nBlock.value()):
            if monomer == -1 or monomer == self.blocks.line(i).value(1):
                ntot += self.blocks.line(i).value(4)
        return ntot
    
    def monomerFraction(self, monomer):
        """ Calculate the volume fraction of specified monomer. """
        if self.__ntot is not None:
            ntot = self.__ntot
        else:
            ntot = self.totalLength()
        nb = self.totalLength(monomer)
        return nb / ntot
    
    def composition(self, nMonomer):
        """ Calculate the volume fraction of all monomers in the molecule. 
        
        Returns
        -------
        frac : numpy.ndarray
            The monomer volume fractions in the molecule.
            Each indexed element holds the fraction for the correspondingly
            indexed monomer type. Monomer 0 will be accessed with frac[0].
        """
        # Avoid repeated calculation of total length
        self.__ntot = self.totalLength()
        # Calculate Fractions
        f = []
        for i in range(nMonomer):
            f.append(self.monomerFraction(i))
        # Remove self.__ntot until next call
        self.__ntot = None
        return np.array(f)
    
    def monomerLengths(self, nMonomer):
        """ Calculate the total length of each monomer type in the molecule.
        
        Returns
        -------
        lengths : numpy.ndarray
            The total length occupied by each monomer type, lumping all
            blocks together. Array index corresponds to monomer index.
        """
        l = []
        for i in range(nMonomer):
            l.append(self.totalLength(i))
        return np.array(l)

class ParamFile(ParamComposite):
    """
    Extension of ParamComposite class to specialize handling as full file.
    """
    
    def __init__(self,filename):
        """ Create ParamFile instance from file `filename` """
        inFile  = open(filename, 'r')
        lines = inFile.readlines() 
        records = []
        for line in lines:
            records.append(Record(line))
        super().__init__()
        self.read(records, 0)
        # Convert polymers to Polymer objects
        npoly = self.Mixture.nPolymer.value()
        if npoly == 1:
            self.Mixture.Polymer = [Polymer(self.Mixture.Polymer)]
        else:
            for i in range(npoly):
                self.Mixture.Polymer[i] = Polymer(self.Mixture.Polymer[i])
        self._N_cell_param, self._cell_param = self._parse_cell_params()
    
    @property
    def dim(self):
        """ Dimensionality of the system in the parameter file (1, 2, or 3) """
        # confirm at least one dimension
        try:
            a = self.mesh.value(0)
        except(IndexError):
            raise(ValueError("System improperly defined to determine dimensionality"))
        
        # check at least 2 dimensions
        try:
            a = self.mesh.value(1)
        except(IndexError):
            return 1
        
        # check at least 3 dimensions
        try:
            a = self.mesh.value(2)
            return 3
        except(IndexError):
            return 2
            pass
    
    __nCellParamRef = { (1,'lamellar')      :   1, \
                        (2,'square')        :   1, \
                        (2,'hexagonal')     :   1, \
                        (2,'rectangular')   :   2, \
                        (2,'oblique')       :   3, \
                        (3,'cubic')         :   1, \
                        (3,'tetragonal')    :   2, \
                        (3,'orthorhombic')  :   3, \
                        (3,'monoclinic')    :   4, \
                        (3,'hexagonal')     :   2, \
                        (3,'trigonal')      :   2, \
                        (3,'triclinic')     :   6    }
    
    def _parse_cell_params(self):
        crys = self.file.unitCell.value(0)
        nparam = self.__class__.__nCellParamRef.get( (self.dim, crys) )
        param = [ self.file.unitCell.value(i+1) for i in range(nparam) ]
        return nparam, np.array(param)
    
    @property
    def N_cell_param(self):
        return self._N_cell_param
    
    @property
    def cell_param(self):
        return self._cell_param
    
    @cell_param.setter
    def cell_param(self, val):
        if not len(val) == self.N_cell_param:
            msg = "Expected {} cell params; got {}, {}."
            raise(ValueError(msg.format(self.N_cell_param, len(val), val)))
        for i in range(self.N_cell_param):
            self._cell_param[i] = val[i]
            self.file.unitCell.value[i+1] = val[i]
    
    
