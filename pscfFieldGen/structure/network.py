
import numpy as np

import pscfFieldGen.filemanagers.pscf as pscf
import pscfFieldGen.util.contexttools as contexttools


class NetworkCrystal:
    """
    A class wrapping symmetry-adapted basis function file for the levelset method.
    """
    
    def __init__(self, filename, core_options=None):
        """
        Initialize the network field.
        
        Parameters
        ----------
        filename : string or Path
            The path to the template field file. The given
            path or string should either be absolute or relative
            to the working directory.
        core_options : list-like of int, optional
            The options for core monomer of the network.
            If not specified, all monomers are enabled as options.
        """
        self._sym = pscf.SymFieldFile(filename)
        self._core_options = core_options
        if core_options is None:
            self._core_opts = [i for i in range(self.N_monomer)]
        # Determine weighting of each star for generation.
        self._star_weights = np.zeros(self.N_star)
        fields = self._sym.fields
        for i in range(1,self.N_star):
            self._star_weights[i] = self._sym.fields[i][0] # use monomer 0 to set weight
    
    @classmethod
    def fromFile(cls, wordstream, entrykey, param):
        """ Initialize and return a new NetworkCrystal from a file.
        
        Parameters
        ----------
        wordstream : util.stringTools.FileParser
            The data stream for the file.
        entrykey : string
            The key marking entry into this block.
            Should be "NetworkCrystal{".
        param : pscfFieldGen.filemanagers.pscf.ParamFile
        """
        if not entrykey == "NetworkCrystal{":
            raise(ValueError("Expected Key 'NetworkCrystal{{'; got '{}'".format(entrykey)))
        nmon = param.N_monomer
        
        core_options = [i for i in range(nmon)] # default
        core_option_set = False
        hasfield = False
        end_block = False
        while not end_block:
            word = next(wordstream)
            if word.lower() == "star_file":
                fname = next(wordstream)
                hasfield = True
            elif word.lower() == "core_option":
                val = wordstream.next_int()
                if not val < nmon:
                    raise(ValueError("Core option {} exceeds nmonomer {}.".format(val,nmon)))
                if not core_option_set:
                    core_options = []
                    core_option_set = True
                core_options.append(val)
            elif word == "}":
                end_block = True
            else:
                msg = "Unrecognized Key '{}' in MotifCrystal{{...}} block."
                raise(ValueError(msg.format(word)))
        if not hasfield:
            filetype = "Symmetry-Adapted Basis Field File"
            msg = "{} template must be provided for NetworkCrystal with keyword 'star_file'."
            raise(RuntimeError(msg.format(filetype)))
        return cls(fname, core_options)
    
    @property
    def N_star(self):
        return self._sym.N_star
    
    @property
    def N_monomer(self):
        return self._sym.N_monomer
    
    def update(self, frac, cell_param=None):
        """
        Update the underlying symmetry-adapted field file.
        
        Based on the system composition given in frac, the
        core monomer is chosen, and basis function coefficients
        are updated to reflect this. This is done to retain the
        weighting of each star relative to that in the input file.
        
        Parameters
        ----------
        frac : list-like
            The overall volume fraction of each monomer in the system.
        cell_param : list-like, Optional
            New cell parameter values for the unit cell. If included,
            no check is done to ensure compatibility with current
            definition; caller is assumed to guarantee this.
        
        Returns
        -------
        core_monomer : int
            Index of the core monomer.
        """
        # choose core monomer
        cmon = self._core_options[0]
        cfrac = frac[cmon]
        for i in self._core_options:
            if frac[i] < cfrac:
                cmon = i
                cfrac = frac[i]
        noncore = 1.0 - cfrac # total fraction of non-core monomers
        for i in range(self.N_star):
            sweight = self._star_weights[i]
            for j in range(1,self.N_monomer):
                if j == cmon:
                    self._sym.fields[i][j] = sweight
                else:   
                    self._sym.fields[i][j] = -sweight * (frac[j] / noncore)
        if cell_param is not None:
            self._sym.cell_param = cell_param
        return cmon
    
    def writeField(self, file_obj):
        """ Output the field to the open, writable File object. """
        self._sym.write(file_obj)

