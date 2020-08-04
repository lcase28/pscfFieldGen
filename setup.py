#!/usr/bin/env python

from distutils.core import setup

setup(  name = "PSCF_Field_Generator",
        version = '0.1',
        description = 'Particle Phase Guess Generator For PSCF Software.',
        author = 'Logan Case',
        packages = ['pscf_field_generator', 
                    'pscf_field_generator.structure',
                    'pscf_field_generator.pscfFileManagers',
                    'pscf_field_generator.util',
                   ]
        
