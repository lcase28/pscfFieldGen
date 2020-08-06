# PSCF Particle Phase Field Generator

A tool to generate PSCF initial guess files for bulk morphologies involving assemblies of
3D spherical or 2D cylindrical particles.

**NOTE:** This is a beta release. See Notes section at the bottom of this file for special
assumptions made in its operation.

## Contents

 * [Requirements](#requirements)
 * [Installation](#Installation)
    * [Obtaining Source Code](#obtaining-source-code)
    * [Modifying Search Paths](#modifying-search-paths)
        * [Adding to PYTHONPATH](#adding-to-pythonpath)
        * [Anaconda Python](#anaconda-python)
 * [Running pscfFieldGen](#running-pscffieldgen)
    * [Model File](#model-file)
    * [Parameter File](#parameter-file)
    * [Use with pscfpp (C++/Cuda Version)](#use-with-pscfpp)
        * [Unit Cell and Crystal System](#unit-cell)
        * [Space Group Name](#space-group)
        * [Branched Polymers](#branched-polymers)
 * [Special Notes](#special-notes)
        

## Requirements

**Use of this tool requires Python 3.4 or later** as it makes use of some of the newer additions
to the standard library.
This tool has been developed using the Anaconda distribution of Python 3.7 on MacOS, Ubuntu Linux,
and Unix. Because of this, **Python 3.7 is the minimum recommended version**; earlier versions
may encounter unanticipated issues with back-compatibility.

The following required modules should be included in most Python distributions
(Version 3.4 or later) as part of the standard library, and should not require
additional installation. They should, however, be available in the active
python environment.

 * abc
 * argparse
 * copy
 * enum
 * itertools
 * pathlib
 * re
 * string
 * sys

In addition, the following libraries are also required:

 * Numpy
 * Scipy
 * Sympy

All three of these libraries are included standard with Anaconda Python. For installation
instructions for other Python distributions, see the project sites for these packages.

## Installation

### Obtaining Source Code

The source code for the tool is hosted on Github. The easiest way to obtain the code is
with a git version control client. If such a client is installed on your computer,
first `cd` into the directory in which you want to place the pscfFieldGen root
directory. From there, the command

```
> git clone [url]
```

will create a complete working copy of the source code in a
subdirectory called `pscfFieldGen/`. Users without a git client can download a 
`.zip` folder from the Github website and extract its contents into an
analogous folder.

### Modifying Search Paths

To allow the operating system and python interpreter to find the pscfFieldGen program, 
you will have to make some modifications to environment variables.

#### Adding to PYTHONPATH

Many python installations make use of the environment variable PYTHONPATH when searching
for modules. To add pscfFieldGen to this search path, use the following command

```
>  PYTHONPATH=$PYTHONPATH:/path/to/root/pscfFieldGen
```

Executing this on the command line only modifies the path until the end of the terminal
session. To make the change permanent, add the above command to the file ~/.bashrc (on linux)
or to ~/.profile (on Mac OS).

#### Anaconda Python

With Anaconda Python and other conda-managed environments, changes to the PYTHONPATH
environment variable often are not reflected in the python interpreter's effective
path. Instead, one must add pscfFieldGen to the environment's site-packages.
If you use multiple environments, activate the one you wish to install to using
`conda activate` before proceeding.

The easiest way to add the tool to site-packages, is using the conda-develop command
included in the conda-build package. Install this package using 

```
> conda install conda-build
```

When that installation completes, enter the following command

```
> conda-develop /path/to/root/pscfFieldGen
```

where "/path/to/root/" represents the absolute path to the directory from which you cloned
the git repository, as that folder should then contain the pscfFieldGen/ subdirectory.
This will create a file called 'conda.pth' in the environment's site-packages which will
contain the path you gave in the last command.

You can also complete this step manually. To do so, first navigate to your environment's 
site-packages directory. For Anaconda's base environment, this is located at 

```
/path/to/anaconda/lib/pythonX.X/site-packages/
```

where "/path/to/anaconda" is the path to anaconda's installation directory (commonly
~/anaconda3 or similar), and "X.X" represents your Python version. Other environments 
would be found at

```
/path/to/anaconda/envs/{NAME_OF_ENVIRONMENT}/lib/pythonX.X/site-packages/
```

Finally, if you have saved an environment outside of the main Anaconda file tree (for example,
to a user home directory tree on a shared supercomputing system), this would be located instead
at 

```
/path/to/environment/lib/pythonX.X/site-packages/
```

Once in the site-packages directory, create a `.pth` file containing the path to pscfFieldGen.
This file can be named anything, as long as it ends with `.pth`. A name such as "pscfFieldGen.pth"
is one possibility.


## Running pscfFieldGen

Running the software requires 2 files:

 * A Model file specifying filenames and particle positions.
 * A PSCF Fortran parameter file.

In order to simplify input for the user, crystallographic and composition information
are taken from a PSCF parameter file. 
Detailed information about the model file is provided in the next subsection.
Presently, only parameter files consistent with
the Fortran version of the software are supported. Despite this, the program can still
generate initial guesses for use with the C++/Cuda version. 
See the section **Use With C++/Cuda Versions** for special instructions on doing so.

After the tool has been installed, and is discoverable by your Python interpreter,
and after you have produced the two necessary input files, the program can be run
using the command

```
> python -m pscfFieldGen -f model_file
```

In the above command, the -m flag tells the python interpreter to look for the module's
`__main__.py` script. The -f flag tells the program that the model file is about to be
specified, and 'model_file' represents the name of your model file.

pscfFieldGen can also be called with a -t or --trace flag to print a detailed trace of 
the software execution to the terminal. 
This would echo the data read from the model file, as well as
the Lattice, and crystal structure details.
In order to redirect this trace to a file, the command can be executed as:

```
> python -m pscfFieldGen -f model_file -t > trace_file
```

where "trace_file" is the name of the file storing the trace data.

Example files for a range of calculations are included in the `examples` directory in
the root of the project repository.

### Model File

The Model file acts as the primary input for the program. Data in this file is specified
by *case-sensitive* keywords. 
Formatting is flexible, requiring only that individual entries be separated
by some amount of whitespace (spaces, tabs, newlines).

Below is an example of what the contents of a model file might look like for a BCC phase.

```
parameter_file      param_kgrid
output_file         rho_kgrid

core_monomer        0

coord_input_style   basis
N_particles         2
particle_positions
        0.0     0.0     0.0
        0.5     0.5     0.5

finish
```

Three fields are required:

 * `parameter_file` : This keyword would be followed by a single file name referencing
the parameter file. The 'file name' in this case can be any path that would allow the
file to be found from the current directory.
 * `N_particles` : This keyword is followed by an integer giving the number of particles
whose positions will be specified in this input file.
 * `particle_positions` : This keyword would be followed by a list of fractional coordinates
for each particle. For a 2-Dimensional system, this means 2(`N_particles`) coordinates are
expected. For a 3-Dimensional system, 3(`N_particles`) coordinates are expected.
Both `parameter_file` and `N_particles` must be specified before `particle_positions`.

Three additional fields are recommended, but not required. If omitted, default values will
be assumed, and a warning message will be printed informing the user that a default will be used.
Each of these fields can be specified anywhere in the file, but a convention
for each is given in its description.

 * `output_file` : This keyword is followed by a single file name to which the generated field
should be written. As with `parameter_file`, this can be any path recognizable from the current
directory. When specified, it is recommended that you place this field immediately following the 
`parameter_file` specification. If not specified, the program defaults to a file 'rho_kgrid'.
 * `coord_input_style` : This keyword is followed by one of two flags, _motif_ or _basis_.
If _motif_ is specified, the given particle positions will be used along with space group symmetry
to generate a full list of particles in the unit cell. If _basis_ is used, the given particle
positions are assumed to be the full set of particles in the unit cell. When specified, it is
advised to specify it immediately before `N_particles`. If omitted, the default is _motif_.
 * `core_monomer` : This keyword specifies, by monomer id, which monomer should be taken to form
the core of the particles in the assembly. Monomers are indexed starting at 0 and counting up.
(This numbering differs from the Fortran numbering, which starts at 1). The default value is 0.
If specified, it is typically included after the file names, and before structure information.

Finally, the keyword `finish` is followed by no data and identifies the end of the model file.
Use of the `finish` keyword is entirely optional, and is included as an aesthetic option for
users who prefer to have explicit file termination markers.

Presence of any unrecognized keywords will raise an error and terminate the program.

### Parameter File

For detailed information regarding the parameter file format, please see the 
[PSCF User Manual](https://pscf.readthedocs.io/en/latest/param.html)

System specifications are taken from a PSCF Parameter File.
This is done to simplify user input, with the assumption that the user will first generate
the parameter file for the desired calculation, and use it to generate the initial guess.
**Currently, only parameter files consistent with the Fortran version of the software are valid.**

Within the parameter file, the `MONOMERS`, `CHAINS`, `SOLVENTS`, `COMPOSITION`, `INTERACTIONS`,
`UNIT_CELL`, `DISCRETIZATION`, and `BASIS` sections are required, along with the `FINISH` keyword.
Calculation or utility commands (such as `ITERATE`, `SWEEP`, or `KGRID_TO_RGRID`) are not required
for the guess generation to work.

### Use With pscfpp

As mentioned, presently only Parameter files for the Fortran version are supported.
Support for the C++/Cuda parameter files will be added, but is not yet available.
However, because the field file format used by the C++/Cuda version of the software
matches that used in the Fortran version, this generator can still be used.

In order to do so, system data must be formatted into a PSCF Fortran-style parameter file.
Most data can be ported directly between the two formats, taking care to follow the
differing formats (such as the organization of `chi` interactions),
punctuation (such as placement of single quotes around string data in the Fortran format),
and keyword labels (such as `mesh` vs `ngrid` for the spatial discretization).
**Three entries will require special attention.**

#### Unit Cell

The first of these is treatment of the unit cell's crystal system identifier.
The Fortran version's parameter file expects the `crystal_system` to be enclosed in
single quotes, while C++/Cuda version does not. When using this tool for a C++/Cuda
calculation, exclude the quotation marks in the Fortran-style parameter file. Thus,
if the C++/Cuda parameter file contains

```
...
    unitCell    cubic   1.9
...
```

a proper PSCF Fortran parameter file would contain

```
...
UNIT_CELL
dim
            2
crystal_system
            'cubic'
N_cell_param
            1
cell_param
            1.9
...
``` 

but the Fortran-style parameter file used for this tool would contain

```
...
UNIT_CELL
dim
            2
crystal_system
            cubic
N_cell_param
            1
cell_param
            1.9
...
``` 

in order to yield the proper kgrid file. Note the lack of single quotes
around the crystal system.

#### Space Group

The second of these is treatment of the `groupName` entry.
The `groupName` (`group_name` in the Fortran file) identifies the space group of the
system. The key difference between the Fortran and C++/Cuda versions is that, while 
Fortran group names contain spaces between distinct symbols, while the C++/Cuda names
separate distinct symbols with underbars. To accommodate this distinction, when generating
the Fortran-style parameter file, one should use the C++/Cuda group name string in the
same location as the Fortran group name string.
For example, if the C++/Cuda parameter file contains

```
...
    groupName      I_m_-3_m
...
```

a proper PSCF Fortran parameter file would contain

```
...
BASIS
group_name
           'I m -3 m'
...
```

but the Fortran-style parameter file for this tool should instead contain

```
...
BASIS
group_name
           I_m_-3_m
...
```


in order to ensure the proper kgrid file format. Note, again, the lack of 
single quotes, and the updated string format.

Please not that the changes just described for the `crystal_system` and `group_name`
entries are only required to make the kgrid file usable
in C++/Cuda calculations as generated. If the user prefers, they can follow the original
PSCF Fortran conventions for these entries and correct the kgrid file after generation.

#### Branched Polymers

The last change relates to the Polymer chain data. The C++/Cuda code is able to handle
branched polymer architectures which are not supported in the Fortran software, meaning
that the polymer chain structure may not be able to be directly translated. The easiest
approach to correcting this is to simply linearize the branched polymer. That is to say,
take the blocks in the order specified in the C++/Cuda parameter file, and treat them as
laying sequentially along a linear multiblock polymer. In this tool, the polymer structure
is only used to determine the overall volume fraction of each monomer species. Once the
volume fractions are calculated, the difference between a linear and branched polymer
is inconsequential in this field generation algorithm.
As an example, if a polymer in the system has 4 branches emanating from one vertex,
the C++/Cuda parameter file might contain

```
...
   Polymer{                    
       nBlock  4               
       nVertex 5               
       blocks  0  0  0  1  0.30
               1  1  1  2  0.30
               2  0  1  3  0.20
               3  1  1  4  0.20
       phi     1.0             
   }                           
...
```

in which case the Fortran-style parameter file would contain

```
...
CHAINS                                     
N_chain                                    
               1                           
N_block                                    
               4                           
block_monomer                              
               1       2       1       2   
block_length                               
               0.30    0.30    0.20    0.20
...
```

in order to generate the proper overall volume fractions.

Once the field file is generated, the C++/Cuda version of PSCF will be able to convert
the kgrid format into the rgrid or basis formats required for the calculations.

## Special Notes

**Input particle positions should be precise to at least 4 decimal places:**
When generating the unit cell structure, particle positions are considered
identical when all components of fractional coordinate position differ by less
than 0.001. If multiple symmetry operation sequences yield a position that has
been seen before (or after consecutive applications of the same operation yield the
original position) the new position is rejected to avoid duplicates. For precise
coordinates (such as 0.0, 0.25, or 0.5), this will not cause a problem; the imprecision
from truncated values (such as 0.3333, or 0.6667) will cascade through symmetry operations
resulting in some error in the resultant positions. When values are truncated above
the positional tolerance, duplicate particles can be missed. _User control of 
this tolerance can later be added_.

