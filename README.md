# PSCF Field Generator

A tool to generate PSCF initial guess files for bulk morphologies involving particle,
network or lamellar structures.

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
        * [Particle Structures](#particlegenerator)
        * [Network Structures](#networkgenerator)
        * [Lamellar Structures](#lamellargenerator)
    * [Parameter File](#parameter-file)
 * [Special Notes](#special-notes)
        

## Requirements

[Back to Top](#pscf-field-generator)

**Use of this tool requires Python 3.5 or later** as it makes use of some of the newer additions
to the standard library.
This tool has been developed using the Anaconda distribution of Python 3.7 on MacOS, Ubuntu Linux,
and Unix. Because of this, **Python 3.7 is the minimum recommended version**; earlier versions
may encounter unanticipated issues with back-compatibility.

The following required modules should be included in most Python distributions
(Version 3.5 or later) as part of the standard library, and should not require
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

 * numpy
 * scipy

All three of these libraries are included standard with Anaconda Python. For installation
instructions for other Python distributions, see the project sites for these packages.

To check that the modules are available in the current python environment, open an
interactive python terminal and attempt to import each library using `import [library_name]`
such as `import abc`.

## Installation

[Back to Top](#pscf-field-generator)

### Obtaining Source Code

[Back to Top](#pscf-field-generator)

The source code for the tool is hosted on Github. The easiest way to obtain the code is
with a git version control client. If such a client is installed on your computer,
first `cd` into the directory in which you want to place the pscfFieldGen root
directory. From there, the command

```
$ git clone https://github.umn.edu/case0234/pscfFieldGen.git
```

will create a complete working copy of the source code in a
subdirectory called `pscfFieldGen/`. Users without a git client can download a 
`.zip` folder from the Github website and extract its contents into an
analogous folder.

### Modifying Search Paths

[Back to Top](#pscf-field-generator)

To allow the operating system and python interpreter to find the pscfFieldGen program, 
you will have to make some modifications to environment variables.

#### Adding to PYTHONPATH

[Back to Top](#pscf-field-generator)

Many python installations make use of the environment variable PYTHONPATH when searching
for modules. To add pscfFieldGen to this search path, use the following command

```
$  PYTHONPATH=$PYTHONPATH:/path/to/root/pscfFieldGen
```

Executing this on the command line only modifies the path until the end of the terminal
session. To make the change permanent, add the above command to the file ~/.bashrc (on linux)
or to ~/.profile (on Mac OS).

#### Anaconda Python

[Back to Top](#pscf-field-generator)

With Anaconda Python and other conda-managed environments, changes to the PYTHONPATH
environment variable often are not reflected in the python interpreter's effective
path. Instead, one must add pscfFieldGen to the environment's site-packages.
If you use multiple environments, activate the one you wish to install to using
`conda activate` before proceeding.

The easiest way to add the tool to site-packages, is using the conda-develop command
included in the conda-build package. Install this package using 

```
$ conda install conda-build
```

When that installation completes, enter the following command

```
$ conda-develop /path/to/root/pscfFieldGen
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

[Back to Top](#pscf-field-generator)

Running the software requires 2 files:

 * A Model file specifying reference files and structures
 * A PSCF parameter file.

In order to simplify input for the user, crystallographic and composition information
are taken from a PSCF parameter file. 
Detailed information about the model file is provided in the next subsection.

After the tool has been installed, and is discoverable by your Python interpreter,
and after you have produced the two necessary input files, the program can be run
using the command

```
$ python -m pscfFieldGen -f model_file
```

In the above command, the -m flag tells the python interpreter to look for the module's
`__main__.py` script. The -f flag tells the program that the model file is about to be
specified, and 'model_file' represents the name of your model file.

By default, the program will echo the contents of the model file as it is read to the 
terminal.
(_Note:_ The echo will not follow the formatting of the model file. Instead, each token
or word read from the file will be output within an event statement of the form
`ECHO   : Token '0' read from file.`)
The level of output from the program can be modified using the `--mode` flag.
This flag is used with one of 5 possible output modes in the following form.

```
$ python -m pscfFieldGen -f model_file [ [--mode | -m] [silent | echo | trace | verbose | debug] ]
```

Above, the mode setting (indicated as optional by the brackets following `model_file`),
requires both the flag (either `--mode` or the abbreviated `-m` as indicated in the first
nested bracket) and an argument (one of the five options listed in the second nested brackets).
Each of these settings, summarized in the table below, can also be accessed with a shortcut
flag that directly indicates the level without explicitly using the `--mode` flag.

Mode Argument | Shortcut | Description
------------- | -------- | -----------
silent  | `--silent | -s`  | No output. Silent operation.
echo    | `--echo | -e`    | (Default) Echo contents of file as read.
trace   | `--trace | -t`   | Output announce major calculation steps and token interpretation.
verbose | `--verbose | -v` | Output results of major calculations.
debug   | `--debug | -d`   | Output granular, intermediate calculation results. (developer use).

For typical users, the _verbose_ mode should be the highest level of output that 
should be needed. At this level, the program will print out data about major
calculation components that the user can then verify. This information can include
the list of core monomers that end up being used, the position and type of each particle
in a particle structure, the lattice on which the calculation will be performed, etc.
This level of output should be sufficient for users to be confident that their inputs 
are being interpreted by the program the way they expect.
Thus, if you wish to run the program without output to the terminal, any of the
following commands would work:

```
$ python -m pscfFieldGen -f model_file --mode silent
$ python -m pscfFieldGen -f model_file -m silent
$ python -m pscfFieldGen -f model_file --silent
$ python -m pscfFieldGen -f model_file -s
```
In order to redirect program output to a file, one would execute

```
$ python -m pscfFieldGen -f model_file -t > trace_file
```

where "trace_file" is the name of the file storing the trace data.

Example files for a range of calculations are included in the `examples` directory in
the root of the project repository.

### Model File

[Back to Top](#pscf-field-generator)

The Model file acts as the primary input for the program.
Its format is similar to that of the C++/CUDA version of PSCF, with bracketed
blocks used to organize data. Each block can contain either data,
which is specified by keywords followed by the data itself,
or nested blocks. Both data keywords and block labels are *case-sensititve*.
Block labels use _CamelCase_ capitalization and their last 
character is a curly bracket (`{`);
data keywords are all lower-case, with words separated by underscore (`_`) characters;
the end of a block is indicated by an isolated closing bracket (`}`).
Formatting is flexible, requiring only that individual entries be separated
by some amount of whitespace (spaces, tabs, newlines).

Below is an example of what the contents of a model file might look like for a BCC phase.

```
PscfFieldGen{

    software        pscf
    parameter_file  param_kgrid
    output_file     rho_kgrid.in
    
    ParticleGenerator{
        
        BasisCrystal{
        
            core_option     0
            core_option     1
            
            Sphere{ position    0.0     0.0     0.0 }
            Sphere{ position    0.5     0.5     0.5 }
        }
    }
}
```

Regardless of the structure being generated, the main block for the program is the 
`PscfFieldGen{ . . . }` block. Within this block, the first three entries (in order)
must be:

 * `software` : This keyword would be followed by a flag indicating the PSCF version
this execution is targeting. Currently flag *pscf* (for the Fortran version) and 
*pscfpp* (for the C++/Cuda versions) are the only acceptable entries.
 * `parameter_file` : This keyword would be followed by a single file name referencing
the parameter file. The 'file name' in this case can be any path that would allow the
file to be found from the current directory.
 * `output_file` : This keyword is followed by a single file name to which the generated field
should be written. As with `parameter_file`, this can be any path recognizable from the current
directory.

Following these entries, the next entry should be a block specifying the structure being
generated. Three options are possible:
`ParticleGenerator{ . . . }` for particle-type (sphere or cylinder phases),
`NetworkGenerator{ . . .}` for network phases using the level-set method,
or `LamellarGenerator{}` for a lamellar phase. Each of these blocks are detailed below.

#### ParticleGenerator

[Back to Top](#pscf-field-generator)

Using the _ParticleGenerator_ block indicates that you intend to produce a field for
a particle phase using the form-factor method. Presently, this block contains only one entry,
which is a nested block specifying the crystal structure. This nested block can be either a
_BasisCrystal_ block or a _MotifCrystal_ block. The _BasisCrystal_ block expects that the user
will specify all particles found within the unit cell, and will simply collect these for use
in the field calculation. The _MotifCrystal_ block, on the other hand, will apply the symmetry
operators of the unit cell space group (read from the parameter file) to particles
read from the file in order to generate the full set of particles in the unit cell.
This feature makes generating complex structures (such as large Frank-Kasper phases) 
much easier, as the user only need specify the Wyckoff positions, rather than the full
particle list.

Internally, the _BasisCrystal_ and _MotifCrystal_ blocks are structured the same way.
Presently, only one data keyword is used in these blocks:

 * `core_option` : This keyword specifies, by monomer id, one monomer to be considered
an option when choosing the core monomer during generation. This keyword can be excluded
entirely (in which case all monomers will be considered options) or repeated any number
of times, until all monomer options are specified. Each use of the keyword will only accept
one monomer id. As shown in the example above, to specify multiple options, the keyword
must be repeated for each option.
Monomers are indexed starting at 0 and counting up.
(This numbering differs from the Fortran numbering, which starts at 1).

During field generation, the monomer with the lowest overall volume fraction among the
available core options is chosen to be placed in the particle cores.
The crystal blocks also accept any number of particles. Each particle is specified within
its own sub-block. Presently, only two particle types are accepted: 

 * Spheres : For three dimensional structures, spheres are defined in
`Sphere{ . . . }` blocks. Presently, the only data in this block is the
keyword `position` followed by 3 numeric values indicating the fractional
coordinates of the sphere center in the unit cell. For example:
`Sphere{  position  0.0  0.0  0.0  }`.
 * Cylinders : For two dimensional structures, cylinders are defined in
`Cylinder2D{ . . . }` blocks. Presently, the only data in this block is the
keyword `position` followed by 2 numeric values indicating the fractional
coordinates of the cylinder axis in the (2D) unit cell. For example:
`Cylinder2D{  position  0.0  0.0  }`.

The core options and particles can be specified in any order that makes sense to the user.

#### NetworkGenerator

[Back to Top](#pscf-field-generator)

This block indicates that a network structure should be generated using the level-set
method. The block is indicated in the input file with a `NetworkGenerator{ . . . }` block.
The level-set method requires converting a small number of symmetry-adapted basis functions
(typically the first non-zero basis) into a real-space grid, and using that to find and
set the field according to the level-set. 
This conversion is done through the Fortran version of PSCF.
Although the C++/CUDA version
is capable of this conversion, a workflow through the Fortran software is the only one
currently implemented for this tool.
Also note that the restriction to Fortran is only for the internal basis-to-grid conversion;
the parameter file driving the method and
and the final RGRID file (specified by `output_file` in the _PscfFieldGen_ block)
can still be compatible with the C++/CUDA codes.

To facilitate this conversion, the user needs to provide a PSCF (Fortran) parameter file
set to perform the conversion. This parameter file should contain only the system
definition and the `FIELD_TO_RGRID` command. 
(It is not critical that the specified polymer structures exactly match those in the
intended calculation, as iterations are not performed during the conversion. 
Thus, the polymer in this conversion file can simply be a multi-block with one
block of each monomer type.)
Within the _NetworkGenerator_ block,
the name of this conversion file is given following the `network_param` keyword.

The user also must provide a template symmetry-adapted basis field file compatible
with the `network_param` file. This template file is used to set the format,
but also to allow the user to define the number and weighting of the basis functions
being defined for the level set. For this latter purpose, the template field file
should be written as if _monomer 0_ were going to be placed in the core of the
network, regardless of which monomer is expected to be in the core in the eventual
generated field. The coefficients assigned to monomer 0 in the template file will
be assigned to the chosen core monomer during field generation, with coefficients
for the remaining monomers chosen to balance this.

If a user intends to use the default set of monomers as core options, the name
of this template field file can be provided directly within the _NetworkGenerator_
block using the keyword `star_file` after the network param file. If, instead, the
user wishes to specify a set of core options, the network param file should be 
followed by a `NetworkCrystal{ . . . }` block. In this case, the `star_file` keyword
would be placed inside the _NetworkCrystal_ block, along with any number of 
`core_option` keywords required to define the preferred set of core options.
The order of the `star_file` and `core_option` keywords does not matter.

Altogether, a possible input for a network phase could look like the following.

```
PscfFieldGen{
    software        pscf
    parameter_file  param_iterate
    output_file     rho_rgrid.in
    
    NetworkGenerator{
        
        network_param   param_convert
        
        NetworkCrystal{
        
            core_option 0
            core_option 1
            
            star_file   rho_template
        }
    }
}
```

*Operation Note* : The internal operations of pscfFieldGen during this level-set
method are performed in a directory called `_network_generator_internal_`.
This directory is created within the same directory from which
the program was invoked (the working directory). Its contents are left in
place as a record of the generation process, but were placed in a separate directory
avoid cluttering the working directory. Deletion of this record after field generation
is allowed if the user does not wish to keep it.

#### LamellarGenerator

[Back to Top](#pscf-field-generator)

Given the simplicity of the Lamellar structure, the _LamellarGenerator_ block is
similarly simple. Unlike with particles or network phases, the lamellar generation
algorithm does not use a concept of a "core" monomer. Rather, in order to ensure the
generated field does not produce non-physical density profiles, the monomer with the
lowest-overall volume fraction is automatically used as a reference.

Because no additional inputs are required for this structure, the _LamellarGenerator_
block can be specified in any of three ways:

 * As an empty block: `LamellarGenerator{    }`
 * As a block-like keyword: `LamellarGenerator{}`  (note the lack of white-space between
the brackets.
 * As a _CamelCase_ keyword: `LamellarGenerator`   (note the absence of brackets altogether).

### Parameter File

[Back to Top](#pscf-field-generator)

For detailed information regarding the parameter file format, please see the 
User manual for the specific version of PSCF.

System specifications are taken from a PSCF Parameter File.
This is done to simplify user input, with the assumption that the user will first generate
the parameter file for the desired calculation, and use it to generate the initial guess.

## Special Notes

[Back to Top](#pscf-field-generator)

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

**For blends, handling of Grand Canonical Ensemble (or input chemical potentials) is experimental:**
For both the Fortran and C++/Cuda versions of PSCF, this generator will handle inputs in the 
Grand Canonical Ensemble (Fortran version) as well as any combination of *phi* and *mu*
specified polymers (C++/Cuda version). Cases using explicit system composition (Canonical
ensemble; *phi* specified for all species) are considered the normal case for this
software. When molecules are specified by chemical potential (Grand Canonical; *mu* specified
for more than one molecule), their contribution to the volume fraction of monomer species
is calculated assuming that all Grand Canonical species are present in equal number (same
number of moles) and share the volume fraction not granted to a canonical (*phi*-specified) 
species. For calculation of monomer volume fractions,
this is analogous to treating a single, canonical "molecular complex" species 
(which contains one molecule of each Grand Canonical species) with sufficient *phi* to
result in total volume fraction of 1.
Reliability of this approach (particularly for mixtures with solvents or 
significantly uneven polymer sizes) has not yet been robustly tested, and field guesses involving
these inputs should be carefully inspected before use.

[Back to Top](#pscf-particle-phase-field-generator)

