# pscf_field_generator

A tool to generate PSCF initial guess files for spherical or cylindrical morphologies.

**NOTE:** This is a beta release. See Notes section at the bottom of this file for special
assumptions made in its operation.

## Installation

Add installation instructions.

## Running the tool.

Running the software requires 2 files:

 * A Model file specifying filenames and particle positions.
 * A PSCF Fortran parameter file.

In order to simplify input for the user, crystallographic and composition information
are taken from a PSCF parameter file. Presently, only parameter files consistent with
the Fortran version of the software are supported.

### Model File

The Model file acts as the primary input for the program. Data in this file is specified
by *case-sensitive* keywords. 
Formatting is flexible, requiring only that individual entries be separated
by some amount of whitespace (spaces, tabs, newlines).

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

### Use With C++/Cuda Versions

As mentioned, presently only Parameter files for the Fortran version are supported.
Support for the C++/Cuda parameter files will be added, but is not yet available.
However, because the field file format used by the C++/Cuda version of the software
matches that used in the Fortran version, this generator can still be used.

In order to do so, system data must be formatted into a PSCF Fortran parameter file.
Most data can be ported directly between the two formats, taking care to follow the
differing formats (such as the organization of `chi` interactions),
punctuation (such as placement of single quotes around string data in the Fortran format),
and keyword labels (such as `mesh` vs `ngrid` for the spatial discretization).
Two sections will require special attention, however.

The first and simplest of these is treatment of the `groupName` entry.
The `groupName` (`group_name` in the Fortran file) identifies the space group of the
system. The key difference between the Fortran and C++/Cuda versions is that, while 
Fortran group names contain spaces between distinct symbols, while the C++/Cuda names
separate distinct symbols with underbars. To accommodate this distinction, when generating
the Fortran-style parameter file, one should use the C++/Cuda group name string in the
same location as the Fortran group name string.
For example, if the C++/Cuda parameter file contains

>   `groupName      I_m_-3_m`

the Fortran-style parameter file should contain

>   `BASIS`
>   `group_name`
>   `           I_m_-3_m`

and **not** the Fortran version's proper

>   `BASIS`
>   `group_name`
>   `           'I m -3 m'`

as one might initially expect. This subtle difference ensures that the user will not
need to change their field file after generation.

The second change is to the Polymer chain data. The C++/Cuda code is able to handle
branched polymer architectures which are not supported in the Fortran software, meaning
that the polymer chain structure may not be able to be directly translated. The easiest
approach to correcting this is to simply linearize the branched polymer. That is to say,
take the blocks in the order specified in the C++/Cuda parameter file, and treat them as
laying sequentially along a linear multiblock polymer. In this tool, the polymer structure
is only used to determine the overall volume fraction of each monomer species. Once the
volume fractions are calculated, the difference between a linear and branched polymer
is inconsequential in the field generation algorithm.
As an example, if a polymer in the system has 4 branches emanating from one vertex,
the C++/Cuda parameter file might contain

>   `   Polymer{                    `
>   `       nBlock  4               `
>   `       nVertex 5               `
>   `       blocks  0  0  0  1  0.30`
>   `               1  1  1  2  0.30`
>   `               2  0  1  3  0.20`
>   `               3  1  1  4  0.20`
>   `       phi     1.0             `
>   `   }                           `

in which case the Fortran-style parameter file would contain

>   `CHAINS                                     `
>   `N_chain                                    `
>   `               1                           `
>   `N_block                                    `
>   `               4                           `
>   `block_monomer                              `
>   `               1       2       1       2   `
>   `block_length                               `
>   `               0.30    0.30    0.20    0.20`

in order to generate the proper overall volume fractions.

Once the field file is generated, the C++/Cuda version of PSCF will be able to convert
the kgrid format into the required rgrid or basis formats required for the calculations.


