# Example Files for Initial Guess Generator

This directory tree contains several examples of how to run the guess generator.

Each sub-directory found here contains a collection of related examples.
Within those collection directories,
each example has its own directory in which to run.
Each example also includes executable scripts to automatically run the example
(assuming all required software are properly installed) as well as to clear all
files generated in the example in order to refresh the directory for another run.
From here, any example can be accessed by entering

```
cd  [collection]/[example]/
```

on the command line. In this case [collection] is the name of the collection's 
directory, and [example] is the name of the specific example within that collection.

The following sections provide a brief overview of the collections explaining 
what each is meant to illustrate. Another README.md file
within each collection provides detailed information regarding the examples and
how to run each one.

## Basis Input

Examples in the directory 'basis_input/' show how to generate an initial guess
by entering the crystals structure's full basis of particle positions. 
These examples all use the Fortran version of PSCF.
Available morphologies are:

 * 2-Dimensional
    * hexagonally packed cylinders (`hex`)
 * 3-Dimensional
    * BCC Spheres (`bcc')
    * Frank-Kasper A15 Spheres (`a15`)
    * Frank-Kasper sigma Spheres (`sigma`)

## Motif Input

Examples in the directory 'motif_input/' show how to generate an initial guess
by entering a motif of particles and letting space group symmetry generate
the rest. This practice is particularly useful for large, complex phases.
These examples all use the Fortran version of PSCF.
Available morphologies are:

 * BCC Spheres (`bcc')
 * Frank-Kasper A15 Spheres (`a15`)
 * Frank-Kasper sigma Spheres (`sigma`)
 * Frank-Kasper c14 Spheres (`c14`)

## Use with PSCF (C++/Cuda)

Examples in the directory 'pscfpp_style/' illustrate how to use the guess generator
for C++/Cuda version calculations.
In order to run these examples, first make sure that PSCF (C++/Cuda) is installed on your
machine, and that `pscfFieldGen` is available as a module in your python environment.
Available morphologies are:

 * 2-Dimensional
    * hexagonally packed cylinders (`hex`)
 * 3-Dimensional
    * BCC Spheres (`bcc')

## Network Phases

Examples in the directory 'network_levelset' illustrate how to run the guess generator
for network phases using the levelset method.
Regardless of which PSCF version your final calculation will be done in,
be sure that the Fortran version is installed, as this is the version used
for the field conversions done during the level-set method.
Available morphologies are:
    
    * 3-Dimensional
        * Gyroid (`gyroid`)

## Monomer Selection

Examples in the directory 'monomer_selection/' reverse the block fractions of some of the
'basis_input' examples such that the second monomer, rather than the first, is the 
minority component. These examples show how to select a monomer other than the default
"monomer 0".
Available morphologies are:

    * 2-Dimensional
        * hexagonally packed cylinders ('hex')
    * 3-dimensional
        * BCC spheres ('bcc')

## Polymer Blends

Examples in the directory 'polymer_blends/' show pscfFieldGen's capability to handle
polymer blends. Each examples blends two or more diblock copolymers in the system.
In each case, blend fractions and block fractions are chosen to allow comparison
to examples in 'basis_input/'. Details of this balancing are given in the collection's 
README.md file.
Available morphologies are:

    * 2-Dimensional
        * hexagonally packed cylinders ('hex')
    * 3-dimensional
        * BCC spheres ('bcc')

