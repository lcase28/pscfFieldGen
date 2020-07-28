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
by keyword. Formatting is flexible, requiring only that each entry be separated by whitespace.

