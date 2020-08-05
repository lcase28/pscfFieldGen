# PSCFpp example: Hexagonally Packed Cylinders

## Generate KGRID and RGRID files

If you wish to simply generate monomer volume fractions in KGRID format
and convert it to RGRID format, use

```
./run
```

or use the following three commands

```
./clean
python -m pscfFieldGen -f model
pscf_pc3d -e -p param -c command
```

on your terminal command line.

## Generate KGRID File and Converge the Field

To generate the field and run the full calculation, enter

```
./runConvergence
```

or use the following three commands

```
./clean
python -m pscfFieldGen -f model
pscf_pc3d -e -p param -c command_long
```

on your terminal command line.


