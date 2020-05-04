# Knit Nested Oscillations
Computes a "knitable" representations of nested oscillations, such that a number of hooks can be connected to form the shape (cardioid).

## Usage

Type:
`python cardioid.py -h` to see the following help for the computation script

```
usage: cardioid.py [-h] [-s STRINGS] [-sa SAVE_AS] [-ho HOOKS] [-np]
                   [-o ORDER]

optional arguments:
  -h, --help            show this help message and exit
  -s STRINGS, --strings STRINGS
                        number of precomputed strings (default=None)
  -sa SAVE_AS, --save_as SAVE_AS
                        save path
  -ho HOOKS, --hooks HOOKS
                        number of hooks (default=499)
  -np, --nprime         do not find nearest prime
  -o ORDER, --order ORDER
                        order of nested oscillations (default=(2, 3))

```
