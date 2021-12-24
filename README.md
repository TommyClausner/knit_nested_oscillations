# Knit Nested Oscillations
Computes a "knit-able" representations of nested oscillations, such that a number of hooks can be connected to form the shape (cardioid). Hence the entire composition can be realized by one continuous string.

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
Examples using the default settings with a minimum of 4000 strings.

Example circle:

![Image of knitted Tommy](https://github.com/TommyClausner/knit_nested_oscillations/blob/master/example_circle.png)

Example rectangle:

![Image of knitted Tommy](https://github.com/TommyClausner/knit_nested_oscillations/blob/master/example_rectangle.png)

Type:
`python hookReader.py -h` to see the following help for the reader script

## Usage of the hook reader
```
usage: hookReader.py [-h] [-f FILE] [-t TMP] [-r]

optional arguments:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  hook order file
  -t TMP, --tmp TMP     temporary file dir
  -r, --restart         whether to restart
```
