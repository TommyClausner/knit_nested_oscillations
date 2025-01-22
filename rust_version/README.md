# Rust version of Knit Nested Oscillations

This version runs much faster than the Python implementation an is therefore
better suited for quick testing of different parameter settings.

## Installation
This repo only contains the source code and needs to be compiled.

1. Install Rust following the steps as described here: https://www.rust-lang.org/tools/install
2. Navigate to the repo directory e.g. via `cd knit_nested_oscillations/rust_version`
   (after having cloned the repo).
3. Compile the command line program via `cargo build --profile release`

## Usage
The compiled version is located in
`knit_nested_oscillations/rust_version/target/release/` and can be called using
multiple ordered arguments:

```bash
cd ./knit_nested_oscillations/rust_version/target/release
OUTPUT=</path/to/output.csv>
HOOKS=500 #(optional; default is 500)
OSC_FACTORS=2,3  #(optional; default is 2)
OSC_PHASES=0.1,0.4  #(optional; default is 0)

./knit_nested_oscillations $OUTPUT $HOOKS $OSC_FACTORS $OSC_PHASES
```

All arguments must be specified in this order. To e.g. provide
`OSC_PHASES`, all previous arguments must be provided as well. In addition,
`OSC_FACTORS` and `OSC_PHASES` must have the same number of comma separated values.
`OSC_FACTORS` can be specified without `OSC_PHASES`. In that case all `OSC_PHASES` 
will be set to 0.

## Results
The result is compatible with the Python version and can thus be used
similarly for visualization and further processing.

### Quick visualization using Python
```python
import numpy as np
import matplotlib.pyplot as plt

num_hooks = 500  # must be equal to what has been set before

# Compute circle coordinates
theta = np.linspace(0, 2 * np.pi, num_hooks)
circle = np.asarray([np.cos(theta), np.sin(theta)]).T

# load data
data = np.genfromtxt('output.csv', delimiter=',', dtype=int)

# Use data to index circle coordinates
for (_, d_from, d_to) in data:
    plt.plot([circle[d_from, 0], circle[d_to, 0]],
             [circle[d_from, 1], circle[d_to, 1]], color="k", linewidth=0.1)

# Cosmetics
plt.axis('equal')
plt.axis('off')
plt.tight_layout()
plt.gca().invert_yaxis()
plt.show()
```

## Documentation
A documentation can be compiled using `cargo doc --open`.