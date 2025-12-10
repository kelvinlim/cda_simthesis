# cda_simthesis

This project contains code for thesis on simulation of ema data

The purpose of this project is to examine the effect of missing data on FGES output, in particular how the the number and types of edges are affected by the strength of the edges as a function of how much data is lost.

Simulated data will be used so that the effect of the strength of edges can be examined.

For simplicity, we will start with 100 samples and then sample from 100% to 40%.

 Simuated data will be created using my fork of CausalPowerAnalysis
https://github.com/lelandwilliams/CausalPowerAnalysis.  My fork is here:
https://github.com/kelvinlim/CausalPowerAnalysis.  It corrects a warning message  when working with a dataframe.

## Generate simulated data

To generate simulated data with range of effectsizes and num_samples.   This is configured in the config.yaml folder.

```
./simdata.py
```

### Output

```
kolim@XJ75J7FXQK cda_hinf8220 %  cd /Users/kolim/Projects/cda_hinf8220 ; /usr/bin/env /Users/kolim/Projects/cda_hinf8220/.venv/bin/python /Use
rs/kolim/.vscode/extensions/ms-python.debugpy-2025.16.0-darwin-arm64/bundled/libs/debugpy/adapter/../../debugpy/launcher 50843 -- simdata.py -
-verbose 3 --cmd sim 
rows-100_vars-14_edges-12_es-0.1_iter-000.csv count: 1
rows-100_vars-14_edges-12_es-0.1_iter-001.csv count: 1
rows-100_vars-14_edges-12_es-0.1_iter-002.csv count: 1
rows-100_vars-14_edges-12_es-0.2_iter-000.csv count: 1
rows-100_vars-14_edges-12_es-0.2_iter-001.csv count: 1
rows-100_vars-14_edges-12_es-0.2_iter-002.csv count: 1
rows-100_vars-14_edges-12_es-0.5_iter-000.csv count: 1138
rows-100_vars-14_edges-12_es-0.5_iter-001.csv count: 5050
rows-100_vars-14_edges-12_es-0.5_iter-002.csv count: 1979
rows-100_vars-14_edges-12_es-0.8_iter-000.csv count: 3152
rows-100_vars-14_edges-12_es-0.8_iter-001.csv count: 1767
kolim@XJ75J7FXQK cda_hinf8220 %  cd /Users/kolim/Projects/cda_hinf8220 ; /usr/bin/env /Users/kolim/Projects/cda_hinf8220/.venv/bin/python /Users/kolim/.vsc
ode/extensions/ms-python.debugpy-2025.16.0-darwin-arm64/bundled/libs/debugpy/adapter/../../debugpy/launcher 51015 -- simdata.py --verbose 3 --cmd sim 
rows-100_vars-14_edges-12_es-0.1_iter-000.csv count: 1
rows-100_vars-14_edges-12_es-0.1_iter-001.csv count: 1
rows-100_vars-14_edges-12_es-0.1_iter-002.csv count: 1
rows-100_vars-14_edges-12_es-0.1_iter-003.csv count: 1
rows-100_vars-14_edges-12_es-0.2_iter-000.csv count: 1
rows-100_vars-14_edges-12_es-0.2_iter-001.csv count: 2
rows-100_vars-14_edges-12_es-0.2_iter-002.csv count: 1
rows-100_vars-14_edges-12_es-0.2_iter-003.csv count: 1
rows-100_vars-14_edges-12_es-0.5_iter-000.csv count: 3674
rows-100_vars-14_edges-12_es-0.5_iter-001.csv count: 11828
rows-100_vars-14_edges-12_es-0.5_iter-002.csv count: 8980
rows-100_vars-14_edges-12_es-0.5_iter-003.csv count: 6673
rows-100_vars-14_edges-12_es-0.8_iter-000.csv count: 31998
rows-100_vars-14_edges-12_es-0.8_iter-001.csv count: 29030
rows-100_vars-14_edges-12_es-0.8_iter-002.csv count: 2895
rows-100_vars-14_edges-12_es-0.8_iter-003.csv count: 12056
rows-100_vars-14_edges-12_es-1.0_iter-000.csv count: 16854
rows-100_vars-14_edges-12_es-1.0_iter-001.csv count: 13019
rows-100_vars-14_edges-12_es-1.0_iter-002.csv count: 14312
rows-100_vars-14_edges-12_es-1.0_iter-003.csv count: 9638
rows-500_vars-14_edges-12_es-0.1_iter-000.csv count: 1
rows-500_vars-14_edges-12_es-0.1_iter-001.csv count: 1
rows-500_vars-14_edges-12_es-0.1_iter-002.csv count: 1
rows-500_vars-14_edges-12_es-0.1_iter-003.csv count: 1
rows-500_vars-14_edges-12_es-0.2_iter-000.csv count: 1
rows-500_vars-14_edges-12_es-0.2_iter-001.csv count: 1
rows-500_vars-14_edges-12_es-0.2_iter-002.csv count: 1
rows-500_vars-14_edges-12_es-0.2_iter-003.csv count: 1
rows-500_vars-14_edges-12_es-0.5_iter-000.csv count: 1354
rows-500_vars-14_edges-12_es-0.5_iter-001.csv count: 12706
rows-500_vars-14_edges-12_es-0.5_iter-002.csv count: 13830
rows-500_vars-14_edges-12_es-0.5_iter-003.csv count: 11960
rows-500_vars-14_edges-12_es-0.8_iter-000.csv count: 15438
rows-500_vars-14_edges-12_es-0.8_iter-001.csv count: 69743
rows-500_vars-14_edges-12_es-0.8_iter-002.csv count: 32190
rows-500_vars-14_edges-12_es-0.8_iter-003.csv count: 30758
rows-500_vars-14_edges-12_es-1.0_iter-000.csv count: 9976
rows-500_vars-14_edges-12_es-1.0_iter-001.csv count: 11230
rows-500_vars-14_edges-12_es-1.0_iter-002.csv count: 15511
rows-500_vars-14_edges-12_es-1.0_iter-003.csv count: 1338
rows-1000_vars-14_edges-12_es-0.1_iter-000.csv count: 1
rows-1000_vars-14_edges-12_es-0.1_iter-001.csv count: 1
rows-1000_vars-14_edges-12_es-0.1_iter-002.csv count: 1
rows-1000_vars-14_edges-12_es-0.1_iter-003.csv count: 1
rows-1000_vars-14_edges-12_es-0.2_iter-000.csv count: 2
rows-1000_vars-14_edges-12_es-0.2_iter-001.csv count: 2
rows-1000_vars-14_edges-12_es-0.2_iter-002.csv count: 1
rows-1000_vars-14_edges-12_es-0.2_iter-003.csv count: 1
rows-1000_vars-14_edges-12_es-0.5_iter-000.csv count: 30174
rows-1000_vars-14_edges-12_es-0.5_iter-001.csv count: 5705
rows-1000_vars-14_edges-12_es-0.5_iter-002.csv count: 1234
rows-1000_vars-14_edges-12_es-0.5_iter-003.csv count: 6959
rows-1000_vars-14_edges-12_es-0.8_iter-000.csv count: 6137
rows-1000_vars-14_edges-12_es-0.8_iter-001.csv count: 22440
rows-1000_vars-14_edges-12_es-0.8_iter-002.csv count: 8755
rows-1000_vars-14_edges-12_es-0.8_iter-003.csv count: 62762
rows-1000_vars-14_edges-12_es-1.0_iter-000.csv count: 30643
rows-1000_vars-14_edges-12_es-1.0_iter-001.csv count: 793
rows-1000_vars-14_edges-12_es-1.0_iter-002.csv count: 9015
rows-1000_vars-14_edges-12_es-1.0_iter-003.csv count: 18074
```

## Selecting cases

Turns out not all datasets create graphs with edges with GFCI

To identify the models that worked, I created a program
try_all_files.ipynb. This program read in all data files
with rows-100 to identify which files worked. Here is the output:

```
An error occurred during model fitting: not enough values to unpack (expected 2, got 0)** On entry to DPOTRI, parameter number  4 had an illegal value

This might be due to an unusable model, such as one with no direct edges.
Model search succeeded for file: sim_data/rows-100_vars-14_edges-12_es-0.2_iter-000.csv
Model search succeeded for file: sim_data/rows-100_vars-14_edges-12_es-0.2_iter-002.csv
An error occurred during model fitting: not enough values to unpack (expected 2, got 0)
This might be due to an unusable model, such as one with no direct edges.
** On entry to DPOTRI, parameter number  4 had an illegal value
An error occurred during model fitting: not enough values to unpack (expected 2, got 0)
This might be due to an unusable model, such as one with no direct edges.
** On entry to DPOTRI, parameter number  4 had an illegal value
Model search succeeded for file: sim_data/rows-100_vars-14_edges-12_es-0.2_iter-003.csv
Model search succeeded for file: sim_data/rows-100_vars-14_edges-12_es-1.0_iter-003.csv
Model search succeeded for file: sim_data/rows-100_vars-14_edges-12_es-0.1_iter-000.csv
Model search succeeded for file: sim_data/rows-100_vars-14_edges-12_es-0.1_iter-001.csv
An error occurred during model fitting: not enough values to unpack (expected 2, got 0)
This might be due to an unusable model, such as one with no direct edges.
** On entry to DPOTRI, parameter number  4 had an illegal value
An error occurred during model fitting: not enough values to unpack (expected 2, got 0)
This might be due to an unusable model, such as one with no direct edges.
** On entry to DPOTRI, parameter number  4 had an illegal value
Model search succeeded for file: sim_data/rows-100_vars-14_edges-12_es-0.1_iter-003.csv
Model search succeeded for file: sim_data/rows-100_vars-14_edges-12_es-0.1_iter-002.csv
Model search succeeded for file: sim_data/rows-100_vars-14_edges-12_es-1.0_iter-001.csv
```

I selected the following four cases - note that txt files are the model files generated
by the CausalPowerAnalysis code.

```
            "sim_data/rows-100_vars-14_edges-12_es-0.1_iter-000.txt",
            "sim_data/rows-100_vars-14_edges-12_es-0.1_iter-001.txt",
            "sim_data/rows-100_vars-14_edges-12_es-0.1_iter-003.txt",
            "sim_data/rows-100_vars-14_edges-12_es-0.1_iter-002.txt",

```

These are used for input into simstandard.py that creates data files with different
beta and rows

Next we use tradsim_gfci.ipynb to identify for rows=100, what is the range
of betas that worked across the four cases.  The range that worked
across all cases was 0.7-2.0

Problem with tradsim_gfci_obj.py. - gettng errors about

Warning: GFci missing setMaxPathLength; skipping

Warning: GFci missing setPossibleMsepSearchDone; skipping

This was due to another version of tetrad being involved

Fixed with #from run_tetrad import TetradWrap

This was due to the old pytetrad code being included and using a different jar file. Removed pytetrad code and problem was solved.

After simulating data, do the computation and plots

```

# computations using simulated data
./tradsim_gfci_obj.py --sim --cmd compute
# create plots
./tradsim_gfci_obj.py --sim --cmd plot

```
## clone graphs from actual data

Due to problems with causalPower simulated graphs not 
being easy to identify causal graphs (few colliders!), try
a different approach.

Created directed graphs from the original 4 real data sets.
Essentially we are cloning the graphs from these 4 datasets.

We use fges for generating the graphs from the real data sets.

Output is in json file for each case.

This format is compatible with tradsim_gfci_obj.py

However will rewrite this to use fges instead

tradsim_fges_obj.py


```
# invoked
./simstandard.py --cmd clonesim
```

## causalpower part 2

Spoke with Erich K on 20251201.

Lag modeling is more complicated.

Examined datasets from cpc pain study. Found
multiple cases with 90 surveys.  
Found case R13 which has 22 nodes and 27 directed edges
based on gfci analysis.

Trying simulation 