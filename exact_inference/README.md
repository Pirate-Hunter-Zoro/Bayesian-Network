# exact_inference
Exact inference implementation for the CS-7313 Bayesian Networks project

## Requirements
This code requires the packages:
```
networkx
```
```
ujson
```

## Usage
```python
python PATH/TO/exact_inference.py [-h] -f FILE [-q QUERY] [-m]
```
Where ```FILE``` is the json file from ```gen-bn``` and ```QUERY``` is the query variable name if you want to specify it. ```QUERY``` is optional and will default to variable 0. ```-m``` is an optional flag that allows multiprocessing of the enumeration-ask algorithm over the values of the query variable. It will decrease runtimes linearly by a factor of ~1/2. If you are not using multiproccessing in your approximate inference implementations, you most likely should not use it here, as any time comparisons will be off by a factor of ~1/2.

## Questions?
Email me at rrg053@utulsa.edu
