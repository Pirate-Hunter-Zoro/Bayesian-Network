# gen-bn

Generates random probabilistic networks with Bernoulli random
variables.

## Requirements

Python 3, numpy, (optional) networkx, matplotlib

## Usage

See

    ./gen-bn --help

for information on parameters to the generator.

## Output format

The network is written by default to `bn.json`, with the following
format:

    {
      <node_id>: {
        "parents": [...]
        "prob": [
           [<assignment>, <probability>]
           ...
        ]
      },
      ...
    }

where:

* `<node_id>` is the numeric identifier for a variable.
* `parents` has a list of variables that `<node_id>` depends on.
* `prob` has a list of conditional probabilities such that
  `P(<node_id>=1|<parents>=<assignment>)=<probability>`. In other
  words, it has the list of probabilities of success for the Bernoulli
  random variable based on every configuration of its parents.
