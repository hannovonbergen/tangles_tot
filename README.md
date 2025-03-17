#Installation

1. Install the tot-dev environment from the environment.yml and activate it
2. Clone the tangles library from https://github.com/tangle-software/tangles.git
3. Install the tangles library or add it to the path

#Tests

You can validate that you have all necessary dependencies by running our test suite.

```bash
pytest
```

#Documentation

The Documentation can be found in the docs folder. 

It is generated using pdoc with the following command.

```bash
pdoc tangles_tot -d google -o ./docs
```