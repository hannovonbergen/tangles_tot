# Installation

We recommend using [uv](https://docs.astral.sh/uv/)

1. Install the dependencies as specified in the pyproject.toml
2. Install the tangles library from https://github.com/tangle-software/tangles.git

# Examples

Interactive jupyter notebooks showing and explaining the usage of the library can be found in the `examples` folder.

# Tests

You can validate that you have all necessary dependencies by running our test suite.

```bash
pytest
```

# Documentation

The Documentation can be found in the docs folder.

It is generated using nbconvert and pdoc with the following command.

```bash
./docs/build_documentation.sh
```
