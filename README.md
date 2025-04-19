# Installation

We recommend using [uv](https://docs.astral.sh/uv/) to install the dependencies as specified in the pyproject.toml.

In the case of uv you would use
```bash
    uv venv .venv
```
to create a virtual environment, which you could then activate.

Afterwards use
```bash
    uv sync
```
to install the dependencies.

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
