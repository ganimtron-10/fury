# pip requirements files

## Index

- [all.txt](all.txt) All the requirements for fury.
- [compute.txt](compute.txt) Computation requirements
- [default.txt](default.txt) Default requirements
- [dev.txt](dev.txt) Developer requirements
- [doc.txt](doc.txt) Documentation requirements
- [medical.txt](medical.txt) Medical libraries requirements
- [optional.txt](optional.txt) Optional requirements
- [plot.txt](plot.txt) Plotting requirements
- [style.txt](style.txt) Requirements for code formatting checks
- [test.txt](test.txt) Requirements for running test suite
- [typing.txt](typing.txt) Requirements for spell checks and docstring checks
- [window.txt](window.txt) Requirements for all the windowing systems

## Examples

### Installing requirements

```bash
pip install -U -r requirements/default.txt
pip install -U -r requirements/optional.txt
```

or

```bash
conda install --yes --file=requirements/default.txt --file=requirements/optional.txt
```

### Running the tests

```bash
pip install -U -r requirements/default.txt
pip install -U -r requirements/test.txt
```

or

```bash
conda install --yes --file=requirements/default.txt --file=requirements/test.txt
```

### Running the Docs

```bash
pip install -U -r requirements/default.txt
pip install -U -r requirements/optional.txt
pip install -U -r requirements/doc.txt
```
