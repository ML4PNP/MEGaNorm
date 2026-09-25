# Contributing to MEGaNorm

Thank you for considering contributing to **MEGaNorm**. We welcome contributions from the community to improve and extend the package.

---

## Ways to Contribute

Contributions can take many forms, including:

* **Bug reports:** If you encounter a bug or unexpected behavior, please [open an issue](https://github.com/ML4PNP/MEGaNorm/issues).
* **Feature requests:** Suggestions for new functionality are welcome and can be proposed through the issue tracker.
* **Code contributions:** Fix bugs, improve existing functionality, or implement new features.
* **Documentation:** Improve or expand the documentation, examples, and tutorials.
* **Testing:** Help develop and improve the automated test suite.

---

## Code of Conduct

All contributors are expected to follow the project's [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Getting Started

1. **Fork the repository** and clone your fork:

   ```bash
   git clone https://github.com/your-username/MEGaNorm.git
   cd MEGaNorm
   ```

2. **Set up a development environment**:

   ```bash
   conda create --name meganorm-dev python=3.12
   conda activate meganorm-dev
   pip install -e ".[dev]"
   ```

3. **Create a new branch** for your contribution:

   ```bash
   git checkout -b feature/my-new-feature
   ```

4. **Make and commit your changes**:

   ```bash
   git add .
   git commit -m "Add feature: my new feature"
   ```

5. **Push the branch to your fork**:

   ```bash
   git push origin feature/my-new-feature
   ```

6. Open a [pull request](https://github.com/ML4PNP/MEGaNorm/pulls) describing the purpose of the changes and any relevant implementation details.

---

## Style Guide

When contributing Python code:

* Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.

* Use [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html).

* Format code using **Black** with the configuration defined in pyproject.toml:

  ```bash
  black .
  ```

* Sort imports using **isort**:

  ```bash
  isort .
  ```

---

## Testing

Install MEGaNorm with its development dependencies and run the full test suite
before submitting a pull request:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Tests are organized by package area under `tests/`, with separate markers for
fast unit tests, integration tests, and slow tests. New functionality and bug
fixes should include focused tests where applicable. The same full suite runs
automatically through GitHub Actions on pushes and pull requests.

---

## Reporting Issues

When reporting a bug, please include enough information to reproduce the problem where possible, including:

* MEGaNorm version
* Python version
* operating system or computing environment
* a minimal example reproducing the problem
* the complete error message or traceback

Please use the [GitHub issue tracker](https://github.com/ML4PNP/MEGaNorm/issues) for bug reports and feature requests.

---

## Thank You

Contributions help improve MEGaNorm and support open and reproducible research in electrophysiological neuroimaging.
