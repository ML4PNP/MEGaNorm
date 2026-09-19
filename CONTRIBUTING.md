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

An automated test suite is currently under development. Where applicable, contributors are encouraged to include tests with new functionality and bug fixes.

Testing instructions will be updated as the automated test infrastructure is introduced.

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
