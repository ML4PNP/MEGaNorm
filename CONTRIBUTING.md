# Contributing to MEGaNorm

Thank you for considering contributing to **MEGaNorm**! We welcome contributions from the community to improve and extend the toolbox. This guide will help you get started.

---

## 🚀 Ways to Contribute

* **Bug reports**: Found a bug? Please [open an issue](https://github.com/ML4PNP/MEGaNorm/issues).
* **Feature requests**: Have an idea for a new feature? We'd love to hear it.
* **Code contributions**: Improve the codebase, fix bugs, or add new functionality.
* **Documentation**: Help improve or expand the documentation and tutorials.
* **Testing**: Write or improve unit tests.

---

## ⚖️ Code of Conduct

Please be respectful and constructive. This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

---

## 🚪 Getting Started

1. **Fork the repository** and clone your fork:

   ```bash
   git clone https://github.com/your-username/MEGaNorm.git
   cd MEGaNorm
   ```

2. **Set up your environment**:

   ```bash
   conda create --name meganorm-dev python=3.12
   conda activate meganorm-dev
   pip install -e .[dev]
   ```

3. **Create a new branch**:

   ```bash
   git checkout -b feature/my-new-feature
   ```

4. **Make your changes**, write tests, and commit:

   ```bash
   git add .
   git commit -m "Add feature: my new feature"
   ```

5. **Push to your fork and open a pull request**:

   ```bash
   git push origin feature/my-new-feature
   ```

   Then open a [Pull Request](https://github.com/ML4PNP/MEGaNorm/pulls).

---

## ✏️ Style Guide

* Format code using **Black**:

  ```bash
  black .
  ```
* Follow [PEP 8](https://peps.python.org/pep-0008/) for Python style
* Use [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html)
* Sort imports with **isort**:

  ```bash
  isort .
  ```

---

## ✅ Running Tests

Make sure your changes don’t break anything:

```bash
pytest tests/
```

Use `pytest --cov` to check test coverage.

---

## 🚀 Thank You!

Your contribution will help advance open, reproducible neuroimaging research. We appreciate your help making MEGaNorm better!
