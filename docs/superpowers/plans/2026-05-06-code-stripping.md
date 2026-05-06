# Code Stripping and Ruff Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all comments and docstrings from Python files and configure ruff for linting.

**Architecture:** Custom script for AST-based cleaning + ruff configuration in pyproject.toml.

**Tech Stack:** Python (ast, tokenize), ruff.

---

### Task 1: Create Stripping Script

**Files:**
- Create: `tools/strip_comments.py`

- [ ] **Step 1: Write stripping script using AST**

```python
import ast
import astor # if available, or just ast.unparse for Python 3.9+

def strip_comments_and_docstrings(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        # Remove docstrings
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
            if (node.body and isinstance(node.body[0], ast.Expr) and 
                isinstance(node.body[0].value, ast.Constant) and 
                isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
    return ast.unparse(tree)
```
*Note: ast.unparse removes all comments as they are not part of the AST.*

- [ ] **Step 2: Test script on a small file**

Run: `python tools/strip_comments.py test.py`
Expected: Output without comments and docstrings.

- [ ] **Step 3: Commit**

```bash
git add tools/strip_comments.py
git commit -m "Tools: Add script to strip comments and docstrings using AST"
```

### Task 2: Apply Stripping to All Source Files

**Files:**
- Modify: `src/**/*.py`, `scripts/**/*.py`, `tests/**/*.py`, `tools/**/*.py`

- [ ] **Step 1: Run stripping script on all Python files**

Run: `find src scripts tests tools -name "*.py" -exec python tools/strip_comments.py {} +`
Expected: All files updated.

- [ ] **Step 2: Verify code still works**

Run: `uv run pytest tests/test_pipeline.py`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add .
git commit -m "Chore: Strip all comments and docstrings from Python source files"
```

### Task 3: Configure and Run Ruff

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add ruff configuration to pyproject.toml**

```toml
[tool.ruff]
line-length = 88
target-version = "py311"
select = ["E", "F", "W", "I"]
ignore = []

[tool.ruff.lint]
fixable = ["ALL"]
```

- [ ] **Step 2: Run ruff check with auto-fix**

Run: `uv run ruff check . --fix`
Expected: Automated fixes applied.

- [ ] **Step 3: Run ruff format**

Run: `uv run ruff format .`
Expected: Code formatted.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml .
git commit -m "Chore: Configure and run ruff for linting and formatting"
```

### Task 4: Final Verification

- [ ] **Step 1: Run tests one last time**

Run: `uv run pytest tests/test_pipeline.py`
Expected: PASS.
