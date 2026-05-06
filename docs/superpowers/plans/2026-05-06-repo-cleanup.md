# Repository Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the repository to clean up the root directory, group files logically, and consolidate dependencies.

**Architecture:** Moving files to `tools/`, `scripts/`, `tests/integration/`, and `~/`. Merging `requirements.txt` into `pyproject.toml`.

**Tech Stack:** Python, uv, bash.

---

### Task 1: Create Directories and Cleanup System Files

**Files:**
- Create: `tools/`, `tests/integration/`
- Delete: `.DS_Store` (root and subdirectories)

- [ ] **Step 1: Create new directories**

Run: `mkdir -p tools tests/integration`
Expected: Directories created.

- [ ] **Step 2: Remove .DS_Store files**

Run: `find . -name ".DS_Store" -delete`
Expected: System files removed.

- [ ] **Step 3: Commit**

```bash
git add tools tests/integration
git commit -m "Chore: Create tools/ and tests/integration/ directories"
```

### Task 2: Move and Delete Root Files

**Files:**
- Move: `debug_perf.py` -> `tools/debug_perf.py`
- Move: `amqtt.yaml` -> `scripts/amqtt.yaml`
- Move: `datasets.md` -> `/Users/yegor/datasets.md`
- Delete: `refactor_imports.py`

- [ ] **Step 1: Move performance debug script**

Run: `mv debug_perf.py tools/debug_perf.py`
Expected: File moved to tools/.

- [ ] **Step 2: Move MQTT config**

Run: `mv amqtt.yaml scripts/amqtt.yaml`
Expected: File moved to scripts/.

- [ ] **Step 3: Move datasets documentation outside the project**

Run: `mv datasets.md /Users/yegor/datasets.md`
Expected: File moved to home directory.

- [ ] **Step 4: Delete refactor script**

Run: `rm refactor_imports.py`
Expected: File deleted.

- [ ] **Step 5: Commit**

```bash
git add tools/debug_perf.py scripts/amqtt.yaml
git rm debug_perf.py amqtt.yaml datasets.md refactor_imports.py
git commit -m "Chore: Move utility scripts and config, delete obsolete files"
```

### Task 3: Organize Integration Tests

**Files:**
- Move: `test_day_images.py` -> `tests/integration/test_day_images.py`
- Move: `test_night_images.py` -> `tests/integration/test_night_images.py`

- [ ] **Step 1: Move day images test**

Run: `mv test_day_images.py tests/integration/test_day_images.py`
Expected: File moved.

- [ ] **Step 2: Move night images test**

Run: `mv test_night_images.py tests/integration/test_night_images.py`
Expected: File moved.

- [ ] **Step 3: Verify tests still run (paths might need adjustment if relative)**

Run: `pytest tests/integration/test_day_images.py --help` (just checking if it loads)
Expected: No immediate import errors.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/
git rm test_day_images.py test_night_images.py
git commit -m "Test: Move integration tests to tests/integration/"
```

### Task 4: Consolidate Dependencies

**Files:**
- Modify: `pyproject.toml`
- Delete: `requirements.txt`

- [ ] **Step 1: Read requirements.txt**

Run: `cat requirements.txt`
Expected: List of dependencies.

- [ ] **Step 2: Update pyproject.toml with dependencies**

Update `pyproject.toml` to include the dependencies from `requirements.txt` in the `dependencies` array.

- [ ] **Step 3: Delete requirements.txt**

Run: `rm requirements.txt`
Expected: File removed.

- [ ] **Step 4: Verify with uv**

Run: `uv lock` (or `uv pip compile` if using it)
Expected: Dependencies resolved.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git rm requirements.txt
git commit -m "Chore: Consolidate dependencies into pyproject.toml"
```

### Task 5: Final Verification

- [ ] **Step 1: Check root directory**

Run: `ls -a`
Expected: Clean root directory.

- [ ] **Step 2: Run all tests**

Run: `pytest tests/test_pipeline.py`
Expected: PASS.
