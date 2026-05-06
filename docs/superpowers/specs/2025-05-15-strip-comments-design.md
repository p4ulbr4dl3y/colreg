# Design Doc: Strip All Comments and Docstrings

Date: 2025-05-15
Topic: Codebase Stripping

## 1. Goal
Remove all Python comments and docstrings from the `src`, `scripts`, `tests`, and `tools` directories to reduce codebase size and potentially obfuscate internal logic/notes.

## 2. Approach
Use the existing `tools/strip_comments.py` script which utilizes Python's `ast` module.

### 2.1. Tool Analysis
- **Mechanism**: The script parses Python code into an AST, removes docstring nodes, and then unparses the AST back into code.
- **Side Effect**: Since it uses `ast.unparse`, all original comments (which are not in the AST) will be removed. Additionally, the entire file will be reformatted according to the standard AST representation (e.g., standardized indentation, spacing around operators, etc.).
- **Self-Modification**: Running the script on the `tools` directory will modify `tools/strip_comments.py` itself. This is safe as the script is already loaded in the interpreter.

## 3. Execution Plan
1.  **Preparation**: Ensure the environment is ready (uv installed, dependencies available).
2.  **Processing**:
    -   `python tools/strip_comments.py src`
    -   `python tools/strip_comments.py scripts`
    -   `python tools/strip_comments.py tests`
    -   `python tools/strip_comments.py tools`
3.  **Validation**:
    -   Run `uv run pytest tests/test_pipeline.py` to ensure core functionality is preserved.
    -   Verify that no syntax errors were introduced by the AST transformation.
4.  **Completion**:
    -   Commit changes with the message: `Chore: Strip all comments and docstrings from Python source files`.

## 4. Risks and Mitigations
- **Risk**: `ast.unparse` might produce code that behaves differently if it relies on specific formatting or if there are bugs in `ast.unparse` for certain edge cases.
- **Mitigation**: Run the full test suite after processing.
- **Risk**: Docstrings might be used for runtime introspection (e.g., help messages).
- **Mitigation**: This is acceptable given the explicit directive to strip them.
