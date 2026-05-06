# Design: Strip Comments and Docstrings Script

## Purpose
A utility script to remove all comments and docstrings from Python files within the project. This is a temporary tool used for code processing.

## Architecture
- **CLI Tool**: Python script located at `tools/strip_comments.py`.
- **Input**: A single file path or a directory path (processed recursively).
- **Engine**: Python's `ast` module for reliable parsing and re-generation.

## Logic Flow
1. **Discovery**: If input is a directory, find all `.py` files.
2. **Parsing**: Load file content and parse into an Abstract Syntax Tree (AST).
3. **Transformation**:
    - Use `ast.NodeTransformer` to visit `Module`, `ClassDef`, and `FunctionDef` nodes.
    - Remove the first statement if it is a docstring.
4. **Re-generation**: Use `ast.unparse()` to generate source code from the modified AST. This process naturally discards all comments.
5. **Output**: Overwrite the original file with the stripped version.

## Error Handling
- Catch and report `SyntaxError` for invalid Python files.
- Catch and report `IOError` for file access issues.

## Testing Strategy
1. Create a `test_strip.py` with various types of comments and docstrings.
2. Run `python tools/strip_comments.py test_strip.py`.
3. Verify output contains only functional code.
