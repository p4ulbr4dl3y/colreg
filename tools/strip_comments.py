"""Утилита для удаления комментариев и строк документации из файлов Python.

Скрипт анализирует структуру AST (абстрактного синтаксического дерева) файлов
и удаляет найденные докстринги для минимизации размера кода или его очистки.
"""

import argparse
import ast
from pathlib import Path


class DocstringRemover(ast.NodeTransformer):
    """Преобразователь AST для удаления строк документации.

    Обходит узлы AST и удаляет узлы выражений, которые являются докстрингами.
    """

    def visit_Module(self, node):
        """Удаляет докстринг модуля."""
        self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, (ast.Str, ast.Constant, ast.Bytes))
        ):
            node.body.pop(0)
        return node

    def visit_ClassDef(self, node):
        """Удаляет докстринг класса."""
        self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, (ast.Str, ast.Constant, ast.Bytes))
        ):
            node.body.pop(0)
        return node

    def visit_FunctionDef(self, node):
        """Удаляет докстринг функции."""
        self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, (ast.Str, ast.Constant, ast.Bytes))
        ):
            node.body.pop(0)
        return node

    def visit_AsyncFunctionDef(self, node):
        """Удаляет докстринг асинхронной функции."""
        return self.visit_FunctionDef(node)


def process_file(file_path: Path):
    """Обрабатывает отдельный файл Python, удаляя из него докстринги."""
    try:
        content = file_path.read_text()
        tree = ast.parse(content)
        DocstringRemover().visit(tree)
        new_content = ast.unparse(tree)
        file_path.write_text(new_content)
        print(f"Processed {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def main():
    """Главная функция парсинга аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Strip comments and docstrings from Python files."
    )
    parser.add_argument("path", help="File or directory path to process")
    args = parser.parse_args()
    path = Path(args.path)
    if path.is_file():
        process_file(path)
    elif path.is_dir():
        for py_file in path.rglob("*.py"):
            process_file(py_file)
    else:
        print(f"Path {path} does not exist or is not a file/directory.")


if __name__ == "__main__":
    main()
