import ast
import sys
from pathlib import Path

class DocstringRemover(ast.NodeTransformer):
    def visit_Module(self, node):
        self.generic_visit(node)
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, (ast.Str, ast.Constant, ast.Bytes))):
            node.body.pop(0)
        return node

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, (ast.Str, ast.Constant, ast.Bytes))):
            node.body.pop(0)
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, (ast.Str, ast.Constant, ast.Bytes))):
            node.body.pop(0)
        return node

    def visit_AsyncFunctionDef(self, node):
        return self.visit_FunctionDef(node)
