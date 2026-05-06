import ast
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


def strip_content(code_str):
    from tools.strip_comments import DocstringRemover

    tree = ast.parse(code_str)
    DocstringRemover().visit(tree)
    return ast.unparse(tree)


def test_strip_basic():
    code = '\n# Comment\n"""Doc."""\ndef f():\n    """Doc."""\n    pass\n'
    result = strip_content(code)
    assert '"""' not in result
    assert "#" not in result
    assert "pass" in result
