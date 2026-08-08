import ast
import io
import sys
import tokenize


def strip(src):
    out = []
    prev = tokenize.INDENT
    last_line = -1
    last_col = 0
    depth = 0
    for tok, text, (sl, sc), (el, ec), _ in tokenize.generate_tokens(io.StringIO(src).readline):
        if sl > last_line:
            last_col = 0
        if sc > last_col:
            out.append(" " * (sc - last_col))
        if tok == tokenize.OP:
            if text in "([{":
                depth += 1
            elif text in ")]}":
                depth -= 1
        if tok == tokenize.COMMENT:
            pass
        elif (tok == tokenize.STRING and depth == 0
              and prev in (tokenize.INDENT, tokenize.NEWLINE, tokenize.DEDENT, tokenize.NL)):
            pass
        else:
            out.append(text)
        prev = tok
        last_col = ec
        last_line = el
    return "".join(out)


def tidy(src):
    lines = [l.rstrip() for l in src.splitlines()]
    keep, blanks = [], 0
    for l in lines:
        if l.strip() == "":
            blanks += 1
            if blanks > 2 or not keep:
                continue
        else:
            blanks = 0
        keep.append(l)
    while keep and keep[-1] == "":
        keep.pop()
    return "\n".join(keep) + "\n"


def drop_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return tree


src = open(sys.argv[1], encoding="utf-8").read()
stripped = tidy(strip(src))
open(sys.argv[2], "w", encoding="utf-8", newline="\n").write(stripped)

a = ast.dump(drop_docstrings(ast.parse(src)))
b = ast.dump(drop_docstrings(ast.parse(stripped)))
print("AST identical after removing docstrings:", a == b)
print("original lines:", len(src.splitlines()), "-> stripped lines:", len(stripped.splitlines()))
print("comments remaining:", sum(1 for l in stripped.splitlines() if l.lstrip().startswith("#")))
compile(stripped, sys.argv[2], "exec")
print("compiles: OK")
