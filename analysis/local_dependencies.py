"""Copy the reachable driver-local type/macro closure, not guessed substitutes."""

from pathlib import Path
import re

from tree_sitter_languages import get_parser
from analysis.header_context import included_macro_names


def walk(node):
    yield node
    for child in node.named_children:
        yield from walk(child)


def identifiers(text):
    raw = text.encode()
    tree = get_parser("c").parse(raw)
    names = {
        raw[node.start_byte:node.end_byte].decode()
        for node in walk(tree.root_node)
        if node.type in {"identifier", "type_identifier"}
    }
    for node in walk(tree.root_node):
        if node.type == "preproc_arg":
            value = raw[node.start_byte:node.end_byte].decode()
            value = re.sub(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"', " ", value, flags=re.S)
            names.update(re.findall(r"\b[A-Za-z_]\w*\b", value))
    return names


def collect_local_definitions(parse_result, function):
    if not parse_result or not function:
        return [], []
    source = Path(parse_result.path).read_bytes()
    root = get_parser("c").parse(source).root_node
    macro_names = included_macro_names(parse_result.path)
    units, owners = [], {}

    def add(node, kind):
        text = source[node.start_byte:node.end_byte].decode()
        if node.type in {"struct_specifier", "union_specifier", "enum_specifier"} and node.parent is not None:
            # Kernel layout annotations can be parsed as declarators, e.g. `} __packed;`.
            # Preserve annotation tokens, never copy the accompanying global object.
            found = False
            for sibling in node.parent.children:
                if sibling == node:
                    found = True
                    continue
                if not found or sibling.type == "comment":
                    continue
                suffix = source[sibling.start_byte:sibling.end_byte].decode()
                first = re.match(r"([A-Za-z_]\w*)", suffix)
                if sibling.type == "attribute_specifier" or (first and first.group(1) in macro_names):
                    text = source[node.start_byte:sibling.end_byte].decode()
                else:
                    break
        names = set()
        if kind == "macro":
            name = node.child_by_field_name("name")
            if name:
                names.add(source[name.start_byte:name.end_byte].decode())
        else:
            for child in walk(node):
                if child.type in {"struct_specifier", "union_specifier", "enum_specifier", "enumerator"}:
                    name = child.child_by_field_name("name")
                    if name and (child.type == "enumerator" or child.child_by_field_name("body")):
                        names.add(source[name.start_byte:name.end_byte].decode())
            if node.type == "type_definition":
                for decl in node.children_by_field_name("declarator"):
                    while decl.child_by_field_name("declarator"):
                        decl = decl.child_by_field_name("declarator")
                    names.add(source[decl.start_byte:decl.end_byte].decode())
        if not names:
            return
        if kind != "macro" and not text.rstrip().endswith(";"):
            text += ";"
        unit = (kind, text, names)
        units.append(unit)
        for name in names:
            owners.setdefault(name, len(units) - 1)

    def collect(node):
        if node.type in {"preproc_def", "preproc_function_def"}:
            add(node, "macro")
        elif node.type == "type_definition":
            add(node, "type")
        elif node.type in {"struct_specifier", "union_specifier", "enum_specifier"}:
            if node.child_by_field_name("body"):
                add(node, "type")
        elif node.type != "function_definition":
            for child in node.named_children:
                collect(child)

    collect(root)
    functions = {item.name: item for item in parse_result.functions}
    closure, pending = set(), [function.name]
    while pending:
        name = pending.pop()
        if name in closure or name not in functions:
            continue
        closure.add(name)
        pending.extend(parse_result.call_graph.get(name, []))
    roots = set().union(*(identifiers(functions[name].code) for name in closure)) if closure else identifiers(function.code)
    selected, visiting, ordered = set(), set(), []

    def visit(index):
        if index in selected or index in visiting:
            return
        visiting.add(index)
        for name in sorted(identifiers(units[index][1])):
            if name in owners and owners[name] != index:
                visit(owners[name])
        visiting.remove(index)
        selected.add(index)
        ordered.append(index)

    for name in sorted(roots):
        if name in owners:
            visit(owners[name])
    macros = [units[index][1].strip() for index in ordered if units[index][0] == "macro"]
    types = [units[index][1].strip() for index in ordered if units[index][0] == "type"]
    # Forward tags allow legal pointer cycles without changing any copied layout.
    forwards = []
    for text in types:
        for kind, name in re.findall(r"\b(struct|union)\s+(\w+)\s*\{", text):
            declaration = f"{kind} {name};"
            if declaration not in forwards:
                forwards.append(declaration)
    return macros, forwards + types
