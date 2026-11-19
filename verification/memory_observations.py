"""Bounded byte-copy evidence for standard C memory operations."""
import re

from validation.test_inspector import C_PARSER, _node_text, _walk


def byte_arrays(source):
    data = source.encode()
    arrays = {}
    for node in _walk(C_PARSER.parse(data).root_node):
        if node.type != 'declaration':
            continue
        type_ = node.child_by_field_name('type')
        # Only byte-sized element types; never assume an architecture's int size.
        if type_ is None or _node_text(data, type_) not in {'char', 'u8', 's8', 'uint8_t', 'int8_t'}:
            continue
        for decl in node.children_by_field_name('declarator'):
            for item in _walk(decl):
                if item.type != 'array_declarator':
                    continue
                name, size = item.child_by_field_name('declarator'), item.child_by_field_name('size')
                if name is not None and name.type == 'identifier' and size is not None:
                    value = integer(_node_text(data, size), {}, {})
                    if value is not None and value >= 0:
                        arrays[_node_text(data, name)] = value
    return arrays


def integer(expression, values, arrays):
    data = ('void f(void){' + expression + ';}').encode()
    statement = next((n for n in _walk(C_PARSER.parse(data).root_node) if n.type == 'expression_statement'), None)
    if statement is None or not statement.named_children:
        return None

    def evaluate(node):
        text = _node_text(data, node)
        if node.type == 'number_literal':
            value = re.sub(r'[uUlL]+$', '', text)
            try:
                return int(value, 16 if value.lower().startswith('0x') else 10)
            except ValueError:
                return None
        if node.type == 'identifier':
            return values.get(text)
        if node.type in {'parenthesized_expression', 'cast_expression'}:
            return evaluate(node.child_by_field_name('value') or node.named_children[-1])
        if node.type == 'sizeof_expression':
            value = node.child_by_field_name('value')
            key = _node_text(data, value).strip('() ') if value is not None else ''
            return arrays.get(key)
        if node.type == 'call_expression':
            name = _node_text(data, node.child_by_field_name('function'))
            args = node.child_by_field_name('arguments').named_children
            if name == 'min_t':
                args = args[1:]
            if name not in {'min', 'min_t'} or len(args) != 2:
                return None
            operands = [evaluate(arg) for arg in args]
            return min(operands) if all(v is not None for v in operands) else None
        return None

    return evaluate(statement.named_children[0])


class Captures(set):
    def __init__(self):
        super().__init__()
        self.byte_ranges = {}

    def invalidate(self, path, overlaps):
        self.difference_update(p for p in list(self) if overlaps(p, path))
        self.byte_ranges = {p: n for p, n in self.byte_ranges.items() if not overlaps(p, path)}

    def observes(self, path, binding):
        from verification.observation_provenance import _overlaps
        for capture in self:
            if not _overlaps(path, capture):
                continue
            extent = self.byte_ranges.get(capture)
            if extent is None:
                return True
            suffix = path[len(capture):] if path.startswith(capture) else ''
            match = re.fullmatch(r'\[(\d+)\]', suffix)
            if match and int(match[1]) < extent:
                return True
            if path == capture and 'MEMEQ' in binding.macro:
                from verification.assertion_quality import _split_call_args
                args = _split_call_args(binding.statement_text)[1:]
                count = integer(args[2], {}, byte_arrays(binding.support_source)) if len(args) >= 3 else None
                if count is not None and 0 < count <= extent:
                    return True
        return False
