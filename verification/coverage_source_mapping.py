"""Map compiled feedback to the exact source snapshot that owns scenario IDs."""
from collections import defaultdict
import re

TOKEN = re.compile(
    r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    r'|[A-Za-z_][A-Za-z_0-9]*|(?:[0-9][A-Za-z_0-9.]*|\.[0-9][A-Za-z_0-9.]*)'
    r'|>>=|<<=|\.\.\.|->|\+\+|--|&&|\|\||<=|>=|==|!=|\*=|/=|%=|\+=|-=|&=|\^=|\|=|<<|>>|##|[^\s]',
    re.S,
)


def tokens(text):
    return [match[0] for match in TOKEN.finditer(text) if not match[0].startswith(("/*", "//"))]


def walk(node):
    pending = [node]
    while pending:
        item = pending.pop()
        yield item
        pending.extend(reversed(item.named_children))


def function_fragment(parser, source, name):
    raw = source.encode()
    node = next((item for item in walk(parser.parser.parse(raw).root_node)
                 if item.type == "function_definition" and parser._get_function_name(item, raw) == name), None)
    if node is None or node.child_by_field_name("body") is None:
        raise ValueError("Target function is absent from source evidence")
    body = node.child_by_field_name("body")
    fragment = raw[body.start_byte:].decode()
    depth = 0
    # Balance lexical braces because GNU statement expressions can shorten a recovered AST node.
    for match in TOKEN.finditer(fragment):
        value = match[0]
        if value.startswith(("/*", "//")):
            continue
        depth += value == "{"
        depth -= value == "}"
        if depth == 0:
            finish = body.start_byte + len(fragment[:match.end()].encode())
            return raw[node.start_byte:finish].decode(), node.start_point[0] + 1, raw[:finish].count(b"\n") + 1
    raise ValueError("Unbalanced target function source")


def macro_definitions(source):
    result = {}
    logical = re.sub(r"\\\r?\n", " ", source)
    pattern = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)(\([^\n]*?\))?([^\n]*)", re.M)
    for match in pattern.finditer(logical):
        params = match[2]
        entry = {"parameters": re.findall(r"[A-Za-z_]\w*", params or ""), "body": match[3].strip()}
        if params and "..." in params:
            entry["variadic"] = True
            if re.search(r"(?:\(|,)\s*\.\.\.\s*\)$", params):
                entry["parameters"].append("__VA_ARGS__")
        if entry not in result.setdefault(match[1], []):
            result[match[1]].append(entry)
    return result


class CoverageFeedbackMappingError(ValueError):
    pass


def feedback_line_map(parser, original, compiled, function):
    before, before_line, _ = function_fragment(parser, original, function)
    after, after_line, _ = function_fragment(parser, compiled, function)
    definitions = macro_definitions(compiled)
    substitutions = {}
    for _, callee, invoke in re.findall(
            r'RACA_BOUNDARY_HOOK_DECL:\s*boundary=(\w+);\s*original=(\w+);\s*invoke=(\w+)', compiled):
        hook = invoke.removesuffix('_invoke') + '_hook'
        plain = tokens('({' + hook + ' ? ' + hook + '(__VA_ARGS__) : ' + callee + '(__VA_ARGS__); })')
        counted = tokens('({' + invoke.removesuffix('_invoke') + '_call_count++; ' +
                         hook + ' ? ' + hook + '(__VA_ARGS__) : ' + callee + '(__VA_ARGS__); })')
        if any(tokens(item['body']) in (plain, counted) for item in definitions.get(invoke, [])):
            substitutions[invoke] = callee

    def positioned(fragment, first, replacements):
        result, offset, line = [], 0, first
        for match in TOKEN.finditer(fragment):
            line += fragment.count('\n', offset, match.start())
            offset = match.start()
            if not match[0].startswith(('/*', '//')):
                result.append((replacements.get(match[0], match[0]), line))
        return result
    original_tokens = positioned(before, before_line, {})
    compiled_tokens = positioned(after, after_line, substitutions)
    if [t for t, _ in original_tokens] != [t for t, _ in compiled_tokens]:
        raise ValueError('Coverage feedback source differs beyond verified framework hook substitutions')
    mapping = defaultdict(set)
    for (_, origin), (_, current) in zip(original_tokens, compiled_tokens):
        mapping[current].add(origin)
    return {line: sorted(origins) for line, origins in mapping.items()}


def map_feedback(parser, original, compiled, function, missed_lines, missed_branches):
    try:
        mapping = feedback_line_map(parser, original, compiled, function)
    except ValueError as error:
        raise CoverageFeedbackMappingError(str(error)) from error
    lines = sorted({origin for line in missed_lines for origin in mapping.get(line, [])})
    branches = [dict(branch, source_mapping={
        'verified': True, 'basis': 'exact_original_function_token_correspondence',
        'original_lines': mapping.get(branch.get('line'), []),
        'edge_outcome_known': False}) for branch in missed_branches]
    return lines, branches
