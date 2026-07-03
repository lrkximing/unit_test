import os
import re
import json
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Set, Optional, Tuple

from tree_sitter_languages import get_parser


# ----------------------------
# Data Structures 
# ----------------------------

@dataclass
class MacroInfo:
    name: str
    code: str
    line: int


@dataclass
class StructInfo:
    """Only in-file struct definitions (not headers)."""
    name: str
    start_line: int
    end_line: int
    code: str


@dataclass
class FunctionInfo:
    """
    Test-oriented function IR (核心输出).
    - calls/type_refs/macro_refs/field_access/global_refs: for test generation context construction
    - external_symbols: suspected from headers or other files (later resolve stage)
    """
    name: str
    start_line: int
    end_line: int
    code: str

    # test-oriented facts
    calls: List[str] = field(default_factory=list)                  # called functions (both internal & external)
    type_refs: List[str] = field(default_factory=list)              # referenced type identifiers (struct/typedef)
    macro_refs: List[str] = field(default_factory=list)             # macro-like identifiers used in logic
    field_access: Dict[str, Set[str]] = field(default_factory=dict) # base_var -> {field1, field2}
    global_refs: List[str] = field(default_factory=list)            # referenced file-level globals/symbols

    # symbol classification
    file_internal_symbols: Dict[str, List[str]] = field(default_factory=dict)  # intersection with file-level defs
    external_symbols: Set[str] = field(default_factory=set)                    # unknown symbols (likely headers)


@dataclass
class FileParseResult:
    path: str
    includes: List[str]

    # file-level definition sets
    file_level_defs: Dict[str, Set[str]]          # {"functions","types","globals","macros","structs"}

    # optional in-file definitions (not headers)
    macros: Dict[str, MacroInfo]                  # name -> MacroInfo (only in this .c)
    structs: Dict[str, StructInfo]                # name -> StructInfo  (only in this .c)

    # function IR list
    functions: List[FunctionInfo]

    # derived graphs
    call_graph: Dict[str, List[str]]              # f -> callees
    groups: List[List[str]]                       # connected components (undirected)


# ----------------------------
# Parser Implementation
# ----------------------------

class CDriverParser:
    """
    Parse a single C driver .c file into a test-oriented IR.
    Notes:
    - This parser does NOT parse headers.
    - It prepares 'external_symbols' + 'includes' as hints for later symbol resolution.
    """

    def __init__(self):
        self.parser = get_parser("c")

    # ---------- basic io ----------
    def _read_code(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _slice(self, code_bytes: bytes, start_b: int, end_b: int) -> str:
        return code_bytes[start_b:end_b].decode("utf-8", errors="ignore")

    # ---------- includes / macros (regex, because preprocessor isn't in AST reliably) ----------
    def _parse_includes(self, code_str: str) -> List[str]:
        pattern = re.compile(r'^\s*#\s*include\s+([<"].+[>"])', re.MULTILINE)
    
        return [m.group(1) for m in pattern.finditer(code_str)]

    def _collect_macros(self, code_str: str) -> Dict[str, MacroInfo]:
        """
        Collect only macros defined in this file.
        (Later you may also want to collect macro uses per function via identifier rules.)
        """
        macros: Dict[str, MacroInfo] = {}
        pattern = re.compile(r'^\s*#\s*define\s+([A-Za-z_]\w*)\s*(.*)$', re.MULTILINE)

        for m in pattern.finditer(code_str):
            name = m.group(1)
            full_line = m.group(0).strip()
            line_no = code_str[:m.start()].count("\n") + 1
            macros[name] = MacroInfo(name=name, code=full_line, line=line_no)
        return macros

    # ---------- struct defs (in-file only) ----------
    def _collect_struct_defs(self, tree, code_bytes: bytes) -> Dict[str, StructInfo]:
        root = tree.root_node
        structs: Dict[str, StructInfo] = {}

        def is_named_struct_def(node) -> bool:
            # struct_specifier with both name and body => "struct foo { ... }"
            return (
                node.type == "struct_specifier"
                and node.child_by_field_name("name") is not None
                and node.child_by_field_name("body") is not None
            )

        stack = [root]
        while stack:
            n = stack.pop()
            if is_named_struct_def(n):
                name_node = n.child_by_field_name("name")
                struct_name = self._slice(code_bytes, name_node.start_byte, name_node.end_byte)

                start_line = n.start_point[0] + 1
                end_line = n.end_point[0] + 1
                struct_code = self._slice(code_bytes, n.start_byte, n.end_byte).rstrip()
                if not struct_code.endswith(";"):
                    struct_code = struct_code + ";"
                struct_code += "\n"

                # if multiple defs with same name appear, keep the first; you can change policy if needed
                if struct_name not in structs:
                    structs[struct_name] = StructInfo(
                        name=struct_name,
                        start_line=start_line,
                        end_line=end_line,
                        code=struct_code,
                    )

            stack.extend(n.children)
        return structs

    # ---------- file-level defs ----------
    def _get_function_name(self, func_node, code_bytes: bytes) -> Optional[str]:
        """
        Extract function name from function_definition node.
        We DFS in its declarator subtree and return first identifier found,
        skipping parameter_list to avoid capturing param names.
        """
        target = None
        for c in func_node.children:
            if "declarator" in c.type:
                target = c
                break
        if target is None:
            return None

        stack = [target]
        while stack:
            n = stack.pop()
            if n.type == "identifier":
                return self._slice(code_bytes, n.start_byte, n.end_byte)
            if n.type == "parameter_list":
                continue
            stack.extend(n.children)
        return None

    def _get_declaration_identifiers(self, node, code_bytes: bytes) -> List[str]:
        """
        Best-effort extraction of identifiers from declarations.
        Used for global vars and params/locals.
        """
        names: Set[str] = set()
        stack = [node]
        while stack:
            n = stack.pop()
            if n.type == "identifier":
                names.add(self._slice(code_bytes, n.start_byte, n.end_byte))
            stack.extend(n.children)
        return sorted(names)

    def _collect_file_level_defs(self, tree, code_bytes: bytes, macros_in_file: Dict[str, MacroInfo],
                                 structs_in_file: Dict[str, StructInfo]) -> Dict[str, Set[str]]:
        """
        Collect file-level defined symbol names:
        - functions (definitions)
        - types (typedef identifiers, and also named in-file structs)
        - globals (top-level declarations)
        - macros (from regex)
        - structs (in-file struct names)
        """
        root = tree.root_node
        defs: Dict[str, Set[str]] = {
            "functions": set(),
            "types": set(),
            "globals": set(),
            "macros": set(macros_in_file.keys()),
            "structs": set(structs_in_file.keys()),
        }

        for child in root.children:
            if child.type == "function_definition":
                fn = self._get_function_name(child, code_bytes)
                if fn:
                    defs["functions"].add(fn)

            elif child.type == "type_definition":
                # typedef ... name;
                # find last identifier in subtree as typedef name (best-effort)
                ids = self._get_declaration_identifiers(child, code_bytes)
                if ids:
                    defs["types"].add(ids[-1])

            elif child.type == "declaration":
                # top-level declaration might be global var or function prototype
                # we conservatively treat identifiers as globals; later filtering will remove known function defs.
                ids = self._get_declaration_identifiers(child, code_bytes)
                for i in ids:
                    defs["globals"].add(i)

        # avoid double-counting function names as globals
        defs["globals"] -= defs["functions"]
        return defs

    # ---------- function-level facts ----------
    def _collect_params(self, func_node, code_bytes: bytes) -> Set[str]:
        params: Set[str] = set()
        stack = [func_node]
        while stack:
            n = stack.pop()
            if n.type == "parameter_declaration":
                for nm in self._get_declaration_identifiers(n, code_bytes):
                    params.add(nm)
            stack.extend(n.children)
        return params

    def _collect_locals(self, func_node, code_bytes: bytes) -> Set[str]:
        locals_: Set[str] = set()
        stack = [func_node]
        while stack:
            n = stack.pop()
            if n.type == "declaration":
                for nm in self._get_declaration_identifiers(n, code_bytes):
                    locals_.add(nm)
            stack.extend(n.children)
        return locals_

    def _collect_calls(self, func_node, code_bytes: bytes) -> Set[str]:
        """
        Collect called function names from call_expression nodes.
        Handles common cases:
        - foo(...)
        - ptr->ops->foo(...)  (we take the last field identifier as callee hint)
        - (*fp)(...)  (hard; we ignore or record as unknown)
        """
        calls: Set[str] = set()
        stack = [func_node]
        while stack:
            n = stack.pop()
            if n.type == "call_expression":
                fn_node = n.child_by_field_name("function")
                if fn_node is not None:
                    callee = self._extract_callee_name(fn_node, code_bytes)
                    if callee:
                        calls.add(callee)
            stack.extend(n.children)
        return calls

    def _extract_callee_name(self, node, code_bytes: bytes) -> Optional[str]:
        """
        Best-effort extraction of callee name from call_expression.function:
        - identifier -> "foo"
        - field_expression -> take last field_identifier
        - parenthesized_expression / pointer_expression -> recurse
        """
        if node.type == "identifier":
            return self._slice(code_bytes, node.start_byte, node.end_byte)

        if node.type == "field_expression":
            # e.g., a->b->foo : field is field_identifier
            field_n = node.child_by_field_name("field")
            if field_n and field_n.type in ("field_identifier", "identifier"):
                return self._slice(code_bytes, field_n.start_byte, field_n.end_byte)

        # unwrap common wrappers
        if node.type in ("parenthesized_expression", "pointer_expression"):
            # try first named child
            for c in node.children:
                if c.is_named:
                    return self._extract_callee_name(c, code_bytes)

        # function_pointer call like (*fp)(...) is hard; return None
        return None

    def _collect_type_refs(self, func_node, code_bytes: bytes) -> Set[str]:
        """
        Collect referenced type identifiers.
        Tree-sitter C usually uses:
        - type_identifier
        - struct_specifier(name)
        """
        types: Set[str] = set()
        stack = [func_node]
        while stack:
            n = stack.pop()
            if n.type == "type_identifier":
                types.add(self._slice(code_bytes, n.start_byte, n.end_byte))
            elif n.type == "struct_specifier":
                name_n = n.child_by_field_name("name")
                if name_n is not None:
                    types.add(self._slice(code_bytes, name_n.start_byte, name_n.end_byte))
            stack.extend(n.children)
        return types

    def _collect_field_access(self, func_node, code_bytes: bytes) -> Dict[str, Set[str]]:
        """
        Collect struct field accesses of the form:
        - a->field
        - a.field
        We store: base_identifier -> {field_identifier,...}
        """
        access: Dict[str, Set[str]] = {}
        stack = [func_node]
        while stack:
            n = stack.pop()
            if n.type == "field_expression":
                arg_n = n.child_by_field_name("argument")
                field_n = n.child_by_field_name("field")
                if arg_n is not None and field_n is not None:
                    base = self._extract_base_identifier(arg_n, code_bytes)
                    fld = self._slice(code_bytes, field_n.start_byte, field_n.end_byte)
                    if base and fld:
                        access.setdefault(base, set()).add(fld)
            stack.extend(n.children)
        return access

    def _extract_base_identifier(self, node, code_bytes: bytes) -> Optional[str]:
        """
        Extract base variable name from expressions like:
        - dev->regmap  => base 'dev'
        - ctx->a->b    => base 'ctx' (best-effort)
        """
        if node.type == "identifier":
            return self._slice(code_bytes, node.start_byte, node.end_byte)

        # unwrap pointer_expression, parenthesized_expression, field_expression
        if node.type in ("pointer_expression", "parenthesized_expression"):
            for c in node.children:
                if c.is_named:
                    return self._extract_base_identifier(c, code_bytes)

        if node.type == "field_expression":
            arg_n = node.child_by_field_name("argument")
            if arg_n is not None:
                return self._extract_base_identifier(arg_n, code_bytes)

        return None

    def _collect_identifiers(self, node, code_bytes: bytes) -> Set[str]:
        """
        Collect all identifier tokens (not including field_identifier).
        Used for symbol classification and macro reference collection.
        """
        ids: Set[str] = set()
        stack = [node]
        while stack:
            n = stack.pop()
            if n.type == "identifier":
                ids.add(self._slice(code_bytes, n.start_byte, n.end_byte))
            stack.extend(n.children)
        return ids

    def _collect_macro_refs_from_identifiers(self, identifiers: Set[str], file_macro_names: Set[str]) -> Set[str]:
        """
        Macro-like identifiers:
        - defined in this file via #define
        - or ALL_CAPS style (common in kernel)
        """
        macros: Set[str] = set()
        for x in identifiers:
            if x in file_macro_names:
                macros.add(x)
            elif re.match(r"^[A-Z_][A-Z0-9_]*$", x):
                macros.add(x)
        return macros

    def _build_function_infos(self, tree, code_str: str, code_bytes: bytes,
                              file_defs: Dict[str, Set[str]]) -> List[FunctionInfo]:
        root = tree.root_node
        funcs: List[FunctionInfo] = []
        file_macro_names = file_defs.get("macros", set())

        for child in root.children:
            if child.type != "function_definition":
                continue

            name = self._get_function_name(child, code_bytes)
            if not name:
                continue

            start_line = child.start_point[0] + 1
            end_line = child.end_point[0] + 1
            func_code = self._slice(code_bytes, child.start_byte, child.end_byte)

            params = self._collect_params(child, code_bytes)
            locals_ = self._collect_locals(child, code_bytes)

            # function-level facts
            calls = self._collect_calls(child, code_bytes)
            type_refs = self._collect_type_refs(child, code_bytes)
            field_access = self._collect_field_access(child, code_bytes)

            # all identifiers for classification
            used_ids = self._collect_identifiers(child, code_bytes)
            used_ids -= (params | locals_ | {name})

            # macro refs from file-local definitions and kernel-style constant identifiers
            macro_refs = self._collect_macro_refs_from_identifiers(used_ids, file_macro_names)

            # internal vs external classification (file-level)
            internal_symbols = {
                "functions": sorted(list(file_defs["functions"] & used_ids)),
                "globals": sorted(list(file_defs["globals"] & used_ids)),
                "types": sorted(list((file_defs["types"] | file_defs["structs"]) & used_ids)),
                "macros": sorted(list(file_defs["macros"] & used_ids)),
            }

            internal_all = (
                file_defs["functions"]
                | file_defs["globals"]
                | file_defs["types"]
                | file_defs["macros"]
                | file_defs["structs"]
            )
            external_symbols = used_ids - internal_all

            # global refs = identifiers that are in file-level globals
            global_refs = sorted(list(file_defs["globals"] & used_ids))

            fi = FunctionInfo(
                name=name,
                start_line=start_line,
                end_line=end_line,
                code=func_code,

                calls=sorted(calls),
                type_refs=sorted(type_refs),
                macro_refs=sorted(macro_refs),
                field_access=field_access,
                global_refs=global_refs,

                file_internal_symbols=internal_symbols,
                external_symbols=external_symbols,
            )
            funcs.append(fi)

        return funcs

    # ---------- graph & grouping ----------
    def _build_call_graph(self, functions: List[FunctionInfo]) -> Dict[str, List[str]]:
        graph: Dict[str, List[str]] = {}
        for f in functions:
            graph[f.name] = list(f.calls)
        return graph

    def _group_by_connected_components(self, call_graph: Dict[str, List[str]]) -> List[List[str]]:
        """
        Group functions by undirected connectivity:
        If A calls B, then A and B are in same group.
        """
        # build undirected adjacency
        adj: Dict[str, Set[str]] = {k: set() for k in call_graph.keys()}
        for u, vs in call_graph.items():
            for v in vs:
                if v not in adj:
                    # v might be external function not defined in file; ignore for grouping
                    continue
                adj[u].add(v)
                adj[v].add(u)

        visited: Set[str] = set()
        groups: List[List[str]] = []

        for node in adj.keys():
            if node in visited:
                continue
            # BFS/DFS
            comp: List[str] = []
            stack = [node]
            visited.add(node)
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for nxt in adj[cur]:
                    if nxt not in visited:
                        visited.add(nxt)
                        stack.append(nxt)
            groups.append(sorted(comp))

        # sort groups by size desc then name
        groups.sort(key=lambda g: (-len(g), g[0] if g else ""))
        return groups

    # ---------- public API ----------
    def parse_file(self, path: str) -> FileParseResult:
        code_str = self._read_code(path)
        code_bytes = code_str.encode("utf-8", errors="ignore")
        tree = self.parser.parse(code_bytes)

        # 1) preprocessor facts (as hints, not analysis unit)
        includes = self._parse_includes(code_str)
        macros_in_file = self._collect_macros(code_str)

        # 2) in-file struct defs (optional but useful as local symbol index)
        structs_in_file = self._collect_struct_defs(tree, code_bytes)

        # 3) file-level defs (functions/types/globals/macros/structs)
        file_defs = self._collect_file_level_defs(tree, code_bytes, macros_in_file, structs_in_file)

        # 4) function-level test-oriented IR
        functions = self._build_function_infos(tree, code_str, code_bytes, file_defs)

        # 5) call graph + groups
        call_graph = self._build_call_graph(functions)
        groups = self._group_by_connected_components(call_graph)

        return FileParseResult(
            path=path,
            includes=includes,
            file_level_defs=file_defs,
            macros=macros_in_file,
            structs=structs_in_file,
            functions=functions,
            call_graph=call_graph,
            groups=groups,
        )


# ----------------------------
# Utilities
# ----------------------------

def save_result(result: FileParseResult, out_path: str) -> None:
    def convert(obj):
        if isinstance(obj, set):
            return list(obj)
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, default=convert, ensure_ascii=False, indent=2)
