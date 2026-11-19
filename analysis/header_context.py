"""Identify macros reachable through the driver's literal include graph."""

from pathlib import Path
import re
from functools import lru_cache


def included_macro_names(driver_path, function_only=False):
    driver = Path(driver_path).resolve()
    info = driver.stat()
    all_names, functions = _included_macros(str(driver), info.st_mtime_ns, info.st_size)
    return set(functions if function_only else all_names)


def included_function_macros(driver_path):
    return included_macro_names(driver_path, function_only=True)


def defined_macro_names(source, function_only=False):
    source = re.sub(r"/\*.*?\*/|//[^\n]*", " ", source, flags=re.S)
    suffix = r"\(" if function_only else r"\b"
    return set(re.findall(r"^\s*#\s*define\s+([A-Za-z_]\w*)" + suffix, source, re.M))


def include_search_roots(driver_path):
    """Use the same literal-header roots for macro identity and source evidence."""
    driver = Path(driver_path).resolve()
    kernel = next((parent for parent in driver.parents if (parent / "include").is_dir()), None)
    roots = [driver.parent]
    if kernel:
        roots.extend([kernel / "include", kernel / "include/uapi", kernel / "include/generated"])
        config = kernel / ".config"
        config_text = config.read_text() if config.is_file() else ""
        architectures = re.findall(r"^CONFIG_([A-Z0-9_]+)=y$", config_text, re.M)
        for arch in sorted((kernel / "arch").glob("*")):
            if arch.name.upper() in architectures:
                roots.extend([arch / "include", arch / "include/uapi", arch / "include/generated"])
    return roots


@lru_cache(maxsize=64)
def _included_macros(driver_path, modified, size):
    driver = Path(driver_path).resolve()
    roots = include_search_roots(driver)
    pending, visited, macros, functions = [driver], set(), set(), set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        try:
            text = path.read_text()
        except (OSError, UnicodeError):
            continue
        text = re.sub(r"/\*.*?\*/|//[^\n]*", " ", text, flags=re.S)
        macros.update(defined_macro_names(text))
        functions.update(defined_macro_names(text, function_only=True))
        for opening, name in re.findall(r'^\s*#\s*include\s*([<"])([^>"\n]+)[>"]', text, re.M):
            search = ([path.parent] if opening == '"' else []) + roots
            for root in search:
                candidate = (root / name).resolve()
                if candidate.is_file():
                    pending.append(candidate)
                    break
    return frozenset(macros), frozenset(functions)
