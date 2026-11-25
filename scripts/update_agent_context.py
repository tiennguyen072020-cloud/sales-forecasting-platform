from __future__ import annotations

import hashlib
import json
import re
import tomllib
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTEXT_FILE = ROOT / "AI agent Workspace" / "project-context.md"
STATE_FILE = ROOT / "AI agent Workspace" / "project-context-state.json"
PYPROJECT_FILE = ROOT / "pyproject.toml"
README_FILE = ROOT / "README.md"
BUILD_FILE = ROOT / "build"
START_MARKER = "<!-- BEGIN GENERATED AGENT CONTEXT -->"
END_MARKER = "<!-- END GENERATED AGENT CONTEXT -->"


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_inventory(package_dir: Path) -> dict[str, object]:
    if not package_dir.exists():
        return {"exists": False, "python_files": 0, "files": []}

    files = sorted(
        path.relative_to(ROOT).as_posix()
        for path in package_dir.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    return {"exists": True, "python_files": len(files), "files": files}


def source_snapshot() -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for package in (ROOT / "ml", ROOT / "api"):
        for path in package.rglob("*.py"):
            if "__pycache__" not in path.parts:
                snapshot[path.relative_to(ROOT).as_posix()] = file_digest(path)
    return snapshot


def build_snapshot() -> dict[str, str]:
    snapshot: dict[str, str] = {}
    build_lib = ROOT / "build" / "lib"
    if not build_lib.exists():
        return snapshot
    for path in build_lib.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            snapshot[path.relative_to(ROOT).as_posix()] = file_digest(path)
    return snapshot


def compare_sets(old: list[str] | set[str], new: list[str] | set[str]) -> tuple[list[str], list[str]]:
    old_set = set(old)
    new_set = set(new)
    return sorted(new_set - old_set), sorted(old_set - new_set)


def format_dependency_changes(label: str, added: list[str], removed: list[str]) -> list[str]:
    if not added and not removed:
        return [f"- {label}: no dependency changes since the previous scan."]
    lines = [f"- {label} changes since the previous scan:"]
    if added:
        lines.append(f"  - Added: `{', '.join(added)}`")
    if removed:
        lines.append(f"  - Removed: `{', '.join(removed)}`")
    return lines


def build_consistency() -> tuple[str, list[str]]:
    if not BUILD_FILE.exists():
        return "not found", []

    build_lib = ROOT / "build" / "lib"
    if not build_lib.exists():
        return "build directory exists, but build/lib is missing", []

    stale_or_missing: list[str] = []
    source_files = 0
    for package_name in ("ml", "api"):
        package = ROOT / package_name
        if not package.exists():
            continue
        for source in package.rglob("*.py"):
            if "__pycache__" in source.parts:
                continue
            source_files += 1
            relative = source.relative_to(ROOT).as_posix()
            built = ROOT / "build" / "lib" / relative
            if not built.exists() or file_digest(source) != file_digest(built):
                stale_or_missing.append(relative)

    if stale_or_missing:
        return "outdated or missing build output for some source files", stale_or_missing
    if source_files == 0:
        return "build/lib exists but no source files were found", []
    return "build/lib matches the current ml/ and api/ Python source files", []


def main() -> None:
    if not PYPROJECT_FILE.exists():
        raise SystemExit("Missing pyproject.toml")
    if not README_FILE.exists():
        raise SystemExit("Missing README.md")
    if not CONTEXT_FILE.exists():
        raise SystemExit("Missing AI agent Workspace/project-context.md")

    with PYPROJECT_FILE.open("rb") as file:
        pyproject = tomllib.load(file)

    project = pyproject.get("project", {})
    runtime_dependencies = sorted(str(item) for item in project.get("dependencies", []))
    optional_dependencies = project.get("optional-dependencies", {})
    dev_dependencies = sorted(str(item) for item in optional_dependencies.get("dev", []))
    python_requirement = project.get("requires-python", "not declared")

    readme = README_FILE.read_text(encoding="utf-8-sig")
    heading_match = re.search(r"^#\s+(.+)$", readme, re.MULTILINE)
    readme_heading = heading_match.group(1).strip() if heading_match else "No H1 heading found"
    has_30_60_90_contract = all(token in readme for token in ("30 days", "60 days", "90 days"))

    ml_inventory = package_inventory(ROOT / "ml")
    api_inventory = package_inventory(ROOT / "api")
    current_sources = source_snapshot()
    current_build = build_snapshot()
    build_status, stale_files = build_consistency()

    previous_state: dict[str, object] = {}
    has_previous_state = STATE_FILE.exists()
    if has_previous_state:
        previous_state = json.loads(STATE_FILE.read_text(encoding="utf-8"))

    previous_dependencies = list(previous_state.get("runtime_dependencies", []))
    previous_dev_dependencies = list(previous_state.get("dev_dependencies", []))
    previous_sources = dict(previous_state.get("source_files", {}))

    if has_previous_state:
        runtime_added, runtime_removed = compare_sets(previous_dependencies, runtime_dependencies)
        dev_added, dev_removed = compare_sets(previous_dev_dependencies, dev_dependencies)
    else:
        runtime_added, runtime_removed = [], []
        dev_added, dev_removed = [], []

    changed_sources = []
    if previous_sources:
        old_paths = set(previous_sources)
        new_paths = set(current_sources)
        changed_sources.extend(sorted(new_paths - old_paths))
        changed_sources.extend(sorted(old_paths - new_paths))
        changed_sources.extend(
            sorted(path for path in old_paths & new_paths if previous_sources[path] != current_sources[path])
        )

    new_state = {
        "scanned_at": datetime.now().date().isoformat(),
        "python_requirement": python_requirement,
        "runtime_dependencies": runtime_dependencies,
        "dev_dependencies": dev_dependencies,
        "source_files": current_sources,
        "build_files": current_build,
        "ml_inventory": ml_inventory,
        "api_inventory": api_inventory,
        "readme_heading": readme_heading,
    }
    STATE_FILE.write_text(json.dumps(new_state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    generated_lines = [
        START_MARKER,
        "## Generated source and build context",
        "",
        f"- Last scanned: `{datetime.now().date().isoformat()}`.",
        f"- README heading: `{readme_heading}`.",
        f"- README supports the 30/60/90-day contract: `{'yes' if has_30_60_90_contract else 'not detected'}`.",
        f"- Python package requirement: `{python_requirement}`.",
        f"- Source package inventory: `ml/` has `{ml_inventory['python_files']}` Python files; `api/` has `{api_inventory['python_files']}` Python files.",
        f"- Build status: `{build_status}`.",
        "- Runtime dependencies:",
    ]
    generated_lines.extend(f"  - `{dependency}`" for dependency in runtime_dependencies)
    generated_lines.append("- Development dependencies:")
    generated_lines.extend(f"  - `{dependency}`" for dependency in dev_dependencies)
    if has_previous_state:
        generated_lines.extend(format_dependency_changes("Runtime dependency", runtime_added, runtime_removed))
        generated_lines.extend(format_dependency_changes("Development dependency", dev_added, dev_removed))
    else:
        generated_lines.extend([
            "- Runtime dependency changes: initial dependency baseline created.",
            "- Development dependency changes: initial dependency baseline created.",
        ])

    if previous_sources:
        if changed_sources:
            generated_lines.append("- Source files changed since the previous scan:")
            generated_lines.extend(f"  - `{path}`" for path in sorted(set(changed_sources)))
        else:
            generated_lines.append("- Source files: no `ml/` or `api/` Python file changes since the previous scan.")
    else:
        generated_lines.append("- Source/dependency change detection: initial baseline created; a later scan can report additions, removals, and modifications.")

    if stale_files:
        generated_lines.append("- Files that need a rebuild or are missing from `build/lib`:")
        generated_lines.extend(f"  - `{path}`" for path in stale_files)

    generated_lines.extend(["", END_MARKER])
    generated_block = "\n".join(generated_lines)

    context = CONTEXT_FILE.read_text(encoding="utf-8")
    if START_MARKER in context and END_MARKER in context:
        pattern = re.compile(re.escape(START_MARKER) + r".*?" + re.escape(END_MARKER), re.DOTALL)
        context = pattern.sub(generated_block, context)
    else:
        marker = "## Continuation protocol"
        if marker in context:
            context = context.replace(marker, generated_block + "\n\n" + marker, 1)
        else:
            context = context.rstrip() + "\n\n" + generated_block + "\n"

    context = re.sub(
        r"^Last verified: [^\.]+\.",
        f"Last scanned: {datetime.now().date().isoformat()}.",
        context,
        count=1,
        flags=re.MULTILINE,
    )
    CONTEXT_FILE.write_text(context, encoding="utf-8")

    print(f"Updated {CONTEXT_FILE.relative_to(ROOT).as_posix()}")
    print(f"Created/updated {STATE_FILE.relative_to(ROOT).as_posix()}")
    print(f"Python requirement: {python_requirement}")
    print(f"ml Python files: {ml_inventory['python_files']}")
    print(f"api Python files: {api_inventory['python_files']}")
    print(f"Build status: {build_status}")


if __name__ == "__main__":
    main()
