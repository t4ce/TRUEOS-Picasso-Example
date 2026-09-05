#!/usr/bin/env python3
"""Host tests for the actual asset bake and Picasso database handoff.

The Blueprint binary cannot link the host test runner. This temporary Cargo
crate imports the production build script and pure parameter serialization,
then runs their Rust tests against Picasso's real in-memory redb backend.
"""

from pathlib import Path
import json
import os
import re
import subprocess
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
BLUEPRINTS = ROOT.parent / "TRUEOS-Blueprints"


def item(path: Path, name: str) -> str:
    source = path.read_text()
    matches = list(re.finditer(
        rf"^(?:pub(?:\([^)]*\))?\s+)?(?:fn|struct|mod)\s+{re.escape(name)}\b",
        source, re.MULTILINE,
    ))
    if len(matches) != 1:
        raise ValueError(f"{path}: expected one production item {name}")
    start = matches[0].start()
    attributes = re.search(r"(?:^#\[[^\n]*\]\n)+\Z", source[:start], re.MULTILINE)
    if attributes:
        start = attributes.start()
    end = re.search(r"^}\s*\n", source[matches[0].end():], re.MULTILINE)
    if not end:
        raise ValueError(f"{path}: no top-level closing brace for {name}")
    return source[start:matches[0].end() + end.end()]


def main() -> None:
    api = BLUEPRINTS / "crates/trueos-v/src/vgpu.rs"
    runtime = ROOT / "src/main.rs"
    source = (
        "#![allow(dead_code)]\n"
        f"#[path = {json.dumps(str(ROOT / 'build.rs'))}]\nmod asset_bake;\n"
        + item(api, "RetainedMaterialParameters")
        + item(runtime, "material_parameter_bytes")
        + item(runtime, "material_parameters_from_bytes")
        + item(runtime, "material_parameter_tests")
    )
    with tempfile.TemporaryDirectory(prefix="picasso-material-tests-") as temporary:
        directory = Path(temporary)
        (directory / "lib.rs").write_text(source)
        (directory / "Cargo.toml").write_text(
            '[package]\nname = "picasso-material-host-tests"\nversion = "0.0.0"\nedition = "2024"\n'
            '[workspace]\n[lib]\npath = "lib.rs"\n'
            '[dependencies]\nbase64 = "=0.22.1"\nbevy_mikktspace = "=0.15.3"\n'
            'gltf = { version = "=1.4.1", default-features = false, features = ["utils"] }\n'
            f'trueos-picasso = {{ path = {json.dumps(str(ROOT.parent / "TRUEOS-Picasso"))}, '
            'default-features = false, features = ["runtime-assets", "test-std"] }\n'
        )
        env = os.environ.copy()
        env["RUSTUP_TOOLCHAIN"] = tomllib.loads(
            (BLUEPRINTS / "rust-toolchain.toml").read_text()
        )["toolchain"]["channel"]
        subprocess.run([
            "cargo", "test", "--offline", "--manifest-path", str(directory / "Cargo.toml"),
            "--target-dir", str(ROOT / "target/material-host-tests"),
        ], cwd=directory, env=env, check=True)


if __name__ == "__main__":
    main()
