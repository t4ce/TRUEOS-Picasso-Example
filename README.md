# PicassoExample

Native `x86_64-unknown-trueos` Blueprint consuming Picasso as a `#![no_std]`
library. Build it with the Blueprint packer so the TRUEOS target specification,
sysroot, ABI guard, and packaging path are applied:

`.cargo/config.toml` points the packer at the canonical
`TRUEOS-Blueprints/apps/target.json` specification.

```sh
cd ../TRUEOS-Blueprints
cargo run -- ../PicassoExample
```

The packer must be launched from the Blueprint repository so its canonical
vendor overlays are available to the native TRUEOS build.

This is a genuinely `#![no_std]` consumer of `trueos-picasso` with its default
host feature disabled. It demonstrates the runtime boundary: already-prepared
Dealer resources are mapped into shared DDR5, CPU code writes an available ring
generation, then the executor owns that generation until its GPU timeline retires.

There is no `std::fs`, glTF parser, redb backend, MASS filesystem adapter, or
CLI in this build. Those are host-only Picasso features used before boot or by
tools.

`Assets/` is an importer ladder from
[t4ce/glTF-Sample-Assets](https://github.com/t4ce/glTF-Sample-Assets), commit
`e3cc9d8fee3ab25e21aafdcedda6558f224afbee`:

- Triangle — minimal geometry
- BoxInterleaved — interleaved attributes
- SimpleSparseAccessor — sparse accessor decoding
- RiggedSimple — skinning/armature data
- DamagedHelmet — PBR texture/material payload

Every copied model retains its upstream `LICENSE.md`; the relevant common
license texts are also in `Assets/`. The assets are input to the host importer,
not compiled into the bare-metal program.
