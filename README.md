# PicassoExample

Native `x86_64-unknown-trueos` Blueprint consuming Picasso as a `#![no_std]`
library. Build it with the Blueprint packer so the TRUEOS target specification,
sysroot, ABI guard, and packaging path are applied:

`.cargo/config.toml` points the packer at the canonical
`TRUEOS-Blueprints/apps/target.json` specification.

```sh
cd ../TRUEOS-Blueprints
cargo run -- ../TRUEOS-Picasso-Example
```

The packer must be launched from the Blueprint repository so its canonical
vendor overlays are available to the native TRUEOS build.

This is a genuinely `#![no_std]` consumer of `trueos-picasso` with its default
host feature disabled. It demonstrates the runtime boundary: already-prepared
Dealer resources are materialized as opaque vGPU buffers in shared DDR5.
Picasso carries only `SharedResourceId` and byte offsets; it never receives a
GPU virtual address. `VVideoRing` owns `VVideoMem` after its `ExecRing` field,
so the mapping outlives every ring view. Its visibility adapter uses
`VVideoMem::flush` and `invalidate` with resource-relative offsets: it flushes
the header plus payload, then flushes the published header control line. The
header remains CPU-owned; after completion it invalidates only GPU-owned payload
bytes, after reading the header metadata.

The packaged binary also runs one end-to-end mixed-topology probe using the
actual `DamagedHelmet.glb` geometry plus three red, green, and blue line-list
primitives. `build.rs` performs the glTF accessor decode and
normalizes its positions at host build time. The `#![no_std]` Blueprint embeds
only prepared position and `u32` index bytes, maps their Picasso ranges to
opaque vGPU buffers, submits the authenticated indexed-render pipeline to a
UI4 surface, waits for its virtual timeline, publishes the frame, and logs when
it crosses UI4's physical SURFLIVE handoff. A versioned indexed-batch contract
keeps the helmet on native triangle-list topology and dispatches each colored
segment as native line-list topology in the same render pass. The lines carry
no transform references, proving that transform scope belongs to a draw rather
than leaking into global render state. The current indexed broker accepts
DMA-backed `Buffer`s; `VVideoMem` remains the shared guest-page backing for
Cubism's execution ring.

The retained-transform proof is intentionally single-geometry: four packed
`TransformId` references select four 48-byte states while every draw continues
to name the same DamagedHelmet vertex and index resources. The GPU animation
program describes a clockwise top-left head, counter-clockwise top-right head,
a bottom-left head that collapses to zero over 0.5 seconds and returns over the
next 0.5 seconds, and an unchanged bottom-right head. All four start at 45%
scale in their respective quadrants. The current vGPU indexed-render ABI does
not yet bind this state table to a vertex shader; until that authenticated
package and submission contract land, the runtime presentation remains one
head plus the three topology-proof lines rather than fabricating four
CPU-transformed vertex copies.

The probe uses Picasso's Blueprint-enabled `FlyCam`: WASD moves at 0.75 world
units per second and holding the primary mouse button while moving the mouse
rotates the view with normalized quaternion composition. The inverse camera
pose is applied to the retained helmet transforms each frame. The static RGB
lines remain a fixed frame reference under the current retained-submit ABI.
Dragging the window's bottom-right resize grip suppresses camera rotation while
still draining its routed mouse samples.

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
