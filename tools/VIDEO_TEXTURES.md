# Hardware video textures

The initial view (key 1, formerly four DamagedHelmet instances) now shows four
quads. They play the embedded `DSC_1879_48_a.mp4`, `DSC_1879_128_a.mp4`, and
`DSC_1879_256_a.mp4`; the fourth quad shares the first stream. The clips retain
MP4 presentation timestamps, B-frames, and full-range BT.709 metadata. Audio is
not played. Other asset views and the existing camera/window controls remain.
Leaving this view drops the three streams. Returning starts them again.

This requires the matching kernel and Blueprint SDK changes. New imports are
strictly additive: `trueos_cabi_vmedia_video_command_v1` and
`trueos_cabi_vgpu_retained_textured_frame_v1`. Old binaries keep their old ABI;
this new demo requires a kernel exporting the two new symbols.

## Resource path

`vmedia::Video::open(device, encoded, looping)` uploads compressed bytes once.
The kernel reuses `vid fs`'s MP4/Annex-B ingress, media-session reservation,
H.264 decoder/DPB, PTS ordering and GuC NV12 conversion. Texture players reserve
from the same three playback slots as shell video. A fourth player returns busy;
there is no additional per-Blueprint allowance. The demo waits and retries if
three slots are unavailable. Other asset keys remain responsive while waiting.

Each stream owns three persistent RGBA8 textures in its device's Picasso Render1
carrier. GuC writes directly into that storage: no decoded pixel readback to the
Blueprint, CPU texture update, per-frame texture allocation, or video atlas.
The decoder source remains pinned until conversion completion. A destination
write lease excludes render readers; renderer references pin the exact mapping.
An unproven GPU completion quarantines both source and destination.

`Video::poll()` returns Pending, a `VideoFrame`, or End/error. Keep the previous
frame on Pending. A frame owns a read lease; drop it after the GPU has retired
its last draw. Dropping the Video handle stops it after any outstanding frames
are dropped. VM teardown cancels the stream and drains producer/GPU ownership.
A full ring applies backpressure without occupying the two shared conversion
lanes, so a stalled consumer cannot block all other players.

`Device::submit_retained_textured_frame_v1` binds up to four independent retained
texture IDs to four ranges/transforms of one mesh, in one render pass. It uses
the existing unlit POS_NORMAL_UV shader. Duplicate IDs are allowed, so sharing
a video across meshes does not require another decoder. The demo uploads four
vertices and six indices once, uses four instances, and waits for render
retirement before replacing frame leases. No PBR maps, lights, or atlas-copy
passes are used. The existing retained transform/depth pass remains in use.

## Verification

Host regression commands:

- TRUEOS: `python3 tools/test_video_texture.py`
- TRUEOS: `python3 tools/test_retained_scene.py`
- Picasso Example: `python3 tools/test_prepared_assets.py`
- Both repositories: `cargo check`; TRUEOS also links with `cargo build`.

The video tests exercise production upload/lease code, stream command/ring
logic, shared-slot admission, owner teardown, and textured draw validation.
GPU execution is mocked in the stream tests. The checked-in asset manifest
records the exact source hashes and metadata.

Physical validation remains required: verify all three videos move on the
quads, the repeated 48px quad stays identical, UV orientation/colour are correct,
looping continues beyond 1,071 frames, camera motion preserves mapping, and
switching away or closing releases all slots. `vid status` identifies
`picasso-texture` versus `ui4-window` sinks; `vid stop <slot>` can cancel either.
Check a shell video plus one/two texture streams also respects the shared cap.
No hardware FPS or visual-correctness claim follows from the host tests.
