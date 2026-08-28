//! Consumer-first Picasso example for a `#![no_std]` TRUEOS application.
//!
//! At boot the platform resolves Dealer IDs to opaque vGPU buffers. Picasso
//! receives only stable resource identities and byte-relative ranges.
#![no_std]

extern crate alloc;

mod demodata;

use alloc::{format, string::String, vec::Vec};
use trueos::ui4_scene::{
    CursorIcon, CursorSource, Damage, Error as Ui4Error, Frame, POINTER_BUTTON_PRIMARY,
    output_dimensions,
};
use trueos::vgpu::{
    BUFFER_USAGE_INDEX, BUFFER_USAGE_MAP_WRITE, BUFFER_USAGE_VERTEX, Buffer, BufferSlice,
    Capabilities, Device, IndexedBatchDrawV2, Queue, QueueClass, RETAINED_VERTEX_LAYOUT_POS_NORMAL,
    RETAINED_VERTEX_LAYOUT_POS_NORMAL_UV, RetainedFrameSubmit, RetainedMesh,
    RetainedMeshDescriptor, RetainedTransformSeed, VVideoMem,
};
use trueos::{
    clock,
    logl::{self, level},
    vsys,
};
use trueos_picasso::ExecRing;
use trueos_picasso::GRID_INDICES;
use trueos_picasso::GRID_VERTICES;
use trueos_picasso::Picasso;
use trueos_picasso::cam::{Camera, FlyCam, Projection, Quaternion};
use trueos_picasso::{CubismError, SharedByteRange, VVideoRingError, VisibilityOps};
use trueos_picasso::{
    ExecutablePrimitive, PreparedRange, PrimitiveTopology, ResourceId, SharedResourceId,
    TransformRefList, TransformStateRange, TransformValue,
};

pub struct PreparedAsset {
    pub name: &'static str,
    pub vertices: &'static [u8],
    pub indices: &'static [u8],
    pub vertex_count: u32,
    pub index_count: u32,
    pub vertex_stride: usize,
    pub base_color_name: &'static str,
    pub base_color_bytes: &'static [u8],
    pub sampled_material: bool,
    pub helmet_program: bool,
}
include!(concat!(env!("OUT_DIR"), "/prepared_assets.rs"));

/// Prepared bytes loaded back through Picasso's runtime database boundary.
/// Static build output is permitted to seed the database, but never to feed a
/// GPU upload directly.
struct DatabasePreparedAsset {
    vertices: Vec<u8>,
    indices: Vec<u8>,
    base_color: Vec<u8>,
}

struct DatabasePreparedCatalog {
    assets: Vec<DatabasePreparedAsset>,
    line_vertices: Vec<u8>,
    line_indices: Vec<u8>,
}

pub const HELMET_VERTEX_COUNT: u32 = ASSETS[0].vertex_count;
pub const HELMET_INDEX_COUNT: u32 = ASSETS[0].index_count;
pub const HELMET_VERTEX_BYTES: u64 = ASSETS[0].vertex_count as u64 * 12;
pub const HELMET_INDEX_BYTES: u64 = ASSETS[0].index_count as u64 * 4;

pub const HELMET_VERTICES: ResourceId = ResourceId(0x0001_0000_0000_0001);
pub const HELMET_INDICES: ResourceId = ResourceId(0x0001_0000_0000_0002);
pub const HELMET_TRANSFORM_REFS: ResourceId = ResourceId(0x0001_0000_0000_0003);
pub const HEAD_INSTANCE_COUNT: u32 = 4;
pub const HEAD_TRANSFORM_STATE_BYTES: u64 =
    HEAD_INSTANCE_COUNT as u64 * core::mem::size_of::<TransformValue>() as u64;

/// Mesh-local `TransformId`s. All four entries reference one retained state
/// table while the vertex/index resources above remain single-copy.
pub static HELMET_TRANSFORM_REFS_U32: &[u8; 16] = &[
    0, 0, 0, 0, // top-left
    1, 0, 0, 0, // top-right
    2, 0, 0, 0, // bottom-left
    3, 0, 0, 0, // bottom-right
];

pub static HELMET_POSNORMAL: &[u8] = ASSETS[0].vertices;
pub static HELMET_INDICES_U32: &[u8] = ASSETS[0].indices;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MixedTopologyDraw {
    index_count: u32,
    first_index: u32,
    base_vertex: i32,
    rgba8_srgb: u32,
    topology: PrimitiveTopology,
}

fn mixed_topology_plan(
    helmet_index_count: u32,
    helmet_vertex_count: u32,
) -> [MixedTopologyDraw; 4] {
    let mut draws = [MixedTopologyDraw {
        index_count: helmet_index_count,
        first_index: 0,
        base_vertex: 0,
        rgba8_srgb: u32::from_le_bytes([210, 215, 225, 255]),
        topology: PrimitiveTopology::TriangleList,
    }; 4];
    for (slot, rgba) in [[255, 32, 32, 255], [32, 255, 32, 255], [32, 96, 255, 255]]
        .into_iter()
        .enumerate()
    {
        draws[slot + 1] = MixedTopologyDraw {
            index_count: 2,
            first_index: helmet_index_count + slot as u32 * 2,
            base_vertex: (helmet_vertex_count + slot as u32 * 2) as i32,
            rgba8_srgb: u32::from_le_bytes(rgba),
            topology: PrimitiveTopology::LineList,
        };
    }
    draws
}

/// Describe the host-prepared DamagedHelmet geometry without runtime parsing.
pub const fn prepared_geometry() -> ExecutablePrimitive {
    ExecutablePrimitive {
        topology: PrimitiveTopology::TriangleList,
        vertices: PreparedRange {
            resource: HELMET_VERTICES,
            offset: 0,
            byte_length: HELMET_VERTEX_BYTES,
            revision: 1,
        },
        indices: Some(PreparedRange {
            resource: HELMET_INDICES,
            offset: 0,
            byte_length: HELMET_INDEX_BYTES,
            revision: 1,
        }),
        index_format: Some(trueos_picasso::IndexFormat::Uint32),
        vertex_stride: 12,
        vertex_count: HELMET_VERTEX_COUNT,
        index_count: HELMET_INDEX_COUNT,
    }
}

/// GPU-side animation operation applied to one retained transform state.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct HeadInstanceProgram {
    pub initial: TransformValue,
    /// Signed Z angular velocity in radians/second. Negative is clockwise in
    /// the clip-space convention used by this example.
    pub angular_velocity_z: f32,
    /// Zero disables pulsing. A nonzero value is one descending or ascending
    /// half-cycle; the complete scale loop lasts twice this duration.
    pub scale_half_period_seconds: f32,
    pub reserved: [u32; 2],
}

// With the fixed upside-down presentation camera, this signed spacing makes
// the helmet stack depart the origin along the displayed green axis.
const HEAD_Y_SPACING: f32 = 1.6;
const WORLD_AXIS_LENGTH: f32 = 2.0;
const PRESENTATION_CAMERA_POSITION: [f32; 3] = [0.0, 0.0, -10.0];
const PRESENTATION_CAMERA_TARGET: [f32; 3] = [0.0; 3];
const PRESENTATION_CAMERA_WORLD_UP: [f32; 3] = [0.0, -1.0, 0.0];
const HEAD_WORLD_TRANSLATIONS: [[f32; 3]; HEAD_INSTANCE_COUNT as usize] = [
    [0.0, 0.0, 0.0],
    [0.0, -HEAD_Y_SPACING, 0.0],
    [0.0, -HEAD_Y_SPACING * 2.0, 0.0],
    [0.0, -HEAD_Y_SPACING * 3.0, 0.0],
];

/// The presentation camera is deliberately fixed: all retained transforms and
/// the static world-axis guide are authored against this same view.
fn presentation_camera() -> Camera {
    Camera {
        // Sit on the world-blue axis and keep the origin centred as the aim
        // point. The negative Z placement gives the blue guide a forward
        // direction through the scene.
        position: PRESENTATION_CAMERA_POSITION,
        // Preserve the aim point, but roll the complete view 180°.
        rotation: look_at_camera_rotation(
            PRESENTATION_CAMERA_POSITION,
            PRESENTATION_CAMERA_TARGET,
            PRESENTATION_CAMERA_WORLD_UP,
        ),
        projection: Projection::Perspective {
            yfov: core::f32::consts::FRAC_PI_3,
            znear: 1.0,
            zfar: Some(100.0),
            aspect_ratio: None,
        },
    }
}

/// Return the camera-to-world rotation whose local -Z axis targets `target`
/// and whose local +Y axis is the world-up closest valid basis direction.
fn look_at_camera_rotation(position: [f32; 3], target: [f32; 3], world_up: [f32; 3]) -> Quaternion {
    let forward = [
        target[0] - position[0],
        target[1] - position[1],
        target[2] - position[2],
    ];
    let forward_length =
        libm::sqrtf(forward[0] * forward[0] + forward[1] * forward[1] + forward[2] * forward[2]);
    let forward = [
        forward[0] / forward_length,
        forward[1] / forward_length,
        forward[2] / forward_length,
    ];
    // Local +X is camera right; local +Y completes the orthonormal frame.
    let right = [
        forward[1] * world_up[2] - forward[2] * world_up[1],
        forward[2] * world_up[0] - forward[0] * world_up[2],
        forward[0] * world_up[1] - forward[1] * world_up[0],
    ];
    let right_length = libm::sqrtf(right[0] * right[0] + right[1] * right[1] + right[2] * right[2]);
    let right = [
        right[0] / right_length,
        right[1] / right_length,
        right[2] / right_length,
    ];
    let up = [
        right[1] * forward[2] - right[2] * forward[1],
        right[2] * forward[0] - right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0],
    ];
    // Matrix columns map local +X, +Y, +Z to world right, up, and -forward.
    let (m00, m01, m02) = (right[0], up[0], -forward[0]);
    let (m10, m11, m12) = (right[1], up[1], -forward[1]);
    let (m20, m21, m22) = (right[2], up[2], -forward[2]);
    let trace = m00 + m11 + m22;
    let rotation = if trace > 0.0 {
        let s = libm::sqrtf(trace + 1.0) * 2.0;
        Quaternion([(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s])
    } else if m00 > m11 && m00 > m22 {
        let s = libm::sqrtf(1.0 + m00 - m11 - m22) * 2.0;
        Quaternion([0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s])
    } else if m11 > m22 {
        let s = libm::sqrtf(1.0 + m11 - m00 - m22) * 2.0;
        Quaternion([(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s])
    } else {
        let s = libm::sqrtf(1.0 + m22 - m00 - m11) * 2.0;
        Quaternion([(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s])
    };
    rotation.normalized()
}

const fn placed_head(y: f32) -> TransformValue {
    TransformValue {
        translation: [0.0, y, 0.0],
        translation_pad: 0.0,
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [0.45, 0.45, 0.45],
        scale_pad: 0.0,
    }
}

/// Four GPU-authored transform programs over one immutable helmet mesh.
///
/// The CPU publishes this compact program once. GPU time advances it forever:
/// clockwise, counter-clockwise, 0.5-second collapse/expand, and identity.
pub const HEAD_INSTANCE_PROGRAMS: [HeadInstanceProgram; HEAD_INSTANCE_COUNT as usize] = [
    HeadInstanceProgram {
        initial: placed_head(0.0),
        angular_velocity_z: -core::f32::consts::FRAC_PI_2,
        scale_half_period_seconds: 0.0,
        reserved: [0; 2],
    },
    HeadInstanceProgram {
        initial: placed_head(-HEAD_Y_SPACING),
        angular_velocity_z: core::f32::consts::FRAC_PI_2,
        scale_half_period_seconds: 0.0,
        reserved: [0; 2],
    },
    HeadInstanceProgram {
        initial: placed_head(-HEAD_Y_SPACING * 2.0),
        angular_velocity_z: 0.0,
        scale_half_period_seconds: 0.5,
        reserved: [0; 2],
    },
    HeadInstanceProgram {
        initial: placed_head(-HEAD_Y_SPACING * 3.0),
        angular_velocity_z: 0.0,
        scale_half_period_seconds: 0.0,
        reserved: [0; 2],
    },
];

/// Bind the mesh's four local references to one restored/mapped shared state
/// table. Neither this descriptor nor Picasso observes its GPU virtual address.
pub const fn head_transform_refs(
    state_resource: SharedResourceId,
    generation: u64,
) -> TransformRefList {
    TransformRefList {
        states: TransformStateRange {
            resource: state_resource,
            offset: 0,
            byte_length: HEAD_TRANSFORM_STATE_BYTES,
            state_count: HEAD_INSTANCE_COUNT,
            state_stride: core::mem::size_of::<TransformValue>() as u32,
            generation,
        },
        references: PreparedRange {
            resource: HELMET_TRANSFORM_REFS,
            offset: 0,
            byte_length: HELMET_TRANSFORM_REFS_U32.len() as u64,
            revision: 1,
        },
        reference_count: HEAD_INSTANCE_COUNT,
    }
}

/// One live, end-to-end DamagedHelmet submission kept resident for UI4 display.
///
/// Picasso contributes only logical resource ranges. This TRUEOS adapter maps
/// those ranges to opaque vGPU buffers; no GPU virtual address crosses it.
pub struct GeometryProbe {
    frame: Frame,
    device: Device,
    queue: Queue,
    _asset_vertex_buffers: [Option<Buffer>; ASSET_COUNT],
    _asset_index_buffers: [Option<Buffer>; ASSET_COUNT],
    line_vertex_buffer: Buffer,
    line_index_buffer: Buffer,
    base_color_textures: [Option<trueos::vmedia::RetainedTexture>; ASSET_COUNT],
    retained_meshes: [Option<RetainedMesh>; ASSET_COUNT],
    selected_asset: usize,
    number_keys: u8,
    flycam: FlyCam,
    resize_drag: Option<ResizeDrag>,
    timeline: u64,
    previous_elapsed_millis: u64,
}

#[derive(Clone, Copy)]
struct ResizeDrag {
    source: CursorSource,
    anchor_x: u32,
    anchor_y: u32,
    width: u32,
    height: u32,
}

impl GeometryProbe {
    fn open(catalog: &DatabasePreparedCatalog) -> Result<Self, GeometryProbeError> {
        const WIDTH: u32 = 640;
        const HEIGHT: u32 = 360;

        if catalog.assets.len() != ASSET_COUNT
            || catalog.line_vertices.len() != core::mem::size_of_val(&GRID_VERTICES)
            || catalog.line_indices.len() != core::mem::size_of_val(&GRID_INDICES)
        {
            return Err(GeometryProbeError::Contract);
        }

        let primitive = prepared_geometry();
        let indices = primitive.indices.ok_or(GeometryProbeError::Contract)?;
        if primitive.vertex_stride != 12
            || primitive.vertex_count == 0
            || primitive.index_count == 0
            || primitive.index_format != Some(trueos_picasso::IndexFormat::Uint32)
            || primitive.vertices.byte_length != HELMET_VERTEX_BYTES
            || indices.byte_length != HELMET_INDICES_U32.len() as u64
        {
            return Err(GeometryProbeError::Contract);
        }

        let frame = Frame::open_streaming(120, 96, WIDTH, HEIGHT)
            .map_err(|error| GeometryProbeError::Ui4("frame-open", error))?;
        let device = Device::open(Capabilities::DEFAULT.union(Capabilities::PRESENT))
            .map_err(|code| GeometryProbeError::Vgpu("device-open", code))?;
        let queue = device
            .create_queue(QueueClass::Render)
            .map_err(|code| GeometryProbeError::Vgpu("queue-create", code))?;
        let mut asset_vertex_buffers = [None; ASSET_COUNT];
        let mut asset_index_buffers = [None; ASSET_COUNT];
        let mut retained_meshes = [None; ASSET_COUNT];
        for slot in 0..ASSET_COUNT {
            let asset = &ASSETS[slot];
            let runtime_asset = &catalog.assets[slot];
            if runtime_asset.vertices.len() != asset.vertices.len()
                || runtime_asset.indices.len() != asset.indices.len()
                || runtime_asset.base_color.len() != asset.base_color_bytes.len()
            {
                return Err(GeometryProbeError::Contract);
            }
            let vertex_buffer = device
                .create_buffer(
                    runtime_asset.vertices.len(),
                    BUFFER_USAGE_MAP_WRITE | BUFFER_USAGE_VERTEX,
                )
                .map_err(|code| GeometryProbeError::Vgpu("asset-vertex-buffer-create", code))?;
            let index_buffer = device
                .create_buffer(
                    runtime_asset.indices.len(),
                    BUFFER_USAGE_MAP_WRITE | BUFFER_USAGE_INDEX,
                )
                .map_err(|code| GeometryProbeError::Vgpu("asset-index-buffer-create", code))?;
            write_exact(device, vertex_buffer, 0, &runtime_asset.vertices)
                .map_err(|code| GeometryProbeError::Vgpu("asset-vertex-upload", code))?;
            write_exact(device, index_buffer, 0, &runtime_asset.indices)
                .map_err(|code| GeometryProbeError::Vgpu("asset-index-upload", code))?;
            let retained_mesh = device
                .create_retained_mesh(
                    vertex_buffer,
                    index_buffer,
                    RetainedMeshDescriptor {
                        vertex_count: asset.vertex_count,
                        index_count: asset.index_count,
                        vertex_layout: if asset.sampled_material {
                            RETAINED_VERTEX_LAYOUT_POS_NORMAL_UV
                        } else {
                            RETAINED_VERTEX_LAYOUT_POS_NORMAL
                        },
                        ..RetainedMeshDescriptor::default()
                    },
                )
                .map_err(|code| GeometryProbeError::Vgpu("retained-mesh-create", code))?;
            asset_vertex_buffers[slot] = Some(vertex_buffer);
            asset_index_buffers[slot] = Some(index_buffer);
            retained_meshes[slot] = Some(retained_mesh);
        }
        let line_vertex_buffer = device
            .create_buffer(
                catalog.line_vertices.len(),
                BUFFER_USAGE_MAP_WRITE | BUFFER_USAGE_VERTEX,
            )
            .map_err(|code| GeometryProbeError::Vgpu("line-vertex-buffer-create", code))?;
        let line_index_buffer = device
            .create_buffer(
                catalog.line_indices.len(),
                BUFFER_USAGE_MAP_WRITE | BUFFER_USAGE_INDEX,
            )
            .map_err(|code| GeometryProbeError::Vgpu("line-index-buffer-create", code))?;

        // Static draws do not inherit the retained carrier's transform seed.
        // Preserve the Picasso DB's world-space source, then project its six
        // axis vertices once for this deliberately fixed presentation camera.
        let camera = presentation_camera();
        let projected_line_vertices =
            project_static_line_vertices(&catalog.line_vertices, camera, WIDTH, HEIGHT)
                .ok_or(GeometryProbeError::Contract)?;
        write_exact(device, line_vertex_buffer, 0, &projected_line_vertices)
            .map_err(|code| GeometryProbeError::Vgpu("line-vertex-upload", code))?;
        write_exact(device, line_index_buffer, 0, &catalog.line_indices)
            .map_err(|code| GeometryProbeError::Vgpu("line-index-upload", code))?;

        let mut probe = Self {
            frame,
            device,
            queue,
            _asset_vertex_buffers: asset_vertex_buffers,
            _asset_index_buffers: asset_index_buffers,
            line_vertex_buffer,
            line_index_buffer,
            base_color_textures: core::array::from_fn(|_| None),
            retained_meshes,
            selected_asset: 0,
            number_keys: 0,
            flycam: FlyCam::new(camera, 0.0),
            resize_drag: None,
            timeline: 0,
            previous_elapsed_millis: 0,
        };
        // Static axis vertices were projected above.  Do not let UI4 input
        // silently move the retained scene away from that shared camera.
        probe.flycam.set_look_sensitivity(0.0);
        for (slot, asset) in ASSETS.iter().enumerate() {
            let encoded = &catalog.assets[slot].base_color;
            if encoded.is_empty() {
                continue;
            }
            let texture = trueos::async_fs::block_on(trueos::vmedia::decode_retained_asset(
                probe.device,
                asset.base_color_name,
                encoded,
            ));
            match texture {
                Ok(texture) => {
                    let info = texture.info();
                    logl::log(
                        level::INFO,
                        format_args!(
                            "PicassoExample: texture proof accepted=1 asset={} role=base-color encoded_bytes={} texture_id=0x{:X} decoded={}x{} stride={} residency={:?} kernel_rgba_readback=0",
                            asset.name,
                            encoded.len(),
                            info.id.raw(),
                            info.width,
                            info.height,
                            info.stride_bytes,
                            info.residency,
                        ),
                    );
                    probe.base_color_textures[slot] = Some(texture);
                }
                Err(code) => {
                    logl::log(
                        level::ERROR,
                        format_args!(
                            "PicassoExample: texture proof accepted=0 asset={} role=base-color encoded_bytes={} error={} action=abort-textured-scene-contract",
                            asset.name,
                            encoded.len(),
                            code,
                        ),
                    );
                    return Err(GeometryProbeError::Vgpu("base-color-texture-decode", code));
                }
            }
        }
        probe.render_frame(0)?;
        Ok(probe)
    }

    /// Advance the compact transform state and render one complete frame.
    /// Geometry remains resident; only these seeds and the UI4 lease vary.
    pub fn render_frame(&mut self, elapsed_millis: u64) -> Result<(), GeometryProbeError> {
        let delta_seconds =
            elapsed_millis.saturating_sub(self.previous_elapsed_millis) as f32 * 0.001;
        self.previous_elapsed_millis = elapsed_millis;
        self.flycam
            .step_ui4(&self.frame, delta_seconds)
            .map_err(|error| GeometryProbeError::Ui4("flycam-ui4", error))?;
        self.service_asset_hotkeys()?;
        self.service_frame_interaction()?;
        let width = self.frame.width();
        let height = self.frame.height();

        self.frame
            .begin_gpu_frame()
            .map_err(|error| GeometryProbeError::Ui4("frame-begin", error))?;
        let surface = self
            .device
            .acquire_ui4_surface(self.frame.window_id())
            .map_err(|code| GeometryProbeError::Vgpu("surface-acquire", code))?;
        let point = self
            .device
            .submit_retained_frame(
                self.queue,
                surface,
                self.retained_meshes[self.selected_asset].ok_or(GeometryProbeError::Contract)?,
                self.line_vertex_buffer,
                self.line_index_buffer,
                RetainedFrameSubmit {
                    base_color_texture: if ASSETS[self.selected_asset].sampled_material {
                        self.base_color_textures[self.selected_asset]
                            .as_ref()
                            .map(|texture| texture.id().raw())
                            .unwrap_or(0)
                    } else {
                        0
                    },
                    clear_rgba8_srgb: u32::from_le_bytes([0, 128, 0, 0]),
                    seed_count: retained_seed_count(ASSETS[self.selected_asset].helmet_program),
                    static_draw_count: 3,
                    seeds: retained_seeds(
                        elapsed_millis,
                        self.flycam.camera,
                        ASSETS[self.selected_asset].helmet_program,
                        width,
                        height,
                    ),
                    static_draws: retained_line_draws(),
                    ..RetainedFrameSubmit::default()
                },
            )
            .map_err(|code| GeometryProbeError::Vgpu("retained-frame-submit", code))?;
        self.device
            .wait(self.queue, point.value)
            .map_err(|code| GeometryProbeError::Vgpu("timeline-wait", code))?;
        self.frame
            .publish(Damage::full(width, height))
            .map_err(|error| GeometryProbeError::Ui4("frame-publish", error))?;
        self.timeline = point.value;
        Ok(())
    }

    fn service_asset_hotkeys(&mut self) -> Result<(), GeometryProbeError> {
        let state = self
            .frame
            .keyboard_state()
            .map_err(|error| GeometryProbeError::Ui4("asset-hotkeys", error))?;
        let current = state.map_or(0, |keyboard| {
            let mut bits = 0u8;
            for slot in 0..ASSET_COUNT {
                if keyboard.is_down(0x1e + slot as u8) {
                    bits |= 1 << slot;
                }
            }
            bits
        });
        let pressed = current & !self.number_keys;
        self.number_keys = current;
        if pressed != 0 {
            let slot = pressed.trailing_zeros() as usize;
            if slot < ASSET_COUNT {
                self.selected_asset = slot;
                logl::log(
                    level::INFO,
                    format_args!(
                        "PicassoExample: asset hotkey={} selected={} vertices={} indices={} instances={}",
                        slot + 1,
                        ASSETS[slot].name,
                        ASSETS[slot].vertex_count,
                        ASSETS[slot].index_count,
                        if ASSETS[slot].helmet_program { 4 } else { 1 }
                    ),
                );
            }
        }
        Ok(())
    }

    /// Consume UI4's one-shot maximize/restore extent and the selected
    /// frame's cursor-sized bottom-right resize grip. The grip has no pixels;
    /// it is only a hit region fully contained in the application frame.
    fn service_frame_interaction(&mut self) -> Result<(), GeometryProbeError> {
        const RESIZE_GRIP_PX: i32 = 16;
        const MIN_WIDTH: u32 = 160;
        const MIN_HEIGHT: u32 = 90;

        let mut requested_extent = None;
        while let Some(event) = self
            .frame
            .take_resize_event()
            .map_err(|error| GeometryProbeError::Ui4("resize-event", error))?
        {
            requested_extent = Some((event.width, event.height));
        }
        if let Some((width, height)) = requested_extent
            && (width != self.frame.width() || height != self.frame.height())
        {
            self.frame
                .resize(width, height)
                .map_err(|error| GeometryProbeError::Ui4("maximize-resize", error))?;
        }

        let (output_width, output_height) =
            output_dimensions().map_err(|error| GeometryProbeError::Ui4("output-extent", error))?;
        let mut drag = self.resize_drag;
        let mut live_extent = None;
        while let Some(event) = self
            .frame
            .take_pointer_event()
            .map_err(|error| GeometryProbeError::Ui4("pointer-event", error))?
        {
            let in_grip = event.local_x >= self.frame.width() as i32 - RESIZE_GRIP_PX
                && event.local_y >= self.frame.height() as i32 - RESIZE_GRIP_PX
                && event.local_x < self.frame.width() as i32
                && event.local_y < self.frame.height() as i32;
            let owns_drag = drag.is_some_and(|active| active.source == event.source);
            // The app owns bottom-right resizing. Everything else remains
            // available to Picasso through the same UI4-routed event; nothing
            // is drained by the library behind the application's back.
            self.flycam
                .handle_ui4_pointer_event(&event, !in_grip && !owns_drag);
            self.frame
                .set_cursor_icon_for(
                    event.source,
                    if in_grip || owns_drag {
                        CursorIcon::ResizeDiagonal
                    } else {
                        CursorIcon::Default
                    },
                )
                .map_err(|error| GeometryProbeError::Ui4("resize-cursor", error))?;

            if event.buttons_pressed & POINTER_BUTTON_PRIMARY != 0 && in_grip {
                drag = Some(ResizeDrag {
                    source: event.source,
                    anchor_x: event.x,
                    anchor_y: event.y,
                    width: self.frame.width(),
                    height: self.frame.height(),
                });
            }
            if let Some(active) = drag
                && active.source == event.source
            {
                let width = (i64::from(active.width) + i64::from(event.x)
                    - i64::from(active.anchor_x))
                .clamp(i64::from(MIN_WIDTH), i64::from(output_width))
                    as u32;
                let height = (i64::from(active.height) + i64::from(event.y)
                    - i64::from(active.anchor_y))
                .clamp(i64::from(MIN_HEIGHT), i64::from(output_height))
                    as u32;
                live_extent = Some((width, height));
                if event.buttons_released & POINTER_BUTTON_PRIMARY != 0
                    || event.buttons_down & POINTER_BUTTON_PRIMARY == 0
                {
                    drag = None;
                }
            }
        }
        self.resize_drag = drag;
        if let Some((width, height)) = live_extent
            && (width != self.frame.width() || height != self.frame.height())
        {
            self.frame
                .resize(width, height)
                .map_err(|error| GeometryProbeError::Ui4("live-resize", error))?;
        }
        Ok(())
    }

    pub const fn timeline(&self) -> u64 {
        self.timeline
    }

    pub fn take_first_presentation(&mut self) -> Result<bool, GeometryProbeError> {
        self.frame
            .take_first_presentation()
            .map_err(|error| GeometryProbeError::Ui4("first-presentation", error))
    }
}

fn retained_line_draws() -> [IndexedBatchDrawV2; trueos::vgpu::MAX_RETAINED_STATIC_DRAWS] {
    core::array::from_fn(|slot| IndexedBatchDrawV2 {
        index_count: 2,
        first_index: slot as u32 * 2,
        base_vertex: slot as i32 * 2,
        rgba8_srgb: u32::from_le_bytes(
            [[255, 32, 32, 255], [32, 255, 32, 255], [32, 96, 255, 255]][slot],
        ),
        topology: trueos::vgpu::PRIMITIVE_TOPOLOGY_LINE_LIST,
        reserved: 0,
    })
}

fn retained_seeds(
    elapsed_millis: u64,
    camera: Camera,
    helmet_program: bool,
    viewport_width: u32,
    viewport_height: u32,
) -> [RetainedTransformSeed; trueos::vgpu::MAX_RETAINED_TRANSFORM_SEEDS] {
    let seconds = elapsed_millis as f32 * 0.001;
    let half_angle = core::f32::consts::FRAC_PI_4 * seconds;
    let clockwise = [-0.0, 0.0, -libm::sinf(half_angle), libm::cosf(half_angle)];
    let counter_clockwise = [0.0, 0.0, libm::sinf(half_angle), libm::cosf(half_angle)];
    let half_cycle = (elapsed_millis % 1_000) as f32;
    let pulse = libm::fabsf(half_cycle - 500.0) / 500.0;
    let [qx, qy, qz, qw] = camera.rotation.normalized().0;
    let view_rotation = Quaternion([-qx, -qy, -qz, qw]);
    let mut seeds = [RetainedTransformSeed::default(); trueos::vgpu::MAX_RETAINED_TRANSFORM_SEEDS];
    for (slot, seed) in seeds
        .iter_mut()
        .take(retained_seed_count(helmet_program) as usize)
        .enumerate()
    {
        let world_translation = if helmet_program {
            HEAD_WORLD_TRANSLATIONS[slot]
        } else {
            [0.0; 3]
        };
        let view_translation = view_rotation.rotate([
            world_translation[0] - camera.position[0],
            world_translation[1] - camera.position[1],
            world_translation[2] - camera.position[2],
        ]);
        let world_rotation = match slot {
            _ if !helmet_program => [0.0, 0.0, 0.0, 1.0],
            0 => clockwise,
            1 => counter_clockwise,
            _ => [0.0, 0.0, 0.0, 1.0],
        };
        let world_scale = if !helmet_program {
            1.0
        } else if slot == 2 {
            0.45 * pulse
        } else {
            0.45
        };
        let (projected_translation, projected_scale) = project_camera_transform(
            camera.projection,
            viewport_width,
            viewport_height,
            view_translation,
            world_scale,
        );
        *seed = RetainedTransformSeed {
            translation: projected_translation,
            scale: projected_scale,
            rotation: (view_rotation * Quaternion(world_rotation)).normalized().0,
            local_radius: 1.0,
            previous_translation: view_translation,
            draw_group: 0,
            flags: (slot as u32) << 16,
        };
    }
    seeds
}

/// Picasso's retained carrier currently owns an identity camera matrix. Fold
/// the runtime camera into each compact model seed so geometry remains GPU
/// transformed without a second vertex upload. All current demo objects share
/// one Z plane, making this center-depth perspective representation exact for
/// placement and stable for the small normalized meshes.
fn project_camera_transform(
    projection: Projection,
    viewport_width: u32,
    viewport_height: u32,
    view_translation: [f32; 3],
    world_scale: f32,
) -> ([f32; 3], [f32; 3]) {
    match projection {
        Projection::Perspective {
            yfov,
            znear,
            zfar,
            aspect_ratio,
        } => {
            let aspect = aspect_ratio
                .unwrap_or_else(|| viewport_width as f32 / viewport_height.max(1) as f32);
            let depth = (-view_translation[2]).max(znear);
            let focal_y = 1.0 / libm::tanf(yfov * 0.5);
            let focal_x = focal_y / aspect;
            let (ndc_z, depth_scale) = match zfar {
                Some(far) => {
                    let range = (far - znear).max(f32::EPSILON);
                    (
                        far / range - far * znear / (range * depth),
                        far * znear / (range * depth * depth),
                    )
                }
                None => (1.0 - znear / depth, znear / (depth * depth)),
            };
            (
                [
                    view_translation[0] * focal_x / depth,
                    // The retained presentation surface uses a top-left
                    // screen-space Y convention after this folded projection.
                    // Negate camera-up here so world +Y rises on screen.
                    -view_translation[1] * focal_y / depth,
                    ndc_z,
                ],
                [
                    world_scale * focal_x / depth,
                    world_scale * focal_y / depth,
                    world_scale * depth_scale,
                ],
            )
        }
        Projection::Orthographic {
            xmag,
            ymag,
            znear,
            zfar,
        } => {
            let depth = -view_translation[2];
            let range = (zfar - znear).max(f32::EPSILON);
            (
                [
                    view_translation[0] / xmag,
                    view_translation[1] / ymag,
                    (depth - znear) / range,
                ],
                [world_scale / xmag, world_scale / ymag, world_scale / range],
            )
        }
    }
}

/// Convert Picasso's database-owned float3 world vertices into the clip-space
/// vertex ABI consumed by retained static draws.  Unlike the retained helmet,
/// that static path has no transform seed of its own.
fn project_static_line_vertices(
    world_vertex_bytes: &[u8],
    camera: Camera,
    viewport_width: u32,
    viewport_height: u32,
) -> Option<Vec<u8>> {
    if world_vertex_bytes.len() != core::mem::size_of_val(&GRID_VERTICES)
        || !world_vertex_bytes.len().is_multiple_of(12)
    {
        return None;
    }
    let [qx, qy, qz, qw] = camera.rotation.normalized().0;
    let view_rotation = Quaternion([-qx, -qy, -qz, qw]);
    let mut projected = Vec::with_capacity(world_vertex_bytes.len());
    for bytes in world_vertex_bytes.chunks_exact(12) {
        let world = [
            f32::from_le_bytes(bytes[0..4].try_into().ok()?),
            f32::from_le_bytes(bytes[4..8].try_into().ok()?),
            f32::from_le_bytes(bytes[8..12].try_into().ok()?),
        ];
        if world.iter().any(|component| !component.is_finite()) {
            return None;
        }
        let view = view_rotation.rotate([
            world[0] * WORLD_AXIS_LENGTH - camera.position[0],
            world[1] * WORLD_AXIS_LENGTH - camera.position[1],
            world[2] * WORLD_AXIS_LENGTH - camera.position[2],
        ]);
        let (clip, _) = project_camera_transform(
            camera.projection,
            viewport_width,
            viewport_height,
            view,
            1.0,
        );
        if clip.iter().any(|component| !component.is_finite()) {
            return None;
        }
        for component in clip {
            projected.extend_from_slice(&component.to_le_bytes());
        }
    }
    Some(projected)
}

const fn retained_seed_count(helmet_program: bool) -> u32 {
    if helmet_program {
        HEAD_INSTANCE_COUNT
    } else {
        1
    }
}

fn line_vertex_bytes() -> &'static [u8] {
    unsafe {
        core::slice::from_raw_parts(
            GRID_VERTICES.as_ptr().cast::<u8>(),
            core::mem::size_of_val(&GRID_VERTICES),
        )
    }
}

fn line_index_bytes() -> &'static [u8] {
    unsafe {
        core::slice::from_raw_parts(
            GRID_INDICES.as_ptr().cast::<u8>(),
            core::mem::size_of_val(&GRID_INDICES),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryProbeError {
    Contract,
    Ui4(&'static str, Ui4Error),
    Vgpu(&'static str, i32),
}

fn write_exact(device: Device, buffer: Buffer, offset: u64, bytes: &[u8]) -> Result<(), i32> {
    let offset = usize::try_from(offset).map_err(|_| trueos::vgpu::ERR_UNSUPPORTED)?;
    let written = device.write_buffer(buffer, offset, bytes)?;
    (written == bytes.len())
        .then_some(())
        .ok_or(trueos::vgpu::ERR_IO)
}

/// TRUEOS-specific materialization adapter. Field order is intentional: the
/// ring is dropped before `memory`, so `VVideoMem` backs every live ring view.
pub struct VVideoRing {
    ring: ExecRing,
    memory: VVideoMem,
}

impl VVideoRing {
    /// Allocate and initialize a new page-pinned shared region.
    pub fn allocate_fresh(
        device: Device,
        resource: SharedResourceId,
        slot_count: usize,
        slot_stride: usize,
        usage: u32,
    ) -> Result<Self, VVideoRingError> {
        let total_bytes = slot_count
            .checked_mul(slot_stride)
            .ok_or(VVideoRingError::Cubism(CubismError::SizeOverflow))?;
        let memory = device
            .allocate_vvideo_mem(total_bytes, usage)
            .map_err(VVideoRingError::Vgpu)?;
        let ring = unsafe {
            ExecRing::from_raw_parts(
                memory.as_ptr() as *mut u8,
                memory.len(),
                resource,
                slot_count,
                slot_stride,
            )
        }
        .map_err(VVideoRingError::Cubism)?;
        unsafe { ring.initialize_fresh() };
        Ok(Self { ring, memory })
    }

    pub const fn ring(&self) -> &ExecRing {
        &self.ring
    }

    /// Opaque kernel buffer identity for submissions outside Picasso.
    pub const fn buffer(&self) -> Buffer {
        self.memory.buffer()
    }

    /// Form a bounds-checked buffer-relative range for an executor.
    pub fn slice(&self, offset: usize, bytes: usize) -> Result<BufferSlice, i32> {
        self.memory.slice(offset, bytes)
    }

    pub fn visibility(&self) -> VVideoVisibility<'_> {
        VVideoVisibility {
            memory: &self.memory,
            resource: self.ring.resource(),
        }
    }
}

/// Cache adapter for a `VVideoRing`. It uses `VVideoMem` offset operations;
/// the GPU address and underlying PPGTT mapping remain inside TRUEOS. Cubism
/// flushes slot data first, then the published header control line. It
/// invalidates just the GPU-owned payload after retirement has read the
/// CPU-owned header metadata.
pub struct VVideoVisibility<'a> {
    memory: &'a VVideoMem,
    resource: SharedResourceId,
}

impl VisibilityOps for VVideoVisibility<'_> {
    fn cpu_make_gpu_visible(&self, range: SharedByteRange) -> Result<(), CubismError> {
        self.maintain(range, true)
    }

    fn cpu_publish_slot(&self, range: SharedByteRange) -> Result<(), CubismError> {
        self.maintain(range, true)
    }

    fn gpu_make_cpu_visible(&self, range: SharedByteRange) -> Result<(), CubismError> {
        self.maintain(range, false)
    }
}

impl VVideoVisibility<'_> {
    fn maintain(&self, range: SharedByteRange, flush: bool) -> Result<(), CubismError> {
        if range.resource != self.resource {
            return Err(CubismError::VisibilityFailed);
        }
        let offset = usize::try_from(range.offset).map_err(|_| CubismError::VisibilityFailed)?;
        let bytes =
            usize::try_from(range.byte_length).map_err(|_| CubismError::VisibilityFailed)?;
        let result = if flush {
            self.memory.flush(offset, bytes)
        } else {
            self.memory.invalidate(offset, bytes)
        };
        result.map_err(|_| CubismError::VisibilityFailed)
    }
}

fn prepared_asset_key(asset: &str, object: &str) -> String {
    format!("{asset}/prepared/{object}")
}

/// Publish the build-prepared representation into the same Picasso-owned
/// runtime database as the exact source files. This is the packaging seam:
/// build-time glTF parsing remains off the Blueprint, while every byte used by
/// the renderer must cross Picasso's database read boundary first.
fn put_prepared_assets(picasso: &Picasso) -> Result<(), trueos_picasso::PicassoError> {
    for asset in &ASSETS {
        picasso.put_embedded_asset(
            &prepared_asset_key(asset.name, "mesh/vertices"),
            asset.vertices,
        )?;
        picasso.put_embedded_asset(
            &prepared_asset_key(asset.name, "mesh/indices"),
            asset.indices,
        )?;
        picasso.put_embedded_asset(
            &prepared_asset_key(asset.name, "material/base-color"),
            asset.base_color_bytes,
        )?;
    }
    picasso.put_embedded_asset("PicassoGrid/prepared/mesh/vertices", line_vertex_bytes())?;
    picasso.put_embedded_asset("PicassoGrid/prepared/mesh/indices", line_index_bytes())?;
    Ok(())
}

fn load_prepared_assets(
    picasso: &Picasso,
) -> Result<Option<DatabasePreparedCatalog>, trueos_picasso::PicassoError> {
    let mut assets = Vec::with_capacity(ASSET_COUNT);
    for asset in &ASSETS {
        let Some(vertices) =
            picasso.embedded_asset(&prepared_asset_key(asset.name, "mesh/vertices"))?
        else {
            return Ok(None);
        };
        let Some(indices) =
            picasso.embedded_asset(&prepared_asset_key(asset.name, "mesh/indices"))?
        else {
            return Ok(None);
        };
        let Some(base_color) =
            picasso.embedded_asset(&prepared_asset_key(asset.name, "material/base-color"))?
        else {
            return Ok(None);
        };
        assets.push(DatabasePreparedAsset {
            vertices,
            indices,
            base_color,
        });
    }
    let Some(line_vertices) = picasso.embedded_asset("PicassoGrid/prepared/mesh/vertices")? else {
        return Ok(None);
    };
    let Some(line_indices) = picasso.embedded_asset("PicassoGrid/prepared/mesh/indices")? else {
        return Ok(None);
    };
    Ok(Some(DatabasePreparedCatalog {
        assets,
        line_vertices,
        line_indices,
    }))
}

fn main() {
    run();
    if !trueos::vshell::shutdown_current_blueprint(
        "PicassoExample terminated after a fatal scene error",
    ) {
        logl::log(
            level::ERROR,
            "PicassoExample: fatal scene return could not request Blueprint shutdown",
        );
    }
}

fn run() {
    // The example owns only the immutable source bytes. Picasso owns their
    // runtime representation behind this public asset-ingestion boundary.
    let picasso = match Picasso::new() {
        Ok(picasso) => picasso,
        Err(error) => {
            logl::log(
                level::ERROR,
                format_args!("PicassoExample: Picasso creation failed: {error:?}"),
            );
            return;
        }
    };
    let mut source_bytes = 0usize;
    for asset in &demodata::ASSETS {
        if let Err(error) = picasso.put_embedded_asset(asset.name, asset.bytes) {
            logl::log(
                level::ERROR,
                format_args!(
                    "PicassoExample: embedded asset rejected asset={} bytes={} error={error:?}",
                    asset.name,
                    asset.bytes.len(),
                ),
            );
            return;
        }
        match picasso.embedded_asset(asset.name) {
            Ok(Some(stored)) if stored == asset.bytes => {}
            Ok(_) => {
                logl::log(
                    level::ERROR,
                    format_args!(
                        "PicassoExample: embedded source round-trip mismatch asset={}",
                        asset.name
                    ),
                );
                return;
            }
            Err(error) => {
                logl::log(
                    level::ERROR,
                    format_args!(
                        "PicassoExample: embedded source read failed asset={} error={error:?}",
                        asset.name
                    ),
                );
                return;
            }
        }
        source_bytes += asset.bytes.len();
    }
    if let Err(error) = put_prepared_assets(&picasso) {
        logl::log(
            level::ERROR,
            format_args!("PicassoExample: prepared asset rejected error={error:?}"),
        );
        return;
    }
    let runtime_catalog = match load_prepared_assets(&picasso) {
        Ok(Some(assets)) => assets,
        Ok(None) => {
            logl::log(
                level::ERROR,
                "PicassoExample: prepared asset missing after Picasso database publication",
            );
            return;
        }
        Err(error) => {
            logl::log(
                level::ERROR,
                format_args!("PicassoExample: prepared asset read failed error={error:?}"),
            );
            return;
        }
    };
    logl::log(
        level::INFO,
        format_args!(
            "PicassoExample: database-backed assets ready assets={} exact_source_bytes={}",
            demodata::ASSETS.len(),
            source_bytes,
        ),
    );

    let mut probe = match GeometryProbe::open(&runtime_catalog) {
        Ok(probe) => probe,
        Err(error) => {
            logl::log(
                level::ERROR,
                format_args!("PicassoExample: geometry probe failed: {error:?}"),
            );
            return;
        }
    };

    logl::log(
        level::INFO,
        format_args!(
            "PicassoExample: prepared textured DamagedHelmet instances + static RGB lines submitted and retired: vertices={} indices={} texture_bound={} timeline={}",
            HELMET_VERTEX_COUNT,
            HELMET_INDEX_COUNT,
            probe.base_color_textures[0].is_some() as u8,
            probe.timeline(),
        ),
    );

    let start = clock::monotonic_millis();
    let mut presentation_logged = false;
    loop {
        vsys::poll_once();
        if let Err(error) = probe.render_frame(clock::monotonic_millis().saturating_sub(start)) {
            logl::log(
                level::ERROR,
                format_args!("PicassoExample: retained animation failed: {error:?}"),
            );
            return;
        }
        if !presentation_logged {
            match probe.take_first_presentation() {
                Ok(true) => {
                    logl::log(
                        level::INFO,
                        "PicassoExample: retained transform + static line frame crossed UI4 SURFLIVE",
                    );
                    presentation_logged = true;
                }
                Ok(false) => {}
                Err(error) => {
                    logl::log(
                        level::ERROR,
                        format_args!("PicassoExample: presentation probe failed: {error:?}"),
                    );
                    return;
                }
            }
        }
        vsys::sleep_ms(16);
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use std::alloc::{Layout, alloc_zeroed, dealloc};

    struct Region {
        ptr: *mut u8,
        layout: Layout,
    }
    impl Region {
        fn new(bytes: usize) -> Self {
            let layout = Layout::from_size_align(bytes, 64).unwrap();
            let ptr = unsafe { alloc_zeroed(layout) };
            assert!(!ptr.is_null());
            Self { ptr, layout }
        }
    }
    impl Drop for Region {
        fn drop(&mut self) {
            unsafe { dealloc(self.ptr, self.layout) }
        }
    }

    #[test]
    fn no_std_consumer_uses_resource_relative_ranges() {
        const SLOT_COUNT: usize = 1;
        const SLOT_STRIDE: usize = 128;
        let region = Region::new(SLOT_COUNT * SLOT_STRIDE);
        let ring = unsafe {
            ExecRing::from_raw_parts(
                region.ptr,
                SLOT_COUNT * SLOT_STRIDE,
                SharedResourceId(9),
                SLOT_COUNT,
                SLOT_STRIDE,
            )
            .unwrap()
        };
        unsafe { ring.initialize_fresh() };
        let visibility = trueos_picasso::CoherentVisibility;
        let mut write = ring.try_acquire().unwrap();
        let index = write.slot_index();
        let generation = write.generation();
        write.payload_mut()[..4].copy_from_slice(b"draw");
        let sealed = write.publish(4, 1, &visibility).unwrap();
        assert_eq!(sealed.payload_range().offset, 64);
        sealed.mark_in_flight(17).unwrap();
        ring.retire(index, generation, 17, &visibility).unwrap();
        let primitive = prepared_geometry();
        assert!(primitive.vertex_count > 3);
        assert!(primitive.index_count > 3);
        assert_eq!(
            primitive.index_format,
            Some(trueos_picasso::IndexFormat::Uint32)
        );
        assert_eq!(
            primitive.vertices.byte_length,
            HELMET_POSITIONS.len() as u64
        );
        assert_eq!(
            primitive.indices.unwrap().byte_length,
            HELMET_INDICES_U32.len() as u64
        );
    }

    #[test]
    fn four_heads_share_geometry_and_reference_retained_gpu_state() {
        let primitive = prepared_geometry();
        let refs = head_transform_refs(SharedResourceId(21), 7);
        assert_eq!(refs.reference_count, 4);
        assert_eq!(refs.states.state_count, 4);
        assert_eq!(refs.states.generation, 7);
        assert_eq!(primitive.vertices.resource, HELMET_VERTICES);
        assert_eq!(primitive.indices.unwrap().resource, HELMET_INDICES);
        assert_eq!(HELMET_TRANSFORM_REFS_U32.len(), 16);
        assert!(
            HEAD_INSTANCE_PROGRAMS
                .iter()
                .all(|program| program.initial.is_valid())
        );
        assert!(HEAD_INSTANCE_PROGRAMS[0].angular_velocity_z < 0.0);
        assert!(HEAD_INSTANCE_PROGRAMS[1].angular_velocity_z > 0.0);
        assert_eq!(HEAD_INSTANCE_PROGRAMS[2].scale_half_period_seconds, 0.5);
        assert_eq!(HEAD_INSTANCE_PROGRAMS[3].angular_velocity_z, 0.0);
        assert_eq!(HEAD_INSTANCE_PROGRAMS[3].scale_half_period_seconds, 0.0);
    }

    #[test]
    fn rgb_lines_are_independent_native_line_draws() {
        let draws = mixed_topology_plan(HELMET_INDEX_COUNT, HELMET_VERTEX_COUNT);
        assert_eq!(draws[0].topology, PrimitiveTopology::TriangleList);
        for (slot, expected_rgba) in [[255, 32, 32, 255], [32, 255, 32, 255], [32, 96, 255, 255]]
            .into_iter()
            .enumerate()
        {
            let draw = draws[slot + 1];
            assert_eq!(draw.topology, PrimitiveTopology::LineList);
            assert_eq!(draw.index_count, 2);
            assert_eq!(draw.first_index, HELMET_INDEX_COUNT + slot as u32 * 2);
            assert_eq!(
                draw.base_vertex,
                (HELMET_VERTEX_COUNT + slot as u32 * 2) as i32
            );
            assert_eq!(draw.rgba8_srgb, u32::from_le_bytes(expected_rgba));
        }
        assert_eq!(GRID_INDICES, [0, 1, 0, 1, 0, 1]);
        let draws = retained_line_draws();
        for (slot, expected_rgba) in [[255, 32, 32, 255], [32, 255, 32, 255], [32, 96, 255, 255]]
            .into_iter()
            .enumerate()
        {
            assert_eq!(draws[slot].first_index, slot as u32 * 2);
            assert_eq!(draws[slot].base_vertex, slot as i32 * 2);
            assert_eq!(draws[slot].index_count, 2);
            assert_eq!(draws[slot].rgba8_srgb, u32::from_le_bytes(expected_rgba));
        }
    }

    #[test]
    fn presentation_camera_stays_on_blue_axis_and_looks_at_origin() {
        let camera = presentation_camera();
        assert_eq!(camera.position, PRESENTATION_CAMERA_POSITION);
        assert_eq!(camera.position[0], 0.0);
        assert_eq!(camera.position[1], 0.0);
        let forward = camera.rotation.rotate([0.0, 0.0, -1.0]);
        let expected_forward = [0.0, 0.0, 1.0];
        for axis in 0..3 {
            assert!((forward[axis] - expected_forward[axis]).abs() < 1.0e-5);
        }
        let distance_to_origin = [
            PRESENTATION_CAMERA_TARGET[0] - camera.position[0],
            PRESENTATION_CAMERA_TARGET[1] - camera.position[1],
            PRESENTATION_CAMERA_TARGET[2] - camera.position[2],
        ];
        assert!(distance_to_origin[0] == 0.0 && distance_to_origin[1] == 0.0);
        assert!(distance_to_origin[2] > 0.0);
        assert!(camera.rotation.rotate([0.0, 1.0, 0.0])[1] < 0.0);
        assert_eq!(
            camera.projection,
            Projection::Perspective {
                yfov: core::f32::consts::FRAC_PI_3,
                znear: 1.0,
                zfar: Some(100.0),
                aspect_ratio: None,
            }
        );

        let seeds = retained_seeds(0, camera, true, 640, 360);
        for slot in 1..HEAD_INSTANCE_COUNT as usize {
            assert!(seeds[slot].translation[1] < seeds[slot - 1].translation[1]);
        }
        assert!(
            seeds[HEAD_INSTANCE_COUNT as usize - 1].translation[1]
                + seeds[HEAD_INSTANCE_COUNT as usize - 1].scale[1]
                < 1.0
        );

        let projected = project_static_line_vertices(line_vertex_bytes(), camera, 640, 360)
            .expect("the six finite Picasso grid vertices project");
        let point = |index: usize| -> [f32; 3] {
            core::array::from_fn(|axis| {
                let offset = index * 12 + axis * 4;
                f32::from_le_bytes(projected[offset..offset + 4].try_into().unwrap())
            })
        };
        let origin = point(0);
        let x_end = point(1);
        let y_origin = point(2);
        let y_end = point(3);
        let z_origin = point(4);
        let z_end = point(5);
        assert_eq!(origin, y_origin);
        assert_eq!(origin, z_origin);
        // The upside-down camera makes world +Y project downward on the
        // top-left-origin retained surface. Looking along +Z makes the blue
        // axis depth-only at its origin, while X remains visibly horizontal.
        assert_ne!(x_end, origin);
        assert!(y_end[1] > origin[1]);
        assert_eq!(z_end, origin);
    }
}
