//! Consumer-first Picasso example for a `#![no_std]` TRUEOS application.
//!
//! At boot the platform resolves Dealer IDs to opaque vGPU buffers. Picasso
//! receives only stable resource identities and byte-relative ranges.
#![no_std]

#[cfg(target_os = "trueos")]
use trueos::ui4_scene::{
    CursorIcon, CursorSource, Damage, Frame, POINTER_BUTTON_PRIMARY, output_dimensions,
};
#[cfg(target_os = "trueos")]
use trueos::vgpu::{
    BUFFER_USAGE_INDEX, BUFFER_USAGE_MAP_WRITE, BUFFER_USAGE_VERTEX, Buffer, BufferSlice,
    Capabilities, Device, IndexedBatchDrawV2, Queue, QueueClass, RetainedFrameSubmit, RetainedMesh,
    RetainedMeshDescriptor, RetainedTransformSeed, VVideoMem,
};
#[cfg(any(test, target_os = "trueos"))]
use trueos_picasso::ExecRing;
#[cfg(any(test, target_os = "trueos"))]
use trueos_picasso::GRID_INDICES;
#[cfg(target_os = "trueos")]
use trueos_picasso::GRID_VERTICES;
#[cfg(target_os = "trueos")]
use trueos_picasso::cam::{Camera, FlyCam, Projection, Quaternion};
#[cfg(target_os = "trueos")]
use trueos_picasso::{CubismError, SharedByteRange, VisibilityOps};
use trueos_picasso::{
    ExecutablePrimitive, PreparedRange, PrimitiveTopology, ResourceId, SharedResourceId,
    TransformRefList, TransformStateRange, TransformValue,
};

include!(concat!(env!("OUT_DIR"), "/damaged_helmet.meta.rs"));

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

pub static HELMET_POSITIONS: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/damaged_helmet.positions.f32le"));
/// Authenticated Helio native-matrix input: Float3 position followed by Float3
/// normal. This remains one immutable geometry copy; the GPU transform pass
/// supplies all four instance matrices separately.
pub static HELMET_POSNORMAL: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/damaged_helmet.posnormal.f32le"));
pub static HELMET_INDICES_U32: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/damaged_helmet.indices.u32le"));

#[cfg(any(test, target_os = "trueos"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MixedTopologyDraw {
    index_count: u32,
    first_index: u32,
    base_vertex: i32,
    rgba8_srgb: u32,
    topology: PrimitiveTopology,
}

#[cfg(any(test, target_os = "trueos"))]
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

const fn placed_head(x: f32, y: f32) -> TransformValue {
    TransformValue {
        translation: [x, y, 0.0],
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
        initial: placed_head(-0.5, 0.5),
        angular_velocity_z: -core::f32::consts::FRAC_PI_2,
        scale_half_period_seconds: 0.0,
        reserved: [0; 2],
    },
    HeadInstanceProgram {
        initial: placed_head(0.5, 0.5),
        angular_velocity_z: core::f32::consts::FRAC_PI_2,
        scale_half_period_seconds: 0.0,
        reserved: [0; 2],
    },
    HeadInstanceProgram {
        initial: placed_head(-0.5, -0.5),
        angular_velocity_z: 0.0,
        scale_half_period_seconds: 0.5,
        reserved: [0; 2],
    },
    HeadInstanceProgram {
        initial: placed_head(0.5, -0.5),
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
#[cfg(target_os = "trueos")]
pub struct GeometryProbe {
    frame: Frame,
    device: Device,
    queue: Queue,
    _helmet_vertex_buffer: Buffer,
    _helmet_index_buffer: Buffer,
    line_vertex_buffer: Buffer,
    line_index_buffer: Buffer,
    retained_mesh: RetainedMesh,
    flycam: FlyCam,
    resize_drag: Option<ResizeDrag>,
    timeline: u64,
    previous_elapsed_millis: u64,
}

#[cfg(target_os = "trueos")]
#[derive(Clone, Copy)]
struct ResizeDrag {
    source: CursorSource,
    anchor_x: u32,
    anchor_y: u32,
    width: u32,
    height: u32,
}

#[cfg(target_os = "trueos")]
impl GeometryProbe {
    pub fn open() -> Result<Self, GeometryProbeError> {
        const WIDTH: u32 = 640;
        const HEIGHT: u32 = 360;

        let primitive = prepared_geometry();
        let indices = primitive.indices.ok_or(GeometryProbeError::Contract)?;
        if primitive.vertex_stride != 12
            || primitive.vertex_count == 0
            || primitive.index_count == 0
            || primitive.index_format != Some(trueos_picasso::IndexFormat::Uint32)
            || primitive.vertices.byte_length != HELMET_POSITIONS.len() as u64
            || indices.byte_length != HELMET_INDICES_U32.len() as u64
        {
            return Err(GeometryProbeError::Contract);
        }

        let mut frame = Frame::open_streaming(120, 96, WIDTH, HEIGHT)
            .map_err(|_| GeometryProbeError::Ui4("frame-open"))?;
        let device = Device::open(Capabilities::DEFAULT.union(Capabilities::PRESENT))
            .map_err(|code| GeometryProbeError::Vgpu("device-open", code))?;
        let queue = device
            .create_queue(QueueClass::Render)
            .map_err(|code| GeometryProbeError::Vgpu("queue-create", code))?;
        let helmet_vertex_buffer = device
            .create_buffer(
                HELMET_POSNORMAL_BYTES as usize,
                BUFFER_USAGE_MAP_WRITE | BUFFER_USAGE_VERTEX,
            )
            .map_err(|code| GeometryProbeError::Vgpu("helmet-vertex-buffer-create", code))?;
        let helmet_index_buffer = device
            .create_buffer(
                indices.byte_length as usize,
                BUFFER_USAGE_MAP_WRITE | BUFFER_USAGE_INDEX,
            )
            .map_err(|code| GeometryProbeError::Vgpu("helmet-index-buffer-create", code))?;
        let line_vertex_buffer = device
            .create_buffer(
                core::mem::size_of_val(&GRID_VERTICES),
                BUFFER_USAGE_MAP_WRITE | BUFFER_USAGE_VERTEX,
            )
            .map_err(|code| GeometryProbeError::Vgpu("line-vertex-buffer-create", code))?;
        let line_index_buffer = device
            .create_buffer(
                core::mem::size_of_val(&GRID_INDICES),
                BUFFER_USAGE_MAP_WRITE | BUFFER_USAGE_INDEX,
            )
            .map_err(|code| GeometryProbeError::Vgpu("line-index-buffer-create", code))?;

        write_exact(device, helmet_vertex_buffer, 0, HELMET_POSNORMAL)
            .map_err(|code| GeometryProbeError::Vgpu("helmet-posnormal-upload", code))?;
        write_exact(device, helmet_index_buffer, 0, HELMET_INDICES_U32)
            .map_err(|code| GeometryProbeError::Vgpu("helmet-index-upload", code))?;
        write_exact(device, line_vertex_buffer, 0, line_vertex_bytes())
            .map_err(|code| GeometryProbeError::Vgpu("line-vertex-upload", code))?;
        write_exact(device, line_index_buffer, 0, line_index_bytes())
            .map_err(|code| GeometryProbeError::Vgpu("line-index-upload", code))?;

        let retained_mesh = device
            .create_retained_mesh(
                helmet_vertex_buffer,
                helmet_index_buffer,
                RetainedMeshDescriptor {
                    vertex_count: primitive.vertex_count,
                    index_count: primitive.index_count,
                    ..RetainedMeshDescriptor::default()
                },
            )
            .map_err(|code| GeometryProbeError::Vgpu("retained-mesh-create", code))?;

        let mut probe = Self {
            frame,
            device,
            queue,
            _helmet_vertex_buffer: helmet_vertex_buffer,
            _helmet_index_buffer: helmet_index_buffer,
            line_vertex_buffer,
            line_index_buffer,
            retained_mesh,
            flycam: FlyCam::new(
                Camera {
                    position: [0.0; 3],
                    rotation: Quaternion::IDENTITY,
                    projection: Projection::Perspective {
                        yfov: core::f32::consts::FRAC_PI_3,
                        znear: 0.01,
                        zfar: Some(100.0),
                        aspect_ratio: None,
                    },
                },
                0.75,
            ),
            resize_drag: None,
            timeline: 0,
            previous_elapsed_millis: 0,
        };
        probe.render_frame(0)?;
        Ok(probe)
    }

    /// Advance the compact transform state and render one complete frame.
    /// Geometry remains resident; only these seeds and the UI4 lease vary.
    pub fn render_frame(&mut self, elapsed_millis: u64) -> Result<(), GeometryProbeError> {
        self.service_frame_interaction()?;
        let delta_seconds =
            elapsed_millis.saturating_sub(self.previous_elapsed_millis) as f32 * 0.001;
        self.previous_elapsed_millis = elapsed_millis;
        let rotation_before_input = self.flycam.camera.rotation;
        self.flycam.step_blueprint(delta_seconds);
        // The frame's bottom-right primary-button drag owns pointer motion.
        // Input is still drained, but resizing must not also rotate the view.
        if self.resize_drag.is_some() {
            self.flycam.camera.rotation = rotation_before_input;
        }
        let width = self.frame.width();
        let height = self.frame.height();

        self.frame
            .begin_gpu_frame()
            .map_err(|_| GeometryProbeError::Ui4("frame-begin"))?;
        let surface = self
            .device
            .acquire_ui4_surface(self.frame.window_id())
            .map_err(|code| GeometryProbeError::Vgpu("surface-acquire", code))?;
        let point = self
            .device
            .submit_retained_frame(
                self.queue,
                surface,
                self.retained_mesh,
                self.line_vertex_buffer,
                self.line_index_buffer,
                RetainedFrameSubmit {
                    clear_rgba8_srgb: u32::from_le_bytes([15, 20, 38, 255]),
                    seed_count: HEAD_INSTANCE_COUNT,
                    static_draw_count: 3,
                    seeds: retained_seeds(elapsed_millis, self.flycam.camera),
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
            .map_err(|_| GeometryProbeError::Ui4("frame-publish"))?;
        self.timeline = point.value;
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
            .map_err(|_| GeometryProbeError::Ui4("resize-event"))?
        {
            requested_extent = Some((event.width, event.height));
        }
        if let Some((width, height)) = requested_extent
            && (width != self.frame.width() || height != self.frame.height())
        {
            self.frame
                .resize(width, height)
                .map_err(|_| GeometryProbeError::Ui4("maximize-resize"))?;
        }

        let (output_width, output_height) =
            output_dimensions().map_err(|_| GeometryProbeError::Ui4("output-extent"))?;
        let mut drag = self.resize_drag;
        let mut live_extent = None;
        while let Some(event) = self
            .frame
            .take_pointer_event()
            .map_err(|_| GeometryProbeError::Ui4("pointer-event"))?
        {
            let in_grip = event.local_x >= self.frame.width() as i32 - RESIZE_GRIP_PX
                && event.local_y >= self.frame.height() as i32 - RESIZE_GRIP_PX
                && event.local_x < self.frame.width() as i32
                && event.local_y < self.frame.height() as i32;
            let owns_drag = drag.is_some_and(|active| active.source == event.source);
            self.frame
                .set_cursor_icon_for(
                    event.source,
                    if in_grip || owns_drag {
                        CursorIcon::ResizeDiagonal
                    } else {
                        CursorIcon::Default
                    },
                )
                .map_err(|_| GeometryProbeError::Ui4("resize-cursor"))?;

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
                .map_err(|_| GeometryProbeError::Ui4("live-resize"))?;
        }
        Ok(())
    }

    pub const fn timeline(&self) -> u64 {
        self.timeline
    }

    pub fn take_first_presentation(&mut self) -> Result<bool, GeometryProbeError> {
        self.frame
            .take_first_presentation()
            .map_err(|_| GeometryProbeError::Ui4("first-presentation"))
    }
}

#[cfg(target_os = "trueos")]
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

#[cfg(target_os = "trueos")]
fn retained_seeds(
    elapsed_millis: u64,
    camera: Camera,
) -> [RetainedTransformSeed; trueos::vgpu::MAX_RETAINED_TRANSFORM_SEEDS] {
    let seconds = elapsed_millis as f32 * 0.001;
    let half_angle = core::f32::consts::FRAC_PI_4 * seconds;
    let clockwise = [-0.0, 0.0, -libm::sinf(half_angle), libm::cosf(half_angle)];
    let counter_clockwise = [0.0, 0.0, libm::sinf(half_angle), libm::cosf(half_angle)];
    let half_cycle = (elapsed_millis % 1_000) as f32;
    let pulse = libm::fabsf(half_cycle - 500.0) / 500.0;
    let translations = [
        [-0.5, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [-0.5, -0.5, 0.0],
        [0.5, -0.5, 0.0],
    ];
    let [qx, qy, qz, qw] = camera.rotation.normalized().0;
    let view_rotation = Quaternion([-qx, -qy, -qz, qw]);
    core::array::from_fn(|slot| {
        let world_translation = translations[slot];
        let view_translation = view_rotation.rotate([
            world_translation[0] - camera.position[0],
            world_translation[1] - camera.position[1],
            world_translation[2] - camera.position[2],
        ]);
        let world_rotation = match slot {
            0 => clockwise,
            1 => counter_clockwise,
            _ => [0.0, 0.0, 0.0, 1.0],
        };
        RetainedTransformSeed {
            translation: view_translation,
            scale: if slot == 2 {
                [0.45 * pulse; 3]
            } else {
                [0.45; 3]
            },
            rotation: (view_rotation * Quaternion(world_rotation)).normalized().0,
            local_radius: 1.0,
            previous_translation: view_translation,
            draw_group: 0,
            flags: (slot as u32) << 16,
        }
    })
}

#[cfg(target_os = "trueos")]
fn line_vertex_bytes() -> &'static [u8] {
    unsafe {
        core::slice::from_raw_parts(
            GRID_VERTICES.as_ptr().cast::<u8>(),
            core::mem::size_of_val(&GRID_VERTICES),
        )
    }
}

#[cfg(target_os = "trueos")]
fn line_index_bytes() -> &'static [u8] {
    unsafe {
        core::slice::from_raw_parts(
            GRID_INDICES.as_ptr().cast::<u8>(),
            core::mem::size_of_val(&GRID_INDICES),
        )
    }
}

#[cfg(target_os = "trueos")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryProbeError {
    Contract,
    Ui4(&'static str),
    Vgpu(&'static str, i32),
}

#[cfg(target_os = "trueos")]
fn write_exact(device: Device, buffer: Buffer, offset: u64, bytes: &[u8]) -> Result<(), i32> {
    let offset = usize::try_from(offset).map_err(|_| trueos::vgpu::ERR_UNSUPPORTED)?;
    let written = device.write_buffer(buffer, offset, bytes)?;
    (written == bytes.len())
        .then_some(())
        .ok_or(trueos::vgpu::ERR_IO)
}

/// TRUEOS-specific materialization adapter. Field order is intentional: the
/// ring is dropped before `memory`, so `VVideoMem` backs every live ring view.
#[cfg(target_os = "trueos")]
pub struct VVideoRing {
    ring: ExecRing,
    memory: VVideoMem,
}

#[cfg(target_os = "trueos")]
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

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg(target_os = "trueos")]
pub enum VVideoRingError {
    Vgpu(i32),
    Cubism(CubismError),
}

/// Cache adapter for a `VVideoRing`. It uses `VVideoMem` offset operations;
/// the GPU address and underlying PPGTT mapping remain inside TRUEOS. Cubism
/// flushes slot data first, then the published header control line. It
/// invalidates just the GPU-owned payload after retirement has read the
/// CPU-owned header metadata.
#[cfg(target_os = "trueos")]
pub struct VVideoVisibility<'a> {
    memory: &'a VVideoMem,
    resource: SharedResourceId,
}

#[cfg(target_os = "trueos")]
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

#[cfg(target_os = "trueos")]
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
    }
}
