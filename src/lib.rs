//! Consumer-first Picasso example for a `#![no_std]` TRUEOS application.
//!
//! At boot the platform resolves Dealer IDs to opaque vGPU buffers. Picasso
//! receives only stable resource identities and byte-relative ranges.
#![no_std]

#[cfg(target_os = "trueos")]
use trueos::ui4_scene::{Damage, Frame};
#[cfg(target_os = "trueos")]
use trueos::vgpu::{
    BUFFER_USAGE_INDEX, BUFFER_USAGE_MAP_WRITE, BUFFER_USAGE_VERTEX, Buffer, BufferSlice,
    Capabilities, Device, IndexedBatchDrawV2, IndexedDrawBatchV2, Queue, QueueClass,
    RenderPipeline, SHADER_PACKAGE_CLIP_POSITION3_IMMEDIATE_RGBA_FNV1A64, VVideoMem,
};
#[cfg(any(test, target_os = "trueos"))]
use trueos_picasso::ExecRing;
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
pub static HELMET_INDICES_U32: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/damaged_helmet.indices.u32le"));

/// Three independent native line-list primitives. They intentionally have no
/// transform references: the mixed batch proves topology and transform scope
/// are per draw rather than accidental global state.
pub static RGB_LINE_VERTICES: [[f32; 3]; 6] = [
    [-0.90, 0.85, 0.0],
    [-0.50, 0.85, 0.0],
    [-0.20, 0.85, 0.0],
    [0.20, 0.85, 0.0],
    [0.50, 0.85, 0.0],
    [0.90, 0.85, 0.0],
];
pub static RGB_LINE_INDICES: [u32; 6] = [0, 1, 0, 1, 0, 1];

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
    _device: Device,
    _queue: Queue,
    _vertex_buffer: Buffer,
    _index_buffer: Buffer,
    _pipeline: RenderPipeline,
    timeline: u64,
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
        let vertex_buffer = device
            .create_buffer(
                primitive.vertices.byte_length as usize
                    + core::mem::size_of_val(&RGB_LINE_VERTICES),
                BUFFER_USAGE_MAP_WRITE | BUFFER_USAGE_VERTEX,
            )
            .map_err(|code| GeometryProbeError::Vgpu("vertex-buffer-create", code))?;
        let index_buffer = device
            .create_buffer(
                indices.byte_length as usize + core::mem::size_of_val(&RGB_LINE_INDICES),
                BUFFER_USAGE_MAP_WRITE | BUFFER_USAGE_INDEX,
            )
            .map_err(|code| GeometryProbeError::Vgpu("index-buffer-create", code))?;

        write_exact(
            device,
            vertex_buffer,
            primitive.vertices.offset,
            HELMET_POSITIONS,
        )
        .map_err(|code| GeometryProbeError::Vgpu("vertex-upload", code))?;
        write_exact(device, index_buffer, indices.offset, HELMET_INDICES_U32)
            .map_err(|code| GeometryProbeError::Vgpu("index-upload", code))?;
        write_exact(
            device,
            vertex_buffer,
            primitive.vertices.byte_length,
            line_vertex_bytes(),
        )
        .map_err(|code| GeometryProbeError::Vgpu("line-vertex-upload", code))?;
        write_exact(
            device,
            index_buffer,
            indices.byte_length,
            line_index_bytes(),
        )
        .map_err(|code| GeometryProbeError::Vgpu("line-index-upload", code))?;

        let shader = device
            .create_shader_module(SHADER_PACKAGE_CLIP_POSITION3_IMMEDIATE_RGBA_FNV1A64)
            .map_err(|code| GeometryProbeError::Vgpu("shader-create", code))?;
        let pipeline = device
            .create_render_pipeline(shader, primitive.vertex_stride, 0)
            .map_err(|code| GeometryProbeError::Vgpu("pipeline-create", code))?;
        device
            .destroy_shader_module(shader)
            .map_err(|code| GeometryProbeError::Vgpu("shader-destroy", code))?;

        frame
            .begin_gpu_frame()
            .map_err(|_| GeometryProbeError::Ui4("frame-begin"))?;
        let surface = device
            .acquire_ui4_surface(frame.window_id())
            .map_err(|code| GeometryProbeError::Vgpu("surface-acquire", code))?;
        let point = device
            .submit_ui4_indexed_batch_v2(
                queue,
                surface,
                pipeline,
                vertex_buffer,
                index_buffer,
                IndexedDrawBatchV2 {
                    vertex_offset: primitive.vertices.offset,
                    index_offset: indices.offset,
                    clear_rgba8_srgb: u32::from_le_bytes([15, 20, 38, 255]),
                    draw_count: 4,
                    draws: mixed_topology_draws(primitive.index_count, primitive.vertex_count),
                    ..IndexedDrawBatchV2::default()
                },
            )
            .map_err(|code| GeometryProbeError::Vgpu("mixed-topology-submit-v2", code))?;
        device
            .wait(queue, point.value)
            .map_err(|code| GeometryProbeError::Vgpu("timeline-wait", code))?;
        frame
            .publish(Damage::full(WIDTH, HEIGHT))
            .map_err(|_| GeometryProbeError::Ui4("frame-publish"))?;

        Ok(Self {
            frame,
            _device: device,
            _queue: queue,
            _vertex_buffer: vertex_buffer,
            _index_buffer: index_buffer,
            _pipeline: pipeline,
            timeline: point.value,
        })
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
fn mixed_topology_draws(
    helmet_index_count: u32,
    helmet_vertex_count: u32,
) -> [IndexedBatchDrawV2; trueos::vgpu::MAX_INDEXED_BATCH_DRAWS] {
    let mut draws = [IndexedBatchDrawV2::default(); trueos::vgpu::MAX_INDEXED_BATCH_DRAWS];
    for (slot, planned) in mixed_topology_plan(helmet_index_count, helmet_vertex_count)
        .into_iter()
        .enumerate()
    {
        draws[slot] = IndexedBatchDrawV2 {
            index_count: planned.index_count,
            first_index: planned.first_index,
            base_vertex: planned.base_vertex,
            rgba8_srgb: planned.rgba8_srgb,
            topology: match planned.topology {
                PrimitiveTopology::LineList => trueos::vgpu::PRIMITIVE_TOPOLOGY_LINE_LIST,
                PrimitiveTopology::TriangleList => trueos::vgpu::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                _ => 0,
            },
            reserved: 0,
        };
    }
    draws
}

#[cfg(target_os = "trueos")]
fn line_vertex_bytes() -> &'static [u8] {
    unsafe {
        core::slice::from_raw_parts(
            RGB_LINE_VERTICES.as_ptr().cast::<u8>(),
            core::mem::size_of_val(&RGB_LINE_VERTICES),
        )
    }
}

#[cfg(target_os = "trueos")]
fn line_index_bytes() -> &'static [u8] {
    unsafe {
        core::slice::from_raw_parts(
            RGB_LINE_INDICES.as_ptr().cast::<u8>(),
            core::mem::size_of_val(&RGB_LINE_INDICES),
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
        assert_eq!(RGB_LINE_INDICES, [0, 1, 0, 1, 0, 1]);
    }
}
