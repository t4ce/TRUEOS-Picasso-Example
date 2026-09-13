//! Three hardware streams, four unlit quads. The fourth quad shares stream 0.
//! Geometry is uploaded once; steady-state CPU traffic is texture IDs and TRS.
use alloc::vec::Vec;
use trueos::{
    vgpu::*,
    vmedia::{Video, VideoFrame, VideoPoll},
};
const ASSETS: [&[u8]; 3] = [
    include_bytes!("../Assets/Video/DSC_1879_48_a.mp4"),
    include_bytes!("../Assets/Video/DSC_1879_128_a.mp4"),
    include_bytes!("../Assets/Video/DSC_1879_256_a.mp4"),
];
const SIZES: [u32; 3] = [48, 128, 256];
// Seen from the example's camera on -Z: normal -Z, image right world +X,
// image top world +Y. Winding agrees with the retained native camera contract.
const VERTICES: [[f32; 8]; 4] = [
    [-1., -1., 0., 0., 0., -1., 0., 1.],
    [-1., 1., 0., 0., 0., -1., 0., 0.],
    [1., 1., 0., 0., 0., -1., 1., 0.],
    [1., -1., 0., 0., 0., -1., 1., 1.],
];
const INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];
pub struct Demo {
    device: Device,
    mesh: Option<RetainedMesh>,
    vertices: Option<Buffer>,
    indices: Option<Buffer>,
    streams: Vec<Video>,
    frames: [Option<VideoFrame>; 3],
}
impl Demo {
    pub fn open(device: Device) -> Result<Self, i32> {
        let mut demo = Self {
            device,
            mesh: None,
            vertices: None,
            indices: None,
            streams: Vec::new(),
            frames: core::array::from_fn(|_| None),
        };
        let vertices: Vec<u8> = VERTICES
            .iter()
            .flatten()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let indices: Vec<u8> = INDICES.iter().flat_map(|i| i.to_le_bytes()).collect();
        let vb =
            device.create_buffer(vertices.len(), BUFFER_USAGE_MAP_WRITE | BUFFER_USAGE_VERTEX)?;
        demo.vertices = Some(vb);
        let ib =
            device.create_buffer(indices.len(), BUFFER_USAGE_MAP_WRITE | BUFFER_USAGE_INDEX)?;
        demo.indices = Some(ib);
        if device.write_buffer(vb, 0, &vertices)? != vertices.len()
            || device.write_buffer(ib, 0, &indices)? != indices.len()
        {
            return Err(ERR_IO);
        }
        demo.mesh = Some(device.create_retained_mesh(
            vb,
            ib,
            RetainedMeshDescriptor {
                vertex_count: 4,
                index_count: 6,
                vertex_layout: RETAINED_VERTEX_LAYOUT_POS_NORMAL_UV,
                topology: PRIMITIVE_TOPOLOGY_TRIANGLE_LIST | RETAINED_MESH_FLAG_DOUBLE_SIDED,
                ..RetainedMeshDescriptor::default()
            },
        )?);
        Ok(demo)
    }
    pub fn running(&self) -> bool {
        !self.streams.is_empty()
    }
    pub fn start(&mut self) -> Result<(), i32> {
        if self.running() {
            return Ok(());
        }
        let mut streams = Vec::new();
        for encoded in ASSETS {
            streams.push(Video::open(self.device, encoded, true)?);
        }
        self.streams = streams;
        Ok(())
    }
    pub fn stop(&mut self) {
        self.frames = core::array::from_fn(|_| None);
        self.streams.clear();
    }
    pub fn poll(&mut self) -> Result<bool, i32> {
        for (i, stream) in self.streams.iter_mut().enumerate() {
            if let VideoPoll::Frame(frame) = stream.poll()? {
                if frame.extent() != [SIZES[i]; 2] {
                    return Err(ERR_UNSUPPORTED);
                }
                self.frames[i] = Some(frame);
            }
        }
        Ok(self.frames.iter().all(Option::is_some))
    }
    pub fn render(
        &self,
        queue: Queue,
        surface: Ui4Surface,
        camera: RetainedCamera,
    ) -> Result<TimelinePoint, i32> {
        let ids = core::array::from_fn(|i| self.frames[i].as_ref().unwrap().texture_id().raw());
        let submit = submission(camera, ids);
        let point = self.device.submit_retained_textured_frame_v1(
            queue,
            surface,
            self.mesh.unwrap(),
            submit,
        )?;
        self.device.wait(queue, point.value)?;
        Ok(point)
    }
}
fn submission(camera: RetainedCamera, textures: [u64; 3]) -> RetainedTexturedFrameV1 {
    let mut frame = RetainedFrameSubmit {
        camera,
        seed_count: 4,
        clear_rgba8_srgb: u32::from_le_bytes([12, 14, 18, 255]),
        ..RetainedFrameSubmit::default()
    };
    for (seed, translation) in frame.seeds.iter_mut().zip(super::HEAD_WORLD_TRANSLATIONS) {
        *seed = RetainedTransformSeed {
            translation,
            previous_translation: translation,
            scale: [1.1; 3],
            rotation: [0., 0., 0., 1.],
            local_radius: 1.5,
            ..RetainedTransformSeed::default()
        };
    }
    RetainedTexturedFrameV1 {
        frame,
        ranges: [RetainedDrawRange {
            first_index: 0,
            index_count: 6,
        }; 4],
        textures: [textures[0], textures[1], textures[2], textures[0]],
    }
}
impl Drop for Demo {
    fn drop(&mut self) {
        // Every render waits for retirement before input can switch scenes.
        self.frames = core::array::from_fn(|_| None);
        self.streams.clear();
        if let Some(mesh) = self.mesh {
            let _ = self.device.destroy_retained_mesh(mesh);
        }
        if let Some(buffer) = self.indices {
            let _ = self.device.destroy_buffer(buffer);
        }
        if let Some(buffer) = self.vertices {
            let _ = self.device.destroy_buffer(buffer);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn four_quads_share_three_streams_without_pbr_or_static_passes() {
        let s = submission(RetainedCamera::default(), [11, 22, 33]);
        assert_eq!(s.textures, [11, 22, 33, 11]);
        assert_eq!(s.frame.seed_count, 4);
        assert_eq!(s.frame.static_draw_count, 0);
        assert_eq!(s.frame.material, RetainedMaterial::default());
        for r in s.ranges {
            assert_eq!(r.first_index, 0);
            assert_eq!(r.index_count, 6);
        }
    }
}
