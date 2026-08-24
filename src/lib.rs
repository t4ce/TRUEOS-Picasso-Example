//! Consumer-first Picasso example for a `#![no_std]` TRUEOS application.
//!
//! `Assets/` is deliberately input to the host-side Dealer/importer. This
//! target does not parse files, call redb, or perform filesystem I/O. At boot
//! the platform resolves these IDs to shared DDR5 allocations and GPUVM maps.
#![no_std]

use trueos_picasso::{
    CubismError, ExecRing, ExecutablePrimitive, GpuAddress, PreparedRange, PrimitiveTopology,
    ResourceId,
};

/// The host importer assigns these stable prepared IDs when it seeds Dealer.
/// They are examples, not a disk-format promise.
pub const TRIANGLE_VERTICES: ResourceId = ResourceId(0x0001_0000_0000_0001);
pub const TRIANGLE_INDICES: ResourceId = ResourceId(0x0001_0000_0000_0002);

/// Describe a prepared glTF primitive without parsing or touching storage.
/// The caller supplies the platform's existing shared-DDR GPU mappings.
pub const fn prepared_triangle(
    vertex_address: GpuAddress,
    index_address: GpuAddress,
) -> ExecutablePrimitive {
    ExecutablePrimitive {
        topology: PrimitiveTopology::TriangleList,
        vertices: PreparedRange {
            resource: TRIANGLE_VERTICES,
            gpu_address: vertex_address,
            offset: 0,
            byte_length: 36,
            revision: 1,
        },
        indices: Some(PreparedRange {
            resource: TRIANGLE_INDICES,
            gpu_address: index_address,
            offset: 0,
            byte_length: 6,
            revision: 1,
        }),
        index_format: Some(trueos_picasso::IndexFormat::Uint16),
        vertex_stride: 12,
        vertex_count: 3,
        index_count: 3,
    }
}

/// Bind the platform's one shared CPU/GPU allocation to Picasso's canonical
/// execution-ring protocol. The allocation is owned and mapped by the vGPU
/// layer; no GuC or driver-private handle crosses this API.
///
/// # Safety
/// `cpu_base` and `gpu_base` must name the same stable shared-DDR allocation.
pub unsafe fn ring_from_shared_mapping(
    cpu_base: *mut u8,
    total_bytes: usize,
    gpu_base: GpuAddress,
    slot_count: usize,
    slot_stride: usize,
) -> Result<ExecRing, CubismError> {
    unsafe { ExecRing::from_raw_parts(cpu_base, total_bytes, gpu_base.0, slot_count, slot_stride) }
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
    fn no_std_consumer_can_seal_and_retire_a_shared_slot() {
        const SLOT_COUNT: usize = 1;
        const SLOT_STRIDE: usize = 128;
        let region = Region::new(SLOT_COUNT * SLOT_STRIDE);
        let ring = unsafe {
            ring_from_shared_mapping(
                region.ptr,
                SLOT_COUNT * SLOT_STRIDE,
                GpuAddress(0x2000_0000),
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
        sealed.mark_in_flight(17).unwrap();
        ring.retire(index, generation, 17, &visibility).unwrap();

        assert_eq!(
            prepared_triangle(GpuAddress(1), GpuAddress(2)).index_count,
            3
        );
    }
}
