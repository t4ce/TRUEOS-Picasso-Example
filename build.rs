use std::{env, fs, path::PathBuf};

fn main() {
    const SOURCE: &str = "Assets/DamagedHelmet/DamagedHelmet.glb";
    println!("cargo:rerun-if-changed={SOURCE}");

    let gltf = gltf::Gltf::open(SOURCE).expect("open DamagedHelmet GLB");
    let blob = gltf.blob.as_deref();
    let mesh = gltf
        .meshes()
        .next()
        .expect("DamagedHelmet must contain a mesh");
    let mut positions = Vec::<[f32; 3]>::new();
    let mut normals = Vec::<[f32; 3]>::new();
    let mut indices = Vec::<u32>::new();

    for primitive in mesh.primitives() {
        assert_eq!(
            primitive.mode(),
            gltf::mesh::Mode::Triangles,
            "the prepared probe supports triangle-list primitives"
        );
        let reader = primitive.reader(|buffer| match buffer.source() {
            gltf::buffer::Source::Bin => blob,
            gltf::buffer::Source::Uri(_) => None,
        });
        let base = u32::try_from(positions.len()).expect("vertex count fits u32");
        let primitive_positions = reader
            .read_positions()
            .expect("DamagedHelmet primitive must contain POSITION")
            .collect::<Vec<_>>();
        let primitive_normals = reader
            .read_normals()
            .expect("DamagedHelmet primitive must contain NORMAL")
            .collect::<Vec<_>>();
        assert_eq!(
            primitive_normals.len(),
            primitive_positions.len(),
            "POSITION/NORMAL accessor counts must match"
        );
        let primitive_indices = reader
            .read_indices()
            .map(|values| values.into_u32().collect::<Vec<_>>())
            .unwrap_or_else(|| {
                (0..u32::try_from(primitive_positions.len()).expect("vertex count fits u32"))
                    .collect()
            });
        positions.extend(primitive_positions);
        normals.extend(primitive_normals);
        indices.extend(
            primitive_indices
                .into_iter()
                .map(|index| index.checked_add(base).expect("combined index fits u32")),
        );
    }

    assert!(!positions.is_empty(), "DamagedHelmet mesh has no vertices");
    assert_eq!(normals.len(), positions.len(), "every position has one normal");
    assert!(!indices.is_empty(), "DamagedHelmet mesh has no indices");
    assert_eq!(indices.len() % 3, 0, "indices must form triangles");
    normalize_to_clip_space(&mut positions);

    let mut vertex_bytes = Vec::with_capacity(positions.len() * 12);
    for position in &positions {
        for component in position {
            vertex_bytes.extend_from_slice(&component.to_le_bytes());
        }
    }
    let mut posnormal_bytes = Vec::with_capacity(positions.len() * 24);
    for (position, normal) in positions.iter().zip(&normals) {
        assert!(
            normal.iter().all(|component| component.is_finite()),
            "mesh normals must be finite"
        );
        for component in position.iter().chain(normal) {
            posnormal_bytes.extend_from_slice(&component.to_le_bytes());
        }
    }
    let mut index_bytes = Vec::with_capacity(indices.len() * 4);
    for index in &indices {
        index_bytes.extend_from_slice(&index.to_le_bytes());
    }

    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set"));
    fs::write(out.join("damaged_helmet.positions.f32le"), &vertex_bytes)
        .expect("write prepared positions");
    fs::write(
        out.join("damaged_helmet.posnormal.f32le"),
        &posnormal_bytes,
    )
    .expect("write prepared position/normal vertices");
    fs::write(out.join("damaged_helmet.indices.u32le"), &index_bytes)
        .expect("write prepared indices");
    fs::write(
        out.join("damaged_helmet.meta.rs"),
        format!(
            "pub const HELMET_VERTEX_COUNT: u32 = {};\n\
             pub const HELMET_INDEX_COUNT: u32 = {};\n\
             pub const HELMET_VERTEX_BYTES: u64 = {};\n\
             pub const HELMET_POSNORMAL_BYTES: u64 = {};\n\
             pub const HELMET_INDEX_BYTES: u64 = {};\n",
            positions.len(),
            indices.len(),
            vertex_bytes.len(),
            posnormal_bytes.len(),
            index_bytes.len(),
        ),
    )
    .expect("write prepared mesh metadata");
}

fn normalize_to_clip_space(positions: &mut [[f32; 3]]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for position in positions.iter() {
        assert!(
            position.iter().all(|component| component.is_finite()),
            "mesh positions must be finite"
        );
        for axis in 0..3 {
            min[axis] = min[axis].min(position[axis]);
            max[axis] = max[axis].max(position[axis]);
        }
    }
    let center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];
    let extent = (max[0] - min[0]).max(max[1] - min[1]).max(max[2] - min[2]);
    assert!(extent > 0.0, "mesh must have a non-zero extent");
    let scale = 1.6 / extent;
    for position in positions {
        for axis in 0..3 {
            position[axis] = (position[axis] - center[axis]) * scale;
        }
    }
}
