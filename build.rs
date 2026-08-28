use std::{
    env, fs,
    path::{Path, PathBuf},
};
const ASSETS: [(&str, &str); 5] = [
    ("DamagedHelmet", "Assets/DamagedHelmet/DamagedHelmet.glb"),
    ("Triangle", "Assets/Triangle/Triangle.gltf"),
    ("BoxInterleaved", "Assets/BoxInterleaved/BoxInterleaved.glb"),
    (
        "SimpleSparseAccessor",
        "Assets/SimpleSparseAccessor/SimpleSparseAccessor.gltf",
    ),
    ("RiggedSimple", "Assets/RiggedSimple/RiggedSimple.glb"),
];

// Stable presentation baseline: retain every map for startup admission, but
// route the active draw through the known opaque position+normal path. The
// base-color-only shader remains baked for the next isolated texture probe.
const ENABLE_SAMPLED_MATERIAL: bool = false;

fn main() {
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set"));
    let mut catalog = String::from(
        "pub const ASSET_COUNT: usize = 5;\npub static ASSETS: [PreparedAsset; ASSET_COUNT] = [\n",
    );
    for (slot, (name, source)) in ASSETS.iter().enumerate() {
        println!("cargo:rerun-if-changed={source}");
        let source_path = Path::new(source);
        let PreparedGeometry {
            vertices,
            indices,
            material,
            vertex_stride,
            primitives: prepared_primitives,
        } = prepare(source_path, ENABLE_SAMPLED_MATERIAL);
        let stem = name.to_ascii_lowercase();
        let vertex_file = format!("{stem}.posnormal.f32le");
        let index_file = format!("{stem}.indices.u32le");
        fs::write(out.join(&vertex_file), &vertices).expect("write vertices");
        fs::write(out.join(&index_file), &indices).expect("write indices");
        let base_color = write_texture(
            out.as_path(),
            &stem,
            name,
            "base-color",
            material.base_color.as_ref(),
        );
        let metallic_roughness = write_texture(
            out.as_path(),
            &stem,
            name,
            "metallic-roughness",
            material.metallic_roughness.as_ref(),
        );
        let emissive = write_texture(
            out.as_path(),
            &stem,
            name,
            "emissive",
            material.emissive.as_ref(),
        );
        let occlusion = write_texture(
            out.as_path(),
            &stem,
            name,
            "occlusion",
            material.occlusion.as_ref(),
        );
        let normal = write_texture(
            out.as_path(),
            &stem,
            name,
            "normal",
            material.normal.as_ref(),
        );
        let primitives = prepared_primitives
            .iter()
            .map(|primitive| format!(
                "PreparedPrimitive {{ topology: PrimitiveTopology::{}, first_vertex: {}, vertex_count: {}, first_index: {}, index_count: {}, double_sided: {} }}",
                primitive.topology,
                primitive.first_vertex,
                primitive.vertex_count,
                primitive.first_index,
                primitive.index_count,
                primitive.double_sided,
            ))
            .collect::<Vec<_>>()
            .join(", ");
        let retained_topology = prepared_primitives
            .first()
            .filter(|first| {
                prepared_primitives
                    .iter()
                    .all(|primitive| primitive.topology == first.topology)
            })
            .map(|primitive| format!("Some(PrimitiveTopology::{})", primitive.topology))
            .unwrap_or_else(|| String::from("None"));
        // One retained mesh currently contains all source primitives. If their
        // cull policies differ, disable culling conservatively: that keeps
        // every glTF `doubleSided` face visible until material batching exists.
        let retained_double_sided = prepared_primitives
            .iter()
            .any(|primitive| primitive.double_sided);
        let sampled_material = ENABLE_SAMPLED_MATERIAL && material.base_color.is_some();
        catalog.push_str(&format!(
            "PreparedAsset {{ name: \"{name}\", vertices: include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{vertex_file}\")), indices: include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{index_file}\")), vertex_count: {vertex_count}, index_count: {index_count}, vertex_stride: {vertex_stride}, material: PreparedMaterial {{ base_color: {base_color}, metallic_roughness: {metallic_roughness}, emissive: {emissive}, occlusion: {occlusion}, normal: {normal}, emissive_factor: {emissive_factor:?} }}, sampled_material: {sampled_material}, helmet_program: {helmet_program}, primitives: &[{primitives}], retained_topology: {retained_topology}, retained_double_sided: {retained_double_sided} }},\n",
            vertex_count = vertices.len() / vertex_stride,
            index_count = indices.len() / 4,
            emissive_factor = material.emissive_factor,
            sampled_material = sampled_material,
            helmet_program = slot == 0,
        ));
    }
    catalog.push_str("];\n");
    fs::write(out.join("prepared_assets.rs"), catalog).expect("write asset catalog");
}

struct PreparedGeometry {
    vertices: Vec<u8>,
    indices: Vec<u8>,
    material: PreparedMaterial,
    vertex_stride: usize,
    primitives: Vec<PreparedPrimitive>,
}

struct PreparedMaterial {
    base_color: Option<(&'static str, Vec<u8>)>,
    metallic_roughness: Option<(&'static str, Vec<u8>)>,
    emissive: Option<(&'static str, Vec<u8>)>,
    occlusion: Option<(&'static str, Vec<u8>)>,
    normal: Option<(&'static str, Vec<u8>)>,
    emissive_factor: [f32; 3],
}

struct PreparedPrimitive {
    topology: &'static str,
    first_vertex: u32,
    vertex_count: u32,
    first_index: u32,
    index_count: u32,
    double_sided: bool,
}

fn prepare(source: &Path, sampled_material: bool) -> PreparedGeometry {
    let bytes = fs::read(source).expect("read asset");
    let parsed = gltf::Gltf::from_slice(&bytes).expect("parse asset");
    let base = source.parent().unwrap_or(Path::new("."));
    let buffers: Vec<Vec<u8>> = parsed
        .buffers()
        .map(|b| match b.source() {
            gltf::buffer::Source::Bin => parsed.blob.clone().expect("BIN chunk"),
            gltf::buffer::Source::Uri(uri) if uri.starts_with("data:") => base64::Engine::decode(
                &base64::engine::general_purpose::STANDARD,
                uri.split_once(',').expect("data URI").1,
            )
            .expect("base64 buffer"),
            gltf::buffer::Source::Uri(uri) => fs::read(base.join(uri)).expect("external buffer"),
        })
        .collect();
    let material = parsed.materials().next();
    let texture = |source: Option<gltf::image::Source<'_>>| {
        source.and_then(|image| load_supported_image(image, &buffers, base))
    };
    let prepared_material = PreparedMaterial {
        base_color: texture(material.as_ref().and_then(|material| {
            material
                .pbr_metallic_roughness()
                .base_color_texture()
                .map(|info| info.texture().source().source())
        })),
        metallic_roughness: texture(material.as_ref().and_then(|material| {
            material
                .pbr_metallic_roughness()
                .metallic_roughness_texture()
                .map(|info| info.texture().source().source())
        })),
        emissive: texture(
            material
                .as_ref()
                .and_then(|material| material.emissive_texture())
                .map(|info| info.texture().source().source()),
        ),
        occlusion: texture(
            material
                .as_ref()
                .and_then(|material| material.occlusion_texture())
                .map(|info| info.texture().source().source()),
        ),
        normal: texture(
            material
                .as_ref()
                .and_then(|material| material.normal_texture())
                .map(|info| info.texture().source().source()),
        ),
        emissive_factor: material
            .as_ref()
            .map_or([0.0; 3], |material| material.emissive_factor()),
    };
    // The active retained shader samples base color alone while every supplied
    // map is still emitted for atomically bundled residency.
    let sampled_material = sampled_material && prepared_material.base_color.is_some();
    let (mut positions, mut normals, mut uvs, mut indices) = (
        Vec::<[f32; 3]>::new(),
        Vec::<[f32; 3]>::new(),
        Vec::<[f32; 2]>::new(),
        Vec::<u32>::new(),
    );
    let mut primitives = Vec::<PreparedPrimitive>::new();
    for mesh in parsed.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|b| Some(buffers[b.index()].as_slice()));
            let base_vertex = positions.len() as u32;
            let p: Vec<_> = reader.read_positions().expect("POSITION").collect();
            let vertex_count = p.len() as u32;
            let local: Vec<u32> = reader
                .read_indices()
                .map(|v| v.into_u32().collect())
                .unwrap_or_else(|| (0..p.len() as u32).collect());
            let n: Vec<_> = reader
                .read_normals()
                .map(|v| v.collect())
                .unwrap_or_else(|| generated_normals(primitive.mode(), &p, &local));
            let uv: Vec<_> = reader
                .read_tex_coords(0)
                .map(|coords| coords.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0; 2]; p.len()]);
            assert_eq!(p.len(), n.len());
            assert_eq!(p.len(), uv.len());
            positions.extend(p);
            normals.extend(n);
            uvs.extend(uv);
            let first_index = indices.len() as u32;
            let index_count = local.len() as u32;
            indices.extend(local.into_iter().map(|i| i + base_vertex));
            primitives.push(PreparedPrimitive {
                topology: primitive_topology(primitive.mode()),
                first_vertex: base_vertex,
                vertex_count,
                first_index,
                index_count,
                double_sided: primitive.material().double_sided(),
            });
        }
    }
    assert!(!positions.is_empty() && !indices.is_empty());
    normalize(&mut positions);
    let vertex_stride = if sampled_material { 32 } else { 24 };
    let mut vb = Vec::with_capacity(positions.len() * vertex_stride);
    for ((p, n), uv) in positions.iter().zip(normals).zip(uvs) {
        for v in p.iter().chain(n.iter()) {
            vb.extend(v.to_le_bytes());
        }
        if sampled_material {
            for v in uv {
                vb.extend(v.to_le_bytes());
            }
        }
    }
    let mut ib = Vec::with_capacity(indices.len() * 4);
    for i in indices {
        ib.extend(i.to_le_bytes());
    }
    PreparedGeometry {
        vertices: vb,
        indices: ib,
        material: prepared_material,
        vertex_stride,
        primitives,
    }
}

fn write_texture(
    out: &Path,
    stem: &str,
    asset_name: &str,
    role: &str,
    texture: Option<&(&'static str, Vec<u8>)>,
) -> String {
    let Some((extension, bytes)) = texture else {
        return String::from("PreparedTexture::NONE");
    };
    let file = format!("{stem}.{role}.{extension}");
    fs::write(out.join(&file), bytes).expect("write prepared material texture");
    format!(
        "PreparedTexture {{ name: \"{asset_name}.{role}.{extension}\", bytes: include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{file}\")) }}"
    )
}

fn primitive_topology(mode: gltf::mesh::Mode) -> &'static str {
    match mode {
        gltf::mesh::Mode::Points => "PointList",
        gltf::mesh::Mode::Lines => "LineList",
        gltf::mesh::Mode::LineLoop => "LineLoop",
        gltf::mesh::Mode::LineStrip => "LineStrip",
        gltf::mesh::Mode::Triangles => "TriangleList",
        gltf::mesh::Mode::TriangleStrip => "TriangleStrip",
        gltf::mesh::Mode::TriangleFan => "TriangleFan",
    }
}

fn load_supported_image(
    source: gltf::image::Source<'_>,
    buffers: &[Vec<u8>],
    base: &Path,
) -> Option<(&'static str, Vec<u8>)> {
    match source {
        gltf::image::Source::View { view, mime_type } => {
            let extension = supported_image_extension(mime_type)?;
            let start = view.offset();
            let end = start.checked_add(view.length())?;
            Some((
                extension,
                buffers
                    .get(view.buffer().index())?
                    .get(start..end)?
                    .to_vec(),
            ))
        }
        gltf::image::Source::Uri { uri, mime_type } => {
            let extension = mime_type.and_then(supported_image_extension).or_else(|| {
                uri.rsplit_once('.')
                    .and_then(|(_, ext)| supported_image_extension(ext))
            })?;
            let bytes = if uri.starts_with("data:") {
                base64::Engine::decode(
                    &base64::engine::general_purpose::STANDARD,
                    uri.split_once(',')?.1,
                )
                .ok()?
            } else {
                fs::read(base.join(uri)).ok()?
            };
            Some((extension, bytes))
        }
    }
}

fn supported_image_extension(value: &str) -> Option<&'static str> {
    match value.to_ascii_lowercase().as_str() {
        "image/jpeg" | "jpeg" | "jpg" => Some("jpg"),
        "image/png" | "png" => Some("png"),
        "image/bmp" | "bmp" => Some("bmp"),
        _ => None,
    }
}

fn generated_normals(mode: gltf::mesh::Mode, p: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut out = vec![[0.0; 3]; p.len()];
    let mut add_triangle = |a: u32, b: u32, c: u32| {
        let a_position = p[a as usize];
        let b_position = p[b as usize];
        let c_position = p[c as usize];
        let u = [
            b_position[0] - a_position[0],
            b_position[1] - a_position[1],
            b_position[2] - a_position[2],
        ];
        let v = [
            c_position[0] - a_position[0],
            c_position[1] - a_position[1],
            c_position[2] - a_position[2],
        ];
        let n = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        for i in [a, b, c] {
            for a in 0..3 {
                out[i as usize][a] += n[a];
            }
        }
    };
    match mode {
        gltf::mesh::Mode::Triangles => {
            for triangle in indices.chunks_exact(3) {
                add_triangle(triangle[0], triangle[1], triangle[2]);
            }
        }
        gltf::mesh::Mode::TriangleStrip => {
            for (index, triangle) in indices.windows(3).enumerate() {
                let [a, b, c] = [triangle[0], triangle[1], triangle[2]];
                if index.is_multiple_of(2) {
                    add_triangle(a, b, c);
                } else {
                    add_triangle(b, a, c);
                }
            }
        }
        gltf::mesh::Mode::TriangleFan => {
            if let Some(&center) = indices.first() {
                for triangle in indices[1..].windows(2) {
                    add_triangle(center, triangle[0], triangle[1]);
                }
            }
        }
        gltf::mesh::Mode::Points
        | gltf::mesh::Mode::Lines
        | gltf::mesh::Mode::LineLoop
        | gltf::mesh::Mode::LineStrip => {}
    }
    for n in &mut out {
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if l > 0.0 {
            for v in n {
                *v /= l;
            }
        } else {
            n[2] = 1.0;
        }
    }
    out
}
fn normalize(p: &mut [[f32; 3]]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for v in p.iter() {
        for a in 0..3 {
            min[a] = min[a].min(v[a]);
            max[a] = max[a].max(v[a]);
        }
    }
    let c = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];
    let e = (max[0] - min[0]).max(max[1] - min[1]).max(max[2] - min[2]);
    assert!(e > 0.0);
    for v in p {
        for a in 0..3 {
            v[a] = (v[a] - c[a]) * 1.6 / e;
        }
    }
}
