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

fn main() {
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set"));
    let mut catalog = String::from(
        "pub const ASSET_COUNT: usize = 5;\npub static ASSETS: [PreparedAsset; ASSET_COUNT] = [\n",
    );
    for (slot, (name, source)) in ASSETS.iter().enumerate() {
        println!("cargo:rerun-if-changed={source}");
        let source_path = Path::new(source);
        let (vertices, indices, base_color, vertex_stride) = prepare(source_path);
        let stem = name.to_ascii_lowercase();
        let vertex_file = format!("{stem}.posnormal.f32le");
        let index_file = format!("{stem}.indices.u32le");
        fs::write(out.join(&vertex_file), &vertices).expect("write vertices");
        fs::write(out.join(&index_file), &indices).expect("write indices");
        let (base_color_name, base_color_bytes) = if let Some((extension, bytes)) = base_color {
            let file = format!("{stem}.basecolor.{extension}");
            fs::write(out.join(&file), bytes).expect("write base-color texture");
            (
                format!("{name}.basecolor.{extension}"),
                format!("include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{file}\"))"),
            )
        } else {
            (String::new(), String::from("&[]"))
        };
        catalog.push_str(&format!("PreparedAsset {{ name: \"{name}\", vertices: include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{vertex_file}\")), indices: include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{index_file}\")), vertex_count: {}, index_count: {}, vertex_stride: {vertex_stride}, base_color_name: \"{base_color_name}\", base_color_bytes: {base_color_bytes}, sampled_material: {}, helmet_program: {} }},\n", vertices.len()/vertex_stride, indices.len()/4, !base_color_name.is_empty(), slot==0));
    }
    catalog.push_str("];\n");
    fs::write(out.join("prepared_assets.rs"), catalog).expect("write asset catalog");
}

fn prepare(source: &Path) -> (Vec<u8>, Vec<u8>, Option<(&'static str, Vec<u8>)>, usize) {
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
    let base_color = parsed
        .materials()
        .find_map(|material| {
            material
                .pbr_metallic_roughness()
                .base_color_texture()
                .map(|info| info.texture().source().source())
        })
        .and_then(|image| load_supported_image(image, &buffers, base));
    let sampled_material = base_color.is_some();
    let (mut positions, mut normals, mut uvs, mut indices) = (
        Vec::<[f32; 3]>::new(),
        Vec::<[f32; 3]>::new(),
        Vec::<[f32; 2]>::new(),
        Vec::<u32>::new(),
    );
    for mesh in parsed.meshes() {
        for primitive in mesh.primitives() {
            assert_eq!(
                primitive.mode(),
                gltf::mesh::Mode::Triangles,
                "only triangle assets are supported"
            );
            let reader = primitive.reader(|b| Some(buffers[b.index()].as_slice()));
            let base_vertex = positions.len() as u32;
            let p: Vec<_> = reader.read_positions().expect("POSITION").collect();
            let local: Vec<u32> = reader
                .read_indices()
                .map(|v| v.into_u32().collect())
                .unwrap_or_else(|| (0..p.len() as u32).collect());
            let n: Vec<_> = reader
                .read_normals()
                .map(|v| v.collect())
                .unwrap_or_else(|| generated_normals(&p, &local));
            let uv: Vec<_> = reader
                .read_tex_coords(0)
                .map(|coords| coords.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0; 2]; p.len()]);
            assert_eq!(p.len(), n.len());
            assert_eq!(p.len(), uv.len());
            positions.extend(p);
            normals.extend(n);
            uvs.extend(uv);
            indices.extend(local.into_iter().map(|i| i + base_vertex));
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
    (vb, ib, base_color, vertex_stride)
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

fn generated_normals(p: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut out = vec![[0.0; 3]; p.len()];
    for tri in indices.chunks_exact(3) {
        let a = p[tri[0] as usize];
        let b = p[tri[1] as usize];
        let c = p[tri[2] as usize];
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let n = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        for &i in tri {
            for a in 0..3 {
                out[i as usize][a] += n[a];
            }
        }
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
