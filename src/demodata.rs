// Additional demo source assets owned by the example. The importer is supplied
// by the caller so this module never opens or owns a database.

pub struct DemoAsset {
    pub name: &'static str,
    pub bytes: &'static [u8],
}

pub fn import_into(mut importer: impl FnMut(&str, &[u8])) {
    for asset in &ASSETS {
        importer(asset.name, asset.bytes);
    }
}

pub static ASSETS: [DemoAsset; 4] = [
    DemoAsset {
        name: "Triangle",
        bytes: include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/Assets/Triangle/Triangle.gltf"
        )),
    },
    DemoAsset {
        name: "BoxInterleaved",
        bytes: include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/Assets/BoxInterleaved/BoxInterleaved.glb"
        )),
    },
    DemoAsset {
        name: "SimpleSparseAccessor",
        bytes: include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/Assets/SimpleSparseAccessor/SimpleSparseAccessor.gltf"
        )),
    },
    DemoAsset {
        name: "RiggedSimple",
        bytes: include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/Assets/RiggedSimple/RiggedSimple.glb"
        )),
    },
];
