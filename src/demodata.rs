//! Source assets owned and embedded by Picasso Example.
//!
//! These exact bytes are passed to Picasso's public embedded-asset API. This
//! module deliberately knows nothing about Picasso's internal storage.

pub struct DemoAsset {
    pub name: &'static str,
    pub bytes: &'static [u8],
}

pub static ASSETS: [DemoAsset; 5] = [
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
    DemoAsset {
        name: "DamagedHelmet",
        bytes: include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/Assets/DamagedHelmet/DamagedHelmet.glb"
        )),
    },
];
