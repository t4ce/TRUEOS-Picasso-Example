#![no_std]

use picasso_example::prepared_triangle;
use trueos::logl::{self, level};
use trueos_picasso::GpuAddress;

fn main() {
    // Placeholder addresses until Dealer resolves the imported asset IDs into
    // the shared DDR allocation visible to both CPU and GPU.
    let primitive = prepared_triangle(GpuAddress(0), GpuAddress(0));

    logl::log(
        level::INFO,
        format_args!(
            "PicassoExample: native TRUEOS primitive ready: vertices={} indices={}",
            primitive.vertex_count, primitive.index_count,
        ),
    );
}
