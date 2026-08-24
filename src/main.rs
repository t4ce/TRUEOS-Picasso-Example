#![no_std]

use trueos::{
    clock,
    logl::{self, level},
    vsys,
};
use trueos_picasso_example::{GeometryProbe, HELMET_INDEX_COUNT, HELMET_VERTEX_COUNT};

fn main() {
    let mut probe = match GeometryProbe::open() {
        Ok(probe) => probe,
        Err(error) => {
            logl::log(
                level::ERROR,
                format_args!("PicassoExample: geometry probe failed: {error:?}"),
            );
            return;
        }
    };

    logl::log(
        level::INFO,
        format_args!(
            "PicassoExample: retained DamagedHelmet instances + static RGB lines submitted and retired: vertices={} indices={} timeline={}",
            HELMET_VERTEX_COUNT,
            HELMET_INDEX_COUNT,
            probe.timeline(),
        ),
    );

    let start = clock::monotonic_millis();
    let mut presentation_logged = false;
    loop {
        vsys::poll_once();
        if let Err(error) = probe.render_frame(clock::monotonic_millis().saturating_sub(start)) {
            logl::log(
                level::ERROR,
                format_args!("PicassoExample: retained animation failed: {error:?}"),
            );
            return;
        }
        if !presentation_logged {
            match probe.take_first_presentation() {
                Ok(true) => {
                    logl::log(
                        level::INFO,
                        "PicassoExample: retained transform + static line frame crossed UI4 SURFLIVE",
                    );
                    presentation_logged = true;
                }
                Ok(false) => {}
                Err(error) => {
                    logl::log(
                        level::ERROR,
                        format_args!("PicassoExample: presentation probe failed: {error:?}"),
                    );
                    return;
                }
            }
        }
        vsys::sleep_ms(16);
    }
}
