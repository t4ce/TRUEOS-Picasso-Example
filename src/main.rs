#![no_std]

use trueos::{
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
            "PicassoExample: DamagedHelmet + RGB line-list batch submitted and retired: vertices={} indices={} timeline={}",
            HELMET_VERTEX_COUNT,
            HELMET_INDEX_COUNT,
            probe.timeline(),
        ),
    );

    let mut presentation_logged = false;
    loop {
        vsys::poll_once();
        if !presentation_logged {
            match probe.take_first_presentation() {
                Ok(true) => {
                    logl::log(
                        level::INFO,
                        "PicassoExample: mixed triangle/line geometry crossed UI4 SURFLIVE",
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
