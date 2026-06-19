//! Shell layout metrics (3-column). Mirrors the Mojo `trainer_shell_metrics`:
//! a fixed left nav (~240) and right status rail (~330) that scale modestly
//! with window width, with the center column taking the remainder.
//!
//! `main.rs` references `NAV_W` / `RAIL_W` directly for its side panels, so
//! those names stay valid as the default (un-scaled) widths. `nav_width` /
//! `rail_width` give the responsive values when a caller has the window width.

/// Default / minimum left-nav width.
pub const NAV_W: f32 = 240.0;
/// Default / minimum right-rail width.
pub const RAIL_W: f32 = 330.0;

/// Below this width we keep the base nav/rail widths; above it we let them grow
/// gently so the dense center column does not hog very wide windows.
#[allow(dead_code)] // consumed once main.rs opts into responsive panel widths
const SCALE_BASELINE_W: f32 = 1480.0;

/// Responsive left-nav width for a given window width (clamped to a sane band).
#[allow(dead_code)] // available for main.rs; it currently uses the NAV_W const
pub fn nav_width(win_w: f32) -> f32 {
    if win_w <= SCALE_BASELINE_W {
        NAV_W
    } else {
        let extra = (win_w - SCALE_BASELINE_W) * 0.04;
        (NAV_W + extra).min(300.0)
    }
}

/// Responsive right-rail width for a given window width (clamped to a sane band).
#[allow(dead_code)] // available for main.rs; it currently uses the RAIL_W const
pub fn rail_width(win_w: f32) -> f32 {
    if win_w <= SCALE_BASELINE_W {
        RAIL_W
    } else {
        let extra = (win_w - SCALE_BASELINE_W) * 0.06;
        (RAIL_W + extra).min(400.0)
    }
}
