//! Design tokens for the Rust Trainer UI handoff (dark default + light alt).
//! Exact values from the design README. `current()` returns the active set; the
//! active theme is a process-global flag (egui is single-threaded).

use eframe::egui::Color32;
use std::sync::atomic::{AtomicU8, Ordering};

static ACTIVE: AtomicU8 = AtomicU8::new(0); // 0 = dark, 1 = light

#[derive(Clone, Copy)]
#[allow(dead_code)] // full design palette; a few tones (ok/err/accent_ink) are not consumed yet
pub struct Tokens {
    pub bg: Color32,
    pub bg_2: Color32,
    pub bg_3: Color32,
    pub panel: Color32,
    pub line: Color32,
    pub line_2: Color32,
    pub ink: Color32,
    pub ink_dim: Color32,
    pub ink_mute: Color32,
    pub accent: Color32,
    pub accent_ink: Color32,
    pub accent_soft: Color32,
    pub ok: Color32,
    pub warn: Color32,
    pub err: Color32,
}

const fn rgb(r: u8, g: u8, b: u8) -> Color32 {
    Color32::from_rgb(r, g, b)
}

/// Dark theme (default).
pub const fn dark() -> Tokens {
    Tokens {
        bg: rgb(20, 17, 15),         // #14110F
        bg_2: rgb(27, 24, 21),       // #1B1815
        bg_3: rgb(34, 30, 26),       // #221E1A
        panel: rgb(27, 24, 21),      // #1B1815
        line: rgb(42, 37, 32),       // #2A2520
        line_2: rgb(51, 45, 39),     // #332D27
        ink: rgb(237, 230, 220),     // #EDE6DC
        ink_dim: rgb(179, 168, 155), // #B3A89B
        ink_mute: rgb(122, 111, 99), // #7A6F63
        accent: rgb(230, 154, 92),   // #E69A5C
        accent_ink: rgb(22, 12, 3),  // #160C03
        accent_soft: Color32::from_rgba_premultiplied(32, 22, 13, 36), // accent @ ~14%
        ok: rgb(110, 195, 148),      // #6EC394
        warn: rgb(215, 185, 94),     // #D7B95E
        err: rgb(217, 106, 84),      // #D96A54
    }
}

/// Light theme (for the toggle).
pub const fn light() -> Tokens {
    Tokens {
        bg: rgb(246, 242, 236),      // #F6F2EC
        bg_2: rgb(255, 255, 255),    // #FFFFFF
        bg_3: rgb(239, 233, 224),    // #EFE9E0
        panel: rgb(255, 255, 255),   // #FFFFFF
        line: rgb(228, 221, 210),    // #E4DDD2
        line_2: rgb(212, 204, 190),  // #D4CCBE
        ink: rgb(26, 22, 19),        // #1A1613
        ink_dim: rgb(92, 83, 74),    // #5C534A
        ink_mute: rgb(138, 128, 117),// #8A8075
        accent: rgb(201, 122, 61),   // #C97A3D
        accent_ink: rgb(255, 248, 240),
        accent_soft: Color32::from_rgba_premultiplied(40, 24, 12, 30),
        ok: rgb(58, 150, 100),
        warn: rgb(166, 132, 40),
        err: rgb(189, 74, 56),
    }
}

pub fn set_active(is_light: bool) {
    ACTIVE.store(u8::from(is_light), Ordering::Relaxed);
}

pub fn is_light() -> bool {
    ACTIVE.load(Ordering::Relaxed) == 1
}

/// The active token set — call from any widget.
pub fn current() -> Tokens {
    if is_light() {
        light()
    } else {
        dark()
    }
}
