#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainerSpec {
    pub id: &'static str,
    pub train_bin: &'static str,
    pub prepare_bin: Option<&'static str>,
    pub sample_bin: Option<&'static str>,
    pub aliases: &'static [&'static str],
    pub local_sampler: bool,
    pub note: &'static str,
}

impl TrainerSpec {
    pub fn matches(&self, id: &str) -> bool {
        let needle = normalize_model_id(id);
        self.id == needle
            || self
                .aliases
                .iter()
                .any(|alias| normalize_model_id(alias) == needle)
    }
}

pub const TRAINERS: &[TrainerSpec] = &[
    TrainerSpec {
        id: "acestep",
        train_bin: "train_acestep",
        prepare_bin: None,
        sample_bin: None,
        aliases: &["ace_step", "ACE_STEP"],
        local_sampler: false,
        note: "trainer only; no EDv2 sampler yet",
    },
    TrainerSpec {
        id: "anima",
        train_bin: "train_anima",
        prepare_bin: Some("prepare_anima"),
        sample_bin: Some("sample_anima"),
        aliases: &["ANIMA"],
        local_sampler: true,
        note: "local sampler module",
    },
    TrainerSpec {
        id: "asymflow",
        train_bin: "train_asymflow",
        prepare_bin: Some("prepare_asymflow"),
        sample_bin: None,
        aliases: &["ASYMFLOW"],
        local_sampler: false,
        note: "AsymFlow helpers lifted into eridiffusion-core",
    },
    TrainerSpec {
        id: "chroma",
        train_bin: "train_chroma",
        prepare_bin: Some("prepare_chroma"),
        sample_bin: Some("sample_chroma"),
        aliases: &["chroma_1", "CHROMA_1"],
        local_sampler: true,
        note: "local sampler module",
    },
    TrainerSpec {
        id: "ernie",
        train_bin: "train_ernie",
        prepare_bin: Some("prepare_ernie"),
        sample_bin: Some("sample_ernie"),
        aliases: &["ernie_image", "ERNIE_IMAGE"],
        local_sampler: true,
        note: "local sampler module",
    },
    TrainerSpec {
        id: "flux",
        train_bin: "train_flux",
        prepare_bin: Some("prepare_flux"),
        sample_bin: Some("sample_flux"),
        aliases: &["flux_1_dev", "FLUX_1_DEV"],
        local_sampler: true,
        note: "local sampler module",
    },
    TrainerSpec {
        id: "hidream_o1",
        train_bin: "train_hidream_o1",
        prepare_bin: Some("prepare_hidream_o1"),
        sample_bin: None,
        aliases: &["hidream", "HIDREAM_O1"],
        local_sampler: false,
        note: "model helpers lifted into eridiffusion-core; sampler still pending",
    },
    TrainerSpec {
        id: "ideogram4",
        train_bin: "train_ideogram",
        prepare_bin: Some("prepare_ideogram"),
        sample_bin: None,
        aliases: &["ideogram", "IDEOGRAM_4"],
        local_sampler: false,
        note: "trainer only in EDv2",
    },
    TrainerSpec {
        id: "klein",
        train_bin: "train_klein",
        prepare_bin: Some("prepare_klein"),
        sample_bin: Some("sample_klein"),
        aliases: &["flux_2", "FLUX_2"],
        local_sampler: true,
        note: "reference vertical",
    },
    TrainerSpec {
        id: "l2p",
        train_bin: "train_l2p",
        prepare_bin: Some("prepare_l2p"),
        sample_bin: None,
        aliases: &["z_image_l2p", "Z_IMAGE_L2P"],
        local_sampler: false,
        note: "trainer includes local L2P sampling helpers",
    },
    TrainerSpec {
        id: "ltx2",
        train_bin: "train_ltx2",
        prepare_bin: Some("prepare_ltx2"),
        sample_bin: Some("sample_ltx2"),
        aliases: &["ltx_2_video", "LTX_2_VIDEO"],
        local_sampler: true,
        note: "local sampler module",
    },
    TrainerSpec {
        id: "qwenimage",
        train_bin: "train_qwenimage",
        prepare_bin: Some("prepare_qwenimage"),
        sample_bin: Some("sample_qwenimage"),
        aliases: &["qwen_image", "QWEN_IMAGE"],
        local_sampler: true,
        note: "local sampler module",
    },
    TrainerSpec {
        id: "sd35",
        train_bin: "train_sd35",
        prepare_bin: Some("prepare_sd35"),
        sample_bin: Some("sample_sd35"),
        aliases: &["stable_diffusion_35", "STABLE_DIFFUSION_35"],
        local_sampler: true,
        note: "local sampler module",
    },
    TrainerSpec {
        id: "sdxl",
        train_bin: "train_sdxl",
        prepare_bin: Some("prepare_sdxl"),
        sample_bin: Some("sample_sdxl"),
        aliases: &["stable_diffusion_xl_10_base", "STABLE_DIFFUSION_XL_10_BASE"],
        local_sampler: true,
        note: "local sampler module",
    },
    TrainerSpec {
        id: "slider_klein",
        train_bin: "train_slider_klein",
        prepare_bin: Some("prepare_klein"),
        sample_bin: Some("sample_klein"),
        aliases: &["SLIDER_KLEIN"],
        local_sampler: true,
        note: "Klein slider trainer",
    },
    TrainerSpec {
        id: "u1",
        train_bin: "train_u1",
        prepare_bin: None,
        sample_bin: Some("sample_u1"),
        aliases: &["sensenova_u1", "SENSENOVA_U1"],
        local_sampler: true,
        note: "folder-mode trainer",
    },
    TrainerSpec {
        id: "wan22",
        train_bin: "train_wan22",
        prepare_bin: Some("prepare_wan22"),
        sample_bin: Some("sample_wan22"),
        aliases: &["wan_22_video", "WAN_22_VIDEO"],
        local_sampler: true,
        note: "local sampler module",
    },
    TrainerSpec {
        id: "zimage",
        train_bin: "train_zimage",
        prepare_bin: Some("prepare_zimage"),
        sample_bin: Some("sample_zimage"),
        aliases: &["z_image", "Z_IMAGE"],
        local_sampler: true,
        note: "local sampler module",
    },
];

pub fn normalize_model_id(id: &str) -> String {
    id.trim()
        .to_ascii_lowercase()
        .replace('-', "_")
        .replace('.', "")
        .replace(' ', "_")
}

pub fn find_trainer(id: &str) -> Option<&'static TrainerSpec> {
    TRAINERS.iter().find(|spec| spec.matches(id))
}

pub fn trainer_ids() -> impl Iterator<Item = &'static str> {
    TRAINERS.iter().map(|spec| spec.id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_unique_canonical_ids() {
        let mut ids: Vec<_> = trainer_ids().collect();
        let original_len = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), original_len);
    }

    #[test]
    fn aliases_match_normalized_ids() {
        assert_eq!(find_trainer("flux-2").unwrap().id, "klein");
        assert_eq!(find_trainer("HiDream").unwrap().id, "hidream_o1");
        assert_eq!(find_trainer("stable diffusion xl 10 base").unwrap().id, "sdxl");
    }
}
