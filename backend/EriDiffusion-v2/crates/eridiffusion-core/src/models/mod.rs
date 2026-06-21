use crate::Result;
use flame_core::{parameter::Parameter, Tensor};

pub mod acestep;
pub mod anima;
pub mod chroma;
pub mod ernie;
pub mod flux;
pub mod asymflux2;
pub mod hidream_o1;
pub mod ideogram;
pub mod ideogram_dit;
pub mod klein;
pub mod l2p;
pub mod ltx2;
pub mod qwenimage;
pub mod sd35;
pub mod sdxl;
pub mod sensenova_u1;
pub mod sensenova_u1_lora;
pub mod wan22;
pub mod wan22_fwd;
pub mod zimage;
pub use acestep::AceStepLoRAModel;
pub use anima::AnimaModel;
pub use chroma::ChromaTrainingModel;
pub use ernie::ErnieModel;
pub use flux::FluxModel;
pub use hidream_o1::{
    HiDreamO1Config, HiDreamO1Model, HiDreamO1WeightLoader, LoraRegistry as HiDreamLoraRegistry,
    MRopePositions as HiDreamMRopePositions,
};
pub use klein::KleinModel;
pub use ltx2::Ltx2Model;
pub use qwenimage::QwenImageTrainingModel;
pub use sd35::SD35Model;
pub use sdxl::SDXLModel;
pub use sensenova_u1::{SenseNovaU1, SenseNovaU1Config, T2IStepOutput as U1T2IStepOutput};
pub use wan22::{
    LoraTarget as Wan22LoraTarget, Wan22Config, Wan22LoraBundle, Wan22Model, Wan22Variant,
};
pub use zimage::ZImageModel;

pub trait TrainableModel: Send + Sync {
    /// `&mut self` so impls can do per-layer weight streaming (BlockOffloader)
    /// inside the forward pass without resorting to interior mutability.
    fn forward(
        &mut self,
        noisy: &Tensor,
        timestep: &Tensor,
        context: &[Tensor],
        pooled: Option<&Tensor>,
    ) -> Result<Tensor>;
    fn parameters(&self) -> Vec<Parameter>;
    fn post_optimizer_step(&mut self);
    fn save_weights(&self, path: &str) -> Result<()>;
    fn load_weights(&mut self, path: &str) -> Result<()>;
}
