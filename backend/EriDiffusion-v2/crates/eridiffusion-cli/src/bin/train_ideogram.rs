//! train_ideogram — Ideogram-4 LoRA trainer (the stage-9 capstone of the
//! parity-verified Ideogram-4 Rust vertical). Mirrors train_klein's loop on the
//! proven Ideogram pieces:
//!   - prepare_ideogram cache: {latent [1,128,h,w], text_embedding [1,L,53248],
//!     text_mask [1,L]} (stages 5-7).
//!   - IdeogramDit (stage 4/8): resident weights + per-block AutogradContext::
//!     checkpoint (bounds VRAM; connects LoRA grads), attach_block_loras (8a/8b).
//!   - flow-match predict (stage 1-3): add_noise -> packed -> MRoPE -> velocity.
//!   - loss = mean MSE in F32 (mean(), NOT mean_all() — grad-preserving).
//!   - lr via eridiffusion_core::training::levers::lr (the shared dispatch).
//!
//! Run (GPU):
//!   LIBTORCH=/home/alex/libs/libtorch LD_LIBRARY_PATH=$LIBTORCH/lib \
//!     cargo run --release --bin train_ideogram -- \
//!       --model <transformer.safetensors> --cache-dir <prepared> --steps 100

use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use eridiffusion_cli::{
    trainer_common,
    trainer_pipeline::{
        apply_autograd_v2_grad_policy, run_simple_step_trainer, SimpleOptimizerConfig,
        SimpleStepTrainer, SimpleTrainLoopConfig, StepLoss,
    },
};
use eridiffusion_core::models::ideogram;
use eridiffusion_core::models::ideogram_dit::IdeogramDit;
use eridiffusion_core::training::features::noise_modifiers;
use flame_core::{CudaDevice, DType, Parameter, Shape, Tensor};

const GH_GW_CH: usize = 128;

#[derive(Parser)]
#[command(name = "train_ideogram", about = "Ideogram-4 LoRA trainer")]
struct Args {
    /// Transformer safetensors (fp8) — diffusion_pytorch_model.safetensors.
    #[arg(long)]
    model: PathBuf,
    /// prepare_ideogram cache dir (<stem>.safetensors with latent/text_embedding).
    #[arg(long)]
    cache_dir: PathBuf,
    #[arg(long, default_value = "100")]
    steps: usize,
    #[arg(long, default_value = "16")]
    rank: usize,
    #[arg(long, default_value = "16.0")]
    lora_alpha: f32,
    #[arg(long, default_value = "1e-4")]
    lr: f32,
    #[arg(long, default_value = "0")]
    warmup_steps: usize,
    #[arg(long, default_value = "adamw")]
    optimizer: String,
    #[arg(long, default_value = "1.0")]
    max_grad_norm: f32,
    #[arg(long, default_value = "42")]
    seed: u64,
    #[arg(long, default_value = "output")]
    output_dir: PathBuf,
}

/// velocity = -out[:, NT:].reshape(gh,gw,128).permute(0,3,1,2)
fn velocity(out: &Tensor, nt: usize, gh: usize, gw: usize) -> anyhow::Result<Tensor> {
    let nimg = gh * gw;
    Ok(out
        .narrow(1, nt, nimg)?
        .contiguous()?
        .reshape(&[1, gh, gw, GH_GW_CH])?
        .permute(&[0, 3, 1, 2])?
        .contiguous()?
        .mul_scalar(-1.0)?)
}

struct IdeogramTrainer {
    args: Args,
    device: Arc<CudaDevice>,
    dit: IdeogramDit,
    params: Vec<Parameter>,
    cache_files: Vec<PathBuf>,
}

impl IdeogramTrainer {
    fn load(args: Args, device: Arc<CudaDevice>) -> anyhow::Result<Self> {
        let cache_files = trainer_common::list_cache_safetensors(&args.cache_dir)?;
        log::info!("[1/3] {} cached samples", cache_files.len());

        log::info!(
            "[2/3] loading Ideogram-4 transformer (resident) + LoRA rank {}",
            args.rank
        );
        let mut dit = IdeogramDit::load(&args.model.to_string_lossy(), device.clone())?;
        let mut params = dit.attach_block_loras(args.rank, args.lora_alpha)?;
        // v2 grad policy (BF16 grads stay BF16) mirrors train_klein.
        apply_autograd_v2_grad_policy(&mut params, false, "params");
        log::info!("    {} trainable LoRA params", params.len());

        Ok(Self {
            args,
            device,
            dit,
            params,
            cache_files,
        })
    }

    fn dataset_len(&self) -> usize {
        self.cache_files.len()
    }
}

impl SimpleStepTrainer for IdeogramTrainer {
    fn trainable_parameters(&self) -> &[Parameter] {
        &self.params
    }

    fn step_loss(&mut self, step: usize) -> anyhow::Result<StepLoss> {
        let cache_path = &self.cache_files[step % self.cache_files.len()];
        let sample = flame_core::serialization::load_file(cache_path, &self.device)?;
        let latent = sample
            .get("latent")
            .ok_or_else(|| anyhow::anyhow!("cache missing latent"))?
            .to_dtype(DType::F32)?; // [1,128,gh,gw]
        let text_emb = sample
            .get("text_embedding")
            .ok_or_else(|| anyhow::anyhow!("cache missing text_embedding"))?;
        let ld = latent.shape().dims().to_vec();
        let (gh, gw) = (ld[2], ld[3]);

        // flow-match: t ~ logit-normal (sigmoid of seeded normal), as in SD3/flow.
        let u = noise_modifiers::randn_f32(Shape::from_dims(&[1]), self.device.clone())?
            .to_vec_f32()?[0];
        let t = 1.0 / (1.0 + (-u).exp()); // sigmoid → (0,1)
        let noise = noise_modifiers::randn_f32(latent.shape().clone(), self.device.clone())?;
        let noisy = ideogram::add_noise(&latent, &noise, t)?;
        let target = ideogram::flow_target(&noise, &latent)?; // noise - clean

        let packed =
            ideogram::build_packed_inputs(&noisy, text_emb, gh, gw, self.device.clone())?;
        let (cos, sin) = ideogram::build_mrope(
            &packed.position_ids,
            ideogram::HEAD_DIM,
            ideogram::MROPE_SECTION,
            ideogram::MROPE_THETA,
            self.device.clone(),
        )?;
        let seq = packed.x.shape().dims()[1];
        let nt = seq - gh * gw;
        let model_t =
            Tensor::from_vec(vec![1.0 - t], Shape::from_dims(&[1]), self.device.clone())?;
        let x_bf = packed.x.to_dtype(DType::BF16)?;
        let llm_bf = packed.llm_full.to_dtype(DType::BF16)?;

        let out = self.dit.forward(
            &x_bf,
            &llm_bf,
            &model_t,
            &packed.indicator,
            &cos,
            &sin,
            None,
            0,
        )?;
        let vel = velocity(&out, nt, gh, gw)?;
        let loss = vel.sub(&target)?.square()?.mean()?; // mean() preserves grad
        let loss_val = loss.to_vec_f32()?[0];
        Ok(StepLoss::with_value(loss, loss_val))
    }

    fn save_final(&mut self) -> anyhow::Result<()> {
        // Save the LoRA in ai-toolkit Ideogram-4 key format (loadable by
        // ai-toolkit / serenitymojo ideogram4_generate_lora).
        let lora_out = self.dit.export_lora_aitoolkit()?;
        let out_path = self.args.output_dir.join("ideogram4_lora.safetensors");
        flame_core::serialization::save_file(&lora_out, &out_path)?;
        log::info!(
            "Saved LoRA ({} tensors, ai-toolkit keys) → {}",
            lora_out.len(),
            out_path.display()
        );
        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    trainer_common::init_logging();
    let args = Args::parse();
    trainer_common::ensure_output_dir(&args.output_dir)?;
    let device = trainer_common::init_cuda();

    let steps = args.steps;
    let warmup_steps = args.warmup_steps;
    let optimizer_config =
        SimpleOptimizerConfig::from_optimizer_name(&args.optimizer, args.lr, args.max_grad_norm)?;

    let mut trainer = IdeogramTrainer::load(args, device)?;
    let loop_config = SimpleTrainLoopConfig::fresh(
        "Ideogram-4",
        steps,
        trainer.dataset_len(),
        1,
        warmup_steps,
    );

    log::info!("[3/3] training {steps} steps");
    run_simple_step_trainer(&mut trainer, loop_config, optimizer_config, None)
}
