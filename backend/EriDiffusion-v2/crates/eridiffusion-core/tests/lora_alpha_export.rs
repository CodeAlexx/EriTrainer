use std::collections::HashMap;

use eridiffusion_core::{adapter::AdapterModule, lora::LoRALinear};
use flame_core::{global_cuda_device, DType};

#[test]
fn plain_lora_alpha_is_export_metadata_not_trainable_leaf() -> eridiffusion_core::Result<()> {
    let device = global_cuda_device();
    let lora = LoRALinear::new(4, 3, 2, 0.5, device, 123)?;

    let named = lora.named_tensors();
    assert_eq!(
        named.iter().map(|(suffix, _)| *suffix).collect::<Vec<_>>(),
        vec!["lora_A.weight", "lora_B.weight"]
    );
    assert_eq!(named.len(), lora.to_parameters().len());

    let exported = lora.export_tensors();
    assert_eq!(
        exported.iter().map(|(suffix, _)| *suffix).collect::<Vec<_>>(),
        vec!["lora_A.weight", "lora_B.weight", "alpha"]
    );
    let alpha = exported
        .iter()
        .find(|(suffix, _)| *suffix == "alpha")
        .expect("export_tensors should include alpha")
        .1
        .to_dtype(DType::F32)?
        .to_vec()?[0];
    assert!((alpha - 0.5).abs() < 1e-6, "alpha={alpha}");

    let mut saved = HashMap::new();
    lora.save_tensors("blocks.0.attn.to_q", &mut saved)?;
    assert!(saved.contains_key("blocks.0.attn.to_q.lora_A.weight"));
    assert!(saved.contains_key("blocks.0.attn.to_q.lora_B.weight"));
    let saved_alpha = saved
        .get("blocks.0.attn.to_q.alpha")
        .expect("save_tensors should include alpha")
        .to_dtype(DType::F32)?
        .to_vec()?[0];
    assert!((saved_alpha - 0.5).abs() < 1e-6, "saved_alpha={saved_alpha}");

    Ok(())
}
