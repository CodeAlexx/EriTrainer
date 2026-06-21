use anyhow::Context;
use std::path::Path;

pub fn ensure_output_dir(path: &Path) -> anyhow::Result<()> {
    let _timing = crate::trainer_metrics::phase("preflight.ensure_output_dir");
    std::fs::create_dir_all(path)
        .with_context(|| format!("create output directory '{}'", path.display()))
}

pub fn require_file(path: &Path, label: &str) -> anyhow::Result<()> {
    if path.is_file() {
        Ok(())
    } else {
        anyhow::bail!("{label} file does not exist: '{}'", path.display())
    }
}

pub fn require_dir(path: &Path, label: &str) -> anyhow::Result<()> {
    if path.is_dir() {
        Ok(())
    } else {
        anyhow::bail!("{label} directory does not exist: '{}'", path.display())
    }
}

pub fn require_file_or_dir(path: &Path, label: &str) -> anyhow::Result<()> {
    if path.is_file() || path.is_dir() {
        Ok(())
    } else {
        anyhow::bail!("{label} path does not exist: '{}'", path.display())
    }
}

pub fn require_positive_steps(steps: usize, flag: &str) -> anyhow::Result<()> {
    if steps > 0 {
        Ok(())
    } else {
        anyhow::bail!("{flag} must be > 0")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn require_positive_steps_rejects_zero() {
        assert!(require_positive_steps(0, "--steps").is_err());
        assert!(require_positive_steps(1, "--steps").is_ok());
    }
}
