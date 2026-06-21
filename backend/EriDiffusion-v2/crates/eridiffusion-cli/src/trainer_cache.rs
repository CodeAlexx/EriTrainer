use anyhow::Context;
use std::path::{Path, PathBuf};

pub fn list_files_with_extensions(
    dir: &Path,
    extensions: &[&str],
    label: &str,
) -> anyhow::Result<Vec<PathBuf>> {
    let _timing = crate::trainer_metrics::phase("cache.list_files");
    crate::trainer_preflight::require_dir(dir, label)?;

    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("read {label} directory '{}'", dir.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .map_or(false, |ext| extensions.iter().any(|wanted| ext == *wanted))
        })
        .collect();
    files.sort();

    if files.is_empty() {
        let suffixes = extensions.join(", ");
        anyhow::bail!(
            "no {label} files with extension(s) [{}] in '{}'",
            suffixes,
            dir.display()
        );
    }

    Ok(files)
}

pub fn list_files_with_extensions_or_empty(
    dir: &Path,
    extensions: &[&str],
    label: &str,
) -> anyhow::Result<Vec<PathBuf>> {
    let _timing = crate::trainer_metrics::phase("cache.list_files_or_empty");
    crate::trainer_preflight::require_dir(dir, label)?;

    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("read {label} directory '{}'", dir.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .map_or(false, |ext| extensions.iter().any(|wanted| ext == *wanted))
        })
        .collect();
    files.sort();
    Ok(files)
}

pub fn list_safetensors(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    list_files_with_extensions(dir, &["safetensors"], "cache")
}

pub fn list_safetensors_or_empty(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    list_files_with_extensions_or_empty(dir, &["safetensors"], "cache")
}

pub fn collect_safetensor_shards(path: &Path, label: &str) -> anyhow::Result<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }

    let label = if label.is_empty() { "safetensors" } else { label };
    list_files_with_extensions(path, &["safetensors"], label)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_files_with_extensions_sorts_and_filters() {
        let root = std::env::temp_dir().join(format!(
            "eritrainer-cache-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("b.safetensors"), b"").unwrap();
        std::fs::write(root.join("a.safetensors"), b"").unwrap();
        std::fs::write(root.join("ignored.txt"), b"").unwrap();

        let files = list_safetensors(&root).unwrap();
        let names: Vec<_> = files
            .iter()
            .map(|path| path.file_name().unwrap().to_str().unwrap().to_owned())
            .collect();

        assert_eq!(names, ["a.safetensors", "b.safetensors"]);
        let _ = std::fs::remove_dir_all(root);
    }
}
