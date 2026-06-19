//! GPU/CPU/RAM sampling for the live-status rail (nvidia-smi + /proc).
//!
//! Sets the static names (`rt.gpu_name`, `rt.gpu_driver`, `rt.cpu_name`) once
//! and refreshes the live numbers (`gpu_util`, `vram_gb`, `vram_total_gb`,
//! `temp_c`, `cpu_util`, `ram_gb`, `ram_total_gb`) each pass. Throttled to
//! ~1 Hz. Any missing tool/file leaves the field at 0 / "unavailable" — never
//! a fabricated number (honesty discipline).

use std::process::Command;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::runtime::Runtime;

/// Snapshot of `/proc/stat` aggregate CPU jiffies, used to derive utilization
/// from the delta between two samples.
#[derive(Clone, Copy)]
struct CpuSample {
    idle: u64,
    total: u64,
}

struct SysState {
    last_refresh: Option<Instant>,
    prev_cpu: Option<CpuSample>,
}

fn state() -> &'static Mutex<SysState> {
    static STATE: OnceLock<Mutex<SysState>> = OnceLock::new();
    STATE.get_or_init(|| {
        Mutex::new(SysState {
            last_refresh: None,
            prev_cpu: None,
        })
    })
}

const REFRESH_INTERVAL: Duration = Duration::from_millis(1000);

pub fn refresh(rt: &mut Runtime) {
    // Throttle to ~1 Hz; cheap early-out on the common per-frame call.
    let mut st = match state().lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    if let Some(last) = st.last_refresh {
        if last.elapsed() < REFRESH_INTERVAL {
            return;
        }
    }
    st.last_refresh = Some(Instant::now());

    // CPU util needs the previous jiffies sample held across calls.
    let prev_cpu = st.prev_cpu;
    let (cpu_util, new_cpu) = sample_cpu_util(prev_cpu);
    st.prev_cpu = new_cpu;
    drop(st);

    refresh_gpu(rt);
    refresh_mem(rt);
    if let Some(util) = cpu_util {
        rt.live.cpu_util = util;
    }
    if rt.cpu_name.is_empty() {
        if let Some(name) = read_cpu_name() {
            rt.cpu_name = name;
        }
    }
}

/// One nvidia-smi query for the first GPU. Leaves fields untouched if the tool
/// is absent or the query fails.
fn refresh_gpu(rt: &mut Runtime) {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output();

    let out = match out {
        Ok(o) if o.status.success() => o,
        _ => return,
    };
    let text = String::from_utf8_lossy(&out.stdout);
    let first = match text.lines().next() {
        Some(l) if !l.trim().is_empty() => l,
        _ => return,
    };

    // `name, driver, util, mem_used(MiB), mem_total(MiB), temp`
    let f: Vec<&str> = first.split(',').map(str::trim).collect();
    if f.len() < 6 {
        return;
    }
    if rt.gpu_name.is_empty() && !f[0].is_empty() {
        rt.gpu_name = f[0].to_string();
    }
    if rt.gpu_driver.is_empty() && !f[1].is_empty() {
        rt.gpu_driver = f[1].to_string();
    }
    if let Ok(v) = f[2].parse::<f32>() {
        rt.live.gpu_util = v;
    }
    if let Ok(mib) = f[3].parse::<f32>() {
        rt.live.vram_gb = mib / 1024.0;
    }
    if let Ok(mib) = f[4].parse::<f32>() {
        rt.live.vram_total_gb = mib / 1024.0;
    }
    if let Ok(v) = f[5].parse::<i32>() {
        rt.live.temp_c = v;
    }
}

/// RAM total/used (GB) from `/proc/meminfo`. Used = MemTotal - MemAvailable.
fn refresh_mem(rt: &mut Runtime) {
    let text = match std::fs::read_to_string("/proc/meminfo") {
        Ok(t) => t,
        Err(_) => return,
    };
    let mut total_kb: Option<u64> = None;
    let mut avail_kb: Option<u64> = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            total_kb = parse_meminfo_kb(rest);
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            avail_kb = parse_meminfo_kb(rest);
        }
        if total_kb.is_some() && avail_kb.is_some() {
            break;
        }
    }
    if let Some(total) = total_kb {
        let total_gb = total as f32 / (1024.0 * 1024.0);
        rt.live.ram_total_gb = total_gb;
        if let Some(avail) = avail_kb {
            let used_gb = (total.saturating_sub(avail)) as f32 / (1024.0 * 1024.0);
            rt.live.ram_gb = used_gb;
        }
    }
}

/// `<number> kB` → kB value.
fn parse_meminfo_kb(rest: &str) -> Option<u64> {
    rest.trim()
        .split_whitespace()
        .next()
        .and_then(|n| n.parse::<u64>().ok())
}

/// Aggregate CPU utilization (%) from `/proc/stat`, computed against the prior
/// sample. Returns `(util, this_sample)`; util is None on the first call (no
/// delta yet) or if the file is unreadable.
fn sample_cpu_util(prev: Option<CpuSample>) -> (Option<f32>, Option<CpuSample>) {
    let text = match std::fs::read_to_string("/proc/stat") {
        Ok(t) => t,
        Err(_) => return (None, prev),
    };
    let line = match text.lines().find(|l| l.starts_with("cpu ")) {
        Some(l) => l,
        None => return (None, prev),
    };

    // Fields: user nice system idle iowait irq softirq steal guest guest_nice
    let vals: Vec<u64> = line
        .split_whitespace()
        .skip(1)
        .filter_map(|v| v.parse::<u64>().ok())
        .collect();
    if vals.len() < 4 {
        return (None, prev);
    }
    let idle = vals[3] + vals.get(4).copied().unwrap_or(0); // idle + iowait
    let total: u64 = vals.iter().sum();
    let cur = CpuSample { idle, total };

    let util = prev.and_then(|p| {
        let dt = cur.total.saturating_sub(p.total);
        let di = cur.idle.saturating_sub(p.idle);
        if dt == 0 {
            None
        } else {
            Some((1.0 - di as f32 / dt as f32) * 100.0)
        }
    });
    (util, Some(cur))
}

/// First "model name" from `/proc/cpuinfo`.
fn read_cpu_name() -> Option<String> {
    let text = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("model name") {
            if let Some((_, name)) = rest.split_once(':') {
                let name = name.trim();
                if !name.is_empty() {
                    return Some(name.to_string());
                }
            }
        }
    }
    None
}
