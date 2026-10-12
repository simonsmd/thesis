//! The build and machine a result was measured on.

use std::{fs, time::SystemTime};

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct Build {
    pub git_rev: &'static str,
    /// Uncommitted changes in `parser/` or `harness/` when the binary was built.
    pub git_dirty: bool,
    pub rustc: &'static str,
    pub rustflags: &'static str,
    pub profile: &'static str,
}

pub fn build() -> Build {
    Build {
        git_rev: env!("HARNESS_GIT_REV"),
        git_dirty: env!("HARNESS_GIT_DIRTY") == "true",
        rustc: env!("HARNESS_RUSTC"),
        rustflags: env!("HARNESS_RUSTFLAGS"),
        profile: env!("HARNESS_PROFILE"),
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct System {
    pub hostname: Option<String>,
    pub kernel: Option<String>,
    pub cpu_model: Option<String>,
    /// Frequency governor of the pinned CPU (or CPU 0).
    pub governor: Option<String>,
    /// `intel_pstate/no_turbo` or the inverse of `cpufreq/boost`.
    pub turbo: Option<bool>,
    pub perf_event_paranoid: Option<i32>,
}

pub fn system(cpu: Option<usize>) -> System {
    let read = |path: &str| fs::read_to_string(path).ok().map(|s| s.trim().to_owned());
    let cpu = cpu.unwrap_or(0);
    let turbo = read("/sys/devices/system/cpu/intel_pstate/no_turbo")
        .map(|v| v == "0")
        .or_else(|| read("/sys/devices/system/cpu/cpufreq/boost").map(|v| v == "1"));
    System {
        hostname: read("/proc/sys/kernel/hostname"),
        kernel: read("/proc/sys/kernel/osrelease"),
        cpu_model: read("/proc/cpuinfo").and_then(|info| {
            info.lines()
                .find_map(|l| l.strip_prefix("model name")?.split_once(':').map(|(_, v)| v.trim().to_owned()))
        }),
        governor: read(&format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")),
        turbo,
        perf_event_paranoid: read("/proc/sys/kernel/perf_event_paranoid").and_then(|v| v.parse().ok()),
    }
}

/// The current UTC time as `2026-10-12T13:04:05Z`.
pub fn timestamp() -> String {
    let (date, time) = now();
    format!("{date}T{time}Z")
}

/// The current UTC time as `20261012-130405`, for file names.
pub fn compact_timestamp() -> String {
    let (date, time) = now();
    format!("{}-{}", date.replace('-', ""), time.replace(':', ""))
}

fn now() -> (String, String) {
    let secs = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0, |d| d.as_secs()) as i64;
    let (days, rem) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    // civil_from_days, from Howard Hinnant's date algorithms
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + (month <= 2) as i64;
    (
        format!("{year:04}-{month:02}-{day:02}"),
        format!("{:02}:{:02}:{:02}", rem / 3600, rem % 3600 / 60, rem % 60),
    )
}
