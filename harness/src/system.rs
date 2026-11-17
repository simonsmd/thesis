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
    /// Cargo features of the harness, e.g. `csveee`.
    pub features: Vec<&'static str>,
}

pub fn build() -> Build {
    Build {
        git_rev: env!("HARNESS_GIT_REV"),
        git_dirty: env!("HARNESS_GIT_DIRTY") == "true",
        rustc: env!("HARNESS_RUSTC"),
        rustflags: env!("HARNESS_RUSTFLAGS"),
        profile: env!("HARNESS_PROFILE"),
        features: env!("HARNESS_FEATURES")
            .split(',')
            .filter(|f| !f.is_empty())
            .collect(),
    }
}

/// The machine a result was measured on. A value that cannot be read is `null`.
#[derive(Debug, Clone, Serialize)]
pub struct System {
    pub hostname: Option<String>,
    /// The name of the installed system, e.g. `Ubuntu 24.04.1 LTS`.
    pub os: Option<String>,
    pub kernel: Option<String>,
    /// The options the kernel was started with. Shows settings such as `isolcpus` or
    /// `mitigations=off`.
    pub kernel_cmdline: Option<String>,
    pub cpu_model: Option<String>,
    /// CPU family, model and stepping numbers, e.g. `25/97/2`. Identifies the CPU more exactly
    /// than its name.
    pub cpu_family_model_stepping: Option<String>,
    pub microcode: Option<String>,
    /// Running inside a virtual machine.
    pub virtual_machine: Option<bool>,
    /// The caches of the pinned CPU (or CPU 0).
    pub caches: Vec<Cache>,
    pub memory_kb: Option<u64>,
    /// Whether any core runs two threads (hyperthreading).
    pub smt_active: Option<bool>,
    /// The CPUs that are online and share a core with the pinned CPU, itself included. Just the
    /// pinned CPU if its sibling is turned off.
    pub core_siblings: Option<String>,
    /// Frequency governor of the pinned CPU (or CPU 0).
    pub governor: Option<String>,
    /// The driver behind the governor, e.g. `intel_pstate` or `amd-pstate-epp`.
    pub scaling_driver: Option<String>,
    pub energy_performance_preference: Option<String>,
    pub max_frequency_khz: Option<u64>,
    /// `intel_pstate/no_turbo` or the inverse of `cpufreq/boost`.
    pub turbo: Option<bool>,
    /// `always`, `madvise` or `never`.
    pub transparent_hugepages: Option<String>,
    pub perf_event_paranoid: Option<i32>,
    /// Load average of the last minute when the run started. Near 0 on an idle machine.
    pub load_at_start: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Cache {
    pub level: Option<u32>,
    /// `Data`, `Instruction` or `Unified`.
    pub kind: Option<String>,
    /// As the kernel prints it, e.g. `48K`.
    pub size: Option<String>,
    /// The CPUs that share this cache.
    pub shared_by: Option<String>,
}

fn read(path: &str) -> Option<String> {
    fs::read_to_string(path).ok().map(|s| s.trim().to_owned())
}

/// The value of the first `name : value` line in `/proc/cpuinfo`.
fn cpuinfo_field(info: &str, name: &str) -> Option<String> {
    info.lines().find_map(|line| {
        let value = line.strip_prefix(name)?.trim_start().strip_prefix(':')?;
        Some(value.trim().to_owned())
    })
}

pub fn system(cpu: Option<usize>) -> System {
    let cpu = cpu.unwrap_or(0);
    let cpu_dir = format!("/sys/devices/system/cpu/cpu{cpu}");
    let cpuinfo = read("/proc/cpuinfo").unwrap_or_default();
    let field = |name: &str| cpuinfo_field(&cpuinfo, name);

    let turbo = read("/sys/devices/system/cpu/intel_pstate/no_turbo")
        .map(|v| v == "0")
        .or_else(|| read("/sys/devices/system/cpu/cpufreq/boost").map(|v| v == "1"));
    let family_model_stepping = match (field("cpu family"), field("model"), field("stepping")) {
        (Some(family), Some(model), Some(stepping)) => Some(format!("{family}/{model}/{stepping}")),
        _ => None,
    };
    let caches = (0..)
        .map(|index| format!("{cpu_dir}/cache/index{index}"))
        .take_while(|dir| fs::metadata(dir).is_ok())
        .map(|dir| Cache {
            level: read(&format!("{dir}/level")).and_then(|v| v.parse().ok()),
            kind: read(&format!("{dir}/type")),
            size: read(&format!("{dir}/size")),
            shared_by: read(&format!("{dir}/shared_cpu_list")),
        })
        .collect();

    #[allow(unused_mut)]
    let mut system = System {
        hostname: read("/proc/sys/kernel/hostname"),
        os: read("/etc/os-release").and_then(|text| {
            let name = text
                .lines()
                .find_map(|line| line.strip_prefix("PRETTY_NAME="))?;
            Some(name.trim_matches('"').to_owned())
        }),
        kernel: read("/proc/sys/kernel/osrelease"),
        kernel_cmdline: read("/proc/cmdline"),
        cpu_model: field("model name"),
        cpu_family_model_stepping: family_model_stepping,
        microcode: field("microcode"),
        virtual_machine: field("flags")
            .map(|flags| flags.split(' ').any(|flag| flag == "hypervisor")),
        caches,
        memory_kb: read("/proc/meminfo").and_then(|text| {
            let total = text
                .lines()
                .find_map(|line| line.strip_prefix("MemTotal:"))?;
            total.trim().trim_end_matches("kB").trim().parse().ok()
        }),
        smt_active: read("/sys/devices/system/cpu/smt/active").map(|v| v == "1"),
        core_siblings: read(&format!("{cpu_dir}/topology/thread_siblings_list")),
        governor: read(&format!("{cpu_dir}/cpufreq/scaling_governor")),
        scaling_driver: read(&format!("{cpu_dir}/cpufreq/scaling_driver")),
        energy_performance_preference: read(&format!(
            "{cpu_dir}/cpufreq/energy_performance_preference"
        )),
        max_frequency_khz: read(&format!("{cpu_dir}/cpufreq/cpuinfo_max_freq"))
            .and_then(|v| v.parse().ok()),
        turbo,
        // the file lists all choices with the active one in brackets: `always [madvise] never`
        transparent_hugepages: read("/sys/kernel/mm/transparent_hugepage/enabled").and_then(
            |text| {
                let (_, rest) = text.split_once('[')?;
                Some(rest.split_once(']')?.0.to_owned())
            },
        ),
        perf_event_paranoid: read("/proc/sys/kernel/perf_event_paranoid")
            .and_then(|v| v.parse().ok()),
        load_at_start: load_average(),
    };
    #[cfg(target_os = "macos")]
    fill_from_sysctl(&mut system);
    system
}

fn load_average() -> Option<f64> {
    let mut load = 0.0;
    // SAFETY: writes one double
    (unsafe { libc::getloadavg(&mut load, 1) } == 1).then_some(load)
}

/// A `sysctl` value. macOS has no `/proc` or `/sys`, it keeps the machine's details there.
#[cfg(target_os = "macos")]
fn sysctl(name: &str) -> Option<String> {
    let out = std::process::Command::new("sysctl")
        .args(["-n", name])
        .output()
        .ok()?;
    let value = String::from_utf8_lossy(&out.stdout).trim().to_owned();
    (out.status.success() && !value.is_empty()).then_some(value)
}

/// The fields macOS has, from `sysctl` and `sw_vers`. The caches are those of the performance
/// cores, where csvbench asks to run.
#[cfg(target_os = "macos")]
fn fill_from_sysctl(system: &mut System) {
    let number = |name: &str| sysctl(name)?.parse::<u64>().ok();
    let sw_vers = |arg: &str| {
        let out = std::process::Command::new("sw_vers")
            .arg(arg)
            .output()
            .ok()?;
        Some(String::from_utf8_lossy(&out.stdout).trim().to_owned())
    };
    system.hostname = sysctl("kern.hostname");
    system.os = match (sw_vers("-productVersion"), sw_vers("-buildVersion")) {
        (Some(version), Some(build)) => Some(format!("macOS {version} ({build})")),
        (Some(version), None) => Some(format!("macOS {version}")),
        _ => None,
    };
    system.kernel = sysctl("kern.osrelease");
    system.cpu_model = sysctl("machdep.cpu.brand_string");
    system.virtual_machine = sysctl("kern.hv_vmm_present").map(|v| v == "1");
    system.memory_kb = number("hw.memsize").map(|bytes| bytes / 1024);
    let cache = |level, kind: &str, name: &str, shared: &str| Cache {
        level: Some(level),
        kind: Some(kind.to_owned()),
        size: number(&format!("hw.perflevel0.{name}")).map(|b| format!("{}K", b / 1024)),
        shared_by: sysctl(&format!("hw.perflevel0.{shared}")).map(|n| format!("{n} cpus")),
    };
    system.caches = vec![
        cache(1, "Data", "l1dcachesize", "cpusperl1"),
        cache(1, "Instruction", "l1icachesize", "cpusperl1"),
        cache(2, "Unified", "l2cachesize", "cpusperl2"),
    ];
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
    let secs = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs()) as i64;
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
