//! Performance counters around benchmark samples, and pinning to one CPU.
//!
//! Both need Linux. Elsewhere [`Counters::open`] and [`pin_to`] fail, and csvbench measures time
//! only.

use std::{fs, io};

#[cfg(target_os = "linux")]
use perf_event::{Builder, Counter, Group, events::Hardware};
use serde::Serialize;

/// Hardware counter values for one sample, all of this thread and over exactly the same interval.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct Reading {
    pub cycles: u64,
    pub instructions: u64,
    pub branches: u64,
    pub branch_misses: u64,
}

/// Hardware counters read as one group, so they are scheduled together and cover the same
/// interval.
///
/// They count user space only, which is what `perf_event_paranoid = 2` allows for any event,
/// software events included; scheduling is measured with [`Usage`] instead. Cycles and
/// instructions have fixed counters, branches and branch misses take two general purpose ones.
#[cfg(target_os = "linux")]
pub struct Counters {
    group: Group,
    cycles: Counter,
    instructions: Counter,
    branches: Counter,
    branch_misses: Counter,
}

#[cfg(target_os = "linux")]
impl Counters {
    /// `cpu` is the CPU the thread is pinned to, if any. On hybrid CPUs the generic hardware
    /// events have to be opened on that core type's PMU.
    pub fn open(cpu: Option<usize>) -> io::Result<Self> {
        let pmu = cpu.and_then(hybrid_pmu_type);
        let mut group = Group::new()?;
        let mut hardware = |event: Hardware| {
            let mut builder = Builder::new(event);
            if let Some(pmu) = pmu {
                // PERF_TYPE_HARDWARE with the PMU type in the upper half of `config`
                builder.attrs_mut().config |= (pmu as u64) << 32;
            }
            group.add(&builder)
        };
        let cycles = hardware(Hardware::CPU_CYCLES)?;
        let instructions = hardware(Hardware::INSTRUCTIONS)?;
        let branches = hardware(Hardware::BRANCH_INSTRUCTIONS)?;
        let branch_misses = hardware(Hardware::BRANCH_MISSES)?;
        Ok(Self {
            group,
            cycles,
            instructions,
            branches,
            branch_misses,
        })
    }

    #[inline]
    pub fn start(&mut self) -> io::Result<()> {
        self.group.reset()?;
        self.group.enable()
    }

    /// Stop counting and read the values.
    ///
    /// Fails if the group did not run for the whole time the thread did: with more events than
    /// free PMU counters the kernel multiplexes them and the values would be estimates, and on a
    /// hybrid CPU the events do not count on the other core type.
    #[inline]
    pub fn stop(&mut self) -> io::Result<Reading> {
        self.group.disable()?;
        let data = self.group.read()?;
        if data.time_running() != data.time_enabled() {
            return Err(io::Error::other(format!(
                "counters only ran {:?} of {:?}: multiplexed, or on the wrong core type",
                data.time_running(),
                data.time_enabled()
            )));
        }
        let value = |c: &Counter| data.get(c).map_or(0, |e| e.value());
        Ok(Reading {
            cycles: value(&self.cycles),
            instructions: value(&self.instructions),
            branches: value(&self.branches),
            branch_misses: value(&self.branch_misses),
        })
    }
}

/// Without `perf_event_open` there are no counters: [`Counters::open`] always fails.
#[cfg(not(target_os = "linux"))]
pub struct Counters(std::convert::Infallible);

#[cfg(not(target_os = "linux"))]
impl Counters {
    pub fn open(_cpu: Option<usize>) -> io::Result<Self> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "perf_event_open is only on Linux",
        ))
    }

    pub fn start(&mut self) -> io::Result<()> {
        match self.0 {}
    }

    pub fn stop(&mut self) -> io::Result<Reading> {
        match self.0 {}
    }
}

/// What the scheduler did to this thread during a sample, from the kernel's own accounting, which
/// needs no perf permissions.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct Usage {
    /// Time the thread was on a CPU, user and kernel (ns). Below the sample's wall time when it
    /// was preempted or blocked; `cycles / cpu_time` is the clock frequency while it ran.
    pub cpu_time: u64,
    /// The thread was switched out by the scheduler.
    pub involuntary_switches: u64,
    /// The thread blocked, e.g. on a page fault that needed IO.
    pub voluntary_switches: u64,
    pub page_faults: u64,
}

impl Usage {
    /// The totals so far; subtract two with [`Usage::since`].
    #[inline]
    pub fn now() -> Usage {
        // SAFETY: both calls only write the structs they are given
        unsafe {
            let mut time: libc::timespec = std::mem::zeroed();
            libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut time);
            let mut usage: libc::rusage = std::mem::zeroed();
            // macOS has no per-thread usage; the whole process is the same while csvbench
            // measures on one thread
            #[cfg(target_os = "linux")]
            libc::getrusage(libc::RUSAGE_THREAD, &mut usage);
            #[cfg(not(target_os = "linux"))]
            libc::getrusage(libc::RUSAGE_SELF, &mut usage);
            Usage {
                cpu_time: time.tv_sec as u64 * 1_000_000_000 + time.tv_nsec as u64,
                involuntary_switches: usage.ru_nivcsw as u64,
                voluntary_switches: usage.ru_nvcsw as u64,
                page_faults: (usage.ru_minflt + usage.ru_majflt) as u64,
            }
        }
    }

    pub fn since(self, start: Usage) -> Usage {
        Usage {
            cpu_time: self.cpu_time - start.cpu_time,
            involuntary_switches: self.involuntary_switches - start.involuntary_switches,
            voluntary_switches: self.voluntary_switches - start.voluntary_switches,
            page_faults: self.page_faults - start.page_faults,
        }
    }
}

/// Whether the CPU has more than one core type, each with its own PMU.
///
/// The hardware counters then only count while the thread runs on the core type they were opened
/// for, so the thread has to be pinned.
pub fn is_hybrid() -> bool {
    fs::exists("/sys/bus/event_source/devices/cpu_core").unwrap_or(false)
}

/// The perf type of the core PMU that `cpu` belongs to, on hybrid CPUs (`cpu_core`/`cpu_atom`).
#[cfg(target_os = "linux")]
fn hybrid_pmu_type(cpu: usize) -> Option<u32> {
    for pmu in ["cpu_core", "cpu_atom"] {
        let dir = format!("/sys/bus/event_source/devices/{pmu}");
        let Ok(cpus) = fs::read_to_string(format!("{dir}/cpus")) else {
            continue;
        };
        if cpu_list_contains(&cpus, cpu) {
            return fs::read_to_string(format!("{dir}/type"))
                .ok()?
                .trim()
                .parse()
                .ok();
        }
    }
    None
}

/// Whether a kernel CPU list like `0-15,18` contains `cpu`.
#[cfg(target_os = "linux")]
fn cpu_list_contains(list: &str, cpu: usize) -> bool {
    list.trim().split(',').any(|range| match range.split_once('-') {
        Some((lo, hi)) => matches!((lo.parse::<usize>(), hi.parse::<usize>()), (Ok(lo), Ok(hi)) if (lo..=hi).contains(&cpu)),
        None => range.parse() == Ok(cpu),
    })
}

/// Pin the calling thread to one CPU.
#[cfg(target_os = "linux")]
pub fn pin_to(cpu: usize) -> io::Result<()> {
    // SAFETY: cpu_set_t is plain data, initialised by CPU_ZERO before use
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
        if libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &set) != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

/// macOS cannot pin threads to a CPU.
#[cfg(not(target_os = "linux"))]
pub fn pin_to(_cpu: usize) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "pinning to a CPU is only possible on Linux",
    ))
}

/// Ask macOS to run the calling thread on the performance cores. It cannot be pinned, but the
/// highest quality of service class keeps it off the efficiency cores while they are free.
#[cfg(target_os = "macos")]
pub fn prefer_performance_cores() -> io::Result<()> {
    // SAFETY: only changes the scheduling class of the calling thread
    let error = unsafe {
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0)
    };
    match error {
        0 => Ok(()),
        e => Err(io::Error::from_raw_os_error(e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sleeping and touching fresh memory happen in the kernel, which the hardware counters do not
    /// see; [`Usage`] has to.
    #[test]
    fn usage_sees_the_scheduler() {
        let start = std::time::Instant::now();
        let before = Usage::now();
        let mut spin = 0u64;
        for i in 0..10_000_000u64 {
            spin = std::hint::black_box(spin.wrapping_add(i));
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
        let memory = std::hint::black_box(vec![1u8; 8 << 20]);
        let usage = Usage::now().since(before);
        let wall = start.elapsed().as_nanos() as u64;
        drop(memory);

        assert!(usage.voluntary_switches >= 1, "{usage:?}");
        assert!(usage.page_faults >= 1000, "{usage:?}");
        // the sleep is not on-CPU time
        assert!(
            usage.cpu_time > 0 && usage.cpu_time + 15_000_000 < wall,
            "{usage:?}, wall {wall}"
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn hardware_counters_count_when_pinned() {
        // unpinned, the thread could move between core types mid-measurement
        // SAFETY: plain libc call without arguments
        let cpu = unsafe { libc::sched_getcpu() } as usize;
        pin_to(cpu).unwrap();
        let Ok(mut counters) = Counters::open(Some(cpu)) else {
            eprintln!("performance counters unavailable, skipping");
            return;
        };
        counters.start().unwrap();
        let mut spin = 0u64;
        for i in 0..10_000_000u64 {
            spin = std::hint::black_box(spin.wrapping_add(i));
        }
        let reading = counters.stop().unwrap();
        assert!(
            reading.instructions >= 10_000_000 && reading.branches >= 10_000_000,
            "{reading:?}"
        );
        assert!(reading.cycles > 0, "{reading:?}");
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn cpu_lists() {
        assert!(cpu_list_contains("0-15\n", 3));
        assert!(cpu_list_contains("0-3,16,18-19", 16));
        assert!(!cpu_list_contains("0-3,16,18-19", 17));
    }
}
