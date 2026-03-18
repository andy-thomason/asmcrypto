/// Tight loop for perf profiling of fp_reduce_wide and fn_reduce_wide in isolation.
///
/// Usage:
///   cargo build --example perf_reduce --release
///   perf record -g --call-graph dwarf -F 997 -- target/release/examples/perf_reduce
///   perf report --stdio -n
use asmcrypto::ecdsa::{bench_fn_reduce_wide, bench_fp_reduce_wide};

fn main() {
    // Representative 512-bit inputs: (p-1)^2 and (n-1)^2 computed offline.
    // These exercise the full folding path in each reduction.
    //
    // fp input: (p-1) * (p-1)  — upper half is large, exercises both K_P folds.
    // p-1 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2E
    // (p-1)^2 in 8×u64 LE:
    let fp_wide: [u64; 8] = [
        0xFFFFFFC200000F84, // w[0]
        0xFFFFFFFDFFFFFC2F, // w[1]
        0xFFFFFFFFFFFFFFFF, // w[2]
        0xFFFFFFFEFFFFFC2E, // w[3]
        0xFFFFFFFEFFFFFC2E, // w[4]  (hi half)
        0xFFFFFFFFFFFFFFFF, // w[5]
        0xFFFFFFFFFFFFFFFF, // w[6]
        0xFFFFFFFFFFFFFFFE, // w[7]
    ];

    // fn input: (n-1) * (n-1)  — exercises the N_COMPL folding path.
    // n-1 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140
    // (n-1)^2 in 8×u64 LE:
    let fn_wide: [u64; 8] = [
        0x9F13B779F6C5CA82, // w[0]
        0xA0D8D9FD5C8D1697, // w[1]
        0x51E9E60A7AB90C54, // w[2]
        0x74DF4FD0E7D5DAAA, // w[3]
        0x40286C57B53F7FC1, // w[4]
        0xBAAEDCE6AF48A03B, // w[5]
        0xFFFFFFFFFFFFFFFD, // w[6]
        0xFFFFFFFFFFFFFFFF, // w[7]
    ];

    // Warm up to stabilise branch predictors and caches.
    for _ in 0..10_000 {
        let _ = bench_fp_reduce_wide(fp_wide);
        let _ = bench_fn_reduce_wide(fn_wide);
    }

    // ── fp_reduce_wide profiling loop (~3 s at ~5 ns/call) ───────────────────
    let n = 300_000_000usize;
    let mut sink_fp = [0u64; 4];
    for _ in 0..n {
        sink_fp = bench_fp_reduce_wide(fp_wide);
    }

    // ── fn_reduce_wide profiling loop ────────────────────────────────────────
    let mut sink_fn = [0u64; 4];
    for _ in 0..n {
        sink_fn = bench_fn_reduce_wide(fn_wide);
    }

    // Prevent dead-code elimination.
    if sink_fp[0] == 0xdeadbeef || sink_fn[0] == 0xdeadbeef {
        eprintln!("(should not print)");
    }
}
