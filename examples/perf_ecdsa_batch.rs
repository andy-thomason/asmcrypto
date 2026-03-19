/// Tight loop for perf profiling and timing of batch ECDSA recovery.
///
/// Usage:
///   cargo build --example perf_ecdsa_batch --release
///   ./target/release/examples/perf_ecdsa_batch
///
/// For perf flamegraph:
///   perf record -g --call-graph dwarf -F 997 -- target/release/examples/perf_ecdsa_batch
///   perf report --stdio -n
use asmcrypto::ecdsa::recover_address as recover_scalar;
use asmcrypto::ecdsa_batch::recover_addresses_batch;

// ─────────────────────────────────────────────────────────────────────────────
// Standard Ethereum ecrecover precompile test vector
// ─────────────────────────────────────────────────────────────────────────────

const HASH: [u8; 32] = [
    0x18, 0xc5, 0x47, 0xe4, 0xf7, 0xb0, 0xf3, 0x25, 0xad, 0x1e, 0x56, 0xf5, 0x7e, 0x26, 0xc7, 0x45,
    0xb0, 0x9a, 0x3e, 0x50, 0x3d, 0x86, 0xe0, 0x0e, 0x52, 0x55, 0xff, 0x7f, 0x71, 0x5d, 0x3d, 0x1c,
];
const R: [u8; 32] = [
    0x73, 0xb1, 0x69, 0x38, 0x92, 0x21, 0x9d, 0x73, 0x6c, 0xab, 0xa5, 0x5b, 0xdb, 0x67, 0x21, 0x6e,
    0x48, 0x55, 0x57, 0xea, 0x6b, 0x6a, 0xf7, 0x5f, 0x37, 0x09, 0x6c, 0x9a, 0xa6, 0xa5, 0xa7, 0x5f,
];
const S: [u8; 32] = [
    0xee, 0xb9, 0x40, 0xb1, 0xd0, 0x3b, 0x21, 0xe3, 0x6b, 0x0e, 0x47, 0xe7, 0x97, 0x69, 0xf0, 0x95,
    0xfe, 0x2a, 0xb8, 0x55, 0xbd, 0x91, 0xe3, 0xa3, 0x87, 0x56, 0xb7, 0xd7, 0x5a, 0x9c, 0x45, 0x49,
];
const V: u8 = 1;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn time_batch(
    label: &str,
    n: usize,
    hashes: [&[u8; 32]; 8],
    rs: [&[u8; 32]; 8],
    ss: [&[u8; 32]; 8],
    vs: [u8; 8],
) {
    // Warm-up
    for _ in 0..50 {
        std::hint::black_box(recover_addresses_batch(hashes, rs, ss, vs));
    }
    let t = std::time::Instant::now();
    for _ in 0..n {
        std::hint::black_box(recover_addresses_batch(hashes, rs, ss, vs));
    }
    let elapsed = t.elapsed();
    let ns_per_batch = elapsed.as_nanos() as f64 / n as f64;
    let us_per_lane = ns_per_batch / 8.0 / 1_000.0;
    let throughput = 8.0 * n as f64 / elapsed.as_secs_f64() / 1_000.0;
    println!(
        "{label:<40} {ns_per_batch:7.0} ns/batch  {us_per_lane:6.2} µs/lane  {throughput:8.0} krecov/s  (n={n})"
    );
}

fn time_scalar_x8(
    label: &str,
    n: usize,
    hashes: [&[u8; 32]; 8],
    rs: [&[u8; 32]; 8],
    ss: [&[u8; 32]; 8],
    vs: [u8; 8],
) {
    // Warm-up
    for _ in 0..50 {
        for i in 0..8 {
            std::hint::black_box(recover_scalar(hashes[i], rs[i], ss[i], vs[i]));
        }
    }
    let t = std::time::Instant::now();
    for _ in 0..n {
        for i in 0..8 {
            std::hint::black_box(recover_scalar(hashes[i], rs[i], ss[i], vs[i]));
        }
    }
    let elapsed = t.elapsed();
    let ns_per_batch = elapsed.as_nanos() as f64 / n as f64;
    let us_per_lane = ns_per_batch / 8.0 / 1_000.0;
    let throughput = 8.0 * n as f64 / elapsed.as_secs_f64() / 1_000.0;
    println!(
        "{label:<40} {ns_per_batch:7.0} ns/batch  {us_per_lane:6.2} µs/lane  {throughput:8.0} krecov/s  (n={n})"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    // Sanity-check: verify the batch result matches the scalar result.
    let addr_scalar = recover_scalar(&HASH, &R, &S, V).expect("scalar ecrecover failed");
    let addrs_batch = recover_addresses_batch([&HASH; 8], [&R; 8], [&S; 8], [V; 8]);
    for (lane, addr) in addrs_batch.iter().enumerate() {
        assert_eq!(
            addr,
            &addr_scalar,
            "batch lane {lane} mismatch: got {} expected {}",
            hex(addr),
            hex(&addr_scalar)
        );
    }
    println!("address: {}", hex(&addr_scalar));
    println!();
    println!(
        "{:<40} {:>14}  {:>12}  {:>14}",
        "variant", "ns/batch", "µs/lane", "krecov/s"
    );
    println!("{}", "-".repeat(88));

    const N: usize = 5_000;

    // ── 8 identical lanes (best-case cache for tables) ────────────────────────
    time_batch(
        "batch x8 (same sig)",
        N,
        [&HASH; 8],
        [&R; 8],
        [&S; 8],
        [V; 8],
    );

    // ── 8× scalar (asmcrypto single-lane, sequential) ─────────────────────────
    time_scalar_x8(
        "scalar x8 (asmcrypto, sequential)",
        N,
        [&HASH; 8],
        [&R; 8],
        [&S; 8],
        [V; 8],
    );

    println!();

    // ── 8 distinct hashes, same (r,s,v) — different u1,u2 per lane ───────────
    let h0: [u8; 32] = std::array::from_fn(|i| (i as u8).wrapping_add(0x10));
    let h1: [u8; 32] = std::array::from_fn(|i| (i as u8).wrapping_add(0x20));
    let h2: [u8; 32] = std::array::from_fn(|i| (i as u8).wrapping_add(0x30));
    let h3: [u8; 32] = std::array::from_fn(|i| (i as u8).wrapping_add(0x40));
    let h4: [u8; 32] = std::array::from_fn(|i| (i as u8).wrapping_add(0x50));
    let h5: [u8; 32] = std::array::from_fn(|i| (i as u8).wrapping_add(0x60));
    let h6: [u8; 32] = std::array::from_fn(|i| (i as u8).wrapping_add(0x70));
    let h7: [u8; 32] = std::array::from_fn(|i| (i as u8).wrapping_add(0x80));

    // Use the same (r,s,v) — they may not all be valid (sqrt may fail), but the
    // timing is still representative of the vectorised path.
    let varied_hashes: [&[u8; 32]; 8] = [&h0, &h1, &h2, &h3, &h4, &h5, &h6, &h7];

    time_batch(
        "batch x8 (varied hashes, same sig)",
        N,
        varied_hashes,
        [&R; 8],
        [&S; 8],
        [V; 8],
    );

    time_scalar_x8(
        "scalar x8 (varied hashes, sequential)",
        N,
        varied_hashes,
        [&R; 8],
        [&S; 8],
        [V; 8],
    );

    println!();
    println!("(krecov/s = thousands of recoveries per second)");
}
