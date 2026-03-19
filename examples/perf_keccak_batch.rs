//! Throughput benchmark: scalar keccak256 × 8 vs AVX-512 keccak256_batch × 8.
//!
//! Usage:
//!   cargo run --example perf_keccak_batch --release

fn main() {
    // Simulate 8 × 64-byte uncompressed pubkey buffers (primary use case).
    let bufs: [[u8; 64]; 8] = std::array::from_fn(|i| [i as u8 + 1; 64]);
    let inputs: [&[u8]; 8] = std::array::from_fn(|i| bufs[i].as_slice());

    let n = 200_000usize;
    let mut sink = [[0u8; 32]; 8];

    // ── Warm-up ───────────────────────────────────────────────────────────
    for _ in 0..1000 {
        sink = asmcrypto::keccak_batch::keccak256_batch(inputs);
    }

    // ── Scalar: 8 independent keccak256 calls ─────────────────────────────
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        for (i, inp) in inputs.iter().enumerate() {
            sink[i] = asmcrypto::keccak::keccak256(inp);
        }
    }
    let us_scalar = t0.elapsed().as_nanos() as f64 / n as f64 / 1_000.0;

    // ── Batch: 8 streams in one AVX-512 call ──────────────────────────────
    let t1 = std::time::Instant::now();
    for _ in 0..n {
        sink = asmcrypto::keccak_batch::keccak256_batch(inputs);
    }
    let us_batch = t1.elapsed().as_nanos() as f64 / n as f64 / 1_000.0;

    println!(
        "keccak256 × 8 scalar:  {us_scalar:.3} µs/batch  ({:.1} ns/hash)",
        us_scalar * 1000.0 / 8.0
    );
    println!(
        "keccak256_batch × 8:   {us_batch:.3} µs/batch  ({:.1} ns/hash)",
        us_batch * 1000.0 / 8.0
    );
    println!("throughput gain:       {:.2}×", us_scalar / us_batch);

    if sink[0][0] == 0xff {
        eprintln!("(should not print)");
    }
}
