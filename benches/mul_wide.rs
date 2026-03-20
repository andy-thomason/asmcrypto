use asmcrypto::ecdsa_scalar::{bench_fn_mul, bench_fp_mul, bench_mul_wide};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

// Two non-trivial 256-bit operands (secp256k1 field prime - 1 and group order).
const A: [u64; 4] = [
    0xFFFFFFFEFFFFFC2E,
    0xFFFFFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFFFF,
];
const B: [u64; 4] = [
    0xBFD25E8CD0364141,
    0xBAAEDCE6AF48A03B,
    0xFFFFFFFFFFFFFFFE,
    0xFFFFFFFFFFFFFFFF,
];

fn bench_mul(c: &mut Criterion) {
    let mut g = c.benchmark_group("mul_wide");
    g.bench_function("schoolbook", |b| {
        b.iter(|| bench_mul_wide(black_box(A), black_box(B)))
    });
    g.finish();
}

fn bench_reduce(c: &mut Criterion) {
    let mut g = c.benchmark_group("reduce");
    g.bench_function("fp_mul", |b| {
        b.iter(|| bench_fp_mul(black_box(A), black_box(B)))
    });
    g.bench_function("fn_mul", |b| {
        b.iter(|| bench_fn_mul(black_box(A), black_box(B)))
    });
    g.finish();
}

criterion_group!(benches, bench_mul, bench_reduce);
criterion_main!(benches);
