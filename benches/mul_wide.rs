use asmcrypto::ecdsa::{bench_fn_reduce_wide, bench_fp_reduce_wide, bench_mul_wide_generic};
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

// 512-bit reduction inputs: (p-1)^2 and (n-1)^2.
const FP_WIDE: [u64; 8] = [
    0xFFFFFFC200000F84,
    0xFFFFFFFDFFFFFC2F,
    0xFFFFFFFFFFFFFFFF,
    0xFFFFFFFEFFFFFC2E,
    0xFFFFFFFEFFFFFC2E,
    0xFFFFFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFFFE,
];
const FN_WIDE: [u64; 8] = [
    0x9F13B779F6C5CA82,
    0xA0D8D9FD5C8D1697,
    0x51E9E60A7AB90C54,
    0x74DF4FD0E7D5DAAA,
    0x40286C57B53F7FC1,
    0xBAAEDCE6AF48A03B,
    0xFFFFFFFFFFFFFFFD,
    0xFFFFFFFFFFFFFFFF,
];

fn bench_mul(c: &mut Criterion) {
    let mut g = c.benchmark_group("mul_wide");
    g.bench_function("generic", |b| {
        b.iter(|| bench_mul_wide_generic(black_box(A), black_box(B)))
    });

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "bmi2",
        target_feature = "adx"
    ))]
    {
        use asmcrypto::ecdsa::bench_mul_wide_adx;
        g.bench_function("adx", |b| {
            b.iter(|| bench_mul_wide_adx(black_box(A), black_box(B)))
        });
    }

    g.finish();
}

fn bench_reduce(c: &mut Criterion) {
    let mut g = c.benchmark_group("reduce");
    g.bench_function("fp_reduce_wide", |b| {
        b.iter(|| bench_fp_reduce_wide(black_box(FP_WIDE)))
    });
    g.bench_function("fn_reduce_wide", |b| {
        b.iter(|| bench_fn_reduce_wide(black_box(FN_WIDE)))
    });
    g.finish();
}

criterion_group!(benches, bench_mul, bench_reduce);
criterion_main!(benches);
