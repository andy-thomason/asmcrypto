//! RDTSC-based execution timeline for ecdsa_clone::recover_address.
//!
//! Breaks the hot path into labelled phases and reports median cycle costs.
//! Usage:
//!   cargo run --example timeline_ecdsa --release

use asmcrypto::ecdsa_clone::{
    BETA, Fe, Gej, PRE_G_DATA, PRE_G128_DATA, Scalar, TABLE_SIZE, WINDOW_A, WINDOW_G,
    build_odd_multiples_table, ecmult, ecmult_wnaf, fe_mul, g_table_get_ge, ge_set_gej_var,
    ge_set_xo_var, gej_add_ge_var, gej_double, scalar_inv_var, scalar_mul, scalar_split_128,
    scalar_split_lambda, table_get_ge, table_get_ge_lambda,
};

// SAFETY: x86_64-only; guaranteed by the crate's supported platform.
#[inline(always)]
fn rdtsc() -> u64 {
    unsafe { core::arch::x86_64::_rdtsc() }
}

/// Collect N rdtsc measurements and return them sorted (for median/percentiles).
fn measure<F: FnMut()>(mut f: F, n: usize) -> Vec<u64> {
    let mut samples = Vec::with_capacity(n);
    for _ in 0..n {
        let t0 = rdtsc();
        f();
        let t1 = rdtsc();
        samples.push(t1 - t0);
    }
    samples.sort_unstable();
    samples
}

fn median(v: &[u64]) -> u64 {
    v[v.len() / 2]
}

fn p95(v: &[u64]) -> u64 {
    v[v.len() * 95 / 100]
}

fn main() {
    // ── Fixed test vector ──────────────────────────────────────────────────
    let hash: [u8; 32] = [
        0x18, 0xc5, 0x47, 0xe4, 0xf7, 0xb0, 0xf3, 0x25, 0xad, 0x1e, 0x56, 0xf5, 0x7e, 0x26, 0xc7,
        0x45, 0xb0, 0x9a, 0x3e, 0x50, 0x3d, 0x86, 0xe0, 0x0e, 0x52, 0x55, 0xff, 0x7f, 0x71, 0x5d,
        0x3d, 0x1c,
    ];
    let r: [u8; 32] = [
        0x73, 0xb1, 0x69, 0x38, 0x92, 0x21, 0x9d, 0x73, 0x6c, 0xab, 0xa5, 0x5b, 0xdb, 0x67, 0x21,
        0x6e, 0x48, 0x55, 0x57, 0xea, 0x6b, 0x6a, 0xf7, 0x5f, 0x37, 0x09, 0x6c, 0x9a, 0xa6, 0xa5,
        0xa7, 0x5f,
    ];
    let s: [u8; 32] = [
        0xee, 0xb9, 0x40, 0xb1, 0xd0, 0x3b, 0x21, 0xe3, 0x6b, 0x0e, 0x47, 0xe7, 0x97, 0x69, 0xf0,
        0x95, 0xfe, 0x2a, 0xb8, 0x55, 0xbd, 0x91, 0xe3, 0xa3, 0x87, 0x56, 0xb7, 0xd7, 0x5a, 0x9c,
        0x45, 0x49,
    ];
    let v = 1u8;

    let (sigr, _) = Scalar::set_b32(&r);
    let (sigs, _) = Scalar::set_b32(&s);
    let (message, _) = Scalar::set_b32(&hash);
    let recid = v & 1;

    // Pre-compute shared inputs (not being timed).
    let brx = sigr.get_b32();
    let (fx, _) = Fe::set_b32_limit(&brx);
    let x_point = ge_set_xo_var(&fx, recid != 0).expect("valid recid");
    let xj = Gej::set_ge(&x_point);
    let rn = scalar_inv_var(&sigr);
    let mut u1 = scalar_mul(&rn, &message);
    u1 = u1.negate();
    let u2 = scalar_mul(&rn, &sigs);

    let n = 10_000usize;

    // ── Warm-up pass ──────────────────────────────────────────────────────
    for _ in 0..500 {
        let _ = asmcrypto::ecdsa_clone::recover_address(&hash, &{
            let mut sig65 = [0u8; 65];
            sig65[0..32].copy_from_slice(&r);
            sig65[32..64].copy_from_slice(&s);
            sig65[64] = v;
            sig65
        });
    }

    // ── §1  ge_set_xo_var  (recover R point: fe_sqrt + fe_inv) ───────────
    let s1 = measure(
        || {
            let _ = ge_set_xo_var(&fx, recid != 0);
        },
        n,
    );

    // ── §2  scalar_inv_var  (safegcd 1/r mod n) ───────────────────────────
    let s2 = measure(
        || {
            let _ = scalar_inv_var(&sigr);
        },
        n,
    );

    // ── §3  scalar_mul × 2  (u1 = -m/r, u2 = s/r) ───────────────────────
    let s3 = measure(
        || {
            let rn2 = scalar_inv_var(&sigr);
            let mut u = scalar_mul(&rn2, &message);
            u = u.negate();
            let _u2 = scalar_mul(&rn2, &sigs);
            core::hint::black_box(u);
        },
        n,
    );
    // Subtract scalar_inv_var cost to isolate the two muls.
    let s3_muls: Vec<u64> = s3
        .iter()
        .zip(s2.iter())
        .map(|(&a, &b)| a.saturating_sub(b))
        .collect::<Vec<_>>()
        .tap_sort();

    // ── §4  scalar_split_lambda + scalar_split_128  (GLV splits) ─────────
    let s4 = measure(
        || {
            let _ = scalar_split_lambda(&u2);
            let _ = scalar_split_128(&u1);
        },
        n,
    );

    // ── §5  build_odd_multiples_table  (8-entry affine A table + aux) ─────
    let s5 = measure(
        || {
            let pre_a = build_odd_multiples_table(&xj);
            let beta = BETA;
            let mut aux = [Fe { n: [0; 5] }; TABLE_SIZE];
            for i in 0..TABLE_SIZE {
                aux[i] = fe_mul(&pre_a[i].x, &beta);
                aux[i].normalize_weak();
            }
            core::hint::black_box(aux);
        },
        n,
    );

    // ── §6  ecmult_wnaf × 4 ───────────────────────────────────────────────
    let (na_1, na_lam) = scalar_split_lambda(&u2);
    let (ng_1, ng_128) = scalar_split_128(&u1);
    let s6 = measure(
        || {
            let _ = ecmult_wnaf(&na_1, WINDOW_A);
            let _ = ecmult_wnaf(&na_lam, WINDOW_A);
            let _ = ecmult_wnaf(&ng_1, WINDOW_G);
            let _ = ecmult_wnaf(&ng_128, WINDOW_G);
        },
        n,
    );

    // ── §7  ecmult main loop  (doublings + table lookups; precompute setup
    //         cost separately so we can subtract it) ─────────────────────
    let pre_a = build_odd_multiples_table(&xj);
    let beta = BETA;
    let mut aux = [Fe { n: [0; 5] }; TABLE_SIZE];
    for i in 0..TABLE_SIZE {
        aux[i] = fe_mul(&pre_a[i].x, &beta);
        aux[i].normalize_weak();
    }
    let (wnaf_na1, bits_na1) = ecmult_wnaf(&na_1, WINDOW_A);
    let (wnaf_nalm, bits_nalm) = ecmult_wnaf(&na_lam, WINDOW_A);
    let (wnaf_ng1, bits_ng1) = ecmult_wnaf(&ng_1, WINDOW_G);
    let (wnaf_ng128, bits_ng128) = ecmult_wnaf(&ng_128, WINDOW_G);
    let bits = bits_na1.max(bits_nalm).max(bits_ng1).max(bits_ng128);

    let s7 = measure(
        || {
            let mut res = Gej::infinity();
            for i in (0..bits).rev() {
                res = gej_double(&res);
                if i < bits_na1 {
                    let n = wnaf_na1[i];
                    if n != 0 {
                        res = gej_add_ge_var(&res, &table_get_ge(&pre_a, n));
                    }
                }
                if i < bits_nalm {
                    let n = wnaf_nalm[i];
                    if n != 0 {
                        res = gej_add_ge_var(&res, &table_get_ge_lambda(&pre_a, &aux, n));
                    }
                }
                if i < bits_ng1 {
                    let n = wnaf_ng1[i];
                    if n != 0 {
                        res = gej_add_ge_var(&res, &g_table_get_ge(&PRE_G_DATA, n));
                    }
                }
                if i < bits_ng128 {
                    let n = wnaf_ng128[i];
                    if n != 0 {
                        res = gej_add_ge_var(&res, &g_table_get_ge(&PRE_G128_DATA, n));
                    }
                }
            }
            core::hint::black_box(res);
        },
        n,
    );

    // ── §8  ge_set_gej_var  (affine normalisation, 1 fe_inv) ─────────────
    let qj = ecmult(&xj, &u2, &u1);
    let s8 = measure(
        || {
            let _ = ge_set_gej_var(&qj);
        },
        n,
    );

    // ── §9  keccak256 ─────────────────────────────────────────────────────
    let pubkey = ge_set_gej_var(&qj);
    let mut buf = [0u8; 64];
    let mut px = pubkey.x;
    px.normalize();
    let mut py = pubkey.y;
    py.normalize();
    buf[0..32].copy_from_slice(&px.get_b32());
    buf[32..64].copy_from_slice(&py.get_b32());
    let s9 = measure(
        || {
            let _ = asmcrypto::keccak::keccak256(&buf);
        },
        n,
    );

    // ── §total  full recover_address ─────────────────────────────────────
    let sig65 = {
        let mut b = [0u8; 65];
        b[0..32].copy_from_slice(&r);
        b[32..64].copy_from_slice(&s);
        b[64] = v;
        b
    };
    let s_total = measure(
        || {
            let _ = asmcrypto::ecdsa_clone::recover_address(&hash, &sig65);
        },
        n,
    );

    // ── Print results ─────────────────────────────────────────────────────
    const GHZ: f64 = 5.4; // approximate TSC frequency from perf stat

    fn cycles_to_ns(c: u64) -> f64 {
        c as f64 / GHZ
    }

    println!("\nPhase breakdown — median cycles @ {GHZ} GHz (n = {n})");
    println!(
        "{:<42}  {:>9}  {:>9}  {:>9}",
        "Phase", "cycles", "ns", "p95 cyc"
    );
    println!("{}", "─".repeat(76));

    let phases: &[(&str, &[u64])] = &[
        ("§1  ge_set_xo_var (R recovery, fe_sqrt)", &s1),
        ("§2  scalar_inv_var (safegcd 1/r)", &s2),
        ("§3  scalar_mul × 2 (u1, u2)", &s3_muls),
        ("§4  scalar_split (GLV + 128-bit)", &s4),
        ("§5  build_odd_multiples_table + aux", &s5),
        ("§6  ecmult_wnaf × 4", &s6),
        ("§7  ecmult main loop (~129 iters)", &s7),
        ("§8  ge_set_gej_var (normalise Q)", &s8),
        ("§9  keccak256", &s9),
        ("TOTAL recover_address", &s_total),
    ];

    let mut sum_phases = 0u64;
    for (label, samples) in phases {
        let med = median(samples);
        let p = p95(samples);
        if !label.starts_with("TOTAL") {
            sum_phases += med;
        }
        println!(
            "{:<42}  {:>9}  {:>8.1}  {:>9}",
            label,
            med,
            cycles_to_ns(med),
            p
        );
    }
    println!("{}", "─".repeat(76));
    println!("{:<42}  {:>9}", "Sum of timed phases", sum_phases);

    let total_med = median(&s_total);
    println!("{:<42}  {:>9}", "TOTAL (direct measurement)", total_med);
}

// ── Helper trait to sort a Vec in place and return it ────────────────────────
trait TapSort {
    fn tap_sort(self) -> Self;
}
impl TapSort for Vec<u64> {
    fn tap_sort(mut self) -> Self {
        self.sort_unstable();
        self
    }
}
