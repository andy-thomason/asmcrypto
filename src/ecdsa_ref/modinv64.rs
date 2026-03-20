//! Safegcd-based modular inverse — variable-time.
//!
//! Port of the secp256k1 C library `modinv64_impl.h` by Peter Dettman,
//! implementing the Bernstein-Yang algorithm from
//! "Fast constant-time gcd computation and modular inversion" (2019).
//!
//! **Only the variable-time path** (`modinv64_var`) is implemented here,
//! matching what `field_5x52_impl.h` uses for `fe_impl_inv_var` and
//! `scalar_4x64_impl.h` uses for `scalar_inverse_var`.
//!
//! The algorithm uses N=62: 256-bit numbers are represented as 5 signed
//! 62-bit limbs (= signed62).  Each outer iteration calls `divsteps_62_var`,
//! which performs up to 62 "division steps" and produces a 2×2 transition
//! matrix.  The matrix is then applied to the running (d, e) and (f, g)
//! pairs.  After g reaches zero, d holds ±modular-inverse of the original g,
//! and a final normalisation step brings it to [0, modulus).
//!
//! C source:
//!   secp256k1-sys-0.11.0/depend/secp256k1/src/modinv64_impl.h
//!   secp256k1-sys-0.11.0/depend/secp256k1/src/modinv64.h
//!
//! Differences from the C original:
//!  - VERIFY_CHECK / VERIFY_BITS assertions are omitted (they are
//!    debug-only in the C library).
//!  - The constant-time variant (`modinv64`) is not ported; only
//!    the variable-time variant (`modinv64_var`) is needed here.
//!  - Rust native `i128` / `u128` replace the `secp256k1_int128` helper.
//!  - Rust `u64::trailing_zeros()` replaces `secp256k1_ctz64_var`.

#![allow(dead_code)]

/// Lower-62-bit mask: 2^62 − 1.
const M62: u64 = u64::MAX >> 2;

// ─────────────────────────────────────────────────────────────────────────────
// § 1  Data types
// ─────────────────────────────────────────────────────────────────────────────

/// 256-bit signed integer stored as five 62-bit signed limbs.
///
/// Limb 0 is least-significant.  Each limb is in (−2^62, 2^62) when
/// normalised, but intermediate values may exceed this range.
///
/// C: `secp256k1_modinv64_signed62`
#[derive(Clone, Copy)]
pub struct Signed62 {
    pub v: [i64; 5],
}

/// Modulus descriptor: the modulus itself plus its inverse mod 2^62.
///
/// `modulus_inv62` = -modulus^{−1} mod 2^62  (the negative inverse).
///
/// C: `secp256k1_modinv64_modinfo`
pub struct ModInfo {
    pub modulus: Signed62,
    /// −modulus^{−1} mod 2^62.
    pub modulus_inv62: u64,
}

/// 2×2 transition matrix M (scaled by 2^62).
/// The i-th map sends [f, g] → [(u·f + v·g) / 2^62,  (q·f + r·g) / 2^62].
///
/// C: `secp256k1_modinv64_trans2x2`
struct Trans2x2 {
    u: i64,
    v: i64,
    q: i64,
    r: i64,
}

// ─────────────────────────────────────────────────────────────────────────────
// § 2  Inner helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Normalise `r` (given as signed62 limbs) into [0, modulus).
///
/// `r` must be in (−2·modulus, modulus) on entry.
/// If `sign < 0` the value is also negated before normalisation.
///
/// C: `secp256k1_modinv64_normalize_62`
fn normalize_62(r: &mut Signed62, sign: i64, modinfo: &ModInfo) {
    let m = &modinfo.modulus;
    let [mut r0, mut r1, mut r2, mut r3, mut r4] = r.v;
    let m62 = M62 as i64;

    // Step 1: add modulus if r is negative → bring into (−modulus, modulus).
    let cond_add = r4 >> 63; // −1 or 0
    r0 += m.v[0] & cond_add;
    r1 += m.v[1] & cond_add;
    r2 += m.v[2] & cond_add;
    r3 += m.v[3] & cond_add;
    r4 += m.v[4] & cond_add;

    // Step 2: negate if requested.
    let cond_neg = sign >> 63; // −1 or 0
    r0 = (r0 ^ cond_neg) - cond_neg;
    r1 = (r1 ^ cond_neg) - cond_neg;
    r2 = (r2 ^ cond_neg) - cond_neg;
    r3 = (r3 ^ cond_neg) - cond_neg;
    r4 = (r4 ^ cond_neg) - cond_neg;

    // Propagate carries so every limb is back in (−2^62, 2^62).
    r1 += r0 >> 62;
    r0 &= m62;
    r2 += r1 >> 62;
    r1 &= m62;
    r3 += r2 >> 62;
    r2 &= m62;
    r4 += r3 >> 62;
    r3 &= m62;

    // Step 3: add modulus again if result is still negative → [0, modulus).
    let cond_add = r4 >> 63;
    r0 += m.v[0] & cond_add;
    r1 += m.v[1] & cond_add;
    r2 += m.v[2] & cond_add;
    r3 += m.v[3] & cond_add;
    r4 += m.v[4] & cond_add;

    // Propagate again.
    r1 += r0 >> 62;
    r0 &= m62;
    r2 += r1 >> 62;
    r1 &= m62;
    r3 += r2 >> 62;
    r2 &= m62;
    r4 += r3 >> 62;
    r3 &= m62;

    r.v = [r0, r1, r2, r3, r4];
}

/// Compute up to 62 Bernstein-Yang division steps (variable-time).
///
/// Returns updated `eta` (= −delta).  The 2×2 matrix `t` accumulates
/// all 62 individual per-step matrices.
///
/// C: `secp256k1_modinv64_divsteps_62_var`
fn divsteps_62_var(mut eta: i64, f0: u64, g0: u64, t: &mut Trans2x2) -> i64 {
    // Matrix elements represented as u64 with wrapping semantics (semantically
    // signed integers in [−2^62, 2^62], but we use unsigned for safe left-shifts).
    let mut u: u64 = 1;
    let mut v: u64 = 0;
    let mut q: u64 = 0;
    let mut r: u64 = 1;

    let mut f = f0;
    let mut g = g0;
    let mut i: i32 = 62; // remaining divstep budget

    loop {
        // Count trailing zeros of g, but at most i (via sentinel bit).
        // C: ctz64_var(g | (UINT64_MAX << i))
        let zeros = (g | (u64::MAX << i as u32)).trailing_zeros() as i32;

        // Perform `zeros` trivial divsteps (g is even → just shift).
        g >>= zeros;
        u <<= zeros;
        v <<= zeros;
        eta -= zeros as i64;
        i -= zeros;

        if i == 0 {
            break;
        }
        // f and g are now both odd.

        if eta < 0 {
            // Negate eta and swap (f, g) → (g, −f); likewise swap matrix columns.
            eta = -eta;
            let tmp = f;
            f = g;
            g = tmp.wrapping_neg();
            let tmp = u;
            u = q;
            q = tmp.wrapping_neg();
            let tmp = v;
            v = r;
            r = tmp.wrapping_neg();

            // Cancel up to 6 bits of g by adding a multiple of f.
            // w = f * g * (f^2 - 2)  (mod 2^min(limit,6))
            let limit = ((eta as i32) + 1).min(i) as u32;
            let m = (u64::MAX >> (64 - limit)) & 63u64;
            let w = f
                .wrapping_mul(g)
                .wrapping_mul(f.wrapping_mul(f).wrapping_sub(2))
                & m;
            g = g.wrapping_add(f.wrapping_mul(w));
            q = q.wrapping_add(u.wrapping_mul(w));
            r = r.wrapping_add(v.wrapping_mul(w));
        } else {
            // Cancel up to 4 bits of g by adding a multiple of f.
            // w = −(f + ((f+1) & 4)*2) * g  (mod 2^min(limit,4))
            let limit = ((eta as i32) + 1).min(i) as u32;
            let m = (u64::MAX >> (64 - limit)) & 15u64;
            let mut w = f.wrapping_add((f.wrapping_add(1) & 4) << 1);
            w = w.wrapping_neg().wrapping_mul(g) & m;
            g = g.wrapping_add(f.wrapping_mul(w));
            q = q.wrapping_add(u.wrapping_mul(w));
            r = r.wrapping_add(v.wrapping_mul(w));
        }
        // The unused matrix row (for f) gets updated on the next iteration via
        // the swap above.  We only track d/e via the q/r/u/v matrix.
    }

    t.u = u as i64;
    t.v = v as i64;
    t.q = q as i64;
    t.r = r as i64;
    eta
}

/// Apply transition matrix `t` (scaled by 2^62) to the (d, e) pair modulo
/// `modinfo.modulus`, dividing off the 2^62 scale factor.
///
/// On input/output d and e are in (−2·modulus, modulus).
///
/// C: `secp256k1_modinv64_update_de_62`
fn update_de_62(d: &mut Signed62, e: &mut Signed62, t: &Trans2x2, modinfo: &ModInfo) {
    let [d0, d1, d2, d3, d4] = d.v;
    let [e0, e1, e2, e3, e4] = e.v;
    let u = t.u;
    let v = t.v;
    let q = t.q;
    let r = t.r;
    let m = &modinfo.modulus;

    // Choose modular correction multiples md, me so that t·[d,e] +
    // modulus·[md,me] is divisible by 2^62.
    //
    // Strategy: if d < 0 add u (resp. q) to md (resp. me); if e < 0 add v
    // (resp. r).  This is the signed-magnitude initialisation.
    let sd = d4 >> 63; // −1 if d < 0, else 0
    let se = e4 >> 63;
    let mut md = (u & sd) + (v & se);
    let mut me = (q & sd) + (r & se);

    // Start accumulating: cd = u*d0 + v*e0,  ce = q*d0 + r*e0.
    let mut cd: i128 = (u as i128) * (d0 as i128) + (v as i128) * (e0 as i128);
    let mut ce: i128 = (q as i128) * (d0 as i128) + (r as i128) * (e0 as i128);

    // Adjust md/me so that the bottom 62 bits of (cd + modulus*md) are zero.
    // C: md -= (modulus_inv62 * (uint64_t)cd + md) & M62
    md -= (modinfo
        .modulus_inv62
        .wrapping_mul(cd as u64)
        .wrapping_add(md as u64)
        & M62) as i64;
    me -= (modinfo
        .modulus_inv62
        .wrapping_mul(ce as u64)
        .wrapping_add(me as u64)
        & M62) as i64;

    // Apply the correction.
    cd += (m.v[0] as i128) * (md as i128);
    ce += (m.v[0] as i128) * (me as i128);

    // The bottom 62 bits should now be zero; shift them away.
    debug_assert!((cd as u64) & M62 == 0);
    debug_assert!((ce as u64) & M62 == 0);
    cd >>= 62;
    ce >>= 62;

    // Limb 1.
    cd += (u as i128) * (d1 as i128) + (v as i128) * (e1 as i128);
    ce += (q as i128) * (d1 as i128) + (r as i128) * (e1 as i128);
    if m.v[1] != 0 {
        cd += (m.v[1] as i128) * (md as i128);
        ce += (m.v[1] as i128) * (me as i128);
    }
    d.v[0] = (cd as u64 & M62) as i64;
    cd >>= 62;
    e.v[0] = (ce as u64 & M62) as i64;
    ce >>= 62;

    // Limb 2.
    cd += (u as i128) * (d2 as i128) + (v as i128) * (e2 as i128);
    ce += (q as i128) * (d2 as i128) + (r as i128) * (e2 as i128);
    if m.v[2] != 0 {
        cd += (m.v[2] as i128) * (md as i128);
        ce += (m.v[2] as i128) * (me as i128);
    }
    d.v[1] = (cd as u64 & M62) as i64;
    cd >>= 62;
    e.v[1] = (ce as u64 & M62) as i64;
    ce >>= 62;

    // Limb 3.
    cd += (u as i128) * (d3 as i128) + (v as i128) * (e3 as i128);
    ce += (q as i128) * (d3 as i128) + (r as i128) * (e3 as i128);
    if m.v[3] != 0 {
        cd += (m.v[3] as i128) * (md as i128);
        ce += (m.v[3] as i128) * (me as i128);
    }
    d.v[2] = (cd as u64 & M62) as i64;
    cd >>= 62;
    e.v[2] = (ce as u64 & M62) as i64;
    ce >>= 62;

    // Limb 4.
    cd += (u as i128) * (d4 as i128) + (v as i128) * (e4 as i128);
    ce += (q as i128) * (d4 as i128) + (r as i128) * (e4 as i128);
    // Limb 4 of modulus is always nonzero for both p and n.
    cd += (m.v[4] as i128) * (md as i128);
    ce += (m.v[4] as i128) * (me as i128);
    d.v[3] = (cd as u64 & M62) as i64;
    cd >>= 62;
    e.v[3] = (ce as u64 & M62) as i64;
    ce >>= 62;

    // The remainder is limb 5 (the full output limb 4).
    d.v[4] = cd as i64;
    e.v[4] = ce as i64;
}

/// Apply transition matrix `t` (scaled by 2^62) to the (f, g) pair,
/// operating on the first `len` limbs.
///
/// C: `secp256k1_modinv64_update_fg_62_var`
fn update_fg_62_var(len: usize, f: &mut Signed62, g: &mut Signed62, t: &Trans2x2) {
    let u = t.u;
    let v = t.v;
    let q = t.q;
    let r = t.r;

    let fi = f.v[0];
    let gi = g.v[0];
    let mut cf: i128 = (u as i128) * (fi as i128) + (v as i128) * (gi as i128);
    let mut cg: i128 = (q as i128) * (fi as i128) + (r as i128) * (gi as i128);

    // Bottom 62 bits must be zero (verify in debug).
    debug_assert!((cf as u64) & M62 == 0);
    debug_assert!((cg as u64) & M62 == 0);
    cf >>= 62;
    cg >>= 62;

    for i in 1..len {
        let fi = f.v[i];
        let gi = g.v[i];
        cf += (u as i128) * (fi as i128) + (v as i128) * (gi as i128);
        cg += (q as i128) * (fi as i128) + (r as i128) * (gi as i128);
        f.v[i - 1] = (cf as u64 & M62) as i64;
        cf >>= 62;
        g.v[i - 1] = (cg as u64 & M62) as i64;
        cg >>= 62;
    }

    f.v[len - 1] = cf as i64;
    g.v[len - 1] = cg as i64;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 3  Public entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the modular inverse of `x` in-place: x ← x^{−1} mod modulus.
///
/// Variable-time.  The result is normalised to [0, modulus).
///
/// C: `secp256k1_modinv64_var`
pub fn modinv64_var(x: &mut Signed62, modinfo: &ModInfo) {
    // d tracks the modular inverse; e is a scratch variable.
    // f starts as modulus, g starts as x.
    let mut d = Signed62 { v: [0; 5] };
    let mut e = Signed62 { v: [1, 0, 0, 0, 0] };
    let mut f = modinfo.modulus;
    let mut g = *x;

    let mut eta: i64 = -1; // eta = −delta; delta starts at 1
    let mut len: usize = 5;

    loop {
        let mut t = Trans2x2 {
            u: 0,
            v: 0,
            q: 0,
            r: 0,
        };
        eta = divsteps_62_var(eta, f.v[0] as u64, g.v[0] as u64, &mut t);
        update_de_62(&mut d, &mut e, &t, modinfo);
        update_fg_62_var(len, &mut f, &mut g, &t);

        // Check whether g == 0.
        if g.v[0] == 0 {
            let mut cond: i64 = 0;
            for j in 1..len {
                cond |= g.v[j];
            }
            if cond == 0 {
                break;
            }
        }

        // Try to shrink the active limb count if the top limb of both f and g
        // is 0 or −1 (i.e. fully represented by the sign extension of limb len−2).
        let fn_ = f.v[len - 1];
        let gn = g.v[len - 1];
        let mut cond: i64 = (len as i64 - 2) >> 63; // −1 when len < 2
        cond |= fn_ ^ (fn_ >> 63); // 0 iff fn_ ∈ {0, −1}
        cond |= gn ^ (gn >> 63);
        if cond == 0 {
            // Merge the sign bit of the top limb into the one below.
            f.v[len - 2] = (f.v[len - 2] as u64 | ((fn_ as u64) << 62)) as i64;
            g.v[len - 2] = (g.v[len - 2] as u64 | ((gn as u64) << 62)) as i64;
            len -= 1;
        }
    }

    // d now holds ±the modular inverse; normalise to [0, modulus).
    normalize_62(&mut d, f.v[len - 1], modinfo);
    *x = d;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 4  Modinfo constants and conversion helpers for Fe and Scalar
//      (Inline here to keep the ecdsa_ref module's public surface clean.)
// ─────────────────────────────────────────────────────────────────────────────

// ── 4a  Field element (secp256k1 prime p) ────────────────────────────────────
//
// p = 2^256 − 2^32 − 977
// In signed62 representation:
//   v[0] = −2^32 − 977  = −0x1000003d1   (62-bit two's-complement → −R)
//   v[1..3] = 0
//   v[4] = 2^(256−4·62) = 256
//
// C: field_5x52_impl.h  `secp256k1_const_modinfo_fe`

/// Modinfo for p (field prime).  Used by `fe_inv_var`.
pub const FE_MODINFO: ModInfo = ModInfo {
    modulus: Signed62 {
        v: [-0x1000003D1i64, 0, 0, 0, 256],
    },
    // C: field_5x52_impl.h  `secp256k1_const_modinfo_fe.modulus_inv62`
    modulus_inv62: 0x27C7_F6E2_2DDA_CACFu64,
};

// ── 4b  Scalar (group order n) ───────────────────────────────────────────────
//
// n = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
// In signed62 representation:
//   v[0] = 0x3FD25E8CD0364141
//   v[1] = 0x2ABB739ABD2280EE
//   v[2] = −0x15  (= 0xFFFFFFFFFFFFFFEB as signed, or just -21)
//   v[3] = 0
//   v[4] = 256
//
// C: scalar_4x64_impl.h  `secp256k1_const_modinfo_scalar`

/// Modinfo for n (group order).  Used by `scalar_inv_var`.
pub const SCALAR_MODINFO: ModInfo = ModInfo {
    modulus: Signed62 {
        v: [
            0x3FD25E8CD0364141i64,
            0x2ABB739ABD2280EEi64,
            -0x15i64,
            0,
            256,
        ],
    },
    // C: scalar_4x64_impl.h  `secp256k1_const_modinfo_scalar.modulus_inv62`
    modulus_inv62: 0x34F2_0099_AA77_4EC1u64,
};
