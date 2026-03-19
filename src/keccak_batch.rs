//! Batch Keccak-256 for x86_64 using AVX-512BW/F.
//!
//! Processes 8 independent byte streams simultaneously by holding the 5×5
//! Keccak-f[1600] state across 25 ZMM registers:
//!
//! ```text
//! ZMM[lane] = [ stream7.lane, stream6.lane, … stream0.lane ]
//! ```
//!
//! Each permutation call therefore advances all 8 states in lockstep with one
//! set of AVX-512 instructions, giving ~8× throughput for the permutation
//! compared to a scalar loop.
//!
//! Variable-length inputs: shared complete blocks are absorbed vectorially;
//! if streams diverge in block count (different lengths), the module extracts
//! per-stream scalar states and finishes them with the reference implementation.

use crate::keccak::keccak256 as keccak256_scalar;

const RATE: usize = 136; // bytes absorbed per permutation

// ── Round constants (ι step) ─────────────────────────────────────────────────
const RC: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808a,
    0x8000000080008000,
    0x000000000000808b,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008a,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000a,
    0x000000008000808b,
    0x800000000000008b,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800a,
    0x800000008000000a,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

// ── AVX-512 implementation ────────────────────────────────────────────────────
#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::{RATE, RC, keccak256_scalar};
    use std::arch::x86_64::*;

    /// 25 ZMM registers holding 8 parallel Keccak-f[1600] states.
    ///
    /// `S[i]` = the i-th lane (x + 5·y) from all 8 streams packed into one ZMM.
    type State = [__m512i; 25];

    // ── rol macro: rotation by 0 is an identity, saves an instruction ────────
    macro_rules! rol {
        ($v:expr, 0) => {
            $v
        };
        ($v:expr, $n:literal) => {
            _mm512_rol_epi64($v, $n)
        };
    }

    #[inline(always)]
    unsafe fn xor5(a: __m512i, b: __m512i, c: __m512i, d: __m512i, e: __m512i) -> __m512i {
        _mm512_xor_si512(
            _mm512_xor_si512(a, b),
            _mm512_xor_si512(_mm512_xor_si512(c, d), e),
        )
    }

    /// One Keccak-f[1600] permutation over 8 parallel states.
    ///
    /// Implements θ → ρ+π → χ → ι for all 24 rounds in-place.
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn permute(s: &mut State) {
        for ri in 0..24usize {
            // ── θ ────────────────────────────────────────────────────────────
            // C[x] = column XOR across all 5 rows
            let c0 = xor5(s[0], s[5], s[10], s[15], s[20]);
            let c1 = xor5(s[1], s[6], s[11], s[16], s[21]);
            let c2 = xor5(s[2], s[7], s[12], s[17], s[22]);
            let c3 = xor5(s[3], s[8], s[13], s[18], s[23]);
            let c4 = xor5(s[4], s[9], s[14], s[19], s[24]);

            // D[x] = C[x-1] ^ rol(C[x+1], 1)
            let d0 = _mm512_xor_si512(c4, _mm512_rol_epi64(c1, 1));
            let d1 = _mm512_xor_si512(c0, _mm512_rol_epi64(c2, 1));
            let d2 = _mm512_xor_si512(c1, _mm512_rol_epi64(c3, 1));
            let d3 = _mm512_xor_si512(c2, _mm512_rol_epi64(c4, 1));
            let d4 = _mm512_xor_si512(c3, _mm512_rol_epi64(c0, 1));

            // A[x,y] ^= D[x]
            s[0] = _mm512_xor_si512(s[0], d0);
            s[5] = _mm512_xor_si512(s[5], d0);
            s[10] = _mm512_xor_si512(s[10], d0);
            s[15] = _mm512_xor_si512(s[15], d0);
            s[20] = _mm512_xor_si512(s[20], d0);
            s[1] = _mm512_xor_si512(s[1], d1);
            s[6] = _mm512_xor_si512(s[6], d1);
            s[11] = _mm512_xor_si512(s[11], d1);
            s[16] = _mm512_xor_si512(s[16], d1);
            s[21] = _mm512_xor_si512(s[21], d1);
            s[2] = _mm512_xor_si512(s[2], d2);
            s[7] = _mm512_xor_si512(s[7], d2);
            s[12] = _mm512_xor_si512(s[12], d2);
            s[17] = _mm512_xor_si512(s[17], d2);
            s[22] = _mm512_xor_si512(s[22], d2);
            s[3] = _mm512_xor_si512(s[3], d3);
            s[8] = _mm512_xor_si512(s[8], d3);
            s[13] = _mm512_xor_si512(s[13], d3);
            s[18] = _mm512_xor_si512(s[18], d3);
            s[23] = _mm512_xor_si512(s[23], d3);
            s[4] = _mm512_xor_si512(s[4], d4);
            s[9] = _mm512_xor_si512(s[9], d4);
            s[14] = _mm512_xor_si512(s[14], d4);
            s[19] = _mm512_xor_si512(s[19], d4);
            s[24] = _mm512_xor_si512(s[24], d4);

            // ── ρ+π → b ──────────────────────────────────────────────────────
            // b[new] = rol(s[old], rot)
            // Mapping precomputed from (x,y) → (y, (2x+3y)%5) with ROTATIONS offsets:
            //   (0,0)→(0,0)=0   (1,1)→(1,0)=44  (2,2)→(2,0)=43  (3,3)→(3,0)=21  (4,4)→(4,0)=14
            //   (3,0)→(0,1)=28  (4,1)→(1,1)=20  (0,2)→(2,1)=3   (1,3)→(3,1)=45  (2,4)→(4,1)=61
            //   (1,0)→(0,2)=1   (2,1)→(1,2)=6   (3,2)→(2,2)=25  (4,3)→(3,2)=8   (0,4)→(4,2)=18
            //   (4,0)→(0,3)=27  (0,1)→(1,3)=36  (1,2)→(2,3)=10  (2,3)→(3,3)=15  (3,4)→(4,3)=56
            //   (2,0)→(0,4)=62  (3,1)→(1,4)=55  (4,2)→(2,4)=39  (0,3)→(3,4)=41  (1,4)→(4,4)=2
            let b0 = rol!(s[0], 0);
            let b1 = rol!(s[6], 44);
            let b2 = rol!(s[12], 43);
            let b3 = rol!(s[18], 21);
            let b4 = rol!(s[24], 14);
            let b5 = rol!(s[3], 28);
            let b6 = rol!(s[9], 20);
            let b7 = rol!(s[10], 3);
            let b8 = rol!(s[16], 45);
            let b9 = rol!(s[22], 61);
            let b10 = rol!(s[1], 1);
            let b11 = rol!(s[7], 6);
            let b12 = rol!(s[13], 25);
            let b13 = rol!(s[19], 8);
            let b14 = rol!(s[20], 18);
            let b15 = rol!(s[4], 27);
            let b16 = rol!(s[5], 36);
            let b17 = rol!(s[11], 10);
            let b18 = rol!(s[17], 15);
            let b19 = rol!(s[23], 56);
            let b20 = rol!(s[2], 62);
            let b21 = rol!(s[8], 55);
            let b22 = rol!(s[14], 39);
            let b23 = rol!(s[15], 41);
            let b24 = rol!(s[21], 2);

            // ── χ ────────────────────────────────────────────────────────────
            // s[x,y] = b[x,y] ^ (~b[x+1,y] & b[x+2,y])
            // _mm512_andnot_si512(a, b) = ~a & b
            s[0] = _mm512_xor_si512(b0, _mm512_andnot_si512(b1, b2));
            s[1] = _mm512_xor_si512(b1, _mm512_andnot_si512(b2, b3));
            s[2] = _mm512_xor_si512(b2, _mm512_andnot_si512(b3, b4));
            s[3] = _mm512_xor_si512(b3, _mm512_andnot_si512(b4, b0));
            s[4] = _mm512_xor_si512(b4, _mm512_andnot_si512(b0, b1));
            s[5] = _mm512_xor_si512(b5, _mm512_andnot_si512(b6, b7));
            s[6] = _mm512_xor_si512(b6, _mm512_andnot_si512(b7, b8));
            s[7] = _mm512_xor_si512(b7, _mm512_andnot_si512(b8, b9));
            s[8] = _mm512_xor_si512(b8, _mm512_andnot_si512(b9, b5));
            s[9] = _mm512_xor_si512(b9, _mm512_andnot_si512(b5, b6));
            s[10] = _mm512_xor_si512(b10, _mm512_andnot_si512(b11, b12));
            s[11] = _mm512_xor_si512(b11, _mm512_andnot_si512(b12, b13));
            s[12] = _mm512_xor_si512(b12, _mm512_andnot_si512(b13, b14));
            s[13] = _mm512_xor_si512(b13, _mm512_andnot_si512(b14, b10));
            s[14] = _mm512_xor_si512(b14, _mm512_andnot_si512(b10, b11));
            s[15] = _mm512_xor_si512(b15, _mm512_andnot_si512(b16, b17));
            s[16] = _mm512_xor_si512(b16, _mm512_andnot_si512(b17, b18));
            s[17] = _mm512_xor_si512(b17, _mm512_andnot_si512(b18, b19));
            s[18] = _mm512_xor_si512(b18, _mm512_andnot_si512(b19, b15));
            s[19] = _mm512_xor_si512(b19, _mm512_andnot_si512(b15, b16));
            s[20] = _mm512_xor_si512(b20, _mm512_andnot_si512(b21, b22));
            s[21] = _mm512_xor_si512(b21, _mm512_andnot_si512(b22, b23));
            s[22] = _mm512_xor_si512(b22, _mm512_andnot_si512(b23, b24));
            s[23] = _mm512_xor_si512(b23, _mm512_andnot_si512(b24, b20));
            s[24] = _mm512_xor_si512(b24, _mm512_andnot_si512(b20, b21));

            // ── ι ────────────────────────────────────────────────────────────
            s[0] = _mm512_xor_si512(s[0], _mm512_set1_epi64(RC[ri] as i64));
        }
    }

    /// XOR one RATE-byte block from each of the 8 streams into `state`, then permute.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn absorb_block(state: &mut State, blocks: [&[u8; RATE]; 8]) {
        for lane in 0..17usize {
            let off = lane * 8;
            // Pack 8 u64 little-endian lane values (one per stream) into one ZMM.
            let v = _mm512_set_epi64(
                i64::from_le_bytes(blocks[7][off..off + 8].try_into().unwrap()),
                i64::from_le_bytes(blocks[6][off..off + 8].try_into().unwrap()),
                i64::from_le_bytes(blocks[5][off..off + 8].try_into().unwrap()),
                i64::from_le_bytes(blocks[4][off..off + 8].try_into().unwrap()),
                i64::from_le_bytes(blocks[3][off..off + 8].try_into().unwrap()),
                i64::from_le_bytes(blocks[2][off..off + 8].try_into().unwrap()),
                i64::from_le_bytes(blocks[1][off..off + 8].try_into().unwrap()),
                i64::from_le_bytes(blocks[0][off..off + 8].try_into().unwrap()),
            );
            state[lane] = _mm512_xor_si512(state[lane], v);
        }
        permute(state);
    }

    /// Extract 32-byte digests for all 8 streams from the first 4 lanes.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn squeeze(state: &State) -> [[u8; 32]; 8] {
        let mut out = [[0u8; 32]; 8];
        // Each of state[0..4] holds the same lane from all 8 streams.
        // Store to a temporary array and scatter into per-stream output.
        let mut tmp = [0i64; 8];
        for lane in 0..4usize {
            _mm512_storeu_si512(tmp.as_mut_ptr() as *mut __m512i, state[lane]);
            for stream in 0..8usize {
                let off = lane * 8;
                out[stream][off..off + 8].copy_from_slice(&(tmp[stream] as u64).to_le_bytes());
            }
        }
        out
    }

    /// Apply Keccak-256 padding to a partial block buffer at byte `offset`.
    fn apply_padding(buf: &mut [u8; RATE], offset: usize) {
        buf[offset] = 0x01;
        for b in buf[offset + 1..RATE - 1].iter_mut() {
            *b = 0;
        }
        buf[RATE - 1] = 0x80;
    }

    /// Extract scalar lane state for one stream from a SIMD state.
    unsafe fn extract_scalar_state(state: &State, stream: usize) -> [u64; 25] {
        let mut out = [0u64; 25];
        let mut tmp = [0i64; 8];
        for lane in 0..25usize {
            _mm512_storeu_si512(tmp.as_mut_ptr() as *mut __m512i, state[lane]);
            out[lane] = tmp[stream] as u64;
        }
        out
    }

    /// Finish hashing `remaining` bytes into a scalar Keccak state (already partially absorbed)
    /// and return the 32-byte digest.  Mirrors keccak256's absorb+pad+squeeze logic.
    fn finish_scalar(mut state: [u64; 25], remaining: &[u8]) -> [u8; 32] {
        // Absorb complete blocks.
        let mut buf = [0u8; RATE];
        let mut offset = 0usize;
        for chunk in remaining.chunks(RATE) {
            if chunk.len() == RATE {
                let block: &[u8; RATE] = chunk.try_into().unwrap();
                for i in 0..17usize {
                    let lane = u64::from_le_bytes(block[8 * i..8 * i + 8].try_into().unwrap());
                    state[i] ^= lane;
                }
                keccak_f1600_scalar(&mut state);
            } else {
                buf[..chunk.len()].copy_from_slice(chunk);
                offset = chunk.len();
            }
        }
        // Pad and final absorb.
        apply_padding(&mut buf, offset);
        for i in 0..17usize {
            let lane = u64::from_le_bytes(buf[8 * i..8 * i + 8].try_into().unwrap());
            state[i] ^= lane;
        }
        keccak_f1600_scalar(&mut state);
        // Squeeze.
        let mut digest = [0u8; 32];
        for i in 0..4usize {
            digest[8 * i..8 * i + 8].copy_from_slice(&state[i].to_le_bytes());
        }
        digest
    }

    /// Scalar Keccak-f[1600] permutation (forwarded from keccak.rs constants).
    fn keccak_f1600_scalar(state: &mut [u64; 25]) {
        const ROTATIONS: [u32; 25] = [
            0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61,
            56, 14,
        ];
        for round in 0..24 {
            let mut c = [0u64; 5];
            for x in 0..5 {
                c[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
            }
            let mut d = [0u64; 5];
            for x in 0..5 {
                d[x] = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
            }
            for y in 0..5 {
                for x in 0..5 {
                    state[x + 5 * y] ^= d[x];
                }
            }
            let mut b = [0u64; 25];
            for y in 0..5usize {
                for x in 0..5usize {
                    b[y + 5 * ((2 * x + 3 * y) % 5)] =
                        state[x + 5 * y].rotate_left(ROTATIONS[x + 5 * y]);
                }
            }
            for y in 0..5 {
                for x in 0..5 {
                    state[x + 5 * y] =
                        b[x + 5 * y] ^ ((!b[(x + 1) % 5 + 5 * y]) & b[(x + 2) % 5 + 5 * y]);
                }
            }
            state[0] ^= RC[round];
        }
    }

    /// Process 8 byte streams through Keccak-256 in parallel using AVX-512.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub(super) unsafe fn keccak256_batch_impl(inputs: [&[u8]; 8]) -> [[u8; 32]; 8] {
        let mut state: State = [_mm512_setzero_si512(); 25];

        // ── Vectorised absorption of all complete blocks shared by every stream ──
        // We process block positions 0..shared_full_blocks where every stream
        // still has a full RATE-byte chunk available.
        let min_len = inputs.iter().map(|s| s.len()).min().unwrap_or(0);
        let shared_full_blocks = min_len / RATE;

        for b in 0..shared_full_blocks {
            let off = b * RATE;
            absorb_block(
                &mut state,
                std::array::from_fn(|i| inputs[i][off..off + RATE].try_into().unwrap()),
            );
        }

        // ── Remaining bytes after the shared prefix ───────────────────────────
        let rems: [&[u8]; 8] = std::array::from_fn(|i| &inputs[i][shared_full_blocks * RATE..]);

        // Fast path: all streams have identical remaining length.
        // This covers the primary use case (8 × 64-byte pubkey buffers).
        if rems[1..].iter().all(|r| r.len() == rems[0].len()) {
            let rem_len = rems[0].len();
            let rem_full = rem_len / RATE;

            // Absorb any further complete blocks.
            for b in 0..rem_full {
                let off = b * RATE;
                absorb_block(
                    &mut state,
                    std::array::from_fn(|i| rems[i][off..off + RATE].try_into().unwrap()),
                );
            }

            // Build and absorb 8 individual padded final blocks.
            let partial_off = rem_full * RATE;
            let partial_len = rem_len - partial_off;
            let mut pads: [[u8; RATE]; 8] = [[0u8; RATE]; 8];
            for i in 0..8 {
                pads[i][..partial_len].copy_from_slice(&rems[i][partial_off..]);
                apply_padding(&mut pads[i], partial_len);
            }
            absorb_block(&mut state, std::array::from_fn(|i| &pads[i]));

            squeeze(&state)
        } else {
            // Slow path: extract per-stream scalar states and finish independently.
            // This path is taken only when inputs have different lengths.
            let mut out = [[0u8; 32]; 8];
            for stream in 0..8 {
                let scalar = extract_scalar_state(&state, stream);
                out[stream] = finish_scalar(scalar, rems[stream]);
            }
            out
        }
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Compute Keccak-256 of 8 byte streams simultaneously using AVX-512BW/F.
///
/// Returns 8 independent 32-byte digests, `out[i]` corresponding to `inputs[i]`.
///
/// All 8 streams are processed in lockstep; the permutation runs once per
/// block position for all 8 states.  When all inputs have the same byte
/// length (the common case for Ethereum address derivation from 64-byte
/// uncompressed public keys), no scalar fallback is needed.
///
/// Falls back to the scalar [`keccak::keccak256`] for each stream when the
/// inputs have different lengths and diverge before the final block.
#[cfg(target_arch = "x86_64")]
pub fn keccak256_batch(inputs: [&[u8]; 8]) -> [[u8; 32]; 8] {
    // Runtime feature detection so the binary runs on older CPUs too.
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
        unsafe { avx512::keccak256_batch_impl(inputs) }
    } else {
        // Scalar fallback on CPUs without AVX-512.
        std::array::from_fn(|i| keccak256_scalar(inputs[i]))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::keccak::keccak256 as keccak256_scalar;

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    /// Batch output must exactly match 8 independent scalar calls.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_batch_matches_scalar_empty() {
        let inputs: [&[u8]; 8] = [b""; 8];
        let batch = keccak256_batch(inputs);
        for digest in &batch {
            assert_eq!(
                hex(digest),
                "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_batch_matches_scalar_various() {
        let msgs: [&[u8]; 8] = [
            b"",
            b"abc",
            b"hello, world",
            b"Transfer(address,address,uint256)",
            &[0u8; 64],  // 64-byte pubkey buffer (primary use case)
            &[0u8; 136], // exactly one block
            &[0u8; 137], // just over one block
            b"Ethereum",
        ];
        let batch = keccak256_batch(msgs);
        for (i, digest) in batch.iter().enumerate() {
            let expected = keccak256_scalar(msgs[i]);
            assert_eq!(
                digest,
                &expected,
                "stream {i} mismatch: got {} expected {}",
                hex(digest),
                hex(&expected)
            );
        }
    }

    /// All-same-length inputs should take the fast vectorised path.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_batch_uniform_64bytes() {
        // Simulate 8 different uncompressed pubkey buffers.
        let bufs: Vec<[u8; 64]> = (0..8u8).map(|i| [i; 64]).collect();
        let inputs: [&[u8]; 8] = std::array::from_fn(|i| bufs[i].as_slice());
        let batch = keccak256_batch(inputs);
        for (i, digest) in batch.iter().enumerate() {
            let expected = keccak256_scalar(inputs[i]);
            assert_eq!(digest, &expected, "stream {i} mismatch");
        }
    }
}
