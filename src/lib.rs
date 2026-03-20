//! AsmCrypto — register-parallel cryptographic primitives.
//!
//! Modules:
//! * [`keccak`]        — Keccak-256 (Ethereum variant, padding byte `0x01`).
//! * [`keccak_batch`]  — AVX-512 batch Keccak-256: 8 streams × 1 permutation.
//! * [`ecdsa`]        — secp256k1 ECDSA public-key and address recovery (4×64-bit field).
//! * [`ecdsa_ref`]    — Exact Rust translation of the C secp256k1 library (5×52-bit field).
//! * [`ecdsa_batch`]  — AVX-512 batch address recovery: 8 signatures in parallel.

pub mod ecdsa;
pub mod ecdsa_batch;
pub mod ecdsa_ref;
pub mod keccak;
pub mod keccak_batch;
