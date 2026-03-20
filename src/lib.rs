#![doc = include_str!("../README.md")]

#[doc(hidden)]
pub mod ecdsa_scalar;
#[doc(hidden)]
pub mod ecdsa_batch;
#[doc(hidden)]
pub mod ecdsa_ref;
#[doc(hidden)]
pub mod keccak_scalar;
#[doc(hidden)]
pub mod keccak_batch;

#[cfg(target_arch = "x86_64")]
pub use ecdsa_batch::recover_addresses_batch;
#[cfg(target_arch = "x86_64")]
pub use keccak_batch::keccak256_batch;
