//! secp256k1 public key used by ECDSA-signed api keys.
//!
//! Natively the key is carried as the compressed SEC1 encoding (33 bytes); in
//! circuit it is the x coordinate as 8 little-endian u32 limbs plus the y
//! parity bit, which is exactly the layout `p3_ecdsa` absorbs into its public
//! digest.

use anyhow::Result;
use num::{BigUint, Zero};
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use super::config::{Builder, F};
use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint, WitnessBigUint};
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::uint::u32::gadgets::arithmetic_u32::CircuitBuilderU32;

pub const ECDSA_PUB_KEY_X_LIMBS: usize = 8;
pub const ECDSA_PUB_KEY_FIELD_ELEMENTS: usize = ECDSA_PUB_KEY_X_LIMBS + 1;
pub const ECDSA_PUB_KEY_COMPRESSED_LEN: usize = 33;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EcdsaPubKey {
    pub x: BigUint,
    pub y_odd: bool,
}

impl EcdsaPubKey {
    pub fn zero() -> Self {
        Self::default()
    }

    /// The zero (empty) key is `x == 0` with even parity; `(0, odd)` is not treated as empty.
    pub fn is_zero(&self) -> bool {
        self.x.is_zero() && !self.y_odd
    }

    pub fn from_compressed(bytes: &[u8; ECDSA_PUB_KEY_COMPRESSED_LEN]) -> Result<Self> {
        if bytes[0] == 2 && bytes[1..].iter().all(|&b| b == 0) {
            return Ok(Self::zero());
        }
        let y_odd = match bytes[0] {
            2 => false,
            3 => true,
            prefix => anyhow::bail!("invalid compressed secp256k1 prefix {prefix}"),
        };
        let x = BigUint::from_bytes_be(&bytes[1..]);
        let key = EcdsaPubKey { x, y_odd };
        if key.to_point().is_none() {
            anyhow::bail!("secp256k1 x coordinate is not on the curve");
        }
        Ok(key)
    }

    pub fn from_point(point: &p3_ecdsa::secp::Point) -> Self {
        EcdsaPubKey {
            x: point.x.clone(),
            y_odd: point.y.bit(0),
        }
    }

    pub fn to_compressed(&self) -> [u8; ECDSA_PUB_KEY_COMPRESSED_LEN] {
        let mut out = [0u8; ECDSA_PUB_KEY_COMPRESSED_LEN];
        out[0] = if self.y_odd { 3 } else { 2 };
        if self.is_zero() {
            return out;
        }
        let x = self.x.to_bytes_be();
        out[1 + (32 - x.len())..].copy_from_slice(&x);
        out
    }

    pub fn to_point(&self) -> Option<p3_ecdsa::secp::Point> {
        if self.is_zero() {
            return None;
        }
        p3_ecdsa::secp::decompress(&self.x, self.y_odd)
    }

    /// `x` as 8 little-endian u32 limbs followed by the parity bit; matches
    /// `EcdsaPubKeyTarget::to_targets`.
    pub fn field_elements(&self) -> [F; ECDSA_PUB_KEY_FIELD_ELEMENTS] {
        let limbs = self.x.to_u32_digits();
        assert!(limbs.len() <= ECDSA_PUB_KEY_X_LIMBS, "x exceeds 256 bits");
        core::array::from_fn(|i| {
            if i < ECDSA_PUB_KEY_X_LIMBS {
                F::from_canonical_u32(limbs.get(i).copied().unwrap_or(0))
            } else {
                F::from_bool(self.y_odd)
            }
        })
    }
}

#[derive(Debug, Clone, Deserialize, Default, serde::Serialize)]
pub struct EcdsaPubKeyTarget {
    pub x: BigUintTarget,
    pub y_odd: BoolTarget,
}

impl EcdsaPubKeyTarget {
    pub fn new(builder: &mut Builder) -> Self {
        EcdsaPubKeyTarget {
            x: builder.add_virtual_biguint_target_safe(ECDSA_PUB_KEY_X_LIMBS),
            y_odd: builder.add_virtual_bool_target_safe(),
        }
    }

    pub fn zero(builder: &mut Builder) -> Self {
        EcdsaPubKeyTarget {
            x: zero_u256(builder),
            y_odd: builder._false(),
        }
    }

    /// Matches `EcdsaPubKey::is_zero`: `x == 0` and `y_odd == false`.
    pub fn is_zero(&self, builder: &mut Builder) -> BoolTarget {
        let x_is_zero = builder.is_zero_biguint(&self.x);
        builder.and_not(x_is_zero, self.y_odd)
    }

    pub fn to_targets(&self) -> [Target; ECDSA_PUB_KEY_FIELD_ELEMENTS] {
        assert_eq!(self.x.limbs.len(), ECDSA_PUB_KEY_X_LIMBS);
        core::array::from_fn(|i| {
            if i < ECDSA_PUB_KEY_X_LIMBS {
                self.x.limbs[i].0
            } else {
                self.y_odd.target
            }
        })
    }

    pub fn select(builder: &mut Builder, flag: BoolTarget, a: &Self, b: &Self) -> Self {
        EcdsaPubKeyTarget {
            x: builder.select_biguint(flag, &a.x, &b.x),
            y_odd: BoolTarget::new_unsafe(builder.select(flag, a.y_odd.target, b.y_odd.target)),
        }
    }

    pub fn connect(builder: &mut Builder, a: &Self, b: &Self) {
        builder.connect_biguint(&a.x, &b.x);
        builder.connect(a.y_odd.target, b.y_odd.target);
    }
}

/// A fixed-width (8 limb) zero, unlike `zero_biguint` which has no limbs.
pub fn zero_u256(builder: &mut Builder) -> BigUintTarget {
    BigUintTarget {
        limbs: (0..ECDSA_PUB_KEY_X_LIMBS)
            .map(|_| builder.zero_u32())
            .collect(),
    }
}

pub trait EcdsaPubKeyTargetWitness<F: PrimeField64> {
    fn set_ecdsa_pub_key_target(&mut self, a: &EcdsaPubKeyTarget, b: &EcdsaPubKey) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> EcdsaPubKeyTargetWitness<F> for T {
    fn set_ecdsa_pub_key_target(&mut self, a: &EcdsaPubKeyTarget, b: &EcdsaPubKey) -> Result<()> {
        self.set_biguint_target(&a.x, &b.x)?;
        self.set_bool_target(a.y_odd, b.y_odd)?;
        Ok(())
    }
}
