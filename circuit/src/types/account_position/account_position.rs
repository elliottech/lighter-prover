// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use num::{BigInt, BigUint, FromPrimitive};
use plonky2::field::extension::Extendable;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::big_u16::{
    BigIntU16Target, CircuitBuilderBigIntU16, CircuitBuilderBiguint16, WitnessBigInt16,
};
use crate::bigint::bigint::{BigIntTarget, CircuitBuilderBigInt, SignTarget, WitnessBigInt};
use crate::bigint::biguint::CircuitBuilderBiguint;
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::circuit_logger::CircuitBuilderLogging;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::deserializers;
use crate::eddsa::gadgets::curve::PartialWitnessCurve;
use crate::hints::CircuitBuilderHints;
use crate::signed::signed_target::{CircuitBuilderSigned, SignedTarget};
use crate::types::config::{BIG_U96_LIMBS, BIGU16_U64_LIMBS, Builder, F};
use crate::types::constants::{
    ENTRY_QUOTE_BITS, MARGIN_FRACTION_BITS, MAX_ORDER_BASE_AMOUNT, ORDER_PRICE_BITS,
    POSITION_HASH_BUCKET_SIZE, POSITION_SIZE_BITS, QUOTE_MULTIPLIER_BITS,
    USDC_TO_COLLATERAL_MULTIPLIER,
};
use crate::types::market_details::MarketDetailsTarget;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct AccountPosition {
    #[serde(rename = "lfrps")]
    #[serde(deserialize_with = "deserializers::int_to_bigint")]
    pub last_funding_rate_prefix_sum: BigInt, // 63 bits

    #[serde(rename = "p")]
    #[serde(deserialize_with = "deserializers::int_to_bigint")]
    pub position: BigInt, // 56 bits

    #[serde(rename = "eq")]
    pub entry_quote: i64, // 56 bits

    #[serde(rename = "imf", default)]
    pub initial_margin_fraction: u16,

    #[serde(rename = "toc", default)]
    pub total_order_count: i64,

    #[serde(rename = "tptoc", default)]
    pub total_position_tied_order_count: i64,

    #[serde(rename = "mmd")]
    pub margin_mode: u8,

    #[serde(rename = "almrg")]
    #[serde(deserialize_with = "deserializers::int_to_bigint")]
    pub allocated_margin: BigInt,
}

#[derive(Debug, Clone, Default)]
pub struct AccountPositionTarget {
    pub last_funding_rate_prefix_sum: BigIntU16Target, // 63 bits
    pub position: BigIntU16Target,                     // 56 bits
    pub entry_quote: Target,                           // 56 bits
    pub initial_margin_fraction: Target,
    pub total_order_count: Target,
    pub total_position_tied_order_count: Target,
    pub margin_mode: Target,
    pub allocated_margin: BigIntTarget, // 96 bits
}

impl AccountPositionTarget {
    pub fn new(builder: &mut Builder) -> Self {
        AccountPositionTarget {
            last_funding_rate_prefix_sum: builder
                .add_virtual_bigint_u16_target_unsafe(BIGU16_U64_LIMBS), // safe because it is read from the state using merkle proofs
            position: builder.add_virtual_bigint_u16_target_unsafe(BIGU16_U64_LIMBS), // safe because it is read from the state using merkle proofs
            entry_quote: builder.add_virtual_target(),
            initial_margin_fraction: builder.add_virtual_target(),
            total_order_count: builder.add_virtual_target(),
            total_position_tied_order_count: builder.add_virtual_target(),
            margin_mode: builder.add_virtual_target(),
            allocated_margin: builder.add_virtual_bigint_target_unsafe(BIG_U96_LIMBS), // safe because it is read from the state using merkle proofs
        }
    }

    /// Caller needs to make sure margin_mode is boolean.
    pub fn is_isolated_unsafe(&self) -> BoolTarget {
        BoolTarget::new_unsafe(self.margin_mode)
    }

    pub fn is_cross_unsafe(&self, builder: &mut Builder) -> BoolTarget {
        builder.not(self.is_isolated_unsafe())
    }

    pub fn print(&self, builder: &mut Builder, tag: &str) {
        builder.println_bigint_u16(
            &self.last_funding_rate_prefix_sum,
            &format!("{} last_funding_rate_prefix_sum", tag),
        );
        builder.println_bigint_u16(&self.position, &format!("{} position", tag));
        builder.println(self.entry_quote, &format!("{} entry_quote", tag));
        builder.println(
            self.initial_margin_fraction,
            &format!("{} initial_margin_fraction", tag),
        );
        builder.println(
            self.total_order_count,
            &format!("{} total_order_count", tag),
        );
        builder.println(
            self.total_position_tied_order_count,
            &format!("{} total_position_tied_order_count", tag),
        );
        builder.println(self.margin_mode, &format!("{} margin_mode", tag));
        builder.println_bigint(&self.allocated_margin, &format!("{} allocated_margin", tag));
    }

    pub fn get_initial_margin_fraction(
        &self,
        builder: &mut Builder,
        default_imr: Target,
        min_imr: Target,
    ) -> Target {
        let position_imr_is_zero = builder.is_zero(self.initial_margin_fraction);
        let position_imr = builder.select(
            position_imr_is_zero,
            default_imr,
            self.initial_margin_fraction,
        );

        let min_imr_lte_position_imr = builder.is_lte(min_imr, position_imr, MARGIN_FRACTION_BITS);
        builder.select(min_imr_lte_position_imr, position_imr, min_imr)
    }

    pub fn is_order_or_position_open(&self, builder: &mut Builder) -> BoolTarget {
        let zero = builder.zero_bigint_u16();
        let is_order_open = builder.is_not_zero(self.total_order_count);
        let is_position_open = builder.is_not_equal_bigint_u16(&self.position, &zero);

        builder.or(is_order_open, is_position_open)
    }

    pub fn empty(builder: &mut Builder) -> Self {
        AccountPositionTarget {
            last_funding_rate_prefix_sum: builder.zero_bigint_u16(),
            position: builder.zero_bigint_u16(),
            entry_quote: builder.zero(),
            initial_margin_fraction: builder.zero(),
            total_order_count: builder.zero(),
            total_position_tied_order_count: builder.zero(),
            margin_mode: builder.zero(),
            allocated_margin: builder.zero_bigint(),
        }
    }

    pub fn calculate_position_tied_order_base_amount(
        &self,
        builder: &mut Builder,
        quote_multiplier: Target,
        price: Target,
        order_quote_limit: Target,
    ) -> Target {
        let abs_position = builder.biguint_u16_to_target(&self.position.abs);

        let max_order_base_amount = builder.constant_u64(MAX_ORDER_BASE_AMOUNT);

        let (max_order_quote_amount_div_quote_multiplier, _) =
            builder.div_rem(order_quote_limit, quote_multiplier, QUOTE_MULTIPLIER_BITS);
        let (max_base_amount_for_quote, _) = builder.div_rem(
            max_order_quote_amount_div_quote_multiplier,
            price,
            ORDER_PRICE_BITS,
        );

        builder.min(
            &[
                abs_position,
                max_order_base_amount,
                max_base_amount_for_quote,
            ],
            63,
        )
    }

    pub fn calculate_aggregated_usdc(&self, builder: &mut Builder) -> BigIntTarget {
        let usdc_to_collateral_multiplier =
            builder.constant_biguint(&BigUint::from_u32(USDC_TO_COLLATERAL_MULTIPLIER).unwrap());
        let aggregated_usdc = builder.euclidian_div_by_biguint(
            &self.allocated_margin,
            &usdc_to_collateral_multiplier,
            BIG_U96_LIMBS,
        );
        let aggregated_entry_quote = BigIntTarget {
            abs: builder.target_to_biguint(self.entry_quote),
            sign: builder.negate_sign(self.position.sign),
        };
        builder.add_bigint_non_carry(&aggregated_usdc, &aggregated_entry_quote, BIG_U96_LIMBS)
    }

    pub fn is_valid(&self, builder: &mut Builder) -> BoolTarget {
        let max_entry_quote =
            builder.constant(F::from_canonical_u64((1u64 << ENTRY_QUOTE_BITS) - 1));
        let max_position_size =
            builder.constant_biguint(&BigUint::from_u64((1u64 << POSITION_SIZE_BITS) - 1).unwrap());

        let position_abs = builder.biguint_u16_to_biguint(&self.position.abs);
        let is_position_size_valid = builder.is_lte_biguint(&position_abs, &max_position_size);
        let is_entry_quote_valid = builder.is_lte(self.entry_quote, max_entry_quote, 64);

        builder.and(is_entry_quote_valid, is_position_size_valid)
    }

    pub fn partial_select_for_cross_risk(
        builder: &mut Builder,
        flag: BoolTarget,
        sub_position: &Self,
        owner_position: &Self,
    ) -> Self {
        Self {
            position: builder.select_bigint_u16(
                flag,
                &sub_position.position,
                &owner_position.position,
            ),
            last_funding_rate_prefix_sum: builder.select_bigint_u16(
                flag,
                &sub_position.last_funding_rate_prefix_sum,
                &owner_position.last_funding_rate_prefix_sum,
            ),
            entry_quote: builder.select(flag, sub_position.entry_quote, owner_position.entry_quote),
            initial_margin_fraction: builder.select(
                flag,
                sub_position.initial_margin_fraction,
                owner_position.initial_margin_fraction,
            ),
            margin_mode: builder.select(flag, sub_position.margin_mode, owner_position.margin_mode),

            // No need to select allocated_margin as insurance fund doesn't have isolated positions
            ..owner_position.clone()
        }
    }

    pub fn select_position(builder: &mut Builder, flag: BoolTarget, a: &Self, b: &Self) -> Self {
        Self {
            position: builder.select_bigint_u16(flag, &a.position, &b.position),
            last_funding_rate_prefix_sum: builder.select_bigint_u16(
                flag,
                &a.last_funding_rate_prefix_sum,
                &b.last_funding_rate_prefix_sum,
            ),
            entry_quote: builder.select(flag, a.entry_quote, b.entry_quote),
            initial_margin_fraction: builder.select(
                flag,
                a.initial_margin_fraction,
                b.initial_margin_fraction,
            ),
            total_order_count: builder.select(flag, a.total_order_count, b.total_order_count),
            total_position_tied_order_count: builder.select(
                flag,
                a.total_position_tied_order_count,
                b.total_position_tied_order_count,
            ),
            margin_mode: builder.select(flag, a.margin_mode, b.margin_mode),
            allocated_margin: builder.select_bigint(flag, &a.allocated_margin, &b.allocated_margin),
        }
    }

    pub fn select_position_bucket(
        builder: &mut Builder,
        flag: BoolTarget,
        a: &[Self; POSITION_HASH_BUCKET_SIZE],
        b: &[Self; POSITION_HASH_BUCKET_SIZE],
    ) -> [Self; POSITION_HASH_BUCKET_SIZE] {
        core::array::from_fn(|i| Self::select_position(builder, flag, &a[i], &b[i]))
    }

    pub fn diff(
        builder: &mut Builder,
        new: &AccountPositionTarget,
        old: &AccountPositionTarget,
    ) -> AccountPositionTarget {
        AccountPositionTarget {
            position: builder.bigint_u16_vector_diff(&new.position, &old.position),
            last_funding_rate_prefix_sum: builder.bigint_u16_vector_diff(
                &new.last_funding_rate_prefix_sum,
                &old.last_funding_rate_prefix_sum,
            ),
            entry_quote: builder.sub(new.entry_quote, old.entry_quote),
            initial_margin_fraction: builder
                .sub(new.initial_margin_fraction, old.initial_margin_fraction),
            total_order_count: builder.sub(new.total_order_count, old.total_order_count),
            total_position_tied_order_count: builder.sub(
                new.total_position_tied_order_count,
                old.total_position_tied_order_count,
            ),
            margin_mode: builder.sub(new.margin_mode, old.margin_mode),
            allocated_margin: builder
                .bigint_vector_diff(&new.allocated_margin, &old.allocated_margin),
        }
    }

    /// Calculates new position by applying the difference to an old position.
    /// If difference is calculated from same old position, result should return
    /// new position from transaction operations
    pub fn apply_diff(
        builder: &mut Builder,
        flag: BoolTarget,
        base: &AccountPositionTarget,
        diff: &AccountPositionTarget,
    ) -> AccountPositionTarget {
        AccountPositionTarget {
            position: builder.bigint_u16_vector_sum(flag, &diff.position, &base.position),
            last_funding_rate_prefix_sum: builder.bigint_u16_vector_sum(
                flag,
                &diff.last_funding_rate_prefix_sum,
                &base.last_funding_rate_prefix_sum,
            ),
            entry_quote: builder.mul_add(flag.target, diff.entry_quote, base.entry_quote),
            initial_margin_fraction: builder.mul_add(
                flag.target,
                diff.initial_margin_fraction,
                base.initial_margin_fraction,
            ),
            total_order_count: builder.mul_add(
                flag.target,
                diff.total_order_count,
                base.total_order_count,
            ),
            total_position_tied_order_count: builder.mul_add(
                flag.target,
                diff.total_position_tied_order_count,
                base.total_position_tied_order_count,
            ),
            margin_mode: builder.mul_add(flag.target, diff.margin_mode, base.margin_mode),
            allocated_margin: builder.bigint_vector_sum(
                flag,
                &diff.allocated_margin,
                &base.allocated_margin,
            ),
        }
    }
}

pub fn random_access_account_position(
    builder: &mut Builder,
    access_index: Target,
    v: Vec<AccountPositionTarget>,
) -> AccountPositionTarget {
    assert!(v.len() % 64 == 0);
    AccountPositionTarget {
        last_funding_rate_prefix_sum: builder.random_access_bigint_u16(
            access_index,
            v.iter()
                .map(|x| x.last_funding_rate_prefix_sum.clone())
                .collect(),
            BIGU16_U64_LIMBS,
        ),
        position: builder.random_access_bigint_u16(
            access_index,
            v.iter().map(|x| x.position.clone()).collect(),
            BIGU16_U64_LIMBS,
        ),
        entry_quote: builder.random_access(access_index, v.iter().map(|x| x.entry_quote).collect()),
        initial_margin_fraction: builder.random_access(
            access_index,
            v.iter().map(|x| x.initial_margin_fraction).collect(),
        ),
        total_order_count: builder.random_access(
            access_index,
            v.iter().map(|x| x.total_order_count).collect(),
        ),
        total_position_tied_order_count: builder.random_access(
            access_index,
            v.iter()
                .map(|x| x.total_position_tied_order_count)
                .collect(),
        ),
        margin_mode: builder.random_access(access_index, v.iter().map(|x| x.margin_mode).collect()),
        allocated_margin: builder.random_access_bigint(
            access_index,
            v.iter()
                .map(|x| x.allocated_margin.clone())
                .collect::<Vec<BigIntTarget>>()
                .as_ref(),
            BIG_U96_LIMBS,
        ),
    }
}

pub fn get_position_unrealized_pnl(
    builder: &mut Builder,
    market_details: &MarketDetailsTarget,
    position_abs: Target,
    position_sign: SignTarget,
    entry_quote: Target,
) -> SignedTarget {
    let multiplier = builder.mul(market_details.quote_multiplier, market_details.mark_price);

    let position_base_notional_signed =
        SignedTarget::new_unsafe(builder.mul(multiplier, position_abs));

    let result = builder.sub_signed(
        position_base_notional_signed,
        SignedTarget::new_unsafe(entry_quote),
    );
    let result_neg = builder.neg_signed(result);

    let position_is_positive = builder.is_sign_positive(position_sign);
    builder.select_signed(position_is_positive, result, result_neg)
}

pub trait AccountPositionTargetWitness<F: PrimeField64 + Extendable<5> + RichField> {
    fn set_position_target(&mut self, a: &AccountPositionTarget, b: &AccountPosition)
    -> Result<()>;
}

impl<T: Witness<F> + PartialWitnessCurve<F>, F: PrimeField64 + Extendable<5> + RichField>
    AccountPositionTargetWitness<F> for T
{
    fn set_position_target(
        &mut self,
        a: &AccountPositionTarget,
        b: &AccountPosition,
    ) -> Result<()> {
        self.set_bigint_u16_target(
            &a.last_funding_rate_prefix_sum,
            &b.last_funding_rate_prefix_sum,
        )?;
        self.set_bigint_u16_target(&a.position, &b.position)?;
        self.set_target(a.entry_quote, F::from_canonical_i64(b.entry_quote))?;
        self.set_target(
            a.initial_margin_fraction,
            F::from_canonical_u16(b.initial_margin_fraction),
        )?;
        self.set_target(a.margin_mode, F::from_canonical_u8(b.margin_mode))?;
        self.set_bigint_target(&a.allocated_margin, &b.allocated_margin)?;
        self.set_target(
            a.total_order_count,
            F::from_canonical_i64(b.total_order_count),
        )?;
        self.set_target(
            a.total_position_tied_order_count,
            F::from_canonical_i64(b.total_position_tied_order_count),
        )?;

        Ok(())
    }
}
