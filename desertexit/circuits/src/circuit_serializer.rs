// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use core::marker::PhantomData;

use circuit::bigint::div_rem::BigUintDivRemGenerator;
use circuit::byte::split_gate::{ByteDecompositionGate, ByteDecompositionGenerator};
use circuit::circuit_logger::LoggingGenerator;
use circuit::ecdsa::curve::curve_types::Curve;
use circuit::ecdsa::gadgets::glv::GLVDecompositionGenerator;
use circuit::eddsa::gadgets::base_field::{QuinticQuotientGenerator, QuinticSqrtGenerator};
use circuit::eddsa::gates::mul_quintic_ext_base::{
    QuinticMultiplicationBaseGenerator, QuinticMultiplicationGate,
};
use circuit::eddsa::gates::square_quintic_ext_base::{
    QuinticSquaringBaseGenerator, QuinticSquaringGate,
};
use circuit::hints::DivRemHintGenerator;
use circuit::nonnative::{
    NonNativeAdditionGenerator, NonNativeInverseGenerator, NonNativeMultipleAddsGenerator,
    NonNativeMultiplicationGenerator, NonNativeSubtractionGenerator,
};
use circuit::poseidon2::{Poseidon2, Poseidon2Gate, Poseidon2Generator};
use circuit::types::config::F;
use circuit::uint::range_check::{RangeCheckGate, RangeCheckGenerator};
use circuit::uint::u16::gates::add_many_u16::{U16AddManyGate, U16AddManyGenerator};
use circuit::uint::u16::gates::arithmetic_u16::{U16ArithmeticGate, U16ArithmeticGenerator};
use circuit::uint::u16::gates::subtraction_u16::{U16SubtractionGate, U16SubtractionGenerator};
use circuit::uint::u16::split::SplitToU16Generator;
use circuit::uint::u32::gadgets::arithmetic_u32::SplitToU32Generator;
use circuit::uint::u32::gates::add_many_u32::{U32AddManyGate, U32AddManyGenerator};
use circuit::uint::u32::gates::arithmetic_u32::{U32ArithmeticGate, U32ArithmeticGenerator};
use circuit::uint::u32::gates::comparison::{ComparisonGate, ComparisonGenerator};
use circuit::uint::u32::gates::interleave_u32::{U32InterleaveGate, U32InterleaveGenerator};
use circuit::uint::u32::gates::subtraction_u32::{U32SubtractionGate, U32SubtractionGenerator};
use circuit::uint::u32::gates::uninterleave_to_u32::{
    UninterleaveToU32Gate, UninterleaveToU32Generator,
};
use circuit::uint::u48::subtraction_u48::{U48SubtractionGate, U48SubtractionGenerator};
use circuit::{impl_gate_serializer, impl_generator_serializer};
use plonky2::field::extension::Extendable;
use plonky2::gadgets::arithmetic::EqualityGenerator;
use plonky2::gadgets::arithmetic_extension::QuotientGeneratorExtension;
use plonky2::gadgets::range_check::LowHighGenerator;
use plonky2::gadgets::split_base::BaseSumGenerator;
use plonky2::gadgets::split_join::{SplitGenerator, WireSplitGenerator};
use plonky2::gates::addition_base::{AdditionBaseGenerator, AdditionGate};
use plonky2::gates::arithmetic_base::{ArithmeticBaseGenerator, ArithmeticGate};
use plonky2::gates::arithmetic_extension::{ArithmeticExtensionGate, ArithmeticExtensionGenerator};
use plonky2::gates::base_sum::{BaseSplitGenerator, BaseSumGate};
use plonky2::gates::constant::ConstantGate;
use plonky2::gates::coset_interpolation::{CosetInterpolationGate, InterpolationGenerator};
use plonky2::gates::equality_base::{EqualityBaseGenerator, EqualityGate};
use plonky2::gates::exponentiation::{ExponentiationGate, ExponentiationGenerator};
use plonky2::gates::lookup::{LookupGate, LookupGenerator};
use plonky2::gates::lookup_table::{LookupTableGate, LookupTableGenerator};
use plonky2::gates::multiplication_base::{MultiplicationBaseGenerator, MultiplicationGate};
use plonky2::gates::multiplication_extension::{MulExtensionGate, MulExtensionGenerator};
use plonky2::gates::noop::NoopGate;
use plonky2::gates::poseidon::{PoseidonGate, PoseidonGenerator};
use plonky2::gates::poseidon_mds::{PoseidonMdsGate, PoseidonMdsGenerator};
use plonky2::gates::public_input::PublicInputGate;
use plonky2::gates::random_access::{RandomAccessGate, RandomAccessGenerator};
use plonky2::gates::reducing::{ReducingGate, ReducingGenerator};
use plonky2::gates::reducing_extension::{
    ReducingExtensionGate, ReducingGenerator as ReducingExtensionGenerator,
};
use plonky2::gates::select_base::{SelectionBaseGenerator, SelectionGate};
use plonky2::hash::hash_types::RichField;
use plonky2::iop::generator::{
    ConstantGenerator, CopyGenerator, NonzeroTestGenerator, RandomValueGenerator,
};
use plonky2::plonk::config::{AlgebraicHasher, GenericConfig};
use plonky2::util::serialization::{GateSerializer, WitnessGeneratorSerializer};

#[derive(Debug)]
pub struct DesertGateSerializer;
impl<F: RichField + Extendable<D> + Poseidon2, const D: usize> GateSerializer<F, D>
    for DesertGateSerializer
{
    impl_gate_serializer! {
        DesertGateSerializer,
        ArithmeticGate,
        ArithmeticExtensionGate<D>,
        BaseSumGate<2>,
        ConstantGate,
        CosetInterpolationGate<F, D>,
        EqualityGate,
        ExponentiationGate<F, D>,
        LookupGate,
        LookupTableGate,
        MulExtensionGate<D>,
        NoopGate,
        PoseidonMdsGate<F, D>,
        PoseidonGate<F, D>,
        PublicInputGate,
        RandomAccessGate<F, D>,
        ReducingExtensionGate<D>,
        ReducingGate<D>,
        BaseSumGate<4>,
        ComparisonGate<F, D>,
        U32AddManyGate<F, D>,
        U32ArithmeticGate<F, D>,
        RangeCheckGate<F, D>,
        U32SubtractionGate<F, D>,
        U16AddManyGate<F, D>,
        U16ArithmeticGate<F, D>,
        U16SubtractionGate<F, D>,
        U48SubtractionGate<F, D>,
        Poseidon2Gate<F, D>,
        AdditionGate,
        MultiplicationGate,
        QuinticMultiplicationGate,
        SelectionGate,
        QuinticSquaringGate,
        ByteDecompositionGate,
        U32InterleaveGate,
        UninterleaveToU32Gate
    }
}

#[derive(Debug, Default)]
pub struct DesertGeneratorSerializer<C: GenericConfig<D>, const D: usize, CC> {
    pub _phantom: PhantomData<C>,
    pub _phantom2: PhantomData<CC>,
}

impl<C, const D: usize, CC> WitnessGeneratorSerializer<F, D> for DesertGeneratorSerializer<C, D, CC>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F> + 'static,
    C::Hasher: AlgebraicHasher<F>,
    CC: Curve,
{
    impl_generator_serializer! {
        DesertGeneratorSerializer,
        ArithmeticBaseGenerator<F, D>,
        ArithmeticExtensionGenerator<F, D>,
        AdditionBaseGenerator<F,D>,
        MultiplicationBaseGenerator<F,D>,
        BaseSplitGenerator<2>,
        BaseSumGenerator<2>,
        ConstantGenerator<F>,
        CopyGenerator,
        EqualityGenerator,
        EqualityBaseGenerator<F, D>,
        ExponentiationGenerator<F, D>,
        InterpolationGenerator<F, D>,
        LookupGenerator,
        LookupTableGenerator,
        LowHighGenerator,
        MulExtensionGenerator<F, D>,
        NonzeroTestGenerator,
        PoseidonGenerator<F, D>,
        PoseidonMdsGenerator<D>,
        QuotientGeneratorExtension<D>,
        RandomAccessGenerator<F, D>,
        RandomValueGenerator,
        ReducingGenerator<D>,
        ReducingExtensionGenerator<D>,
        SplitGenerator,
        WireSplitGenerator,
        QuinticSqrtGenerator,
        QuinticQuotientGenerator,
        DivRemHintGenerator<F, D>,
        SplitToU32Generator<F, D>,
        BigUintDivRemGenerator<F, D>,
        Poseidon2Generator<F, D>,
        BaseSplitGenerator<4>,
        ComparisonGenerator<F, D>,
        U32SubtractionGenerator<F, D>,
        U32AddManyGenerator<F, D>,
        U32ArithmeticGenerator<F, D>,
        U16SubtractionGenerator<F, D>,
        U16AddManyGenerator<F, D>,
        U16ArithmeticGenerator<F, D>,
        SplitToU16Generator<F, D>,
        U48SubtractionGenerator<F, D>,
        RangeCheckGenerator<F, D>,
        LoggingGenerator<F, D>,
        SelectionBaseGenerator<F, D>,
        QuinticSquaringBaseGenerator<F,D>,
        QuinticMultiplicationBaseGenerator<F,D>,
        ByteDecompositionGenerator,
        GLVDecompositionGenerator<F, D>,
        U32InterleaveGenerator,
        UninterleaveToU32Generator,
        NonNativeMultiplicationGenerator<F, D, CC::BaseField>,
        NonNativeSubtractionGenerator<F, D, CC::BaseField>,
        NonNativeMultipleAddsGenerator<F, D, CC::BaseField>,
        NonNativeInverseGenerator<F, D, CC::BaseField>,
        NonNativeAdditionGenerator<F, D, CC::BaseField>,
        NonNativeMultiplicationGenerator<F, D, CC::ScalarField>,
        NonNativeSubtractionGenerator<F, D, CC::ScalarField>,
        NonNativeMultipleAddsGenerator<F, D, CC::ScalarField>,
        NonNativeInverseGenerator<F, D, CC::ScalarField>,
        NonNativeAdditionGenerator<F, D, CC::ScalarField>
    }
}
