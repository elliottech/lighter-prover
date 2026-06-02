// Portions of this file are derived from plonky2
// Copyright (c) 2022-2025 The Plonky2 Authors
// Licensed under the MIT License. See THIRD_PARTY_NOTICES for details.

// Modifications copyright (c) 2025 Elliot Technologies, Inc.
// This file has been modified from its original version.

use core::marker::PhantomData;

use plonky2::field::extension::Extendable;
use plonky2::field::types::PrimeField;
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

use crate::bigint::div_rem::BigUintDivRemGenerator;
use crate::blob::evaluate_bitstream::{EvaluateBitstreamBaseGenerator, EvaluateBitstreamGate};
use crate::byte::split_gate::{ByteDecompositionGate, ByteDecompositionGenerator};
use crate::circuit_logger::LoggingGenerator;
use crate::delta::evaluate_sequence::{EvaluateSequenceBaseGenerator, EvaluateSequenceGate};
use crate::ecdsa::curve::curve_types::Curve;
use crate::ecdsa::gadgets::glv::GLVDecompositionGenerator;
use crate::eddsa::gadgets::base_field::{QuinticQuotientGenerator, QuinticSqrtGenerator};
use crate::eddsa::gates::mul_quintic_ext_base::{
    QuinticMultiplicationBaseGenerator, QuinticMultiplicationGate,
};
use crate::eddsa::gates::square_quintic_ext_base::{
    QuinticSquaringBaseGenerator, QuinticSquaringGate,
};
use crate::hints::DivRemHintGenerator;
use crate::nonnative::{
    NonNativeAdditionGenerator, NonNativeDivisionGenerator, NonNativeInverseGenerator,
    NonNativeMulDivGenerator, NonNativeMultipleAddsGenerator, NonNativeMultiplicationGenerator,
    NonNativeSubtractionGenerator,
};
use crate::poseidon2::{Poseidon2, Poseidon2Gate, Poseidon2Generator};
use crate::types::config::F;
use crate::uint::range_check::{RangeCheckGate, RangeCheckGenerator};
use crate::uint::u4::split::SplitToU4Generator;
use crate::uint::u16::gates::add_many_u16::{U16AddManyGate, U16AddManyGenerator};
use crate::uint::u16::gates::arithmetic_u16::{U16ArithmeticGate, U16ArithmeticGenerator};
use crate::uint::u16::gates::subtraction_u16::{U16SubtractionGate, U16SubtractionGenerator};
use crate::uint::u16::split::SplitToU16Generator;
use crate::uint::u32::gadgets::arithmetic_u32::SplitToU32Generator;
use crate::uint::u32::gates::add_many_u32::{U32AddManyGate, U32AddManyGenerator};
use crate::uint::u32::gates::arithmetic_u32::{U32ArithmeticGate, U32ArithmeticGenerator};
use crate::uint::u32::gates::comparison::{ComparisonGate, ComparisonGenerator};
use crate::uint::u32::gates::interleave_u32::{U32InterleaveGate, U32InterleaveGenerator};
use crate::uint::u32::gates::subtraction_u32::{U32SubtractionGate, U32SubtractionGenerator};
use crate::uint::u32::gates::uninterleave_to_b32::UninterleaveToB32Gate;
use crate::uint::u32::gates::uninterleave_to_u32::{
    UninterleaveToU32Gate, UninterleaveToU32Generator,
};
use crate::uint::u48::subtraction_u48::{U48SubtractionGate, U48SubtractionGenerator};

#[macro_export]
// Macro taken from plonky2 but use std Vec instead of supporting no_std. (I think plonky2 implementation is bugged and doesn't export
// Vec object so we can't import their macro here)
macro_rules! impl_gate_serializer {
    ($target:ty, $($gate_types:ty),+) => {
        fn read_gate(
            &self,
            buf: &mut plonky2::util::serialization::Buffer,
            common: &plonky2::plonk::circuit_data::CommonCircuitData<F, D>,
        ) -> plonky2::util::serialization::IoResult<plonky2::gates::gate::GateRef<F, D>> {
            let tag = plonky2::util::serialization::Read::read_u32(buf)?;
            read_gate_impl!(buf, tag, common, $($gate_types),+)
        }

        fn write_gate(
            &self,
            buf: &mut Vec<u8>,
            gate: &plonky2::gates::gate::GateRef<F, D>,
            common: &plonky2::plonk::circuit_data::CommonCircuitData<F, D>,
        ) -> plonky2::util::serialization::IoResult<()> {
            let tag = get_gate_tag_impl!(gate, $($gate_types),+)?;

            plonky2::util::serialization::Write::write_u32(buf, tag)?;
            gate.0.serialize(buf, common)?;
            Ok(())
        }
    };
}

#[macro_export]
// Macro taken from plonky2 but use std Vec instead of supporting no_std. (I think plonky2 implementation is bugged and doesn't export
// Vec object so we can't import their macro here)
macro_rules! impl_generator_serializer {
    ($target:ty, $($generator_types:ty),+) => {
        fn read_generator(
            &self,
            buf: &mut plonky2::util::serialization::Buffer,
            common: &plonky2::plonk::circuit_data::CommonCircuitData<F, D>,
        ) -> plonky2::util::serialization::IoResult<plonky2::iop::generator::WitnessGeneratorRef<F, D>> {
            let tag = plonky2::util::serialization::Read::read_u32(buf)?;
            read_generator_impl!(buf, tag, common, $($generator_types),+)
        }

        fn write_generator(
            &self,
            buf: &mut Vec<u8>,
            generator: &plonky2::iop::generator::WitnessGeneratorRef<F, D>,
            common: &plonky2::plonk::circuit_data::CommonCircuitData<F, D>,
        ) -> plonky2::util::serialization::IoResult<()> {
            let tag = get_generator_tag_impl!(generator, $($generator_types),+)?;

            plonky2::util::serialization::Write::write_u32(buf, tag)?;
            generator.0.serialize(buf, common)?;
            Ok(())
        }
    };
}

/// A gate serializer that can be used to serialize all default gates supported
/// by the `plonky2` library and also custom gates defined here, like U32 gates
/// and Poseion2
/// Being a unit struct, it can be simply called as
/// ```rust
/// use circuit::circuit_serializer::BlockGateSerializer;
/// let gate_serializer = BlockGateSerializer;
/// ```
#[derive(Debug)]
pub struct BlockGateSerializer;
impl<F: RichField + Extendable<D> + Poseidon2, const D: usize> GateSerializer<F, D>
    for BlockGateSerializer
{
    impl_gate_serializer! {
        BlockGateSerializer,
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

/// A generator serializer that can be used to serialize all default generators supported
/// by the `plonky2` library and also custom generators. It can simply be called as
/// ```rust
/// use circuit::circuit_serializer::BlockGeneratorSerializer;
/// use plonky2::plonk::config::PoseidonGoldilocksConfig;
///
/// const D: usize = 2;
/// type C = PoseidonGoldilocksConfig;
/// type CC = circuit::ecdsa::curve::secp256k1::Secp256K1;
/// let generator_serializer = BlockGeneratorSerializer::<C, D, CC>::default();
/// ```
#[derive(Debug, Default)]
pub struct BlockGeneratorSerializer<C: GenericConfig<D>, const D: usize, CC> {
    pub _phantom: PhantomData<C>,
    pub _phantom2: PhantomData<CC>,
}

impl<C, const D: usize, CC> WitnessGeneratorSerializer<F, D> for BlockGeneratorSerializer<C, D, CC>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F> + 'static,
    C::Hasher: AlgebraicHasher<F>,
    CC: Curve,
{
    impl_generator_serializer! {
        BlockGeneratorSerializer,
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

/// A gate serializer that can be used to serialize all default gates supported
/// by the `plonky2` library and also custom gates defined here, like U32 gates
/// and Poseion2
/// Being a unit struct, it can be simply called as
/// ```rust
/// use circuit::circuit_serializer::RecursionGateSerializer;
/// let gate_serializer = RecursionGateSerializer;
/// ```
#[derive(Debug)]
pub struct RecursionGateSerializer;
impl<F: RichField + Extendable<D> + Poseidon2, const D: usize> GateSerializer<F, D>
    for RecursionGateSerializer
{
    impl_gate_serializer! {
        RecursionGateSerializer,
        ArithmeticGate,
        ArithmeticExtensionGate<D>,
        BaseSumGate<2>,
        ConstantGate,
        CosetInterpolationGate<F, D>,
        ExponentiationGate<F, D>,
        EqualityGate,
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
        U48SubtractionGate<F, D>,
        Poseidon2Gate<F, D>,
        U32InterleaveGate,
        UninterleaveToB32Gate,
        UninterleaveToU32Gate,
        SelectionGate,
        QuinticMultiplicationGate,
        QuinticSquaringGate,
        AdditionGate,
        MultiplicationGate,
        ByteDecompositionGate
    }
}

/// A generator serializer that can be used to serialize all default generators supported
/// by the `plonky2` library and also custom generators. It can simply be called as
/// ```rust
/// use circuit::circuit_serializer::RecursionGeneratorSerializer;
/// use plonky2::plonk::config::PoseidonGoldilocksConfig;
///
/// const D: usize = 2;
/// type C = PoseidonGoldilocksConfig;
/// let generator_serializer = RecursionGeneratorSerializer::<C, D>::default();
/// ```
#[derive(Debug, Default)]
pub struct RecursionGeneratorSerializer<C: GenericConfig<D>, const D: usize> {
    pub _phantom: PhantomData<C>,
}

impl<C, const D: usize> WitnessGeneratorSerializer<F, D> for RecursionGeneratorSerializer<C, D>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F> + 'static,
    C::Hasher: AlgebraicHasher<F>,
{
    impl_generator_serializer! {
        RecursionGeneratorSerializer,
        ArithmeticBaseGenerator<F, D>,
        ArithmeticExtensionGenerator<F, D>,
        AdditionBaseGenerator<F,D>,
        MultiplicationBaseGenerator<F,D>,
        BaseSplitGenerator<2>,
        BaseSumGenerator<2>,
        ConstantGenerator<F>,
        CopyGenerator,
        EqualityGenerator,
        EqualityBaseGenerator<F,D>,
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
        RangeCheckGenerator<F, D>,
        U32AddManyGenerator<F, D>,
        U32ArithmeticGenerator<F, D>,
        U48SubtractionGenerator<F, D>,
        LoggingGenerator<F, D>,
        SelectionBaseGenerator<F, D>,
        QuinticSquaringBaseGenerator<F,D>,
        QuinticMultiplicationBaseGenerator<F,D>,
        U32InterleaveGenerator,
        UninterleaveToU32Generator,
        ByteDecompositionGenerator
    }
}

/// A gate serializer that can be used to serialize all default gates supported
/// by the `plonky2` library and also custom gates defined here, like U32 gates
/// and Poseion2
/// Being a unit struct, it can be simply called as
/// ```rust
/// use circuit::circuit_serializer::InnerWrapperGateSerializer;
/// let gate_serializer = InnerWrapperGateSerializer;
/// ```
#[derive(Debug)]
pub struct InnerWrapperGateSerializer;
impl<F: RichField + Extendable<D> + Poseidon2, const D: usize> GateSerializer<F, D>
    for InnerWrapperGateSerializer
{
    impl_gate_serializer! {
        InnerWrapperGateSerializer,
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
        ComparisonGate<F, D>,
        U32AddManyGate<F, D>,
        U32ArithmeticGate<F, D>,
        RangeCheckGate<F, D>,
        U32SubtractionGate<F, D>,
        Poseidon2Gate<F, D>,
        U48SubtractionGate<F, D>,
        SelectionGate,
        QuinticMultiplicationGate,
        QuinticSquaringGate,
        AdditionGate,
        U32InterleaveGate,
        UninterleaveToU32Gate,
        ByteDecompositionGate,
        EvaluateBitstreamGate
    }
}

/// A generator serializer that can be used to serialize all default generators supported
/// by the `plonky2` library and also custom generators. It can simply be called as
/// ```rust
/// use circuit::circuit_serializer::InnerWrapperGeneratorSerializer;
/// use plonky2::plonk::config::PoseidonGoldilocksConfig;
/// use plonky2::field::goldilocks_field::GoldilocksField;
///
/// const D: usize = 2;
/// type C = PoseidonGoldilocksConfig;
/// type FF = GoldilocksField;
/// let generator_serializer = InnerWrapperGeneratorSerializer::<C, D, FF>::default();
/// ```
#[derive(Debug, Default)]
pub struct InnerWrapperGeneratorSerializer<C: GenericConfig<D>, const D: usize, FF> {
    pub _phantom: PhantomData<C>,
    pub _phantom2: PhantomData<FF>,
}

impl<C, const D: usize, FF> WitnessGeneratorSerializer<F, D>
    for InnerWrapperGeneratorSerializer<C, D, FF>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F> + 'static,
    C::Hasher: AlgebraicHasher<F>,
    FF: PrimeField,
{
    impl_generator_serializer! {
        InnerWrapperGeneratorSerializer,
        ArithmeticBaseGenerator<F, D>,
        ArithmeticExtensionGenerator<F, D>,
        AdditionBaseGenerator<F,D>,
        MultiplicationBaseGenerator<F,D>,
        BaseSplitGenerator<2>,
        BaseSumGenerator<2>,
        ConstantGenerator<F>,
        CopyGenerator,
        EqualityGenerator,
        EqualityBaseGenerator<F,D>,
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
        SplitToU4Generator<F, D>,
        RangeCheckGenerator<F, D>,
        U32AddManyGenerator<F, D>,
        U32ArithmeticGenerator<F, D>,
        NonNativeMultiplicationGenerator<F, D, FF>,
        NonNativeSubtractionGenerator<F, D, FF>,
        NonNativeMultipleAddsGenerator<F, D, FF>,
        NonNativeInverseGenerator<F, D, FF>,
        NonNativeAdditionGenerator<F, D, FF>,
        NonNativeMulDivGenerator<F, D, FF>,
        NonNativeDivisionGenerator<F, D, FF>,
        LoggingGenerator<F, D>,
        U48SubtractionGenerator<F, D>,
        SelectionBaseGenerator<F, D>,
        QuinticSquaringBaseGenerator<F,D>,
        QuinticMultiplicationBaseGenerator<F,D>,
        U32InterleaveGenerator,
        UninterleaveToU32Generator,
        ByteDecompositionGenerator,
        EvaluateBitstreamBaseGenerator<F, D>
    }
}

/// A generator serializer that can be used to serialize all default generators supported
/// by the `plonky2` library. It can simply be called as
/// ```rust
/// use circuit::circuit_serializer::DefaultPoseidonBN128GeneratorSerializer;
/// use circuit::poseidon_bn128::plonky2_config::PoseidonBN128GoldilocksConfig;
///
/// const D: usize = 2;
/// let generator_serializer = DefaultPoseidonBN128GeneratorSerializer::<PoseidonBN128GoldilocksConfig, D>::default();
/// ```
/// Applications using custom generators should define their own serializer implementing
/// the `WitnessGeneratorSerializer` trait. This can be easily done through the
/// `impl_generator_serializer` macro.
#[derive(Debug, Default)]
pub struct DefaultPoseidonBN128GeneratorSerializer<C: GenericConfig<D>, const D: usize> {
    pub _phantom: PhantomData<C>,
}

impl<F, C, const D: usize> WitnessGeneratorSerializer<F, D>
    for DefaultPoseidonBN128GeneratorSerializer<C, D>
where
    F: RichField + Extendable<D> + Poseidon2,
    C: GenericConfig<D, F = F> + 'static,
{
    impl_generator_serializer! {
        DefaultPoseidonBN128GeneratorSerializer,
        ArithmeticBaseGenerator<F, D>,
        ArithmeticExtensionGenerator<F, D>,
        AdditionBaseGenerator<F,D>,
        MultiplicationBaseGenerator<F,D>,
        BaseSplitGenerator<2>,
        BaseSumGenerator<2>,
        ConstantGenerator<F>,
        CopyGenerator,
        EqualityGenerator,
        EqualityBaseGenerator<F,D>,
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
        SelectionBaseGenerator<F, D>,
        QuinticSquaringBaseGenerator<F,D>,
        QuinticMultiplicationBaseGenerator<F,D>,
        Poseidon2Generator<F, D>
    }
}

pub struct DeltaGateSerializer;
impl<F: RichField + Extendable<D> + Poseidon2, const D: usize> GateSerializer<F, D>
    for DeltaGateSerializer
{
    impl_gate_serializer! {
        DeltaGateSerializer,
        ArithmeticGate,
        ArithmeticExtensionGate<D>,
        BaseSumGate<2>,
        ConstantGate,
        CosetInterpolationGate<F, D>,
        EqualityGate,
        ExponentiationGate<F, D>,
        MulExtensionGate<D>,
        NoopGate,
        PoseidonMdsGate<F, D>,
        PoseidonGate<F, D>,
        PublicInputGate,
        RandomAccessGate<F, D>,
        ReducingExtensionGate<D>,
        ReducingGate<D>,
        RangeCheckGate<F, D>,
        U32SubtractionGate<F, D>,
        Poseidon2Gate<F, D>,
        AdditionGate,
        MultiplicationGate,
        QuinticMultiplicationGate,
        SelectionGate,
        QuinticSquaringGate,
        EvaluateSequenceGate,
        U48SubtractionGate<F, D>,
        ByteDecompositionGate,
        U16SubtractionGate<F, D>
    }
}

#[derive(Debug, Default)]
pub struct DeltaGeneratorSerializer<C: GenericConfig<D>, const D: usize> {
    pub _phantom: PhantomData<C>,
}

impl<C, const D: usize> WitnessGeneratorSerializer<F, D> for DeltaGeneratorSerializer<C, D>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F> + 'static,
    C::Hasher: AlgebraicHasher<F>,
{
    impl_generator_serializer! {
        DeltaGeneratorSerializer,
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
        EvaluateSequenceBaseGenerator<F, D>
    }
}
