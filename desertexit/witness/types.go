// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

package main

import (
	"math/big"

	p2 "github.com/elliottech/poseidon_crypto/hash/poseidon2_goldilocks_plonky2"
)

// Blobs from genesis, and final state of validium and state roots
type BlobDataInput struct {
	AccountIndex       int64 `json:"ai"`
	AssetIndex         int16 `json:"asi"`
	MasterAccountIndex int64 `json:"mai"`

	BlobBytes []string `json:"blobs"` // [types.BlobBytesSize]byte{}

	InitialAccountPubDataLeaves []*PubdataAccountWitness `json:"ial"`

	LastValidiumRoot string `json:"vr"`
}

const (
	MasterAccount = iota
	SubAccount    = 1
	PublicPool    = 2
	InsuranceFund = 3 // Insurance Fund Public Pool

	AssetListSize  = (1 << AssetIndexBits)
	AssetIndexBits = 6
	MinAssetIndex  = int16(1)
	USDCAssetIndex = int16(3)
	MaxAssetIndex  = int16(AssetListSize - 2)

	NilAccountType = MasterAccount

	DesertWitnessAccounts                   = 17   // 1 main account + 16 public pool shares
	DesertBinaryOptionsPositions            = 1024 // protocol cap on binary options positions per account
	TreasuryAccountIndex                    = int64(0)
	InsuranceFundOperatorAccountIndex       = int64(1)
	AccountTreeHeight                       = 48
	EmptyL1Address                          = "0x0000000000000000000000000000000000000000"
	PositionListSize                        = (1 << 8) - 1 // NilMarketIndex is not included
	SharesListSize                          = int(16)
	InitialPoolShareValue             int64 = 1_000           // 0.001 USDC
	MaxAccountIndex                         = 281474976710654 // (1 << 48) - 2
	NilAccountIndex                         = MaxAccountIndex + 1
	PositionBucketSize                      = 16

	MarketTreeHeight            = 12
	MarketIndexBits             = 12
	MarketSlotStatusBits        = 2
	MarketTypeBinaryOptions     = 2
	ExpiredMarketStatus         = uint8(0)
	ActiveMarketStatus          = uint8(1)
	InSettlementMarketStatus    = uint8(2)
	MinBinaryOptionsMarketIndex = int16(1000)
	MaxBinaryOptionsMarketIndex = int16(2000)

	BlobVersionMarketDeltas      = uint16(1) // market deltas: slot, active flag, public market index
	BlobVersionBinaryOptions     = uint16(2) // slot status, price, cap, QEM in market deltas; BO position deltas
	BlobVersionIndex             = 0
	BlobVersionByteSize          = int64(2)
	BlobReservedBytesSize        = int64(32)
	BlobFilledBytesSize          = 4096 * 31
	BlobBytesSize                = 4096 * 32
	FundingByteSize              = 9 // 1 sign byte, 8 bytes for the funding rate
	MarkPriceByteSize            = 4
	QuoteMultiplierByteSize      = 2
	BlobQuoteMultipliersByteSize = PositionListSize * QuoteMultiplierByteSize
	BlobMarkPricesByteSize       = PositionListSize * MarkPriceByteSize
	BlobFundingsByteSize         = PositionListSize * FundingByteSize
	BlobMarketsByteSize          = BlobFundingsByteSize + BlobMarkPricesByteSize + BlobQuoteMultipliersByteSize
	BlobDeltasByteSize           = int(BlobFilledBytesSize - BlobMarketsByteSize - BlobReservedBytesSize - BlobVersionByteSize)
)

var (
	USDCToCollateralMultiplierBig = big.NewInt(int64(1_000_000))
	NilHash                       = []byte{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
)

type DesertWitness struct {
	Accounts           [DesertWitnessAccounts]*PubdataAccountWitness `json:"acc"`
	AssetIndex         int16                                         `json:"ai"`
	MasterAccountIndex int64                                         `json:"mai"`
	TotalAccountValue  string                                        `json:"tav"`

	AccountPubDataTreeRoot     p2.HashOut                                                    `json:"apdtr"`
	AccountPubDataMerkleProofs [DesertWitnessAccounts][AccountTreeHeight]p2.NumericalHashOut `json:"mpapd"`

	AllPublicMarketDetails [PositionListSize]*PubdataMarketDetailWitness `json:"pmda"`
	MarketPubDataTreeRoot  p2.HashOut                                    `json:"mpdtr"`

	// Binary options positions of the main account sorted by market slot, each proven against the account's
	// market pub data root and the market pub data tree root
	BinaryOptionsPositions [DesertBinaryOptionsPositions]*PubdataBinaryOptionsPositionWitness `json:"bop"`

	ValidiumRoot p2.HashOut `json:"vr"`
	StateRoot    p2.HashOut `json:"sr"`
}

type PubdataAccountWitness struct {
	AccountIndex       int64                                    `json:"ai"`
	L1Address          string                                   `json:"l1,omitempty"`
	AccountType        uint8                                    `json:"at,omitempty"`
	AggregatedBalances map[int16]*big.Int                       `json:"abal,omitempty"` // in USDC
	Positions          map[uint8]*PubdataAccountPositionWitness `json:"ap,omitempty"`
	PublicPoolShares   []*PubdataPublicPoolShareWitness         `json:"pps,omitempty"`
	PublicPoolInfo     *PubdataPublicPoolInfoWitness            `json:"ppi,omitempty"`

	// Binary options position sizes keyed by market slot, the leaves of the account market pub data tree.
	// The circuit only needs the root, the main account's positions are carried in DesertWitness.BinaryOptionsPositions.
	BinaryOptionsPositions map[int16]int64 `json:"-"`
	MarketPubDataRoot      p2.HashOut      `json:"mpdr"`
}

// Published binary options market state, the preimage of the market pub data tree leaf
type PubdataMarketWitness struct {
	Status                   uint8  `json:"st,omitempty"`
	Price                    uint32 `json:"p,omitempty"`
	SettlementCap            uint32 `json:"sc,omitempty"`
	QuoteExtensionMultiplier int64  `json:"qem,omitempty"`
}

type PubdataBinaryOptionsPositionWitness struct {
	MarketIndex int16 `json:"mi"`
	Size        int64 `json:"s,omitempty"`
	PubdataMarketWitness

	// Siblings from the leaf up to the root, matching the account pub data proofs
	AccountMarketPubDataMerkleProof [MarketTreeHeight]p2.NumericalHashOut `json:"mpampd"`
	MarketPubDataMerkleProof        [MarketTreeHeight]p2.NumericalHashOut `json:"mpmpd"`
}

type PubdataMarketDeltaWitness struct {
	MarketIndex int16
	PubdataMarketWitness
}

type PubdataAccountPositionWitness struct {
	LastFundingRatePrefixSum int64 `json:"lfrps,omitempty"`
	Position                 int64 `json:"p,omitempty"`
}

type PubdataPublicPoolShareWitness struct {
	PublicPoolIndex int64 `json:"ppi,omitempty"`
	ShareAmount     int64 `json:"sa,omitempty"`
}

type PubdataPublicPoolInfoWitness struct {
	TotalShares    int64 `json:"ppi_tsa,omitempty"`
	OperatorShares int64 `json:"ppi_os,omitempty"`
}

type PubdataMarketDetailWitness struct {
	FundingRatePrefixSum int64  `json:"f,omitempty"`
	MarkPrice            uint32 `json:"mp,omitempty"`
	QuoteMultiplier      uint16 `json:"qm,omitempty"`
}

func EmptyPubdataAccountWitness(accountIndex int64) *PubdataAccountWitness {
	return &PubdataAccountWitness{
		AccountIndex:       accountIndex,
		L1Address:          EmptyL1Address,
		AccountType:        MasterAccount,
		AggregatedBalances: make(map[int16]*big.Int),
		Positions:          make(map[uint8]*PubdataAccountPositionWitness),
		PublicPoolShares:   make([]*PubdataPublicPoolShareWitness, 0),
		PublicPoolInfo:     &PubdataPublicPoolInfoWitness{},

		BinaryOptionsPositions: make(map[int16]int64),
		MarketPubDataRoot:      emptyMarketTreeRoot(),
	}
}

func EmptyPubdataBinaryOptionsPositionWitness() *PubdataBinaryOptionsPositionWitness {
	proof := nilMarketMerkleProof()
	return &PubdataBinaryOptionsPositionWitness{
		MarketIndex:                     MinBinaryOptionsMarketIndex,
		AccountMarketPubDataMerkleProof: proof,
		MarketPubDataMerkleProof:        proof,
	}
}

func IsBinaryOptionsMarketSlot(marketIndex int16) bool {
	return marketIndex >= MinBinaryOptionsMarketIndex && marketIndex <= MaxBinaryOptionsMarketIndex
}
