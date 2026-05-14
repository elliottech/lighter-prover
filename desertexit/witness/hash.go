// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

package main

import (
	"fmt"
	"math/big"

	g "github.com/elliottech/poseidon_crypto/field/goldilocks"
	p2 "github.com/elliottech/poseidon_crypto/hash/poseidon2_goldilocks_plonky2"
	eth "github.com/ethereum/go-ethereum/common"
)

func getPubDataLeafHash(elem *PubdataAccountWitness) []byte {
	partialHashForPubData := getAccountPartialHashBlob(elem)

	address := new(big.Int).SetBytes(eth.FromHex(elem.L1Address)).FillBytes(make([]byte, 20))
	accountAddressBig := new(big.Int).SetBytes(address)
	accountAddressU32Limbs := bigToU32Limbs(accountAddressBig, 5)
	accountAddressElements := []g.GoldilocksField{
		g.GoldilocksField(accountAddressU32Limbs[0]),
		g.GoldilocksField(accountAddressU32Limbs[1]),
		g.GoldilocksField(accountAddressU32Limbs[2]),
		g.GoldilocksField(accountAddressU32Limbs[3]),
		g.GoldilocksField(accountAddressU32Limbs[4]),
	}

	elementsForPubData := make([]g.GoldilocksField, 0)

	elementsForPubData = append(elementsForPubData, partialHashForPubData[:]...)
	elementsForPubData = append(elementsForPubData, accountAddressElements...)
	elementsForPubData = append(elementsForPubData, g.GoldilocksField(elem.AccountType))

	// Construct aggregated balances tree and get the root
	assetDeltaSMTItems := make([]Item, 0)
	aggregatedBalancesTree, err := NewSparseMerkleTree(NewHasherPool(p2.NewPoseidon2), AssetIndexBits, NilHash)
	if err != nil {
		fmt.Printf("failed to create new tree by assets, %v", err)
		panic("failed to create new tree by assets, err:" + err.Error())
	}
	for assetIndex, balance := range elem.AggregatedBalances {
		if balance.Sign() == 0 {
			continue
		}
		assetDeltaSMTItems = append(assetDeltaSMTItems, Item{
			Key: uint64(assetIndex), // nolint:gosec
			Val: computeAssetBalanceLeafHash(balance),
		})
	}
	version := Version(1)
	if err = aggregatedBalancesTree.MultiSetWithVersion(assetDeltaSMTItems, version); err != nil {
		panic("failed to set asset balances in SMT, err:" + err.Error())
	}
	if _, err = aggregatedBalancesTree.Commit(&version); err != nil {
		panic("failed to commit asset balances in SMT, err:" + err.Error())
	}
	aggregatedBalancesRootF, err := p2.HashOutFromLittleEndianBytes(aggregatedBalancesTree.Root())
	if err != nil {
		panic(fmt.Sprintf("failed to convert aggregated balances root to field element, err: %v", err))
	}
	elementsForPubData = append(elementsForPubData, aggregatedBalancesRootF[:]...)

	return p2.HashNoPad(elementsForPubData).ToLittleEndianBytes()
}

func computeAssetBalanceLeafHash(balance *big.Int) []byte {
	if balance.Sign() == 0 {
		return NilHash
	}

	balanceLimbs := bigToU32Limbs(balance, 3)
	return p2.HashNoPad([]g.GoldilocksField{
		g.NonCannonicalGoldilocksField(int64(balance.Sign())),
		g.GoldilocksField(balanceLimbs[0]),
		g.GoldilocksField(balanceLimbs[1]),
		g.GoldilocksField(balanceLimbs[2]),
	}).ToLittleEndianBytes()
}

func getAccountPartialHashBlob(elem *PubdataAccountWitness) p2.HashOut {
	positionBucketHashesForPubData := getPositionsBucketHashesForAccountBlob(elem.Positions)
	publicPoolHashParametersForPubData := getPublicPoolHashParametersBlob(elem.PublicPoolShares)
	publicPoolInfoHashParametersForPubData := getPublicPoolInfoHashParametersBlob(elem.PublicPoolInfo)

	elementsForPubData := make([]g.GoldilocksField, 0)
	elementsForPubData = append(elementsForPubData, positionBucketHashesForPubData...)
	elementsForPubData = append(elementsForPubData, publicPoolHashParametersForPubData...)
	elementsForPubData = append(elementsForPubData, publicPoolInfoHashParametersForPubData...)
	return p2.HashNoPad(elementsForPubData)
}

func getPositionsBucketHashesForAccountBlob(positionInfo map[uint8]*PubdataAccountPositionWitness) []g.GoldilocksField {
	positions := convertFixedPositionsBlob(positionInfo)

	elementsForPubData := make([]g.GoldilocksField, 0)
	for i := 0; i < PositionListSize+1; i += PositionBucketSize {
		currentBucketHashElementsForPubData := make([]g.GoldilocksField, 0)
		for j := 0; j < PositionBucketSize; j++ {
			posIdx := i + j

			lasFundingRatePrefixSumAbs := uint64(abs(positions[posIdx].LastFundingRatePrefixSum))
			currentBucketHashElementsForPubData = append(currentBucketHashElementsForPubData, g.GoldilocksField(lasFundingRatePrefixSumAbs&0xFFFF))
			currentBucketHashElementsForPubData = append(currentBucketHashElementsForPubData, g.GoldilocksField((lasFundingRatePrefixSumAbs>>16)&0xFFFF))
			currentBucketHashElementsForPubData = append(currentBucketHashElementsForPubData, g.GoldilocksField((lasFundingRatePrefixSumAbs>>32)&0xFFFF))
			currentBucketHashElementsForPubData = append(currentBucketHashElementsForPubData, g.GoldilocksField((lasFundingRatePrefixSumAbs>>48)&0xFFFF))

			lastFundingRatePrefixSumSign := g.ZeroF()
			if positions[posIdx].LastFundingRatePrefixSum > 0 {
				lastFundingRatePrefixSumSign = g.OneF()
			} else if positions[posIdx].LastFundingRatePrefixSum < 0 {
				lastFundingRatePrefixSumSign = g.NegOneF()
			}
			currentBucketHashElementsForPubData = append(currentBucketHashElementsForPubData, lastFundingRatePrefixSumSign)

			positionAbs := uint64(abs(positions[posIdx].Position))
			currentBucketHashElementsForPubData = append(currentBucketHashElementsForPubData, g.GoldilocksField(positionAbs&0xFFFF))
			currentBucketHashElementsForPubData = append(currentBucketHashElementsForPubData, g.GoldilocksField((positionAbs>>16)&0xFFFF))
			currentBucketHashElementsForPubData = append(currentBucketHashElementsForPubData, g.GoldilocksField((positionAbs>>32)&0xFFFF))
			currentBucketHashElementsForPubData = append(currentBucketHashElementsForPubData, g.GoldilocksField((positionAbs>>48)&0xFFFF))
			positionSign := g.ZeroF()
			if positions[posIdx].Position > 0 {
				positionSign = g.OneF()
			} else if positions[posIdx].Position < 0 {
				positionSign = g.NegOneF()
			}
			currentBucketHashElementsForPubData = append(currentBucketHashElementsForPubData, positionSign)
		}
		currentBucketHashForPubData := p2.HashNoPad(currentBucketHashElementsForPubData)
		elementsForPubData = append(elementsForPubData, currentBucketHashForPubData[:]...)
	}

	return elementsForPubData
}

func convertFixedPositionsBlob(positionInfo map[uint8]*PubdataAccountPositionWitness) (position [PositionListSize + 1]*PubdataAccountPositionWitness) {
	for i := uint8(0); i < PositionListSize; i++ {
		if positionInfo[i] != nil {
			position[i] = &PubdataAccountPositionWitness{
				LastFundingRatePrefixSum: positionInfo[i].LastFundingRatePrefixSum,
				Position:                 positionInfo[i].Position,
			}
		} else {
			position[i] = &PubdataAccountPositionWitness{
				LastFundingRatePrefixSum: 0,
				Position:                 0,
			}
		}
	}
	position[PositionListSize] = &PubdataAccountPositionWitness{
		LastFundingRatePrefixSum: 0,
		Position:                 0,
	}
	return position
}

func getPublicPoolHashParametersBlob(publicPoolShares []*PubdataPublicPoolShareWitness) []g.GoldilocksField {
	shares := convertFixedPublicPoolSharesBlob(publicPoolShares)

	elementsForPubData := make([]g.GoldilocksField, 0)

	for i := 0; i < SharesListSize; i++ {
		elementsForPubData = append(elementsForPubData, []g.GoldilocksField{
			g.GoldilocksField(shares[i].PublicPoolIndex),
			g.GoldilocksField(shares[i].ShareAmount),
		}[:]...)
	}

	return elementsForPubData
}

func convertFixedPublicPoolSharesBlob(poolShares []*PubdataPublicPoolShareWitness) (shares [SharesListSize]*PubdataPublicPoolShareWitness) {
	for i := 0; i < int(SharesListSize); i++ {
		if i < len(poolShares) {
			shares[i] = &PubdataPublicPoolShareWitness{
				PublicPoolIndex: poolShares[i].PublicPoolIndex,
				ShareAmount:     poolShares[i].ShareAmount,
			}
		} else {
			shares[i] = &PubdataPublicPoolShareWitness{
				PublicPoolIndex: 0,
				ShareAmount:     0,
			}
		}
	}
	return shares
}

func getPublicPoolInfoHashParametersBlob(publicPoolInfo *PubdataPublicPoolInfoWitness) []g.GoldilocksField {
	if publicPoolInfo == nil {
		return []g.GoldilocksField{g.ZeroF(), g.ZeroF()}
	}

	return []g.GoldilocksField{
		g.GoldilocksField(publicPoolInfo.TotalShares),
		g.GoldilocksField(publicPoolInfo.OperatorShares),
	}
}
