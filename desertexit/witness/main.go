// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math/big"
	"os"
	"sort"

	p2 "github.com/elliottech/poseidon_crypto/hash/poseidon2_goldilocks_plonky2"
	eth "github.com/ethereum/go-ethereum/common"
)

func main() {
	// Read input file
	input := &BlobDataInput{}
	fileBytes, err := os.ReadFile("../desertinput.json")
	if err != nil {
		panic(fmt.Sprintf("failed to read desertinput.json: %v", err))
	}
	if err := json.Unmarshal(fileBytes, input); err != nil {
		panic(fmt.Sprintf("failed to unmarshal desertinput.json: %v", err))
	}
	accountIndex := input.AccountIndex
	if accountIndex < 0 || accountIndex >= MaxAccountIndex {
		panic(fmt.Sprintf("invalid account index %d, must be between 0 and %d", accountIndex, MaxAccountIndex-1))
	}
	assetIndex := input.AssetIndex
	if assetIndex < MinAssetIndex || assetIndex > MaxAssetIndex {
		panic(fmt.Sprintf("invalid asset index %d, must be between %d and %d", assetIndex, MinAssetIndex, MaxAssetIndex))
	}

	// Initialize account pub data tree
	accountPubDataTree, err := initializeAccountPubDataTree(input)
	if err != nil {
		fmt.Printf("unable to create new account pub data tree: %v", err)
		panic("unable to create new account pub data tree, err:" + err.Error())
	}

	accountPubDatas := make(map[int64]*PubdataAccountWitness)
	var allPublicMarketDetails [PositionListSize]*PubdataMarketDetailWitness
	for i := 0; i < PositionListSize; i++ {
		allPublicMarketDetails[i] = &PubdataMarketDetailWitness{}
	}

	for i, blobBytes := range input.BlobBytes {
		// Get bytes from blob
		markPriceBytes, fundingBytes, quoteMultiplierBytes, accountPubDataBytes, err := getBlobBytes(blobBytes)
		if err != nil {
			panic("failed to reverse compute blob data, err:" + err.Error())
		}

		// Parse market details
		for j := 0; j < PositionListSize; j++ {
			allPublicMarketDetails[j].MarkPrice = binary.BigEndian.Uint32(markPriceBytes[j*MarkPriceByteSize:])
			allPublicMarketDetails[j].FundingRatePrefixSum = readInt64WithSignBytes(fundingBytes, j*FundingByteSize)
			allPublicMarketDetails[j].QuoteMultiplier = binary.BigEndian.Uint16(quoteMultiplierBytes[j*QuoteMultiplierByteSize:])
		}

		// Parse account deltas and build account pub data objects
		accountDeltas, err := bytesToAccountDeltas(accountPubDataBytes)
		if err != nil {
			fmt.Printf("failed to decompress pubdata, %v", err)
			panic("failed to decompress pubdata, err:" + err.Error())
		}

		// Construct hashes to be set in the pub data tree
		accountPubDataSMTItems := make([]Item, 0)
		for accountIndex, delta := range accountDeltas {
			if existingDelta, exists := accountPubDatas[accountIndex]; exists {
				// Merge deltas
				publicPoolShares := make(map[int64]int64)
				for _, share := range existingDelta.PublicPoolShares {
					publicPoolShares[share.PublicPoolIndex] = share.ShareAmount
				}
				existingDelta.AccountIndex = delta.AccountIndex
				if delta.L1Address != EmptyL1Address {
					existingDelta.L1Address = delta.L1Address
				}
				if delta.AccountType != 0 {
					existingDelta.AccountType = delta.AccountType
				}
				for marketIndex, posDelta := range delta.Positions {
					if existingPosDelta, posExists := existingDelta.Positions[marketIndex]; posExists {
						existingPosDelta.LastFundingRatePrefixSum += posDelta.LastFundingRatePrefixSum
						existingPosDelta.Position += posDelta.Position
					} else {
						existingDelta.Positions[marketIndex] = &PubdataAccountPositionWitness{
							LastFundingRatePrefixSum: posDelta.LastFundingRatePrefixSum,
							Position:                 posDelta.Position,
						}
					}
				}
				for assetIndex, balanceDelta := range delta.AggregatedBalances {
					if existingBalance, balanceExists := existingDelta.AggregatedBalances[assetIndex]; balanceExists {
						existingDelta.AggregatedBalances[assetIndex] = new(big.Int).Add(existingBalance, balanceDelta)
					} else {
						existingDelta.AggregatedBalances[assetIndex] = new(big.Int).Set(balanceDelta)
					}
				}
				for _, share := range delta.PublicPoolShares {
					if existingShareDelta, shareExists := publicPoolShares[share.PublicPoolIndex]; shareExists {
						publicPoolShares[share.PublicPoolIndex] = existingShareDelta + share.ShareAmount
					} else {
						publicPoolShares[share.PublicPoolIndex] = share.ShareAmount
					}
				}
				existingDelta.PublicPoolShares = make([]*PubdataPublicPoolShareWitness, 0, len(publicPoolShares))
				for poolIndex, shareDelta := range publicPoolShares {
					if shareDelta != 0 {
						existingDelta.PublicPoolShares = append(existingDelta.PublicPoolShares, &PubdataPublicPoolShareWitness{
							PublicPoolIndex: poolIndex,
							ShareAmount:     shareDelta,
						})
					}
				}
				if delta.PublicPoolInfo != nil {
					if existingDelta.PublicPoolInfo == nil {
						existingDelta.PublicPoolInfo = &PubdataPublicPoolInfoWitness{
							TotalShares:    0,
							OperatorShares: 0,
						}
					}
					existingDelta.PublicPoolInfo.TotalShares += delta.PublicPoolInfo.TotalShares
					existingDelta.PublicPoolInfo.OperatorShares += delta.PublicPoolInfo.OperatorShares
				}
			} else {
				accountPubDatas[accountIndex] = delta
			}

			// sort existingDelta.PublicPoolShares by PublicPoolIndex
			sort.Slice(accountPubDatas[accountIndex].PublicPoolShares, func(i, j int) bool {
				return accountPubDatas[accountIndex].PublicPoolShares[i].PublicPoolIndex < accountPubDatas[accountIndex].PublicPoolShares[j].PublicPoolIndex
			})
			// Make publicPoolShares up to 16
			for i = len(accountPubDatas[accountIndex].PublicPoolShares); i < SharesListSize; i++ {
				accountPubDatas[accountIndex].PublicPoolShares = append(accountPubDatas[accountIndex].PublicPoolShares, &PubdataPublicPoolShareWitness{
					PublicPoolIndex: 0,
					ShareAmount:     0,
				})
			}

			accountPubDataSMTItems = append(accountPubDataSMTItems, Item{
				Key: uint64(accountIndex), // nolint:gosec
				Val: getPubDataLeafHash(accountPubDatas[accountIndex]),
			})
		}

		// Set account pub data tree
		version := Version(i)
		if err := accountPubDataTree.MultiSetWithVersion(accountPubDataSMTItems, version); err != nil {
			panic(fmt.Sprintf("accountPubDataTree.MultiSetWithVersion error: %v", err))
		}
		if _, err := accountPubDataTree.Commit(&version); err != nil {
			panic(fmt.Sprintf("accountPubDataTree.Commit error: %v", err))
		}
	}

	if accountPubDatas[int64(accountIndex)] == nil {
		panic(fmt.Sprintf("account index %d not found in pubdata", accountIndex))
	}

	mainAccount := accountPubDatas[int64(accountIndex)]
	accounts := [DesertWitnessAccounts]*PubdataAccountWitness{}
	accounts[0] = mainAccount
	for i := 0; i < SharesListSize; i++ {
		if i >= len(mainAccount.PublicPoolShares) || mainAccount.PublicPoolShares[i].ShareAmount == 0 {
			accounts[i+1] = EmptyPubdataAccountWitness(NilAccountIndex)
		} else {
			accounts[i+1] = accountPubDatas[mainAccount.PublicPoolShares[i].PublicPoolIndex]
		}
	}
	accountPubDataMerkleProofs := [DesertWitnessAccounts][AccountTreeHeight]p2.NumericalHashOut{}
	for i := 0; i < DesertWitnessAccounts; i++ {
		proof, _ := accountPubDataTree.GetProof(uint64(accounts[i].AccountIndex))
		accountPubDataMerkleProofs[i], _ = accountMerkleProofFromBytes(proof)
	}

	accountPubDataTreeRoot, _ := p2.HashOutFromLittleEndianBytes(accountPubDataTree.Root())
	lastValidiumRoot, _ := p2.HashOutFromLittleEndianBytes(eth.Hex2Bytes(input.LastValidiumRoot))
	allPublicMarketDetailsHash := allPublicMarketDetailsHash(allPublicMarketDetails)

	totalBalance := big.NewInt(0)
	if assetIndex == USDCAssetIndex {
		totalBalance = getUsdcBalanceForWitness(accounts, allPublicMarketDetails)
	} else if assetBal, exists := accounts[0].AggregatedBalances[assetIndex]; exists {
		totalBalance = assetBal
	}

	desertWitness := &DesertWitness{
		Accounts:           accounts,
		TotalAccountValue:  totalBalance.String(),
		AssetIndex:         assetIndex,
		MasterAccountIndex: input.MasterAccountIndex,

		AccountPubDataTreeRoot:     accountPubDataTreeRoot,
		AccountPubDataMerkleProofs: accountPubDataMerkleProofs,

		AllPublicMarketDetails: allPublicMarketDetails,

		ValidiumRoot: lastValidiumRoot,
		StateRoot:    p2.HashNToOne([]p2.HashOut{accountPubDataTreeRoot, allPublicMarketDetailsHash, lastValidiumRoot}),
	}
	jsonBytes, err := json.Marshal(desertWitness)
	if err != nil {
		panic(fmt.Sprintf("failed to marshal desertWitness to JSON: %v", err))
	}
	if err := os.WriteFile("../artifacts/desertwitness.json", jsonBytes, 0644); err != nil {
		panic(fmt.Sprintf("failed to write desertWitness to disk: %v", err))
	}

	return
}
