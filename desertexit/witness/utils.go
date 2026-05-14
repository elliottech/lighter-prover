package main

import (
	"encoding/binary"
	"fmt"
	"math/big"
	"slices"
	"sort"
	"sync"

	g "github.com/elliottech/poseidon_crypto/field/goldilocks"
	p2 "github.com/elliottech/poseidon_crypto/hash/poseidon2_goldilocks_plonky2"
	eth "github.com/ethereum/go-ethereum/common"
	"github.com/panjf2000/ants/v2"
)

func AddBig(x *big.Int, y *big.Int) *big.Int {
	return new(big.Int).Add(x, y)
}

func MulBig(factor1 *big.Int, factor2 *big.Int) *big.Int {
	return new(big.Int).Mul(factor1, factor2)
}

func DivBig(a, b *big.Int) *big.Int {
	return new(big.Int).Div(a, b)
}

func uint56ToBytes(val int64) []byte {
	result := make([]byte, 7)
	for i := 0; i < 7; i++ {
		result[i] = byte(val & 0xFF)
		val >>= 8
	}
	return result
}

func bigToU32Limbs(bi *big.Int, numLimbs uint8) []uint32 {
	littleEndianBytes := bi.Bytes()
	slices.Reverse(littleEndianBytes)

	if len(littleEndianBytes)%4 != 0 {
		littleEndianBytes = append(littleEndianBytes, make([]byte, 4-len(littleEndianBytes)%4)...)
	}
	if len(littleEndianBytes) < int(numLimbs*4) {
		littleEndianBytes = append(littleEndianBytes, make([]byte, int(numLimbs*4)-len(littleEndianBytes))...)
	}

	littleEndianBytes = littleEndianBytes[:numLimbs*4]

	var limbs []uint32
	for i := 0; i < len(littleEndianBytes); i += 4 {
		limb := uint32(littleEndianBytes[i]) |
			uint32(littleEndianBytes[i+1])<<8 |
			uint32(littleEndianBytes[i+2])<<16 |
			uint32(littleEndianBytes[i+3])<<24
		limbs = append(limbs, limb)
	}
	return limbs
}

func abs(x int64) int64 {
	if x > 0 {
		return x
	}
	return -x
}

func accountMerkleProofFromBytes(proof [][]byte) (res [AccountTreeHeight]p2.NumericalHashOut, err error) {
	if len(proof) != AccountTreeHeight {
		return res, fmt.Errorf("invalid account size")
	}

	res = [AccountTreeHeight]p2.NumericalHashOut{}
	for i := uint8(0); i < AccountTreeHeight; i++ {
		proof, err := p2.HashOutFromLittleEndianBytes(proof[i])
		if err != nil {
			return res, fmt.Errorf("failed to parse proof err: %w", err)
		}
		res[i] = proof.ToUint64Array()
	}
	return res, nil
}

func bytesToAccountDeltas(compressedPubdata []byte) (deltas map[int64]*PubdataAccountWitness, err error) {
	pdd := NewPubdataDecoder(compressedPubdata)
	deltas = make(map[int64]*PubdataAccountWitness)
	lastAccountIndex := int64(0)
	isFirstIteration := true
	for pdd.GetRemainingLength() > 1 { // if only one half-byte is remaining, from byte-> half-byte conversion
		accountDelta := EmptyPubdataAccountWitness(0)
		{
			diff := pdd.DecompressTarget()
			if diff == 0 && !isFirstIteration {
				return deltas, nil
			}
			accountDelta.AccountIndex = lastAccountIndex + int64(diff) // nolint:gosec
			isFirstIteration = false
		}

		val := pdd.DecompressTarget()
		hasL1Address := (val & 1) == 1
		hasPublicPoolInfo := (val & 2) == 2
		accountDelta.AccountType = uint8(val >> 2) // nolint:gosec

		if hasL1Address {
			l1Address := make([]byte, 20)
			val := pdd.DecompressTarget()
			copy(l1Address[0:7], uint56ToBytes(int64(val))[0:7]) // nolint:gosec
			val = pdd.DecompressTarget()
			copy(l1Address[7:14], uint56ToBytes(int64(val))[0:7]) // nolint:gosec
			val = pdd.DecompressTarget()
			copy(l1Address[14:20], uint56ToBytes(int64(val))[0:6]) // nolint:gosec
			slices.Reverse(l1Address)
			accountDelta.L1Address = eth.Bytes2Hex(l1Address)
		}

		if hasPublicPoolInfo {
			val := pdd.DecompressTarget()
			operatorShareAmountNeg, totalShareAmountNeg := val&1, (val>>1)&1
			totalShareAmountAbs := pdd.DecompressTarget()
			operatorShareAmountAbs := pdd.DecompressTarget()
			accountDelta.PublicPoolInfo = &PubdataPublicPoolInfoWitness{
				TotalShares:    int64(totalShareAmountAbs),    // nolint:gosec
				OperatorShares: int64(operatorShareAmountAbs), // nolint:gosec
			}
			if totalShareAmountNeg == 1 {
				accountDelta.PublicPoolInfo.TotalShares *= -1
			}
			if operatorShareAmountNeg == 1 {
				accountDelta.PublicPoolInfo.OperatorShares *= -1
			}
		}
		{
			positionCount := pdd.DecompressTarget()
			for i := uint64(0); i < positionCount; i++ {
				marketIndex := pdd.DecompressTarget()
				fundingAbs := pdd.DecompressTarget()
				val := pdd.DecompressTarget()
				fundingCarryBits := (marketIndex >> 8)
				marketIndex &= (1 << 8) - 1
				fundingAbs |= fundingCarryBits << 60
				fundingNeg, posDeltaNeg, posDeltaAbs := val&1, (val>>1)&1, val>>2            // nolint:gosec
				accountDelta.Positions[uint8(marketIndex)] = &PubdataAccountPositionWitness{ // nolint:gosec
					Position:                 int64(posDeltaAbs), // nolint:gosec
					LastFundingRatePrefixSum: int64(fundingAbs),  // nolint:gosec
				}
				if posDeltaNeg == 1 {
					accountDelta.Positions[uint8(marketIndex)].Position *= -1 // nolint:gosec
				}
				if fundingNeg == 1 {
					accountDelta.Positions[uint8(marketIndex)].LastFundingRatePrefixSum *= -1 // nolint:gosec
				}
			}
		}
		{
			assetCount := pdd.DecompressTarget()
			for i := uint64(0); i < assetCount; i++ {
				limb1 := pdd.DecompressTarget()
				assetIndex, isNeg, balanceLimb1 := int16(limb1&((1<<AssetIndexBits)-1)), (limb1>>AssetIndexBits)&1 == 1, limb1>>(AssetIndexBits+1)
				balanceLimb2 := pdd.DecompressTarget()
				balanceDelta := new(big.Int).SetUint64(balanceLimb2)
				balanceDelta.Lsh(balanceDelta, 48).Or(balanceDelta, new(big.Int).SetUint64(balanceLimb1))
				if isNeg {
					balanceDelta.Neg(balanceDelta)
				}
				accountDelta.AggregatedBalances[assetIndex] = balanceDelta
			}
		}
		{
			shareCount := pdd.DecompressTarget()
			for i := uint64(0); i < shareCount; i++ {
				shareDeltaNegAndPublicPoolIndexDiff := pdd.DecompressTarget()
				shareDeltaNeg := shareDeltaNegAndPublicPoolIndexDiff & 1
				publicPoolIndexDiff := shareDeltaNegAndPublicPoolIndexDiff >> 1
				shareDeltaAbs := pdd.DecompressTarget()
				accountDelta.PublicPoolShares = append(accountDelta.PublicPoolShares, &PubdataPublicPoolShareWitness{
					PublicPoolIndex: MaxAccountIndex - int64(publicPoolIndexDiff), // nolint:gosec
					ShareAmount:     int64(shareDeltaAbs),                         // nolint:gosec
				})
				if shareDeltaNeg == 1 {
					accountDelta.PublicPoolShares[len(accountDelta.PublicPoolShares)-1].ShareAmount *= -1
				}
			}
			// Sort pool shares for pub data tree, as they can be unsorted in deltas and we construct from deltas.
			sort.Slice(accountDelta.PublicPoolShares, func(i, j int) bool {
				return accountDelta.PublicPoolShares[i].PublicPoolIndex < accountDelta.PublicPoolShares[j].PublicPoolIndex
			})
		}
		deltas[accountDelta.AccountIndex] = accountDelta
		lastAccountIndex = accountDelta.AccountIndex
	}

	return deltas, nil
}

func readInt64WithSignBytes(data []byte, offset int) (val int64) {
	sign, offset := data[offset], offset+1
	absoluteValue := binary.BigEndian.Uint64(data[offset:])
	if sign > 0 {
		return -int64(absoluteValue) //nolint:gosec
	}
	return int64(absoluteValue) //nolint:gosec
}

func getBlobBytes(blobBytes string) (
	markPriceBytes, fundingBytes, quoteMultiplierBytes, accountPubDataBytes []byte, err error,
) {
	blobData := [BlobBytesSize]byte{}
	copy(blobData[:], eth.Hex2Bytes(blobBytes))

	if len(blobData) != BlobBytesSize {
		return nil, nil, nil, nil, fmt.Errorf("invalid blob data size: %d", len(blobData))
	}
	pubData := make([]byte, 0, BlobFilledBytesSize)
	for i := 0; i < BlobBytesSize; i++ {
		if i%32 != 0 {
			pubData = append(pubData, blobData[i])
		}
	}
	markPriceBytes = pubData[BlobVersionByteSize+BlobReservedBytesSize : BlobVersionByteSize+BlobReservedBytesSize+BlobMarkPricesByteSize]
	fundingBytes = pubData[BlobVersionByteSize+BlobReservedBytesSize+BlobMarkPricesByteSize : BlobVersionByteSize+BlobReservedBytesSize+BlobMarkPricesByteSize+BlobFundingsByteSize]                                                           //nolint:lll
	quoteMultiplierBytes = pubData[BlobVersionByteSize+BlobReservedBytesSize+BlobMarkPricesByteSize+BlobFundingsByteSize : BlobVersionByteSize+BlobReservedBytesSize+BlobMarkPricesByteSize+BlobFundingsByteSize+BlobQuoteMultipliersByteSize] //nolint:lll
	accountPubDataBytes = pubData[BlobVersionByteSize+BlobReservedBytesSize+BlobMarkPricesByteSize+BlobFundingsByteSize+BlobQuoteMultipliersByteSize:]

	return markPriceBytes, fundingBytes, quoteMultiplierBytes, accountPubDataBytes, nil
}

func initializeAccountPubDataTree(input *BlobDataInput) (SparseMerkleTree, error) {
	pool, err := ants.NewPool(100, ants.WithPanicHandler(func(p interface{}) {
		panic("worker exits from a panic")
	}))
	if err != nil {
		return nil, err
	}

	var opts []Option
	opts = append(opts, []Option{GoRoutinePool(pool)}...)

	hasher := &Hasher{
		pool: sync.Pool{
			New: func() interface{} {
				return p2.NewPoseidon2()
			},
		},
	}
	accountPubDataTree, _ := NewSparseMerkleTree(hasher, AccountTreeHeight, NilHash)

	// If desert input contains account pub data leaves, initialize the tree with them. Otherwise
	// initialize with empty data for treasury and insurance fund operator.
	version := Version(0)
	accountPubDataSMTItems := make([]Item, 0)

	for _, leaf := range input.InitialAccountPubDataLeaves {
		accountPubDataSMTItems = append(accountPubDataSMTItems, Item{
			Key: uint64(leaf.AccountIndex), // nolint:gosec
			Val: getPubDataLeafHash(leaf),
		})
	}

	if err := accountPubDataTree.MultiSetWithVersion(accountPubDataSMTItems, version); err != nil {
		return nil, fmt.Errorf("accountPubDataTree.MultiSetWithVersion error: %w", err)
	}
	if _, err := accountPubDataTree.Commit(&version); err != nil {
		return nil, fmt.Errorf("accountPubDataTree.Commit error: %w", err)
	}

	return accountPubDataTree, nil
}

func getUsdcBalanceForWitness(
	accounts [DesertWitnessAccounts]*PubdataAccountWitness,
	publicMarketDetails [PositionListSize]*PubdataMarketDetailWitness,
) *big.Int {
	mainAccount := accounts[0]

	usdcBalance, exists := mainAccount.AggregatedBalances[USDCAssetIndex]
	if !exists {
		usdcBalance = big.NewInt(0)
	}

	extendedCollateral := MulBig(usdcBalance, USDCToCollateralMultiplierBig)
	positionsTavComponent := getPositionsTavComponent(mainAccount.Positions, publicMarketDetails)
	publicPoolsTavComponent := getPublicPoolsTavComponent(accounts, publicMarketDetails)

	mainAccountTav := AddBig(AddBig(extendedCollateral, positionsTavComponent), publicPoolsTavComponent)

	fmt.Println("Extended Collateral", extendedCollateral)
	fmt.Println("Positions TAV Component", positionsTavComponent)
	fmt.Println("Public Pools TAV Component", publicPoolsTavComponent)
	fmt.Println("Main Account TAV (extended)", mainAccountTav)

	if mainAccountTav.Sign() < 0 {
		fmt.Println("Main Account TAV is negative, returning 0")
		return big.NewInt(0)
	}

	// If main account is public pool, return only operator's share of the TAV.
	if mainAccount.AccountType == PublicPool {
		if mainAccount.PublicPoolInfo.TotalShares == 0 {
			return new(big.Int).SetInt64(0)
		}
		totalShares := new(big.Int).SetInt64(mainAccount.PublicPoolInfo.TotalShares)
		operatorShares := new(big.Int).SetInt64(mainAccount.PublicPoolInfo.OperatorShares)
		mainAccountTav = DivBig(MulBig(mainAccountTav, operatorShares), totalShares)

		fmt.Println("Main Account is public pool. Proportioned TAV (extended)", mainAccountTav)
	}

	return DivBig(mainAccountTav, USDCToCollateralMultiplierBig)
}

func getPositionsTavComponent(
	positions map[uint8]*PubdataAccountPositionWitness,
	publicMarketDetails [PositionListSize]*PubdataMarketDetailWitness,
) *big.Int {
	positionNotionalsSum, fundingsSum := big.NewInt(0), big.NewInt(0)
	for marketIndex, position := range positions {
		if position.Position == 0 && position.LastFundingRatePrefixSum == 0 {
			continue
		}

		markPrice := publicMarketDetails[marketIndex].MarkPrice
		fundingRatePrefixSum := publicMarketDetails[marketIndex].FundingRatePrefixSum
		quoteMultiplier := publicMarketDetails[marketIndex].QuoteMultiplier

		positionNotional := big.NewInt(abs(position.Position) * int64(markPrice) * int64(quoteMultiplier))
		if position.Position <= 0 {
			positionNotional.Neg(positionNotional)
		}
		positionNotionalsSum = AddBig(positionNotionalsSum, positionNotional)

		fundingDelta := MulBig(
			MulBig(big.NewInt(position.Position), big.NewInt(int64(quoteMultiplier))),
			big.NewInt(position.LastFundingRatePrefixSum-fundingRatePrefixSum),
		)
		fundingsSum = AddBig(fundingsSum, fundingDelta)
	}
	positionNotionalsSumExtended := MulBig(positionNotionalsSum, USDCToCollateralMultiplierBig)

	return AddBig(positionNotionalsSumExtended, fundingsSum)
}

func getPublicPoolsTavComponent(
	accounts [DesertWitnessAccounts]*PubdataAccountWitness,
	publicMarketDetails [PositionListSize]*PubdataMarketDetailWitness,
) *big.Int {
	poolValuesSum := big.NewInt(0)
	for i, poolShare := range accounts[0].PublicPoolShares {
		poolAccount := accounts[i+1]

		shareAmount := poolShare.ShareAmount
		if shareAmount == 0 {
			continue
		}

		poolPositionsTavComponent := getPositionsTavComponent(poolAccount.Positions, publicMarketDetails)

		poolUsdcBalance := big.NewInt(0)
		if balance, exists := poolAccount.AggregatedBalances[USDCAssetIndex]; exists {
			poolUsdcBalance = balance
		}
		poolExtendedCollateral := MulBig(poolUsdcBalance, USDCToCollateralMultiplierBig)
		poolValue := AddBig(poolPositionsTavComponent, poolExtendedCollateral)

		var bigShareAmount *big.Int
		if poolAccount.PublicPoolInfo.TotalShares == 0 {
			bigShareAmount = big.NewInt(0)
		} else {
			bigShareAmount = DivBig(
				MulBig(poolValue, big.NewInt(shareAmount)),
				MulBig(big.NewInt(poolAccount.PublicPoolInfo.TotalShares), USDCToCollateralMultiplierBig),
			)
		}

		poolValuesSum = AddBig(poolValuesSum, MulBig(bigShareAmount, USDCToCollateralMultiplierBig))
	}
	return poolValuesSum
}

func allPublicMarketDetailsHash(allMarketsInfoBefore [PositionListSize]*PubdataMarketDetailWitness) p2.HashOut {
	elements := make([]g.GoldilocksField, 0)
	for i := 0; i < PositionListSize; i++ {
		if allMarketsInfoBefore[i] == nil {
			elements = append(elements, getPublicMarketDetailsHashParameters(&PubdataMarketDetailWitness{
				MarkPrice:            0,
				FundingRatePrefixSum: 0,
				QuoteMultiplier:      0,
			})...)
		} else {
			elements = append(elements, getPublicMarketDetailsHashParameters(&PubdataMarketDetailWitness{
				MarkPrice:            allMarketsInfoBefore[i].MarkPrice,
				FundingRatePrefixSum: allMarketsInfoBefore[i].FundingRatePrefixSum,
				QuoteMultiplier:      allMarketsInfoBefore[i].QuoteMultiplier,
			})...)
		}
	}
	return p2.HashNoPad(elements)
}

func getPublicMarketDetailsHashParameters(marketInfo *PubdataMarketDetailWitness) []g.GoldilocksField {
	fundingRatePrefixSum := uint64(abs(marketInfo.FundingRatePrefixSum))
	fundingRatePrefixSumSign := g.ZeroF()
	if marketInfo.FundingRatePrefixSum > 0 {
		fundingRatePrefixSumSign = g.OneF()
	} else if marketInfo.FundingRatePrefixSum < 0 {
		fundingRatePrefixSumSign = g.NegOneF()
	}

	return []g.GoldilocksField{
		g.GoldilocksField(fundingRatePrefixSum & 0xFFFF),
		g.GoldilocksField((fundingRatePrefixSum >> 16) & 0xFFFF),
		g.GoldilocksField((fundingRatePrefixSum >> 32) & 0xFFFF),
		g.GoldilocksField((fundingRatePrefixSum >> 48) & 0xFFFF),
		fundingRatePrefixSumSign,
		g.GoldilocksField(marketInfo.MarkPrice),
		g.GoldilocksField(marketInfo.QuoteMultiplier),
	}
}
