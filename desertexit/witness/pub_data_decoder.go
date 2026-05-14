// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

package main

// Little endian bitset implementation
// Bits are separated into uint32 buckets.
type PubdataDecoder struct {
	limbs  []uint32
	offset int
}

const PubDataLimbBitSize = 4
const PubDataLimbMask = (1 << PubDataLimbBitSize) - 1

func NewPubdataDecoder(pubData []byte) *PubdataDecoder {
	return &PubdataDecoder{
		limbs:  getLimbs(pubData),
		offset: 0,
	}
}

func (pdd *PubdataDecoder) DecompressTarget() uint64 {
	size := pdd.limbs[pdd.offset]
	pdd.offset++
	val := uint64(0)
	for i := 0; i < int(size); i++ {
		val = (val << 4) | uint64(pdd.limbs[pdd.offset])
		pdd.offset++
	}
	return val
}

func getLimbs(pubData []byte) []uint32 {
	limbs := make([]uint32, 0)
	for i := 0; i < len(pubData); i++ {
		low := uint32(pubData[i] & PubDataLimbMask)                          // nolint:gosec
		high := uint32((pubData[i] >> PubDataLimbBitSize) & PubDataLimbMask) // nolint:gosec
		limbs = append(limbs, low)
		limbs = append(limbs, high)
	}
	return limbs
}

func (pdd *PubdataDecoder) GetRemainingLength() int {
	return len(pdd.limbs) - pdd.offset
}
