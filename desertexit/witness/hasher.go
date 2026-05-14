// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

package main

import (
	"hash"
	"sync"
)

func NewHasherPool(init func() hash.Hash) *Hasher {
	return &Hasher{
		pool: sync.Pool{
			New: func() interface{} {
				return init()
			},
		},
	}
}

type Hasher struct {
	pool sync.Pool
}

func (h *Hasher) Hash(inputs ...[]byte) []byte {
	hasher := h.pool.Get().(hash.Hash)
	defer h.pool.Put(hasher)

	hasher.Reset()
	for i := range inputs {
		_, err := hasher.Write(inputs[i])
		if err != nil {
			panic(err)
		}
	}
	return hasher.Sum(nil)
}
