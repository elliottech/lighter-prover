// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

package circuit

import (
	"fmt"
	"math/big"

	"github.com/consensys/gnark/frontend"
	gl "github.com/elliottech/gnark-plonky2-verifier/goldilocks"
	"github.com/elliottech/gnark-plonky2-verifier/types"
	"github.com/elliottech/gnark-plonky2-verifier/variables"
	"github.com/elliottech/gnark-plonky2-verifier/verifier"
)

type VerifierCircuit struct {
	Commitment              frontend.Variable `gnark:",public"`
	PublicInputs            []gl.Variable
	Proof                   variables.Proof
	VerifierOnlyCircuitData variables.VerifierOnlyCircuitData `gnark:"-"` // constant verifier data/key

	// This is configuration for the circuit, it is a constant not a variable
	CommonCircuitData types.CommonCircuitData
}

func (c *VerifierCircuit) Define(api frontend.API) error {
	verifierChip := verifier.NewVerifierChip(api, c.CommonCircuitData)
	verifierChip.Verify(c.Proof, c.PublicInputs, c.VerifierOnlyCircuitData)

	// Inner circuit outputs the commitment(keccak256 hash output) as byte array
	if len(c.PublicInputs) != 32 {
		panic(fmt.Errorf("public input size should be 32, got %d", len(c.PublicInputs)))
	}

	// Safe to assume that each public input is a valid byte because inner circuits constrain it

	// Combine the bytes to get the commitment
	commitment := frontend.Variable(0)
	i := 0
	for {
		if i >= 32 {
			break
		}
		u8 := c.PublicInputs[i].Limb

		commitment = api.MulAcc(commitment, u8, big.NewInt(0).Lsh(big.NewInt(1), uint(8*(31-i))))
		i += 1
	}

	api.AssertIsEqual(c.Commitment, commitment)

	return nil
}
