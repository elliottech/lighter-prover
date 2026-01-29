// Portions of this file are derived from sp1
// Copyright (c) 2023 Succinct Labs
// Licensed under the MIT License. See THIRD_PARTY_NOTICES for details.

// Copyright 2020-2025 Consensys Software Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Original file: https://github.com/Consensys/gnark-crypto/blob/v0.19.0/ecc/bn254/kzg/marshal.go
// Modifications: Modified SRS reader functions in [ecc/bn254/kzg/marshal.go] to support
// legacy SRS serialization format.
//
// See THIRD_PARTY_NOTICES for details.

package builder

import (
	"io"
	"math/bits"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	kzg_bn254 "github.com/consensys/gnark-crypto/ecc/bn254/kzg"
	"github.com/consensys/gnark-crypto/kzg"
	"github.com/consensys/gnark/constraint"
)

// Taken from https://github.com/succinctlabs/sp1/blob/15b274359964f6afada9b7293cdc2864a7177054/crates/recursion/gnark-ffi/go/sp1/trusted_setup/trusted_setup.go#L151
func ToLagrange(scs constraint.ConstraintSystem, canonicalSRS kzg.SRS) kzg.SRS {
	var lagrangeSRS kzg.SRS

	switch srs := canonicalSRS.(type) {
	case *kzg_bn254.SRS:
		var err error
		sizeSystem := scs.GetNbPublicVariables() + scs.GetNbConstraints()
		nextPowerTwo := 1 << bits.Len(uint(sizeSystem))
		newSRS := &kzg_bn254.SRS{Vk: srs.Vk}
		newSRS.Pk.G1, err = kzg_bn254.ToLagrangeG1(srs.Pk.G1[:nextPowerTwo])
		if err != nil {
			panic(err)
		}
		lagrangeSRS = newSRS
	default:
		panic("unrecognized curve")
	}

	return lagrangeSRS
}

// ReadFrom decodes SRS data from reader.
func ReadFromSRSFile(srs *kzg_bn254.SRS, r io.Reader, readLines bool) (int64, error) {
	// decode the VerifyingKey
	var pn, vn int64
	var err error
	if pn, err = srs.Pk.ReadFrom(r); err != nil {
		return pn, err
	}
	vn, err = ReadFromVerifyingKey(&srs.Vk, r, readLines)
	if err != nil {
		return pn, err
	}

	// If lines are not read, precompute them
	if !readLines {
		srs.Vk.Lines[0] = bn254.PrecomputeLines(srs.Vk.G2[0])
		srs.Vk.Lines[1] = bn254.PrecomputeLines(srs.Vk.G2[1])
	}

	return pn + vn, nil
}

// ReadFrom decodes VerifyingKey data from reader.
func ReadFromVerifyingKey(vk *kzg_bn254.VerifyingKey, r io.Reader, readLines bool) (int64, error) {
	// decode the VerifyingKey
	dec := bn254.NewDecoder(r)
	nLines := 66
	toDecode := make([]interface{}, 0, 4*nLines+3)
	toDecode = append(toDecode, &vk.G2[0])
	toDecode = append(toDecode, &vk.G2[1])
	toDecode = append(toDecode, &vk.G1)

	if readLines {
		for k := 0; k < 2; k++ {
			for j := 0; j < 2; j++ {
				for i := nLines - 1; i >= 0; i-- {
					toDecode = append(toDecode, &vk.Lines[k][j][i].R0)
					toDecode = append(toDecode, &vk.Lines[k][j][i].R1)
				}
			}
		}
	}

	for _, v := range toDecode {
		if err := dec.Decode(v); err != nil {
			return dec.BytesRead(), err
		}
	}

	return dec.BytesRead(), nil
}
