// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/consensys/gnark-crypto/ecc"
	kzg_bn254 "github.com/consensys/gnark-crypto/ecc/bn254/kzg"
	"github.com/consensys/gnark-crypto/kzg"
	"github.com/consensys/gnark/backend/plonk"
	"github.com/consensys/gnark/constraint"
	"github.com/elliottech/gnark-plonky2-verifier/trusted_setup"
	"github.com/elliottech/lighter-prover/snark/builder"
)

// go run snark/main.go -circuit-data common_circuit_data.json -verifier-circuit-data verifier_only_circuit_data.json -srs srs_file -generate-keys -pis proof_with_public_inputs.json
func main() {
	name := flag.String("name", "final", "name prefix for output files")
	circuitDataPath := flag.String("circuit-data", "", "path to the circuit data")
	verifierCircuitDataPath := flag.String("verifier-circuit-data", "", "path to the verifier circuit data")
	outputPath := flag.String("output", ".", "folder to save the output")
	generatKeys := flag.Bool("generate-keys", false, "generate proving and verification keys")
	srsPath := flag.String("srs", "", "path to the srs file")
	proofWithPublicInputsPath := flag.String("pis", "", "path to the proof with public inputs")
	placeHolder := flag.String("place-holder", "true", "use place holder pis")
	flag.Parse()

	if *circuitDataPath == "" {
		panic("circuit data path is required")
	}
	if *verifierCircuitDataPath == "" {
		panic("verifier circuit data path is required")
	}

	var r1CS constraint.ConstraintSystem
	var innerCircuitDigest string
	var err error

	if *placeHolder == "true" {
		r1CS, innerCircuitDigest, err = builder.BuildCircuitPlaceHolder(*circuitDataPath, *verifierCircuitDataPath)
	} else {
		if *proofWithPublicInputsPath == "" {
			panic("proof with public inputs path is required")
		}
		r1CS, innerCircuitDigest, err = builder.BuildCircuit(*circuitDataPath, *verifierCircuitDataPath, *proofWithPublicInputsPath)
	}
	if err != nil {
		panic(fmt.Errorf("failed to build circuit: %v", err))
	}

	fileName := fmt.Sprintf("%s/%s::%s.r1cs", *outputPath, *name, innerCircuitDigest)
	r1CSFile, err := os.Create(fileName)
	if err != nil {
		panic(fmt.Errorf("failed to create output file: %v", err))
	}
	defer r1CSFile.Close()
	_, err = r1CS.WriteTo(r1CSFile)
	if err != nil {
		panic(fmt.Errorf("failed to write R1CS to file: %v", err))
	}

	if !*generatKeys {
		return
	}

	if *srsPath == "" {
		panic("srs path is required")
	}

	var srs = &kzg_bn254.SRS{}
	var srsLagrange kzg.SRS = kzg.NewSRS(ecc.BN254)
	if _, err := os.Stat(*srsPath); os.IsNotExist(err) {
		trusted_setup.DownloadAndSaveAztecIgnitionSrs(174, *srsPath, false)
	}

	if _, err := os.Stat(*srsPath); os.IsNotExist(err) {
		panic(fmt.Errorf("srs file not found: %v", *srsPath))
	}
	srsFile, err := os.Open(*srsPath)
	if err != nil {
		panic(fmt.Errorf("failed to open srs file: %v", err))
	}
	defer srsFile.Close()
	_, err = builder.ReadFromSRSFile(srs, srsFile, false)
	if err != nil {
		panic(fmt.Errorf("failed to read srs file: %v", err))
	}
	_, err = srsFile.Seek(0, io.SeekStart)
	if err != nil {
		panic(fmt.Errorf("failed to seek srs file: %v", err))
	}
	_, err = builder.ReadFromSRSFile(srsLagrange.(*kzg_bn254.SRS), srsFile, false)
	if err != nil {
		panic(fmt.Errorf("failed to read srs lagrange file: %v", err))
	}
	// convert G1 points to lagrange form
	srsLagrange = builder.ToLagrange(r1CS, srs)

	var vk plonk.VerifyingKey
	var pk plonk.ProvingKey
	pk, vk, err = plonk.Setup(r1CS, srs, srsLagrange)
	if err != nil {
		panic(fmt.Errorf("failed to setup plonk: %v", err))
	}

	log.Println("Saving pk, vk, and verifier contract")

	pkFileName := fmt.Sprintf("%s/%s::%s.pk", *outputPath, *name, innerCircuitDigest)
	fPK, err := os.Create(pkFileName)
	if err != nil {
		panic(fmt.Errorf("failed to create proving key file: %v", err))
	}
	defer fPK.Close()
	pk.WriteTo(fPK)

	vkFileName := fmt.Sprintf("%s/%s::%s.vk", *outputPath, *name, innerCircuitDigest)
	fVK, err := os.Create(vkFileName)
	if err != nil {
		panic(fmt.Errorf("failed to create verifying key file: %v", err))
	}
	defer fVK.Close()
	vk.WriteTo(fVK)

	verifierContractFileName := fmt.Sprintf("%s/%s::%s.sol", *outputPath, *name, innerCircuitDigest)
	fSolidity, err := os.Create(verifierContractFileName)
	if err != nil {
		panic(fmt.Errorf("failed to create solidity file: %v", err))
	}
	if err = vk.ExportSolidity(fSolidity); err != nil {
		panic(err)
	}
	defer fSolidity.Close()
}
