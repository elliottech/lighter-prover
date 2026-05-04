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
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/scs"
	"github.com/elliottech/gnark-plonky2-verifier/trusted_setup"
	"github.com/elliottech/gnark-plonky2-verifier/types"
	"github.com/elliottech/gnark-plonky2-verifier/variables"
	"github.com/elliottech/lighter-prover/desertexit/circuits/wrapper/circuit"
	snarkBuilder "github.com/elliottech/lighter-prover/snark/builder"
)

func main() {
	circuitDataPath := flag.String("circuit-data", "", "path to the circuit data")
	verifierCircuitDataPath := flag.String("verifier-circuit-data", "", "path to the verifier circuit data")
	srsPath := flag.String("srs", "", "path to the srs file")
	innerCircuitDigest := flag.String("inner-circuit-digest", "", "digest for circuit outputs (optional)")
	outputPath := flag.String("output-path", ".", "output directory for generated files")
	flag.Parse()

	if *circuitDataPath == "" {
		panic("circuit data path is required")
	}
	if *verifierCircuitDataPath == "" {
		panic("verifier circuit data path is required")
	}
	if *srsPath == "" {
		panic("srs path is required")
	}

	// Do not build the circuits if the files already exist
	digest := *innerCircuitDigest
	if digest != "" {
		r1csFile := fmt.Sprintf("%s/desertwrapper::%s.r1cs", *outputPath, digest)
		pkFile := fmt.Sprintf("%s/desertwrapper::%s.pk", *outputPath, digest)
		vkFile := fmt.Sprintf("%s/desertwrapper::%s.vk", *outputPath, digest)
		solFile := fmt.Sprintf("%s/desertwrapper::%s.sol", *outputPath, digest)

		allExist := true
		for _, f := range []string{r1csFile, pkFile, vkFile, solFile} {
			if _, err := os.Stat(f); os.IsNotExist(err) {
				log.Printf("File %s does not exist, will build circuits.", f)
				allExist = false
			}
		}
		if allExist {
			log.Printf("All circuit files for digest %s exist, skipping build.", digest)
			return
		}
	}

	log.Println("Building circuit...")
	r1CS, computedDigest, err := BuildCircuitPlaceHolder(*circuitDataPath, *verifierCircuitDataPath)
	if err != nil {
		panic(fmt.Errorf("failed to build circuit: %v", err))
	}
	digest = computedDigest

	var srs = &kzg_bn254.SRS{}
	var srsLagrange kzg.SRS = kzg.NewSRS(ecc.BN254)
	if _, err := os.Stat(*srsPath); os.IsNotExist(err) {
		if *srsPath == "" {
			log.Println("SRS file doesn't exist, downloading...")
		} else {
			log.Printf("SRS file %s doesn't exist, downloading...", *srsPath)
		}
		trusted_setup.DownloadAndSaveAztecIgnitionSrs(174, *srsPath, false)
	}
	if _, err := os.Stat(*srsPath); os.IsNotExist(err) {
		panic(fmt.Errorf("srs file not found: %v", *srsPath))
	}
	srsFile, err := os.Open(*srsPath)
	if err != nil {
		panic(fmt.Errorf("failed to open srs file: %v", err))
	}
	log.Println("Reading SRS file...")
	defer srsFile.Close()
	_, err = snarkBuilder.ReadFromSRSFile(srs, srsFile, false)
	if err != nil {
		panic(fmt.Errorf("failed to read srs file: %v", err))
	}
	_, err = srsFile.Seek(0, io.SeekStart)
	if err != nil {
		panic(fmt.Errorf("failed to seek srs file: %v", err))
	}
	_, err = snarkBuilder.ReadFromSRSFile(srsLagrange.(*kzg_bn254.SRS), srsFile, false)
	if err != nil {
		panic(fmt.Errorf("failed to read srs lagrange file: %v", err))
	}
	// convert G1 points to lagrange form
	srsLagrange = snarkBuilder.ToLagrange(r1CS, srs)

	fileName := fmt.Sprintf("%s/desertwrapper::%s.r1cs", *outputPath, digest)
	fmt.Printf("Circuit built successfully. Writing R1CS to file %s", fileName)
	r1CSFile, err := os.Create(fileName)
	if err != nil {
		panic(fmt.Errorf("failed to create output file: %v", err))
	}
	defer r1CSFile.Close()
	_, err = r1CS.WriteTo(r1CSFile)
	if err != nil {
		panic(fmt.Errorf("failed to write R1CS to file: %v", err))
	}

	log.Println("Creating proving and verifying keys...")
	pk, vk, err := plonk.Setup(r1CS, srs, srsLagrange)
	if err != nil {
		panic(fmt.Errorf("failed to setup plonk: %v", err))
	}

	pkFileName := fmt.Sprintf("%s/desertwrapper::%s.pk", *outputPath, digest)
	log.Println("Keys generated. Writing proving key to", pkFileName)
	fPK, err := os.Create(pkFileName)
	if err != nil {
		panic(fmt.Errorf("failed to create proving key file: %v", err))
	}
	defer fPK.Close()
	pk.WriteTo(fPK)

	vkFileName := fmt.Sprintf("%s/desertwrapper::%s.vk", *outputPath, digest)
	log.Println("Writing verifying key to", vkFileName)
	fVK, err := os.Create(vkFileName)
	if err != nil {
		panic(fmt.Errorf("failed to create verifying key file: %v", err))
	}
	defer fVK.Close()
	vk.WriteTo(fVK)

	verifierContractFileName := fmt.Sprintf("%s/desertwrapper::%s.sol", *outputPath, digest)
	log.Println("Writing verifier contract to", verifierContractFileName)
	fSolidity, err := os.Create(verifierContractFileName)
	if err != nil {
		panic(fmt.Errorf("failed to create solidity file: %v", err))
	}
	if err = vk.ExportSolidity(fSolidity); err != nil {
		panic(err)
	}
	defer fSolidity.Close()

	log.Println("All files generated successfully")
}

// Returns the R1CS and the circuit digest that is going to be verified. It uses circuit data to generate a place holder proof.
func BuildCircuitPlaceHolder(commonCircuitDataPath, verifierCircuitDataPath string) (constraint.ConstraintSystem, string, error) {
	commonCircuitData := types.ReadCommonCircuitData(commonCircuitDataPath)
	verifierOnlyCircuitDataRaw := types.ReadVerifierOnlyCircuitData(verifierCircuitDataPath)
	verifierOnlyCircuitData := variables.DeserializeVerifierOnlyCircuitData(verifierOnlyCircuitDataRaw)
	proof, publicInputs := snarkBuilder.PlaceHolderProof(commonCircuitData)

	circuit := circuit.VerifierCircuit{
		Commitment:              frontend.Variable(0),
		PublicInputs:            publicInputs,
		Proof:                   proof,
		VerifierOnlyCircuitData: verifierOnlyCircuitData,
		CommonCircuitData:       commonCircuitData,
	}

	builder := scs.NewBuilder[constraint.U64]
	r1cs, err := frontend.Compile(ecc.BN254.ScalarField(), builder, &circuit)
	if err != nil {
		return nil, "", fmt.Errorf("failed to compile circuit: %v", err)
	}

	return r1cs, verifierOnlyCircuitDataRaw.CircuitDigest, nil
}
