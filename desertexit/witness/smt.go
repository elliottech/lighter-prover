// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

package main

import (
	"bytes"
	"errors"
	"fmt"
	"math/big"
	"sync"

	"github.com/panjf2000/ants/v2"
)

// Option is a function that configures
type Option func(*LighterSparseMerkleTree)

func GoRoutinePool(pool *ants.Pool) Option {
	return func(s *LighterSparseMerkleTree) {
		s.goroutinePool = pool
	}
}

type NilHashes struct {
	Hashes [][]byte
}

func (h *NilHashes) Get(depth uint8) []byte {
	if len(h.Hashes)-1 < int(depth) {
		return nil
	}
	return h.Hashes[depth]
}

func constructNilHashes(maxDepth uint8, nilHash []byte, hasher *Hasher) *NilHashes {
	hashes := make([][]byte, maxDepth+1)
	hashes[maxDepth] = nilHash
	for i := 1; i <= int(maxDepth); i++ {
		nHash := hasher.Hash(nilHash, nilHash)
		hashes[maxDepth-uint8(i)] = nHash
		nilHash = nHash
	}
	return &NilHashes{hashes}
}

type LighterSparseMerkleTree struct {
	version Version

	root          *TreeNode
	maxDepth      uint8
	nilHashes     *NilHashes
	hasher        *Hasher
	goroutinePool *ants.Pool
}

func NewSparseMerkleTree(hasher *Hasher, maxDepth uint8, nilHash []byte) (SparseMerkleTree, error) {
	if maxDepth == 0 {
		return nil, ErrInvalidDepth
	}

	s := &LighterSparseMerkleTree{
		maxDepth:  maxDepth,
		nilHashes: constructNilHashes(maxDepth, nilHash, hasher),
		hasher:    hasher,
	}

	s.root = NewTreeNode(0, 0, s.nilHashes, s.hasher, nil)

	var err error
	if s.goroutinePool == nil {
		s.goroutinePool, err = ants.NewPool(128, ants.WithPanicHandler(func(i interface{}) {
			panic(i)
		}))
		if err != nil {
			return nil, err
		}
	}

	return s, nil
}

func (tree *LighterSparseMerkleTree) Get(key uint64, version *Version) ([]byte, error) {
	if tree.IsEmpty() {
		return nil, ErrEmptyRoot
	}

	if key > tree.getMaxKey() {
		return nil, ErrInvalidKey
	}

	if version == nil {
		version = &tree.version
	}

	if tree.version != *version {
		return nil, ErrInvalidVersion
	}

	currentNode := tree.root
	for depth := uint8(0); depth < tree.maxDepth; depth++ {
		if currentNode == nil {
			return tree.nilHashes.Get(tree.maxDepth), nil
		}
		path := (key >> (tree.maxDepth - (depth + 1)))
		nibble := path & 0x0000000000000001 // 1 bit for left or right
		currentNode = currentNode.Children[nibble]
	}
	if currentNode == nil {
		return tree.nilHashes.Get(tree.maxDepth), nil
	}
	ret, _ := currentNode.Root()
	return ret, nil
}

func (tree *LighterSparseMerkleTree) SetWithVersion(key uint64, val []byte, newVersion Version) error {
	if key > tree.getMaxKey() {
		return ErrInvalidKey
	}

	if newVersion <= tree.version {
		return ErrVersionTooLow
	}
	targetNode := tree.root
	depth := uint8(0)
	for ; depth < tree.maxDepth; depth++ {
		path := (key >> (tree.maxDepth - (depth + 1)))
		nibble := path & 0x0000000000000001 // 1 bit for left or right
		targetNode.ExtendNode(nibble, path, tree.nilHashes, tree.hasher)
		targetNode = targetNode.Children[nibble]
	}
	targetNode.Set(val, newVersion)
	for targetNode.depth > 0 {
		parent := targetNode.Parent
		parent.recompute(newVersion)
		targetNode = parent
	}
	return nil
}

// MultiSetWithVersion sets k,v pairs in parallel with a specific version.
//
// 1. generate all intermediate nodes, with lock;
// 2. set all leaves, without lock;
// 3. re-compute hash, from leaves to root
func (tree *LighterSparseMerkleTree) MultiSetWithVersion(items []Item, newVersion Version) error {
	size := len(items)
	if size == 0 {
		return nil
	}
	// also check len(items) not exceed 2^maxDepth - 1
	// also check no duplicated keys
	leavesChan := make(chan *TreeNode, size)
	// should we initialize all intermediate nodes when New SMT? so we can skip this step
	maxKey := tree.getMaxKey()
	keys := make(map[uint64]bool, size)
	wg := sync.WaitGroup{}
	for _, item := range items {
		it := item
		if it.Key > maxKey {
			return ErrInvalidKey
		}
		if _, exists := keys[it.Key]; exists {
			return fmt.Errorf("%w: key %d already exists", ErrDuplicateKey, it.Key)
		}
		keys[it.Key] = true
		wg.Add(1)
		err := tree.goroutinePool.Submit(func() {
			defer wg.Done()
			leavesChan <- tree.setIntermediateAndLeaves(it, newVersion)
		})
		if err != nil {
			return err
		}
	}

	wg.Wait()
	wg.Add(size)
	// For treeNode, the concurrency set to the number of leaf nodes
	for i := 0; i < size; i++ {
		leaf := <-leavesChan
		err := tree.goroutinePool.Submit(func() {
			defer wg.Done()
			tree.recompute(leaf)
		})
		if err != nil {
			return fmt.Errorf("failed to submit recompute task: %w", err)
		}
	}
	wg.Wait()
	return nil
}

// return leaf node
func (tree *LighterSparseMerkleTree) setIntermediateAndLeaves(item Item, newVer Version) *TreeNode {
	var (
		key = item.Key
		val = item.Val
	)
	targetNode := tree.root
	// find middle nodes
	for depth := uint8(0); int(depth) < int(tree.maxDepth); depth++ {
		// path <= 2^maxDepth - 1
		path := key >> (int(tree.maxDepth) - int(depth+1))
		// position in treeNode, nibble <= 0xf
		nibble := path & 0x0000000000000001 // 1 bit for left or right
		// mark internals that will change and create a new treeNode in targetNode
		targetNode.markAndExtendNode(nibble, path, tree.nilHashes, tree.hasher)
		targetNode = targetNode.Children[nibble]
	}
	// update hash of leaf node
	targetNode.Set(val, newVer)
	return targetNode
}

func (tree *LighterSparseMerkleTree) recompute(leaf *TreeNode) {
	version := leaf.Version
	currentNode := leaf

	for currentNode.depth > 0 {
		parent := currentNode.Parent
		if !parent.recomputeMultiSet(version, int(currentNode.path&0x0000000000000001)) { // 1 bit for left or right
			return
		}
		currentNode = parent
	}
}

func (tree *LighterSparseMerkleTree) IsEmpty() bool {
	hash, _ := tree.root.Root()
	return bytes.Equal(hash, tree.nilHashes.Get(0))
}

func (tree *LighterSparseMerkleTree) Root() []byte {
	hash, _ := tree.root.Root()
	return hash
}

func (tree *LighterSparseMerkleTree) GetProof(key uint64) (Proof, error) {
	proofs := make([][]byte, 0, tree.maxDepth)
	if tree.IsEmpty() {
		for i := tree.maxDepth; i > 0; i-- {
			proofs = append(proofs, tree.nilHashes.Get(i))
		}
		return proofs, nil
	}

	if key > tree.getMaxKey() {
		return nil, ErrInvalidKey
	}

	var neighborNode *TreeNode
	targetNode := tree.root
	for depth := uint8(0); depth < tree.maxDepth; depth++ {
		path := (key >> (tree.maxDepth - (depth + 1)))
		nibble := path & 0x0000000000000001 // 1 bit for left or right

		if targetNode == nil {
			// No active nodes under this path, return nil hash
			proofs = append(proofs, tree.nilHashes.Get(depth+1))
		} else {
			neighborNode = targetNode.Children[nibble^1]
			targetNode = targetNode.Children[nibble]
			if neighborNode == nil {
				proofs = append(proofs, tree.nilHashes.Get(depth+1))
			} else {
				hash, _ := neighborNode.Root()
				proofs = append(proofs, hash)
			}
		}
	}

	return reverseBytes(proofs[:]), nil
}

func (tree *LighterSparseMerkleTree) VerifyProof(key uint64, proof Proof) bool {
	if key > tree.getMaxKey() {
		return false
	}

	keyVal, err := tree.Get(key, nil)
	if err != nil && !errors.Is(err, ErrEmptyRoot) {
		return false
	}
	if len(keyVal) == 0 {
		keyVal = tree.nilHashes.Get(tree.maxDepth)
	}
	var helpers = make([]int, 0, tree.maxDepth)

	for depth := uint8(0); depth < tree.maxDepth; depth++ {
		path := key >> (int(tree.maxDepth) - (int(depth) + 1))
		nibble := path & 0x0000000000000001 // 1 bit for left or right
		helpers = append(helpers, int(nibble)%2)
	}
	helpers = reverseInts(helpers)
	if len(proof) != len(helpers) {
		return false
	}

	root := tree.Root()
	node := keyVal
	for i := 0; i < len(proof); i++ {
		switch helpers[i] {
		case 0:
			node = tree.hasher.Hash(node, proof[i])
		case 1:
			node = tree.hasher.Hash(proof[i], node)
		default:
			return false
		}
	}

	return bytes.Equal(root, node)
}

func (tree *LighterSparseMerkleTree) LatestVersion() Version {
	return tree.version
}

func (tree *LighterSparseMerkleTree) RecentVersion() Version {
	return tree.version
}

func (tree *LighterSparseMerkleTree) getMaxKey() uint64 {
	return new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), uint(tree.maxDepth)), big.NewInt(1)).Uint64()
}

func (tree *LighterSparseMerkleTree) Versions() []Version {
	tree.root.mu.RLock()
	defer tree.root.mu.RUnlock()
	return []Version{tree.root.Version}
}

func (tree *LighterSparseMerkleTree) Reset() {}

func (tree *LighterSparseMerkleTree) Commit(recentVersion *Version) (Version, error) {
	return tree.CommitWithNewVersion(recentVersion, nil)
}

func (tree *LighterSparseMerkleTree) CommitWithNewVersion(_ *Version, newVersion *Version) (Version, error) {
	var newVer Version
	if newVersion == nil {
		newVer = tree.version + 1
	} else {
		newVer = *newVersion
	}
	tree.version = newVer
	return newVer, nil
}

func (tree *LighterSparseMerkleTree) CommitGenesis() error {
	tree.version = 0
	return nil
}

func (tree *LighterSparseMerkleTree) Rollback(version Version) error {
	if version != tree.version {
		return ErrVersionTooOld
	}
	return nil
}

type (
	Version uint64

	Item struct {
		Key uint64
		Val []byte
	}

	Proof [][]byte

	SparseMerkleTree interface {
		Get(key uint64, version *Version) ([]byte, error)
		SetWithVersion(key uint64, val []byte, newVersion Version) error
		MultiSetWithVersion(items []Item, newVersion Version) error
		IsEmpty() bool
		Root() []byte
		GetProof(key uint64) (Proof, error)
		VerifyProof(key uint64, proof Proof) bool
		LatestVersion() Version
		RecentVersion() Version
		Reset()
		Commit(recentVersion *Version) (Version, error)
		CommitWithNewVersion(recentVersion *Version, newVersion *Version) (Version, error)
		CommitGenesis() error
		Rollback(version Version) error
		Versions() []Version
	}
)

func reverseBytes(value [][]byte) [][]byte {
	for i, j := 0, len(value)-1; i < j; i, j = i+1, j-1 {
		value[i], value[j] = value[j], value[i]
	}
	return value
}

func reverseInts(value []int) []int {
	for i, j := 0, len(value)-1; i < j; i, j = i+1, j-1 {
		value[i], value[j] = value[j], value[i]
	}
	return value
}

var (
	ErrEmptyRoot = errors.New("empty root")

	ErrVersionTooOld = errors.New("the version is lower than the rollback version")

	ErrInvalidVersion = errors.New("invalid version")

	ErrVersionTooHigh = errors.New("the version is higher than the latest version")

	ErrVersionTooLow = errors.New("the version is lower than the latest version")

	ErrNodeNotFound = errors.New("tree node not found")

	ErrVersionMismatched = errors.New("the version is mismatched with the database")

	ErrUnexpected = errors.New("unexpected error")

	ErrInvalidKey = errors.New("invalid key")

	ErrInvalidDepth = errors.New("depth must be a multiple of 4")

	ErrExtendNode = errors.New("extending node error")

	ErrNotImplemented = errors.New("not implemented")

	ErrInvalidNibble = errors.New("nibble must be either left or right child")

	ErrDuplicateKey = errors.New("duplicate key")
)
