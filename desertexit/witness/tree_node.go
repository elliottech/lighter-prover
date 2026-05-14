// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

package main

import (
	"bytes"
	"sync"
)

type InternalNode []byte

const (
	LeftChildId  = 0
	RightChildId = 1
)

type TreeNode struct {
	mu sync.RWMutex

	Version         Version
	Hash            []byte
	hasActiveLeaves bool

	Children [2]*TreeNode
	Parent   *TreeNode // Parent node, nil for root

	markMask uint64

	nilHash      []byte
	childNilHash []byte

	path   uint64
	depth  uint8
	hasher *Hasher
}

func NewTreeNode(depth uint8, path uint64, nilHashes *NilHashes, hasher *Hasher, parent *TreeNode) *TreeNode {
	treeNode := &TreeNode{
		nilHash:      nilHashes.Get(depth),
		childNilHash: nilHashes.Get(depth + 1),
		Hash:         nilHashes.Get(depth),
		path:         path,
		depth:        depth,
		hasher:       hasher,
		Parent:       parent,
		markMask:     0,
	}
	return treeNode
}

// Root Get hash of a node
func (node *TreeNode) Root() ([]byte, bool) {
	node.mu.RLock()
	defer node.mu.RUnlock()
	return node.root(), node.hasActiveLeaves
}

func (node *TreeNode) HasActiveLeaves() bool {
	node.mu.RLock()
	defer node.mu.RUnlock()
	return node.hasActiveLeaves
}

// Root Get hash of a node without a lock
func (node *TreeNode) root() []byte {
	return node.Hash
}

func (node *TreeNode) Set(hash []byte, version Version) {
	node.mu.Lock()
	defer node.mu.Unlock()

	node.Version = version
	node.Hash = hash
	node.hasActiveLeaves = !bytes.Equal(hash, node.nilHash)
}

// Extend node with lock
func (node *TreeNode) ExtendNode(nibble uint64, path uint64, nilhashes *NilHashes, hasher *Hasher) {
	node.mu.Lock()
	defer node.mu.Unlock()
	node.extendNode(nibble, path, nilhashes, hasher)
}

// Extend node without lock
func (node *TreeNode) extendNode(nibble uint64, path uint64, nilhashes *NilHashes, hasher *Hasher) {
	if node.Children[nibble] != nil {
		return
	}
	node.Children[nibble] = NewTreeNode(node.depth+1, path, nilhashes, hasher, node)
}

func (node *TreeNode) recompute(version Version) {
	node.mu.Lock()
	defer node.mu.Unlock()

	leftHash, rightHash := node.childNilHash, node.childNilHash
	leftHasActiveLeaves, rightHasActiveLeaves := false, false
	if node.Children[LeftChildId] != nil {
		leftHash, leftHasActiveLeaves = node.Children[LeftChildId].Root()
		if !leftHasActiveLeaves {
			// Clear left child if it has no active leaves
			node.Children[LeftChildId].Parent = nil
			node.Children[LeftChildId] = nil
		}
	}
	if node.Children[RightChildId] != nil {
		rightHash, rightHasActiveLeaves = node.Children[RightChildId].Root()
		if !rightHasActiveLeaves {
			// Clear right child if it has no active leaves
			node.Children[RightChildId].Parent = nil
			node.Children[RightChildId] = nil
		}
	}

	// Update the node
	node.hasActiveLeaves = leftHasActiveLeaves || rightHasActiveLeaves
	if node.hasActiveLeaves {
		node.Hash = node.hasher.Hash(leftHash, rightHash)
	} else {
		node.Hash = node.nilHash
	}
	node.Version = version
}

func (node *TreeNode) markAndExtendNode(nibble uint64, path uint64, nilhashes *NilHashes, hasher *Hasher) {
	node.mu.Lock()
	defer node.mu.Unlock()
	node.markMask |= 1 << nibble
	node.extendNode(nibble, path, nilhashes, hasher)
}

func (node *TreeNode) recomputeMultiSet(version Version, nibble int) bool {
	node.mu.Lock()
	defer node.mu.Unlock()
	if node.markMask == 0 {
		// Nothing to recompute
		return false
	}
	node.markMask ^= (1 << nibble)
	if !node.Children[nibble].HasActiveLeaves() {
		// Free memory
		node.Children[nibble].Parent = nil
		node.Children[nibble] = nil
	}
	if node.markMask != 0 {
		// One of the children didn't finish executing, continue after the last child
		return false
	}

	leftHash, rightHash := node.childNilHash, node.childNilHash
	leftHasActiveLeaves, rightHasActiveLeaves := false, false
	if node.Children[LeftChildId] != nil {
		leftHash, leftHasActiveLeaves = node.Children[LeftChildId].Root()
		if !leftHasActiveLeaves {
			// Clear left child if it has no active leaves
			node.Children[LeftChildId].Parent = nil
			node.Children[LeftChildId] = nil
		}
	}
	if node.Children[RightChildId] != nil {
		rightHash, rightHasActiveLeaves = node.Children[RightChildId].Root()
		if !rightHasActiveLeaves {
			// Clear right child if it has no active leaves
			node.Children[RightChildId].Parent = nil
			node.Children[RightChildId] = nil
		}
	}

	// Update the node
	node.hasActiveLeaves = leftHasActiveLeaves || rightHasActiveLeaves
	if node.hasActiveLeaves {
		node.Hash = node.hasher.Hash(leftHash, rightHash)
	} else {
		node.Hash = node.nilHash
	}
	node.Version = version

	return true
}
