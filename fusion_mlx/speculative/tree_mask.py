# SPDX-License-Identifier: Apache-2.0
"""PR-I: Tree-Mask spec-decode verification + Draft Virtual Append Offset.

Replaces the linear first-mismatch verify path (dflash/verifier.py,
prompt_lookup) with:

  - Tree-Mask attention: verify multiple draft candidates (a tree, not a
    chain) in ONE forward pass. Each query position attends only to its
    ancestors in the draft tree (causal tree mask), so all branches are
    verified simultaneously. Returns the longest accepted root-to-leaf path.
  - Draft Virtual Append Offset: draft tokens are proposed into a virtual
    offset on top of the real cache (no commit). On verify: atomic commit
    advances the real offset by the accepted length; rejected suffix is
    zero-copy rolled back (trim, no copy).

Degrade switch (default OFF — prototype):
  FUSION_SHIM_TREE_MASK=1  — enable tree-mask verify + virtual append

When OFF, callers use the stock linear first-mismatch verifier — zero
behavior change. Golden reference harness (PR-F) verifies KL < 1e-6 vs
stock linear verify on single-chain trees (tree degenerates to chain).

The Metal ragged-candidate kernel (true ragged Q packing, no padding) is
Tier-2 deferred — this module builds the dense tree mask + uses stock
broadcast attention. The vendored steel attention ``block_token_mask``
(glm_moe_dsa) is the eventual kernel target; interface stable here.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import mlx.core as mx

logger = logging.getLogger(__name__)


def _env_on(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


def is_tree_mask_enabled() -> bool:
    return _env_on("FUSION_SHIM_TREE_MASK")


# ---------------------------------------------------------------------------
# DraftTree — topology of draft candidates.
#
# A draft tree is a list of nodes; node 0 is the root (the last confirmed
# token). Each node carries its parent index and a draft token id. A chain
# (linear spec decode) is the degenerate tree where node i's parent is i-1.
#
# The tree lets a drafter propose multiple continuations from a shared
# prefix (e.g. top-k branches at each step) and verify them all in one
# target forward pass.
# ---------------------------------------------------------------------------


@dataclass
class DraftNode:
    token: int
    parent: int  # -1 for root


@dataclass
class DraftTree:
    """Draft candidate tree.

    Node 0 is the ROOT = the last confirmed token (anchor, not a draft
    candidate). Nodes 1..n are draft candidates. accepted_tokens returned
    by verify_tree_logits excludes the root (path[1:]).

    A chain tree (from_chain) with root = last confirmed token verifies
    identically to the stock linear first-mismatch path — used for the
    golden reference check.
    """

    nodes: list[DraftNode] = field(default_factory=list)

    @classmethod
    def from_chain(cls, root_token: int, chain_tokens: list[int]) -> DraftTree:
        """Build a degenerate chain tree (parent = i-1).

        Node 0 = root_token (last confirmed). Nodes 1..n = chain_tokens.
        A chain tree verifies identically to the stock linear first-mismatch
        path — used for the golden reference KL check.
        """
        nodes = [DraftNode(token=root_token, parent=-1)]
        for i, t in enumerate(chain_tokens):
            nodes.append(DraftNode(token=t, parent=i))
        return cls(nodes=nodes)

    @classmethod
    def from_branches(
        cls,
        root_token: int,
        shared_prefix: list[int],
        branches: list[list[int]],
    ) -> DraftTree:
        """Build a tree from a root + shared prefix + k branches.

        Node 0 = root_token (last confirmed). Nodes 1..k = shared_prefix
        (chain from root). Each branch extends from the last prefix node.
        """
        nodes: list[DraftNode] = [DraftNode(token=root_token, parent=-1)]
        cur_parent = 0
        for t in shared_prefix:
            nodes.append(DraftNode(token=t, parent=cur_parent))
            cur_parent = len(nodes) - 1
        prefix_tail = cur_parent
        for branch in branches:
            cur_parent = prefix_tail
            for t in branch:
                nodes.append(DraftNode(token=t, parent=cur_parent))
                cur_parent = len(nodes) - 1
        return cls(nodes=nodes)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def tokens(self) -> list[int]:
        return [n.token for n in self.nodes]

    def parent_array(self) -> mx.array:
        return mx.array([n.parent for n in self.nodes], dtype=mx.int32)

    def ancestors_of(self, idx: int) -> list[int]:
        """Return [root, ..., idx] inclusive — the path from root to idx."""
        path = []
        cur = idx
        while cur >= 0:
            path.append(cur)
            cur = self.nodes[cur].parent
        path.reverse()
        return path


# ---------------------------------------------------------------------------
# Tree causal mask.
#
# mask[i, j] = 1 if node j is an ancestor of (or equal to) node i, else 0.
# Query i attends to key j iff j is on i's root-path. This is the tree
# generalization of the lower-triangular causal mask.
# ---------------------------------------------------------------------------


def build_tree_attention_mask(tree: DraftTree) -> mx.array:
    """Build the (n, n) tree causal attention mask.

    mask[i, j] = 1.0 where node j is an ancestor-or-self of node i.
    """
    n = tree.n_nodes
    if n == 0:
        return mx.zeros((0, 0), dtype=mx.float32)
    parents = [node.parent for node in tree.nodes]
    rows = []
    for i in range(n):
        anc = set(tree.ancestors_of(i))
        row = [1.0 if j in anc else 0.0 for j in range(n)]
        rows.append(row)
    return mx.array(rows, dtype=mx.float32)


def apply_tree_mask(scores: mx.array, mask: mx.array) -> mx.array:
    """Apply tree causal mask to attention scores: -inf where mask==0.

    scores: (..., n_q, n_k) attention logits.
    mask:   (n_q, n_k) tree mask (1 = attend, 0 = blocked).
    """
    neg_inf = mx.array(-float("inf"), dtype=scores.dtype)
    return mx.where(mask.astype(mx.bool_), scores, neg_inf)


# ---------------------------------------------------------------------------
# Tree verification.
#
# verify_tree runs ONE target forward over the draft tree's token sequence
# (in tree order), gathers argmax at each node position, then walks the tree
# to find the longest accepted root-to-leaf path. A node is "accepted" if
# the target's argmax at that node's parent position matches the node's
# draft token.
#
# This generalizes linear first-mismatch: for a chain tree, it returns the
# same accepted prefix + bonus token as _decide_accepted_prefix.
#
# The actual attention kernel (tree mask applied during the target forward)
# is Tier-2 deferred. verify_tree here operates on the target logits the
# caller already produced (the caller may apply the tree mask via a custom
# attention call, or the target's standard causal mask suffices when the
# tree is a chain). The interface is stable so a future ragged Metal kernel
# can drop in behind verify_tree_logits.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TreeVerifyResult:
    accepted_tokens: tuple[int, ...]
    bonus_token: int
    accepted_len: int
    accepted_path: tuple[int, ...]  # node indices root..leaf
    n_verified: int


def verify_tree_logits(
    tree: DraftTree,
    target_argmax: list[int],
) -> TreeVerifyResult:
    """Decide accepted path from target argmax at each node position.

    target_argmax[i] = the token the target predicts at node i's position
    (i.e. the next token after node i). A child node c of parent p is
    accepted if target_argmax[p] == tree.nodes[c].token.

    For the root (parent=-1), the bonus token is target_argmax[0] — the
    token the target predicts after the last confirmed token (node 0 is
    the last confirmed token itself, so target_argmax[0] is the first
    predicted token).

    Walks the tree BFS/DFS to find the longest accepted root-to-leaf path.
    Returns the accepted tokens + a bonus token (the target's prediction
    at the end of the accepted path).
    """
    n = tree.n_nodes
    if n == 0:
        return TreeVerifyResult(
            accepted_tokens=(),
            bonus_token=-1,
            accepted_len=0,
            accepted_path=(),
            n_verified=0,
        )
    if len(target_argmax) != n:
        raise ValueError(
            f"target_argmax length {len(target_argmax)} must equal " f"tree nodes {n}"
        )

    # children[p] = list of child node indices.
    children: list[list[int]] = [[] for _ in range(n)]
    for i, node in enumerate(tree.nodes):
        if node.parent >= 0:
            children[node.parent].append(i)

    # BFS from root (node 0), accepting children whose draft token matches
    # the target's argmax at the parent position.
    # Node 0 (root = last confirmed token) is always "accepted" (it's the
    # anchor, not a draft prediction).
    best_path = [0]
    stack = [[0]]
    while stack:
        path = stack.pop()
        node = path[-1]
        accepted_children = []
        for c in children[node]:
            if target_argmax[node] == tree.nodes[c].token:
                accepted_children.append(c)
        if not accepted_children:
            if len(path) > len(best_path):
                best_path = path
            continue
        for c in accepted_children:
            stack.append(path + [c])

    accepted_path = tuple(best_path)
    accepted_tokens = tuple(tree.nodes[i].token for i in best_path[1:])
    leaf = best_path[-1]
    bonus_token = target_argmax[leaf]
    return TreeVerifyResult(
        accepted_tokens=accepted_tokens,
        bonus_token=bonus_token,
        accepted_len=len(accepted_tokens),
        accepted_path=accepted_path,
        n_verified=n,
    )


# ---------------------------------------------------------------------------
# Draft Virtual Append Offset.
#
# During spec decode, the drafter proposes tokens that the target has NOT
# yet verified. Writing them into the real KV cache commits unverified KV
# state; on rejection, the verifier must trim (copy-out) the rejected tail.
#
# Virtual Append Offset decouples draft proposal from real cache commit:
#   - propose(n) advances a *virtual* offset on top of the real cache offset
#     — draft KV is staged but the real offset does not move.
#   - commit(n) atomically advances the real offset by n (accepted tokens
#     become visible); the staged KV beyond n is discarded.
#   - rollback() discards ALL staged draft KV — zero-copy (just resets the
#     virtual offset; no trim/copy needed because the real offset never moved).
#
# This wraps any FusionPagedKVCache (or CoWPagedKVCache). When the switch
# is OFF, propose/commit/rollback are no-ops and the caller uses the stock
# update_and_fetch + trim path directly.
# ---------------------------------------------------------------------------


class DraftVirtualAppendOffset:
    """Virtual append offset on top of a paged KV cache.

    Usage:
        voff = DraftVirtualAppendOffset(cache)
        voff.propose(draft_keys, draft_values)  # stage, real offset unmoved
        # ... target verify ...
        if accepted_len > 0:
            voff.commit(accepted_len)   # advance real offset
        else:
            voff.rollback()             # discard staged, zero-copy
    """

    def __init__(self, cache):
        self.cache = cache
        self._virtual_offset = 0
        self._staged_k = None
        self._staged_v = None
        self._staged_steps = 0
        self._committed = 0
        self._rolled_back = 0

    @property
    def real_offset(self) -> int:
        return self.cache.offset

    @property
    def virtual_offset(self) -> int:
        return self.cache.offset + self._virtual_offset

    @property
    def staged_steps(self) -> int:
        return self._staged_steps

    def propose(self, keys: mx.array, values: mx.array) -> int:
        """Stage draft KV without advancing the real cache offset.

        keys/values: (B, n_kv_heads, num_steps, k_head_dim) draft KV.
        Returns the virtual offset after staging (real offset unchanged).
        """
        if not is_tree_mask_enabled():
            # Stock path: write directly into the real cache (no virtual).
            self.cache.update_and_fetch(keys, values)
            return self.cache.offset
        self._staged_k = keys
        self._staged_v = values
        self._staged_steps = keys.shape[2]
        self._virtual_offset = self._staged_steps
        logger.debug(
            "draft_voff propose: staged %d steps, real_offset=%d virt_offset=%d",
            self._staged_steps,
            self.real_offset,
            self.virtual_offset,
        )
        return self.virtual_offset

    def commit(self, n: int) -> int:
        """Atomically advance the real cache offset by n accepted steps.

        Writes the first n staged steps into the real cache (update_and_fetch)
        and discards the rest. Returns the new real offset.
        """
        if not is_tree_mask_enabled():
            # Stock path: already written by propose; just account.
            self._virtual_offset = 0
            self._staged_k = None
            self._staged_v = None
            self._staged_steps = 0
            return self.cache.offset
        n = max(0, min(n, self._staged_steps))
        if n > 0 and self._staged_k is not None:
            self.cache.update_and_fetch(
                self._staged_k[..., :n, :], self._staged_v[..., :n, :]
            )
        self._committed += n
        self._staged_k = None
        self._staged_v = None
        self._staged_steps = 0
        self._virtual_offset = 0
        logger.info(
            "draft_voff commit: advanced real offset by %d -> %d",
            n,
            self.cache.offset,
        )
        return self.cache.offset

    def rollback(self) -> int:
        """Discard all staged draft KV. Zero-copy: real offset never moved."""
        if not is_tree_mask_enabled():
            return self.cache.offset
        self._rolled_back += self._staged_steps
        self._staged_k = None
        self._staged_v = None
        self._staged_steps = 0
        self._virtual_offset = 0
        logger.debug(
            "draft_voff rollback: discarded staged, real_offset=%d",
            self.cache.offset,
        )
        return self.cache.offset

    def stats(self) -> dict:
        return {
            "committed_total": self._committed,
            "rolled_back_total": self._rolled_back,
            "staged_steps": self._staged_steps,
            "virtual_offset": self._virtual_offset,
            "real_offset": self.real_offset,
            "tree_mask_enabled": is_tree_mask_enabled(),
        }


__all__ = [
    "DraftNode",
    "DraftTree",
    "DraftVirtualAppendOffset",
    "TreeVerifyResult",
    "apply_tree_mask",
    "build_tree_attention_mask",
    "is_tree_mask_enabled",
    "verify_tree_logits",
]
