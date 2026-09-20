# Blob data schema

Each blob is 4096 × 32 bytes. Byte 0 of every 32-byte field element is unused (kept `0x00` so the element is
canonical); dropping it yields 4096 × 31 = 126,976 bytes of pub data, laid out as:

| Offset  | Size    | Field                                                               |
|---------|---------|---------------------------------------------------------------------|
| 0       | 2       | `version` (uint16 BE): `0` = V0, `1` = V1, `2` = V2                 |
| 2       | 32      | reserved, all zero                                                  |
| 34      | 1,020   | mark prices: 255 × uint32 BE, one per market index `0..254`         |
| 1,054   | 2,295   | funding rate prefix sums: 255 × (1 sign byte, `1` = negative; uint64 BE abs) |
| 3,349   | 510     | quote multipliers: 255 × uint16 BE                                  |
| 3,859   | 123,117 | compressed deltas (see below), zero padded                          |

Market sections always hold the full post-block value for every market; only the deltas section is compressed.

## Compressed deltas

The deltas section is a stream of 4-bit limbs (low nibble of each byte first). A *target* is encoded as
`[n][limb_1 .. limb_n]`: one nibble giving the count `n`, then `n` nibbles, most significant first. Every field
below is one target unless stated otherwise.

All values are *deltas* over the block (new − old); magnitudes are stored, signs travel in separate bits.

### Market deltas (version >= V1)

One entry per market slot whose `public_market_index` or slot status changed in the block (market created,
moved into settlement or expired); unchanged markets are not listed. Each entry carries the full published
state of the slot, so the latest entry for a slot is the slot's market pub data leaf.

V1:

```
market_delta_count
repeat market_delta_count:
  active << 12 | market_index      (market_index: 12 bits, active: 1 bit)
  public_market_index
```

V2 (binary options):

```
market_delta_count
repeat market_delta_count:
  price << 14 | status << 12 | market_index   (market_index: 12 bits, status: 2 bits, price: 32 bits)
  public_market_index
  settlement_cap
  quote_extension_multiplier
```

Slot status: `0` expired, `1` active, `2` in settlement. `price` is the binary options default price while
active, the settlement price while in settlement and `0` otherwise; a yes share pays `price × qem`, a no share
`(settlement_cap − price) × qem`. `settlement_cap` and `quote_extension_multiplier` are fixed at market creation
and zero for perps / spot slots. The market pub data leaf of a slot is `Poseidon2(2, status, price, settlement_cap,
quote_extension_multiplier)`, or the nil hash when all four are zero.

### Account deltas (version >= V0)

Only accounts with at least one non-zero item below appear, sorted by index; a `0` index diff after the first
account terminates the stream. Optional parts:

- `l1_address`: only when the account is created in the block (first L1 deposit to a new address, sub-account /
  pool creation).
- `account_type`: non-zero only when a non-master account is created (`0` = master or unchanged).
- `public_pool_info`: only for a pool whose total / operator shares changed (share mint / burn).
- `position` entry: position size or funding prefix sum changed (one market per tx, several per block possible).
- `binary_options_position` entry (V2): binary options position size changed. Sizes are signed, a negative size
  is a NO position.
- `asset` entry: net balance changed. USDC aggregates spot + isolated margin + position collateral; LIT includes
  pending unlocks. Zero net deltas are dropped.
- `share` entry: the account's shares in a public pool changed (pool deposit / withdraw).

```
repeat:
  account_index_diff               (added to the previous account index)
  flags                            bit0 = has_l1_address, bit1 = has_public_pool_info, bits 2.. = account_type
  if has_l1_address:      3 targets = byte-reversed address split into 7 / 7 / 6 byte little-endian chunks
  if has_public_pool_info: sign_bits (bit0 = operator_shares_neg, bit1 = total_shares_neg),
                           |total_shares_delta|, |operator_shares_delta|
  position_count
  repeat position_count:
    market_index | funding_carry << 8   (funding bits 60..63 ride in the high bits)
    |funding_prefix_sum_delta| low 60 bits
    |position_delta| << 2 | position_neg << 1 | funding_neg
  if version >= V2:
    binary_options_position_count
    repeat binary_options_position_count:
      market_index                   (market slot, 1000..2000)
      |size_delta| << 1 | size_neg
  asset_count
  repeat asset_count:
    balance_lo48 << 7 | balance_neg << 6 | asset_index   (asset_index: 6 bits)
    balance_hi                            (balance = balance_hi << 48 | balance_lo48)
  share_count
  repeat share_count:
    pool_index_diff << 1 | share_neg      (pool_index = MaxAccountIndex - pool_index_diff)
    |share_delta|
```

Account types: `0` master, `1` sub-account, `2` public pool, `3` insurance fund, `4` staking pool, `5` treasury.

Reference decoder: `witness/utils.go` (`getBlobBytes`, `bytesToDeltas`).
