//! The 16-bit range checks, as `(column, scale)` pairs: the checked value is
//! `scale * main[column]`.

use super::layout::*;

pub fn row_lookups() -> Vec<(usize, u64)> {
    let mut v: Vec<(usize, u64)> = (ACCX..WORK + WORK_VALUES * LIMBS).map(|c| (c, 1)).collect();
    v.extend((SLOT_BASE..MAIN_WIDTH).map(|c| (c, 1)));
    assert_eq!(v.len(), ROW_LOOKUPS);
    v
}

/// A scaled lookup `scale * c < 2^16` only bounds `c` when `c` itself is also
/// looked up (`c < 2^16`, so the product cannot wrap): every scaled column
/// appears with scale 1 too. `z`'s limbs `ZLO + 256 * ZHI` are bounded by their
/// bytes (both `< 256`).
pub fn reg_lookups() -> Vec<(usize, u64)> {
    let mut v: Vec<(usize, u64)> = (0..REG_LIMBS_END).map(|c| (c, 1)).collect();
    v.push((Y_HALF, 1));
    for c in ZLO..ZHI + LIMBS {
        v.push((c, 1));
        v.push((c, 256));
    }
    assert_eq!(v.len(), REG_LOOKUPS);
    v
}
