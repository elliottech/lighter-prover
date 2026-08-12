//! Minimal little-endian cursor reader shared by the wire-format decoders
//! (challenge hints, signature-batch public inputs).
//!
//! Every read is bounds-checked against the remaining slice and returns
//! [`crate::Error::TruncatedWirePayload`] on underflow, so callers never need to
//! pre-validate length before decoding untrusted, attacker-controlled bytes.

/// A forward-only cursor over a byte slice, used to decode fixed little-endian
/// wire formats without panicking on truncated input.
pub(crate) struct ByteReader<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> ByteReader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }

    /// Number of bytes not yet consumed.
    pub(crate) fn remaining(&self) -> usize {
        self.bytes.len() - self.cursor
    }

    /// Reads exactly `N` bytes into a fixed-size array, advancing the cursor.
    pub(crate) fn read_array<const N: usize>(&mut self) -> Result<[u8; N], crate::Error> {
        let bytes = self.read_bytes(N)?;
        Ok(bytes
            .try_into()
            .expect("read_bytes returned requested length"))
    }

    /// Reads `len` bytes as a slice, advancing the cursor.
    ///
    /// Returns [`crate::Error::TruncatedWirePayload`] if fewer than `len` bytes
    /// remain, including when `cursor + len` would overflow `usize`.
    pub(crate) fn read_bytes(&mut self, len: usize) -> Result<&'a [u8], crate::Error> {
        let end = self
            .cursor
            .checked_add(len)
            .ok_or(crate::Error::TruncatedWirePayload)?;
        if end > self.bytes.len() {
            return Err(crate::Error::TruncatedWirePayload);
        }
        let bytes = &self.bytes[self.cursor..end];
        self.cursor = end;
        Ok(bytes)
    }

    /// Reads a little-endian `u16`, advancing the cursor by 2 bytes.
    pub(crate) fn read_u16(&mut self) -> Result<u16, crate::Error> {
        Ok(u16::from_le_bytes(self.read_array()?))
    }

    /// Reads a little-endian `u32`, advancing the cursor by 4 bytes.
    pub(crate) fn read_u32(&mut self) -> Result<u32, crate::Error> {
        Ok(u32::from_le_bytes(self.read_array()?))
    }

    /// Reads a little-endian `u64`, advancing the cursor by 8 bytes.
    pub(crate) fn read_u64(&mut self) -> Result<u64, crate::Error> {
        Ok(u64::from_le_bytes(self.read_array()?))
    }
}
