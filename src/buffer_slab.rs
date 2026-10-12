//! # BufferSlab
//!
//! Pool of fixed-size buffers with O(1) tracking through a hierarchical bitmap.
//!
//! ## Design
//!
//! - `!Send + !Sync` through the raw pointer: single-threaded, or wrap it explicitly
//! - `allocate(&self)`: several guards can be alive at the same time
//! - `AutoGuard` frees its slot on `Drop`: no manual `free`
//! - Virtual space reserved up front with an anonymous `mmap`, initial pages pre-faulted
//!   with `madvise(MADV_POPULATE_WRITE)`: the `storage` pointer never moves.
//!
//! ## Example
//!
//! ```
//! use hislab::buffer_slab::BufferSlab;
//!
//! let pool = BufferSlab::<4096>::new(4, 128).unwrap();
//!
//! let mut g0 = pool.allocate();
//! let mut g1 = pool.allocate(); // several guards at once
//! g0.as_slice_mut().fill(0xAB);
//! assert_eq!(g0.index(), 0);
//! assert_eq!(g1.index(), 1);
//! // dropping g0 and g1 frees their slots
//! ```

use std::cell::UnsafeCell;

use memmap2::MmapMut;

use crate::bitmap_tree::BitmapTree;

// ============================================================================
// BufferSlab
// ============================================================================

/// Pool of fixed-size buffers of `N` bytes.
///
/// The whole virtual space is reserved at construction with an anonymous `mmap`.
/// The pages covering the first `initial_capacity` slots are pre-faulted right away
/// with `madvise(MADV_POPULATE_WRITE)`; the following pages are committed on demand.
///
/// `!Send + !Sync` by construction (`*mut u8` field): wrapping it in a `Mutex` or
/// `RwLock` is up to the caller if needed.
pub struct BufferSlab<const N: usize> {
    /// Stable pointer to the start of the storage; never moves.
    storage: *mut u8,
    /// Maximum number of slots; sets the size of the virtual reservation.
    virtual_capacity: u32,
    /// Hierarchical tracking bitmap, mutable through `&self` via `UnsafeCell`.
    /// Sound because `!Sync` forbids any concurrent access.
    tree: UnsafeCell<BitmapTree>,
    /// Keeps the mmap mapping alive. Never modified after construction.
    _mmap: MmapMut,
}

// ============================================================================
// AutoGuard
// ============================================================================

/// RAII guard over a buffer allocated from a [`BufferSlab`].
///
/// Frees its slot automatically on `Drop`. Holds `&'a BufferSlab`, so the borrow
/// checker guarantees the slab outlives the guard.
///
/// Several guards can coexist.
pub struct AutoGuard<'a, const N: usize> {
    ptr: *mut u8,
    idx: u32,
    slab: &'a BufferSlab<N>,
}

// ============================================================================
// Impl BufferSlab
// ============================================================================

impl<const N: usize> BufferSlab<N> {
    /// Creates a pool with `virtual_capacity` slots reserved in virtual memory.
    ///
    /// The pages covering the first `initial_capacity` slots are pre-faulted right away
    /// with `madvise(MADV_POPULATE_WRITE)`, so the first allocations take no page fault.
    /// The following pages are committed on demand by the kernel.
    ///
    /// The storage pointer never moves: indices returned by `allocate` stay valid for
    /// the whole lifetime of the pool.
    ///
    /// # Panics
    ///
    /// - If `initial_capacity > virtual_capacity`
    /// - If `virtual_capacity * N` overflows `usize`: invalid parameters, the virtual
    ///   space cannot be reserved
    ///
    /// # Errors
    ///
    /// Returns an error if `mmap` fails (system out of memory).
    pub fn new(initial_capacity: u32, virtual_capacity: u32) -> Result<Self, std::io::Error> {
        assert!(N > 0, "N must be > 0");
        assert!(
            initial_capacity <= virtual_capacity,
            "initial_capacity must be <= virtual_capacity"
        );

        // Intentional panic: virtual_capacity * N > usize::MAX is a caller bug,
        // not a condition to recover from at runtime.
        let virtual_bytes = (virtual_capacity as usize)
            .checked_mul(N)
            .expect("virtual_capacity * N dépasse usize::MAX");

        let mmap = MmapMut::map_anon(virtual_bytes)?;
        let storage = mmap.as_ptr() as *mut u8;

        // Pre-fault the initial pages so the first allocations take no page fault.
        // MADV_POPULATE_WRITE asks the kernel to resolve write faults right away
        // (Linux 5.14+). Best effort: errors are ignored (older kernel, unsupported
        // madvise).
        if initial_capacity > 0 {
            let initial_bytes = initial_capacity as usize * N;
            unsafe {
                libc::madvise(
                    storage as *mut libc::c_void,
                    initial_bytes,
                    libc::MADV_POPULATE_WRITE | libc::MADV_HUGEPAGE,
                );
            }
        }

        Ok(Self {
            storage,
            virtual_capacity,
            tree: UnsafeCell::new(BitmapTree::new()),
            _mmap: mmap,
        })
    }

    /// Maximum virtual capacity (total number of reserved slots).
    #[inline]
    pub fn virtual_capacity(&self) -> u32 {
        self.virtual_capacity
    }

    /// Allocates a free buffer and returns its RAII guard.
    ///
    /// Panics if `virtual_capacity` is reached (pool exhausted).
    /// O(1) through the hierarchical bitmap.
    #[inline]
    pub fn allocate(&self) -> AutoGuard<'_, N> {
        self.try_allocate()
            .expect("BufferSlab: virtual_capacity épuisée")
    }

    /// Tries to allocate a free buffer. Returns `None` if `virtual_capacity` is reached.
    ///
    /// O(1) through the hierarchical bitmap.
    #[inline]
    pub fn try_allocate(&self) -> Option<AutoGuard<'_, N>> {
        let tree = unsafe { &mut *self.tree.get() };
        let idx = tree.find_free_idx();
        if idx >= self.virtual_capacity {
            return None;
        }
        tree.set_bit(idx);
        let ptr = unsafe { self.storage.add(idx as usize * N) };
        Some(AutoGuard {
            ptr,
            idx,
            slab: self,
        })
    }

    /// Frees a slot by index. Called by the `Drop` of [`AutoGuard`].
    #[inline]
    fn free_at(&self, idx: u32) {
        debug_assert!(
            unsafe { &*self.tree.get() }.is_set(idx),
            "free_at: slot {} n'était pas alloué",
            idx
        );
        unsafe { &mut *self.tree.get() }.clear_bit(idx);
    }
}

// ============================================================================
// Impl AutoGuard
// ============================================================================

impl<const N: usize> AutoGuard<'_, N> {
    #[inline(always)]
    const fn is_ptr_power_of_two() -> bool {
        N.is_power_of_two()
    }

    #[inline(always)]
    unsafe fn hint_aligned(&self) {
        // 1. Checked at compile time: is N a power of two?

        // 2. LLVM only sees the hint when it is true
        if Self::is_ptr_power_of_two() {
            // LLVM intrinsic for the alignment hint
            unsafe {
                std::hint::assert_unchecked(self.ptr as usize & (N - 1) == 0);
            }
        }
    }

    /// Shared slice over the `N` bytes of the buffer.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            self.hint_aligned();
            std::slice::from_raw_parts(self.ptr, N)
        }
    }

    /// Mutable slice over the `N` bytes of the buffer.
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        unsafe {
            self.hint_aligned();
            std::slice::from_raw_parts_mut(self.ptr, N)
        }
    }

    /// Index of the buffer in the pool.
    #[inline]
    pub fn index(&self) -> u32 {
        self.idx
    }

    /// Raw pointer to the start of the buffer.
    ///
    /// Valid while the guard is alive.
    #[inline]
    pub fn as_ptr(&self) -> *mut u8 {
        unsafe { self.hint_aligned() };
        self.ptr
    }
}

impl<const N: usize> Drop for AutoGuard<'_, N> {
    #[inline]
    fn drop(&mut self) {
        self.slab.free_at(self.idx);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_drop_basic() {
        let pool = BufferSlab::<4096>::new(4, 4).expect("should allocate");

        let g0 = pool.allocate();
        let g1 = pool.allocate();
        let g2 = pool.allocate();
        let g3 = pool.allocate();

        assert_eq!(g0.index(), 0);
        assert_eq!(g1.index(), 1);
        assert_eq!(g2.index(), 2);
        assert_eq!(g3.index(), 3);

        assert!(pool.try_allocate().is_none());

        drop(g1);
        let g1b = pool.allocate();
        assert_eq!(g1b.index(), 1);
    }

    #[test]
    fn test_multiple_guards_simultaneous() {
        let pool = BufferSlab::<64>::new(8, 8).expect("should allocate");
        let guards: Vec<_> = (0..8).map(|_| pool.allocate()).collect();
        assert!(pool.try_allocate().is_none());
        drop(guards);
        assert!(pool.try_allocate().is_some());
    }

    #[test]
    fn test_beyond_initial_capacity() {
        // Allocating beyond initial_capacity works (pages committed on demand)
        let pool = BufferSlab::<64>::new(2, 16).expect("should allocate");
        let guards: Vec<_> = (0..16).map(|_| pool.allocate()).collect();
        assert!(pool.try_allocate().is_none());
        drop(guards);
    }

    #[test]
    fn test_ptr_alignment() {
        let pool = BufferSlab::<4096>::new(4, 4).expect("should allocate");
        let g0 = pool.allocate();
        let g1 = pool.allocate();
        assert_eq!(g1.as_ptr() as usize - g0.as_ptr() as usize, 4096);
    }

    #[test]
    fn test_rw_slice() {
        let pool = BufferSlab::<64>::new(2, 2).expect("should allocate");
        let mut g = pool.allocate();
        g.as_slice_mut().fill(0xBE);
        assert!(g.as_slice().iter().all(|&b| b == 0xBE));
    }

    #[test]
    fn test_virtual_capacity_exhausted() {
        let pool = BufferSlab::<64>::new(4, 600).expect("should allocate");
        let guards: Vec<_> = (0..600).map(|_| pool.allocate()).collect();
        assert!(pool.try_allocate().is_none());
        drop(guards);
        assert!(pool.try_allocate().is_some());
    }
}
