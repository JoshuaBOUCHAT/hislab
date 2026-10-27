//! # HiSlab
//!
//! A high-performance slab allocator using hierarchical bitmaps for O(1) operations.
//!
//! ## Example
//!
//! ```
//! use hislab::HiSlab;
//!
//! let slab = HiSlab::<i32>::new(0, 65536).unwrap();
//!
//! let idx = slab.insert(42);
//! assert_eq!(slab[idx], 42);
//!
//! let val = slab.remove(idx);
//! assert_eq!(val, Some(42));
//! assert!(slab.get(idx).is_none());
//! ```

use std::cell::{Cell, UnsafeCell};
use std::ops::{Index, IndexMut};

use crate::bit_block::BitBlock;
use crate::bitmap_tree::BitmapTree;
use memmap2::MmapMut;

mod bit_block;
mod bitmap_tree;
pub mod buffer_slab;
#[cfg(test)]
mod test;

// ============================================================================
// HiSlab
// ============================================================================

/// A slab allocator with O(1) insert and remove using hierarchical bitmaps.
///
/// `HiSlab` stores elements in one contiguous region reserved up front with an
/// anonymous `mmap` (indices are stable, nothing is ever moved), and tracks free
/// slots with a 4-level bitmap hierarchy, so a free slot is found in constant time
/// regardless of fragmentation.
///
/// `!Send + !Sync` by construction (raw pointer): wrap it in a `Mutex` if needed.
/// In exchange, `insert` and `remove` take `&self` (interior mutability through
/// `UnsafeCell`, sound because `!Sync` forbids any concurrent access).
///
/// For tagging support, see [`TaggedHiSlab`].
pub struct HiSlab<T: 'static> {
    data: *mut T,
    virtual_capacity: u32,
    /// Hierarchical bitmap, mutable through `&self` via `UnsafeCell`.
    /// Sound because `!Sync` forbids any concurrent access.
    pub(crate) tree: UnsafeCell<BitmapTree>,
    mmap: MmapMut,
    /// Slots occupés.
    len: Cell<u32>,
    /// Au-delà de ce slot, rien n'a jamais été écrit (ou tout a été rendu) : borne de
    /// [`HiSlab::release_above`].
    touched_end: Cell<u32>,
}

impl<T> HiSlab<T> {
    /// Creates a new empty `HiSlab`.
    ///
    /// `virtual_capacity * size_of::<T>()` bytes are reserved right away in virtual
    /// memory. The pages covering `initial_capacity` slots are pre-faulted with
    /// `madvise(MADV_POPULATE_WRITE)`; the following pages are committed on demand.
    ///
    /// # Panics
    ///
    /// - If `initial_capacity > virtual_capacity`
    /// - If `virtual_capacity * size_of::<T>()` overflows `usize`
    ///
    /// # Errors
    ///
    /// Returns an error if `mmap` fails (system out of memory).
    pub fn new(initial_capacity: u32, virtual_capacity: u32) -> Result<Self, std::io::Error> {
        assert!(
            initial_capacity <= virtual_capacity,
            "initial_capacity must be <= virtual_capacity"
        );

        let virtual_bytes = (virtual_capacity as usize)
            .checked_mul(std::mem::size_of::<T>())
            .expect("virtual_capacity * size_of::<T>() dépasse usize::MAX");

        let mmap = MmapMut::map_anon(virtual_bytes)?;
        let ptr = mmap.as_ptr() as *mut T;

        if initial_capacity > 0 {
            let initial_bytes = initial_capacity as usize * std::mem::size_of::<T>();
            unsafe {
                libc::madvise(
                    ptr as *mut libc::c_void,
                    initial_bytes,
                    libc::MADV_POPULATE_WRITE | libc::MADV_HUGEPAGE,
                );
            }
        }

        Ok(Self {
            data: ptr,
            virtual_capacity,
            tree: UnsafeCell::new(BitmapTree::new()),
            mmap,
            len: Cell::new(0),
            touched_end: Cell::new(initial_capacity),
        })
    }
}

impl<T> HiSlab<T> {
    /// Inserts a value and returns its index.
    ///
    /// The returned index is stable and can be used to access the value
    /// until it is removed.
    #[inline(always)]
    pub fn insert(&self, val: T) -> u32 {
        // SAFETY: !Sync guarantees single-threaded access; UnsafeCell provides interior mutability.
        let idx = unsafe { &mut *self.tree.get() }.reserve_free_idx();
        assert!(
            idx < self.virtual_capacity,
            "HiSlab: virtual_capacity épuisée"
        );
        unsafe { std::ptr::write(self.data.add(idx as usize), val) };
        self.len.set(self.len.get() + 1);
        if idx >= self.touched_end.get() {
            self.touched_end.set(idx + 1);
        }
        idx
    }

    /// Removes the element at the given index and returns it, or `None` if the slot is empty.
    ///
    /// The slot becomes available for future insertions.
    pub fn remove(&self, idx: u32) -> Option<T> {
        if !self.is_occupied(idx) {
            return None;
        }
        // SAFETY: !Sync guarantees single-threaded access.
        unsafe { &mut *self.tree.get() }.clear_bit(idx);
        self.len.set(self.len.get() - 1);
        Some(unsafe { std::ptr::read(self.data.add(idx as usize)) })
    }

    /// Number of occupied slots.
    #[inline]
    pub fn len(&self) -> u32 {
        self.len.get()
    }

    /// `true` if no slot is occupied.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len.get() == 0
    }

    /// Highest occupied slot, or `None` if the slab is empty. Scans the level-1 bitmap
    /// downwards one 64-bit word at a time.
    pub fn last_occupied(&self) -> Option<u32> {
        let lvl1 = &unsafe { &*self.tree.get() }.lvl1;
        // SAFETY: see `copy_occupancy`, a `Vec<BitBlock>` is a contiguous run of u64.
        let words =
            unsafe { std::slice::from_raw_parts(lvl1.as_ptr() as *const u64, lvl1.len() * 8) };
        let w = words.iter().rposition(|&w| w != 0)?;
        Some((w * 64) as u32 + 63 - words[w].leading_zeros())
    }

    /// Compaction: moves the element of the occupied slot `src` into the lowest free slot,
    /// if that slot is below `src`, and returns it (`None`: no free slot below, nothing
    /// moved). The value is moved bitwise: the caller fixes whatever refers to `src`.
    ///
    /// # Safety
    /// `src` is occupied, and no reference into the slab is alive.
    pub unsafe fn move_down(&self, src: u32) -> Option<u32> {
        debug_assert!(self.is_occupied(src));
        let tree = unsafe { &mut *self.tree.get() };
        let dst = tree.find_free_idx();
        if dst >= src {
            return None;
        }
        tree.set_bit(dst);
        unsafe { std::ptr::copy_nonoverlapping(self.data.add(src as usize), self.data.add(dst as usize), 1) };
        tree.clear_bit(src);
        Some(dst)
    }

    /// Highest occupied slot strictly below `below`, or `None`.
    pub fn prev_occupied(&self, below: u32) -> Option<u32> {
        if below == 0 {
            return None;
        }
        let lvl1 = &unsafe { &*self.tree.get() }.lvl1;
        // SAFETY: see `copy_occupancy`.
        let words =
            unsafe { std::slice::from_raw_parts(lvl1.as_ptr() as *const u64, lvl1.len() * 8) };
        let last = (below - 1) as usize;
        let mut w = (last / 64).min(words.len().checked_sub(1)?);
        let mut word = if w == last / 64 { words[w] & (u64::MAX >> (63 - last % 64)) } else { words[w] };
        loop {
            if word != 0 {
                return Some((w * 64) as u32 + 63 - word.leading_zeros());
            }
            w = w.checked_sub(1)?;
            word = words[w];
        }
    }

    /// Compaction in a pipeline: reserves the lowest free slot without writing it (its
    /// bit is set, its content undefined until [`HiSlab::move_to`]).
    ///
    /// # Safety
    /// The slot is filled by `move_to` or given back by [`HiSlab::unreserve`] before any
    /// other use of the slab.
    pub unsafe fn reserve_lowest(&self) -> u32 {
        unsafe { &mut *self.tree.get() }.reserve_free_idx()
    }

    /// Gives back a slot reserved by [`HiSlab::reserve_lowest`] and never filled.
    ///
    /// # Safety
    /// `idx` was reserved and not filled.
    pub unsafe fn unreserve(&self, idx: u32) {
        unsafe { &mut *self.tree.get() }.clear_bit(idx);
    }

    /// Moves the element of the occupied slot `src` into the reserved slot `dst` (bitwise).
    ///
    /// # Safety
    /// `src` is occupied, `dst` reserved by [`HiSlab::reserve_lowest`], no reference into the
    /// slab is alive.
    pub unsafe fn move_to(&self, src: u32, dst: u32) {
        unsafe {
            std::ptr::copy_nonoverlapping(self.data.add(src as usize), self.data.add(dst as usize), 1);
            (*self.tree.get()).clear_bit(src);
        }
        if dst >= self.touched_end.get() {
            self.touched_end.set(dst + 1);
        }
    }

    /// Address of slot `idx`, for prefetching (no check, never dereferenced here).
    #[inline(always)]
    pub fn slot_ptr(&self, idx: u32) -> *const T {
        self.data.wrapping_add(idx as usize)
    }

    /// Returns to the system the memory of every whole 2 MiB page lying at or above slot
    /// `from` (`madvise(MADV_DONTNEED)`; the pages fault back in, zeroed, if slots there
    /// are used again). Returns the bytes released.
    ///
    /// # Safety
    /// Every slot at or above `from` is free.
    pub unsafe fn release_above(&self, from: u32) -> usize {
        const PAGE: usize = 2 << 20;
        debug_assert!(self.last_occupied().is_none_or(|l| l < from));
        let size = std::mem::size_of::<T>();
        let base = self.data as usize;
        let start = (base + from as usize * size).next_multiple_of(PAGE);
        let end = (base + self.touched_end.get() as usize * size).next_multiple_of(PAGE);
        if end <= start {
            return 0;
        }
        unsafe { libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_DONTNEED) };
        self.touched_end.set(((start - base) / size) as u32);
        end - start
    }

    /// Returns `true` if the slot at the given index is occupied.
    #[inline(always)]
    pub fn is_occupied(&self, idx: u32) -> bool {
        unsafe { &*self.tree.get() }.is_set(idx)
    }

    /// Returns a reference to the element at the given index, or `None` if empty.
    pub fn get(&self, idx: u32) -> Option<&T> {
        if self.is_occupied(idx) {
            unsafe { Some(self.data.add(idx as usize).as_ref()?) }
        } else {
            None
        }
    }

    /// Returns a mutable reference to the element at the given index, or `None` if empty.
    pub fn get_mut(&mut self, idx: u32) -> Option<&mut T> {
        if self.is_occupied(idx) {
            unsafe { Some(self.data.add(idx as usize).as_mut()?) }
        } else {
            None
        }
    }
}

impl<T> Index<u32> for HiSlab<T> {
    type Output = T;
    fn index(&self, idx: u32) -> &Self::Output {
        self.get(idx)
            .expect("Index out of bounds or element removed")
    }
}

impl<T> IndexMut<u32> for HiSlab<T> {
    fn index_mut(&mut self, idx: u32) -> &mut Self::Output {
        self.get_mut(idx)
            .expect("Index out of bounds or element removed")
    }
}

impl<T> HiSlab<T> {
    /// Returns a reference without checking if the slot is occupied.
    ///
    /// # Safety
    /// The caller must ensure the index is valid and occupied.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, idx: u32) -> &T {
        unsafe { self.data.add(idx as usize).as_ref().unwrap() }
    }

    /// Returns a mutable reference without checking if the slot is occupied.
    ///
    /// # Safety
    /// The caller must ensure the index is valid and occupied.
    #[inline(always)]
    pub unsafe fn get_unchecked_mut(&mut self, idx: u32) -> &mut T {
        unsafe { self.data.add(idx as usize).as_mut().unwrap() }
    }
}

impl<T> HiSlab<T> {
    /// Copies the level-1 occupancy bitmap into `out` (reused across calls):
    /// slot `i` is occupied ⇔ bit `i % 64` of word `i / 64`. A single memcpy.
    /// Useful as a point-in-time presence snapshot, e.g. at the start of an
    /// incremental save. Returns the number of slots covered (`out.len() * 64`).
    pub fn copy_occupancy(&self, out: &mut Vec<u64>) -> u32 {
        let lvl1 = &unsafe { &*self.tree.get() }.lvl1;
        // SAFETY: BitBlock = [u64; 8] en repr(align(64)), taille 64 :
        // un Vec<BitBlock> est une suite contiguë de u64.
        let words =
            unsafe { std::slice::from_raw_parts(lvl1.as_ptr() as *const u64, lvl1.len() * 8) };
        out.clear();
        out.extend_from_slice(words);
        (words.len() * 64) as u32
    }

    /// Number of slots covered by the occupancy bitmap: every occupied slot is
    /// below it. Grows with the slab, never shrinks.
    #[inline]
    pub fn slot_span(&self) -> u32 {
        (unsafe { &*self.tree.get() }.lvl1.len() << 9) as u32
    }

    /// Bytes of the slot range ever written and not handed back by
    /// [`HiSlab::release_above`]: what the slab's data can hold resident (memory
    /// accounting; the pages are committed by the kernel on first write, without any
    /// allocator call). The bitmaps live on the heap and are not included.
    #[inline]
    pub fn touched_bytes(&self) -> usize {
        self.touched_end.get() as usize * std::mem::size_of::<T>()
    }

    /// Bytes of the slab's data actually resident, measured with `mincore` over the
    /// touched range up to the next 2 MiB boundary (a transparent huge page taken on
    /// first write spills up to it). One syscall and one byte per 4 KiB: for statistics,
    /// not for hot paths. Falls back to [`HiSlab::touched_bytes`] if `mincore` fails.
    pub fn resident_bytes(&self) -> usize {
        const PAGE: usize = 4096;
        let touched = self.touched_bytes();
        if touched == 0 {
            return 0;
        }
        let base = self.data as usize;
        let end = (base + touched).next_multiple_of(2 << 20).min(base + self.mmap.len());
        let len = end.next_multiple_of(PAGE) - base;
        let mut vec = vec![0u8; len / PAGE];
        // SAFETY: `[base, base + len)` lies in the slab's mapping (page-aligned base).
        let r = unsafe { libc::mincore(base as *mut libc::c_void, len, vec.as_mut_ptr()) };
        if r != 0 {
            return touched;
        }
        vec.iter().filter(|&&v| v & 1 != 0).count() * PAGE
    }

    /// First occupied slot at or after `from`, or `None`. Scans the level-1 bitmap
    /// one 64-bit word at a time: resumable traversal by slices, e.g. a collector
    /// visiting a bounded number of slots per call.
    pub fn next_occupied(&self, from: u32) -> Option<u32> {
        let lvl1 = &unsafe { &*self.tree.get() }.lvl1;
        // SAFETY: see `copy_occupancy`, a `Vec<BitBlock>` is a contiguous run of u64.
        let words =
            unsafe { std::slice::from_raw_parts(lvl1.as_ptr() as *const u64, lvl1.len() * 8) };
        let mut w = from as usize / 64;
        let mut word = *words.get(w)? & (!0u64 << (from % 64));
        while word == 0 {
            w += 1;
            word = *words.get(w)?;
        }
        Some((w * 64) as u32 + word.trailing_zeros())
    }

    /// Iterates over all occupied slots with maximum performance.
    ///
    /// This is faster than using the iterator for simple operations
    /// as it avoids iterator overhead.
    pub fn for_each_occupied<F>(&self, mut f: F)
    where
        F: FnMut(u32, &T),
    {
        for (b_idx, block) in unsafe { &*self.tree.get() }.lvl1.iter().enumerate() {
            for (w_idx, &word) in block.data.iter().enumerate() {
                if word == 0 {
                    continue;
                }

                let mut temp_word = word;
                let base_idx = (b_idx << 9) | (w_idx << 6);

                while temp_word != 0 {
                    let bit = temp_word.trailing_zeros();
                    let final_idx = base_idx | (bit as usize);

                    unsafe {
                        f(final_idx as u32, self.data.add(final_idx).as_ref().unwrap());
                    }

                    temp_word &= temp_word - 1;
                }
            }
        }
    }
}

pub struct SlabIter<'a, T> {
    slab: *const T,
    lvl1: &'a [BitBlock],
    block_idx: usize,
    word_idx: usize,
    current_word: u64,
}

impl<'a, T> SlabIter<'a, T> {
    fn new(slab: &'a HiSlab<T>) -> Self {
        // SAFETY: HiSlab is !Sync; single-threaded access guaranteed.
        // The BitmapTree lives at least 'a since slab: &'a HiSlab<T>.
        let tree: &'a BitmapTree = unsafe { &*slab.tree.get() };
        let first_word = tree.lvl1.first().map(|b| b.data[0]).unwrap_or(0);
        Self {
            slab: slab.data,
            lvl1: &tree.lvl1,
            block_idx: 0,
            word_idx: 0,
            current_word: first_word,
        }
    }
}

impl<'a, T: 'static> Iterator for SlabIter<'a, T> {
    type Item = (u32, &'a T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.current_word == 0 {
            self.word_idx += 1;

            if self.word_idx >= 8 {
                self.word_idx = 0;
                self.block_idx += 1;
            }

            self.current_word = self.lvl1.get(self.block_idx)?.data[self.word_idx];
        }

        let bit = self.current_word.trailing_zeros();
        let final_idx = (self.block_idx << 9) | (self.word_idx << 6) | (bit as usize);

        self.current_word &= self.current_word - 1;

        unsafe { Some((final_idx as u32, self.slab.add(final_idx).as_ref().unwrap())) }
    }
}

impl<'a, T> IntoIterator for &'a HiSlab<T> {
    type Item = (u32, &'a T);
    type IntoIter = SlabIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        SlabIter::new(self)
    }
}

pub struct SlabIterMut<'a, T> {
    data_ptr: *mut T,
    lvl1: &'a [BitBlock],
    block_idx: usize,
    word_idx: usize,
    current_word: u64,
    _marker: std::marker::PhantomData<&'a mut T>,
}

impl<'a, T> SlabIterMut<'a, T> {
    fn new(slab: &'a mut HiSlab<T>) -> Self {
        // SAFETY: we hold &'a mut HiSlab<T>, exclusive access guaranteed.
        let tree: &'a BitmapTree = unsafe { &*slab.tree.get() };
        let first_word = tree.lvl1.first().map(|b| b.data[0]).unwrap_or(0);
        Self {
            data_ptr: slab.data,
            lvl1: &tree.lvl1,
            block_idx: 0,
            word_idx: 0,
            current_word: first_word,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a, T> Iterator for SlabIterMut<'a, T> {
    type Item = (u32, &'a mut T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.current_word == 0 {
            self.word_idx += 1;

            if self.word_idx >= 8 {
                self.word_idx = 0;
                self.block_idx += 1;
            }

            self.current_word = self.lvl1.get(self.block_idx)?.data[self.word_idx];
        }

        let bit = self.current_word.trailing_zeros();
        let final_idx = (self.block_idx << 9) | (self.word_idx << 6) | (bit as usize);
        self.current_word &= self.current_word - 1;

        unsafe { Some((final_idx as u32, &mut *self.data_ptr.add(final_idx))) }
    }
}

impl<'a, T> IntoIterator for &'a mut HiSlab<T> {
    type Item = (u32, &'a mut T);
    type IntoIter = SlabIterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        SlabIterMut::new(self)
    }
}

pub struct SlabIntoIter<T> {
    data: *mut T,
    lvl1: Vec<BitBlock>,
    block_idx: usize,
    word_idx: usize,
    current_word: u64,
    _mmap: MmapMut,
}

impl<T> Iterator for SlabIntoIter<T> {
    type Item = (u32, T);
    fn next(&mut self) -> Option<Self::Item> {
        while self.current_word == 0 {
            self.word_idx += 1;

            if self.word_idx >= 8 {
                self.word_idx = 0;
                self.block_idx += 1;
            }

            self.current_word = self.lvl1.get(self.block_idx)?.data[self.word_idx];
        }

        let bit = self.current_word.trailing_zeros();
        let final_idx = (self.block_idx << 9) | (self.word_idx << 6) | (bit as usize);

        self.current_word &= self.current_word - 1;

        let value = unsafe { std::ptr::read(self.data.add(final_idx)) };
        Some((final_idx as u32, value))
    }
}

impl<T> IntoIterator for HiSlab<T> {
    type Item = (u32, T);
    type IntoIter = SlabIntoIter<T>;

    fn into_iter(mut self) -> Self::IntoIter {
        let data = self.data;
        let tree = self.tree.get_mut();
        let lvl1 = std::mem::take(&mut tree.lvl1);
        let _lvl2 = std::mem::take(&mut tree.lvl2);
        let _lvl3 = std::mem::take(&mut tree.lvl3);
        let first_word = lvl1.first().map(|b| b.data[0]).unwrap_or(0);
        // Safety: ptr::read + forget is the standard pattern for moving out of a
        // Drop type field-by-field.
        let mmap = unsafe { std::ptr::read(&self.mmap) };
        std::mem::forget(self);

        SlabIntoIter {
            data,
            lvl1,
            block_idx: 0,
            word_idx: 0,
            current_word: first_word,
            _mmap: mmap,
        }
    }
}

impl<T> Drop for SlabIntoIter<T> {
    fn drop(&mut self) {
        for _ in self.by_ref() {}
    }
}

impl<T> Drop for HiSlab<T> {
    fn drop(&mut self) {
        for (b_idx, block) in self.tree.get_mut().lvl1.iter().enumerate() {
            for (w_idx, &word) in block.data.iter().enumerate() {
                if word == 0 {
                    continue;
                }
                let mut temp_word = word;
                let base_idx = (b_idx << 9) | (w_idx << 6);
                while temp_word != 0 {
                    let bit = temp_word.trailing_zeros();
                    let final_idx = base_idx | (bit as usize);
                    unsafe {
                        std::ptr::drop_in_place(self.data.add(final_idx));
                    }
                    temp_word &= temp_word - 1;
                }
            }
        }
    }
}

// ============================================================================
// Random selection (feature "rand") — HiSlab
// ============================================================================

#[cfg(feature = "rand")]
mod random {
    use super::{BitBlock, HiSlab};
    use rand::Rng;

    impl BitBlock {
        #[inline]
        pub fn popcnt(&self) -> u32 {
            self.data.iter().map(|w| w.count_ones()).sum()
        }
    }

    #[inline]
    fn select_nth_bit_u64(word: u64, n: u32) -> usize {
        #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
        {
            use std::arch::x86_64::_pdep_u64;
            unsafe {
                let mask = 1u64 << n;
                let deposited = _pdep_u64(mask, word);
                deposited.trailing_zeros() as usize
            }
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
        {
            let mut remaining = n;
            let mut w = word;
            while w != 0 {
                let bit_pos = w.trailing_zeros();
                if remaining == 0 {
                    return bit_pos as usize;
                }
                remaining -= 1;
                w &= w - 1;
            }
            unreachable!("n should be < popcnt(word)")
        }
    }

    impl<T> HiSlab<T> {
        /// Total number of occupied elements.
        #[inline]
        pub fn count_occupied(&self) -> usize {
            unsafe { &*self.tree.get() }
                .lvl1
                .iter()
                .map(|b| b.popcnt() as usize)
                .sum()
        }

        /// Picks a random occupied element.
        pub fn random_occupied<R: Rng>(&self, rng: &mut R) -> Option<(u32, &T)> {
            let tree = unsafe { &*self.tree.get() };
            let block_counts: Vec<u32> = tree.lvl1.iter().map(|b| b.popcnt()).collect();
            let total: u32 = block_counts.iter().sum();

            if total == 0 {
                return None;
            }

            let mut choice = rng.gen_range(0..total);

            let mut block_idx = 0;
            for (i, &cnt) in block_counts.iter().enumerate() {
                if choice < cnt {
                    block_idx = i;
                    break;
                }
                choice -= cnt;
            }

            let block = &tree.lvl1[block_idx];
            let mut word_choice = choice;

            let mut word_idx = 0;
            for (i, &word) in block.data.iter().enumerate() {
                let pop = word.count_ones();
                if word_choice < pop {
                    word_idx = i;
                    break;
                }
                word_choice -= pop;
            }

            let word = block.data[word_idx];
            let bit_pos = select_nth_bit_u64(word, word_choice);

            let final_idx = ((block_idx << 9) | (word_idx << 6) | bit_pos) as u32;

            unsafe { Some((final_idx, &*self.data.add(final_idx as usize))) }
        }

        /// Picks a random occupied element (mutable).
        pub fn random_occupied_mut<R: Rng>(&mut self, rng: &mut R) -> Option<(u32, &mut T)> {
            let tree = unsafe { &*self.tree.get() };
            let block_counts: Vec<u32> = tree.lvl1.iter().map(|b| b.popcnt()).collect();
            let total: u32 = block_counts.iter().sum();

            if total == 0 {
                return None;
            }

            let mut choice = rng.gen_range(0..total);

            let mut block_idx = 0;
            for (i, &cnt) in block_counts.iter().enumerate() {
                if choice < cnt {
                    block_idx = i;
                    break;
                }
                choice -= cnt;
            }

            let block = &tree.lvl1[block_idx];
            let mut word_choice = choice;

            let mut word_idx = 0;
            for (i, &word) in block.data.iter().enumerate() {
                let pop = word.count_ones();
                if word_choice < pop {
                    word_idx = i;
                    break;
                }
                word_choice -= pop;
            }

            let word = block.data[word_idx];
            let bit_pos = select_nth_bit_u64(word, word_choice);

            let final_idx = ((block_idx << 9) | (word_idx << 6) | bit_pos) as u32;

            unsafe { Some((final_idx, &mut *self.data.add(final_idx as usize))) }
        }

        /// Picks N random occupied elements (with replacement).
        pub fn random_occupied_many<R: Rng>(&self, rng: &mut R, count: usize) -> Vec<(u32, &T)> {
            let mut results = Vec::with_capacity(count);
            for _ in 0..count {
                if let Some(item) = self.random_occupied(rng) {
                    results.push(item);
                } else {
                    break;
                }
            }
            results
        }

        /// Picks N random occupied elements without replacement.
        pub fn random_occupied_unique<R: Rng>(&self, rng: &mut R, count: usize) -> Vec<(u32, &T)> {
            let total = self.count_occupied();
            if count >= total {
                return self.into_iter().collect();
            }

            use std::collections::HashSet;

            let mut selected_indices = HashSet::with_capacity(count);
            let mut results = Vec::with_capacity(count);

            let block_counts: Vec<u32> = unsafe { &*self.tree.get() }
                .lvl1
                .iter()
                .map(|b| b.popcnt())
                .collect();
            let total_u32 = total as u32;

            while results.len() < count {
                let choice = rng.gen_range(0..total_u32);

                let final_idx = self.choice_to_index(&block_counts, choice);

                if selected_indices.insert(final_idx) {
                    unsafe {
                        results.push((final_idx, &*self.data.add(final_idx as usize)));
                    }
                }
            }

            results
        }

        #[inline]
        fn choice_to_index(&self, block_counts: &[u32], mut choice: u32) -> u32 {
            let tree = unsafe { &*self.tree.get() };
            let mut block_idx = 0;
            for (i, &cnt) in block_counts.iter().enumerate() {
                if choice < cnt {
                    block_idx = i;
                    break;
                }
                choice -= cnt;
            }

            let block = &tree.lvl1[block_idx];
            let mut word_idx = 0;
            for (i, &word) in block.data.iter().enumerate() {
                let pop = word.count_ones();
                if choice < pop {
                    word_idx = i;
                    break;
                }
                choice -= pop;
            }

            let word = block.data[word_idx];
            let bit_pos = select_nth_bit_u64(word, choice);

            ((block_idx << 9) | (word_idx << 6) | bit_pos) as u32
        }
    }
}

// ============================================================================
// TaggedHiSlab
// ============================================================================

/// A slab allocator with tagging support, built on top of [`HiSlab`].
///
/// `TaggedHiSlab` wraps a [`HiSlab`] and maintains a second bitmap tree to track
/// "tagged" elements, enabling O(1) operations and efficient iteration over
/// tagged elements only.
///
/// All base slab operations are delegated to the inner [`HiSlab`].
pub struct TaggedHiSlab<T: 'static> {
    inner: HiSlab<T>,
    tagged_tree: UnsafeCell<BitmapTree>,
}

impl<T> TaggedHiSlab<T> {
    /// Creates a new empty `TaggedHiSlab`.
    ///
    /// See [`HiSlab::new`] for the meaning of `initial_capacity` and `virtual_capacity`.
    pub fn new(initial_capacity: u32, virtual_capacity: u32) -> Result<Self, std::io::Error> {
        Ok(Self {
            inner: HiSlab::new(initial_capacity, virtual_capacity)?,
            tagged_tree: UnsafeCell::new(BitmapTree::new()),
        })
    }
}

impl<T: 'static> TaggedHiSlab<T> {
    /// Inserts a value and returns its index. The element is not tagged.
    #[inline(always)]
    pub fn insert(&self, val: T) -> u32 {
        let idx = self.inner.insert(val);
        // Sécurité : effacer le tag si le slot réutilisé était taggé
        unsafe { &mut *self.tagged_tree.get() }.clear_bit(idx);
        idx
    }

    /// Removes the element at the given index and returns it, or `None` if the slot is empty.
    ///
    /// Also clears the tagged flag if set.
    pub fn remove(&self, idx: u32) -> Option<T> {
        let result = self.inner.remove(idx);
        if result.is_some() {
            unsafe { &mut *self.tagged_tree.get() }.clear_bit(idx);
        }
        result
    }

    /// Returns `true` if the slot at the given index is occupied.
    #[inline(always)]
    pub fn is_occupied(&self, idx: u32) -> bool {
        self.inner.is_occupied(idx)
    }

    /// Returns a reference to the element at the given index, or `None` if empty.
    pub fn get(&self, idx: u32) -> Option<&T> {
        self.inner.get(idx)
    }

    /// Returns a mutable reference to the element at the given index, or `None` if empty.
    pub fn get_mut(&mut self, idx: u32) -> Option<&mut T> {
        self.inner.get_mut(idx)
    }

    /// Returns a reference without checking if the slot is occupied.
    ///
    /// # Safety
    /// The caller must ensure the index is valid and occupied.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, idx: u32) -> &T {
        unsafe { self.inner.get_unchecked(idx) }
    }

    /// Returns a mutable reference without checking if the slot is occupied.
    ///
    /// # Safety
    /// The caller must ensure the index is valid and occupied.
    #[inline(always)]
    pub unsafe fn get_unchecked_mut(&mut self, idx: u32) -> &mut T {
        unsafe { self.inner.get_unchecked_mut(idx) }
    }

    /// See [`HiSlab::copy_occupancy`].
    pub fn copy_occupancy(&self, out: &mut Vec<u64>) -> u32 {
        self.inner.copy_occupancy(out)
    }

    /// See [`HiSlab::len`].
    #[inline]
    pub fn len(&self) -> u32 {
        self.inner.len()
    }

    /// See [`HiSlab::is_empty`].
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// See [`HiSlab::last_occupied`].
    pub fn last_occupied(&self) -> Option<u32> {
        self.inner.last_occupied()
    }

    /// See [`HiSlab::move_down`]. The tag follows the element.
    ///
    /// # Safety
    /// Same as [`HiSlab::move_down`].
    pub unsafe fn move_down(&self, src: u32) -> Option<u32> {
        let dst = unsafe { self.inner.move_down(src) }?;
        let tags = unsafe { &mut *self.tagged_tree.get() };
        if tags.is_set(src) {
            tags.clear_bit(src);
            tags.set_bit(dst);
        }
        Some(dst)
    }

    /// See [`HiSlab::prev_occupied`].
    pub fn prev_occupied(&self, below: u32) -> Option<u32> {
        self.inner.prev_occupied(below)
    }

    /// See [`HiSlab::reserve_lowest`].
    ///
    /// # Safety
    /// Same as [`HiSlab::reserve_lowest`].
    pub unsafe fn reserve_lowest(&self) -> u32 {
        unsafe { self.inner.reserve_lowest() }
    }

    /// See [`HiSlab::unreserve`].
    ///
    /// # Safety
    /// Same as [`HiSlab::unreserve`].
    pub unsafe fn unreserve(&self, idx: u32) {
        unsafe { self.inner.unreserve(idx) }
    }

    /// See [`HiSlab::move_to`]. The tag follows the element.
    ///
    /// # Safety
    /// Same as [`HiSlab::move_to`].
    pub unsafe fn move_to(&self, src: u32, dst: u32) {
        unsafe { self.inner.move_to(src, dst) };
        let tags = unsafe { &mut *self.tagged_tree.get() };
        if tags.is_set(src) {
            tags.clear_bit(src);
            tags.set_bit(dst);
        }
    }

    /// See [`HiSlab::slot_ptr`].
    #[inline(always)]
    pub fn slot_ptr(&self, idx: u32) -> *const T {
        self.inner.slot_ptr(idx)
    }

    /// See [`HiSlab::release_above`].
    ///
    /// # Safety
    /// Same as [`HiSlab::release_above`].
    pub unsafe fn release_above(&self, from: u32) -> usize {
        unsafe { self.inner.release_above(from) }
    }

    /// See [`HiSlab::slot_span`].
    #[inline]
    pub fn slot_span(&self) -> u32 {
        self.inner.slot_span()
    }

    /// See [`HiSlab::touched_bytes`].
    #[inline]
    pub fn touched_bytes(&self) -> usize {
        self.inner.touched_bytes()
    }

    /// See [`HiSlab::resident_bytes`].
    pub fn resident_bytes(&self) -> usize {
        self.inner.resident_bytes()
    }

    /// See [`HiSlab::next_occupied`].
    pub fn next_occupied(&self, from: u32) -> Option<u32> {
        self.inner.next_occupied(from)
    }

    /// Iterates over all occupied slots with maximum performance.
    pub fn for_each_occupied<F>(&self, f: F)
    where
        F: FnMut(u32, &T),
    {
        self.inner.for_each_occupied(f)
    }
}

impl<T: 'static> Index<u32> for TaggedHiSlab<T> {
    type Output = T;
    fn index(&self, idx: u32) -> &Self::Output {
        self.get(idx)
            .expect("Index out of bounds or element removed")
    }
}

impl<T: 'static> IndexMut<u32> for TaggedHiSlab<T> {
    fn index_mut(&mut self, idx: u32) -> &mut Self::Output {
        self.get_mut(idx)
            .expect("Index out of bounds or element removed")
    }
}

impl<'a, T: 'static> IntoIterator for &'a TaggedHiSlab<T> {
    type Item = (u32, &'a T);
    type IntoIter = SlabIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        SlabIter::new(&self.inner)
    }
}

impl<'a, T: 'static> IntoIterator for &'a mut TaggedHiSlab<T> {
    type Item = (u32, &'a mut T);
    type IntoIter = SlabIterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        SlabIterMut::new(&mut self.inner)
    }
}

impl<T: 'static> IntoIterator for TaggedHiSlab<T> {
    type Item = (u32, T);
    type IntoIter = SlabIntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        // tagged_tree contient uniquement des bitmaps (pas de données), il sera
        // droppé normalement. inner.into_iter() prend ownership des données.
        let TaggedHiSlab {
            inner,
            tagged_tree: _,
        } = self;
        inner.into_iter()
    }
}

// ============================================================================
// TaggedHiSlab — méthodes de tagging
// ============================================================================

impl<T: 'static> TaggedHiSlab<T> {
    /// Inserts a value with the tagged flag set and returns its index.
    #[inline(always)]
    pub fn insert_tagged(&self, val: T) -> u32 {
        let idx = self.inner.insert(val);
        unsafe { &mut *self.tagged_tree.get() }.set_bit(idx);
        idx
    }

    /// Tags an existing element at the given index.
    ///
    /// Returns `true` if the element was successfully tagged (it exists and wasn't already tagged).
    /// Returns `false` if the slot is empty or already tagged.
    #[inline]
    pub fn tag(&self, idx: u32) -> bool {
        if !self.inner.is_occupied(idx) {
            return false;
        }
        let tagged = unsafe { &mut *self.tagged_tree.get() };
        if tagged.is_set(idx) {
            return false;
        }
        tagged.set_bit(idx);
        true
    }

    /// Untags an element at the given index.
    ///
    /// Returns `true` if the element was successfully untagged.
    /// Returns `false` if the slot is empty or wasn't tagged.
    #[inline]
    pub fn untag(&self, idx: u32) -> bool {
        if !self.inner.is_occupied(idx) {
            return false;
        }
        let tagged = unsafe { &mut *self.tagged_tree.get() };
        if !tagged.is_set(idx) {
            return false;
        }
        tagged.clear_bit(idx);
        true
    }

    /// Returns `true` if the slot at the given index is tagged.
    #[inline(always)]
    pub fn is_tagged(&self, idx: u32) -> bool {
        unsafe { &*self.tagged_tree.get() }.is_set(idx)
    }

    /// Iterates over all tagged elements with maximum performance.
    pub fn for_each_tagged<F>(&self, mut f: F)
    where
        F: FnMut(u32, &T),
    {
        unsafe { &*self.tagged_tree.get() }.for_each_set(|idx| unsafe {
            f(idx, self.inner.get_unchecked(idx));
        });
    }

    /// Returns an iterator over all tagged elements.
    pub fn iter_tagged(&self) -> TaggedIter<'_, T> {
        TaggedIter::new(self)
    }

    /// Returns a mutable iterator over all tagged elements.
    pub fn iter_tagged_mut(&mut self) -> TaggedIterMut<'_, T> {
        TaggedIterMut::new(self)
    }

    /// Retains only the tagged elements for which the predicate returns `true`.
    ///
    /// Elements for which the predicate returns `false` are removed from the slab
    /// (both untagged and deleted). This is useful for TTL expiration:
    ///
    /// ```ignore
    /// slab.retain_tagged(|idx, entity| {
    ///     if entity.ttl_expired() {
    ///         false // Remove from slab
    ///     } else {
    ///         true // Keep
    ///     }
    /// });
    /// ```
    pub fn retain_tagged<F>(&mut self, mut f: F)
    where
        F: FnMut(u32, &mut T) -> bool,
    {
        let mut to_remove = Vec::new();

        for (b_idx, block) in self.tagged_tree.get_mut().lvl1.iter().enumerate() {
            for (w_idx, &word) in block.data.iter().enumerate() {
                if word == 0 {
                    continue;
                }

                let mut temp_word = word;
                let base_idx = (b_idx << 9) | (w_idx << 6);

                while temp_word != 0 {
                    let bit = temp_word.trailing_zeros();
                    let idx = (base_idx | (bit as usize)) as u32;

                    let val = unsafe { self.inner.get_unchecked_mut(idx) };
                    if !f(idx, val) {
                        to_remove.push(idx);
                    }

                    temp_word &= temp_word - 1;
                }
            }
        }

        for idx in to_remove {
            self.remove(idx);
        }
    }

    /// Similar to `retain_tagged`, but only untags elements instead of removing them.
    ///
    /// Elements for which the predicate returns `false` are untagged but stay in the slab.
    pub fn retain_tag<F>(&mut self, mut f: F)
    where
        F: FnMut(u32, &mut T) -> bool,
    {
        let mut to_untag = Vec::new();

        for (b_idx, block) in self.tagged_tree.get_mut().lvl1.iter().enumerate() {
            for (w_idx, &word) in block.data.iter().enumerate() {
                if word == 0 {
                    continue;
                }

                let mut temp_word = word;
                let base_idx = (b_idx << 9) | (w_idx << 6);

                while temp_word != 0 {
                    let bit = temp_word.trailing_zeros();
                    let idx = (base_idx | (bit as usize)) as u32;

                    let val = unsafe { self.inner.get_unchecked_mut(idx) };
                    if !f(idx, val) {
                        to_untag.push(idx);
                    }

                    temp_word &= temp_word - 1;
                }
            }
        }

        for idx in to_untag {
            unsafe { &mut *self.tagged_tree.get() }.clear_bit(idx);
        }
    }

    /// Counts the number of tagged elements.
    #[cfg(feature = "rand")]
    #[inline]
    pub fn count_tagged(&self) -> usize {
        unsafe { &*self.tagged_tree.get() }.count_set()
    }
}

// ============================================================================
// TaggedIter — itérateur immutable sur les éléments taggés
// ============================================================================

pub struct TaggedIter<'a, T> {
    data: *mut T,
    lvl1: &'a [BitBlock],
    block_idx: usize,
    word_idx: usize,
    current_word: u64,
}

impl<'a, T> TaggedIter<'a, T> {
    fn new(slab: &'a TaggedHiSlab<T>) -> Self {
        // SAFETY: TaggedHiSlab is !Sync; single-threaded access guaranteed.
        // The BitmapTree lives at least 'a since slab: &'a TaggedHiSlab<T>.
        let tree: &'a BitmapTree = unsafe { &*slab.tagged_tree.get() };
        let first_word = tree.lvl1.first().map(|b| b.data[0]).unwrap_or(0);
        Self {
            data: slab.inner.data,
            lvl1: &tree.lvl1,
            block_idx: 0,
            word_idx: 0,
            current_word: first_word,
        }
    }
}

impl<'a, T: 'static> Iterator for TaggedIter<'a, T> {
    type Item = (u32, &'a T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.current_word == 0 {
            self.word_idx += 1;

            if self.word_idx >= 8 {
                self.word_idx = 0;
                self.block_idx += 1;
            }

            self.current_word = self.lvl1.get(self.block_idx)?.data[self.word_idx];
        }

        let bit = self.current_word.trailing_zeros();
        let final_idx = (self.block_idx << 9) | (self.word_idx << 6) | (bit as usize);
        self.current_word &= self.current_word - 1;

        unsafe { Some((final_idx as u32, self.data.add(final_idx).as_ref().unwrap())) }
    }
}

// ============================================================================
// TaggedIterMut — itérateur mutable sur les éléments taggés
// ============================================================================

pub struct TaggedIterMut<'a, T> {
    data_ptr: *mut T,
    lvl1: &'a [BitBlock],
    block_idx: usize,
    word_idx: usize,
    current_word: u64,
    _marker: std::marker::PhantomData<&'a mut T>,
}

impl<'a, T> TaggedIterMut<'a, T> {
    fn new(slab: &'a mut TaggedHiSlab<T>) -> Self {
        // SAFETY: we hold &'a mut TaggedHiSlab<T>, exclusive access guaranteed.
        let tree: &'a BitmapTree = unsafe { &*slab.tagged_tree.get() };
        let first_word = tree.lvl1.first().map(|b| b.data[0]).unwrap_or(0);
        Self {
            data_ptr: slab.inner.data,
            lvl1: &tree.lvl1,
            block_idx: 0,
            word_idx: 0,
            current_word: first_word,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a, T> Iterator for TaggedIterMut<'a, T> {
    type Item = (u32, &'a mut T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.current_word == 0 {
            self.word_idx += 1;

            if self.word_idx >= 8 {
                self.word_idx = 0;
                self.block_idx += 1;
            }

            self.current_word = self.lvl1.get(self.block_idx)?.data[self.word_idx];
        }

        let bit = self.current_word.trailing_zeros();
        let final_idx = (self.block_idx << 9) | (self.word_idx << 6) | (bit as usize);
        self.current_word &= self.current_word - 1;

        unsafe { Some((final_idx as u32, &mut *self.data_ptr.add(final_idx))) }
    }
}

// ============================================================================
// Random selection (feature "rand") — TaggedHiSlab
// ============================================================================

#[cfg(feature = "rand")]
impl<T> TaggedHiSlab<T> {
    /// Total number of occupied elements.
    #[inline]
    pub fn count_occupied(&self) -> usize {
        self.inner.count_occupied()
    }

    /// Picks a random occupied element.
    pub fn random_occupied<R: rand::Rng>(&self, rng: &mut R) -> Option<(u32, &T)> {
        self.inner.random_occupied(rng)
    }

    /// Picks a random occupied element (mutable).
    pub fn random_occupied_mut<R: rand::Rng>(&mut self, rng: &mut R) -> Option<(u32, &mut T)> {
        self.inner.random_occupied_mut(rng)
    }

    /// Picks N random occupied elements (with replacement).
    pub fn random_occupied_many<R: rand::Rng>(&self, rng: &mut R, count: usize) -> Vec<(u32, &T)> {
        self.inner.random_occupied_many(rng, count)
    }

    /// Picks N random occupied elements without replacement.
    pub fn random_occupied_unique<R: rand::Rng>(
        &self,
        rng: &mut R,
        count: usize,
    ) -> Vec<(u32, &T)> {
        self.inner.random_occupied_unique(rng, count)
    }

    /// Selects a random tagged element.
    /// Returns None if no elements are tagged.
    pub fn random_tagged<R: rand::Rng>(&self, rng: &mut R) -> Option<(u32, &T)> {
        let idx = unsafe { &*self.tagged_tree.get() }.random_set(rng)?;
        unsafe { Some((idx, self.inner.get_unchecked(idx))) }
    }

    /// Selects a random tagged element (mutable version).
    /// Returns None if no elements are tagged.
    pub fn random_tagged_mut<R: rand::Rng>(&mut self, rng: &mut R) -> Option<(u32, &mut T)> {
        let idx = unsafe { &*self.tagged_tree.get() }.random_set(rng)?;
        unsafe { Some((idx, self.inner.get_unchecked_mut(idx))) }
    }

    /// Selects N random tagged elements (with possible duplicates).
    pub fn random_tagged_many<R: rand::Rng>(&self, rng: &mut R, count: usize) -> Vec<(u32, &T)> {
        let mut results = Vec::with_capacity(count);
        for _ in 0..count {
            if let Some(item) = self.random_tagged(rng) {
                results.push(item);
            } else {
                break;
            }
        }
        results
    }

    /// Selects N unique random tagged elements.
    pub fn random_tagged_unique<R: rand::Rng>(&self, rng: &mut R, count: usize) -> Vec<(u32, &T)> {
        use std::collections::HashSet;

        let total = self.count_tagged();
        if count >= total {
            let mut results = Vec::with_capacity(total);
            let mut indices = Vec::with_capacity(total);
            unsafe { &*self.tagged_tree.get() }.for_each_set(|idx| {
                indices.push(idx);
            });
            for idx in indices {
                unsafe {
                    results.push((idx, self.inner.get_unchecked(idx)));
                }
            }
            return results;
        }

        let mut selected = HashSet::with_capacity(count);
        let mut results = Vec::with_capacity(count);

        while results.len() < count {
            if let Some(idx) = unsafe { &*self.tagged_tree.get() }.random_set(rng) {
                if selected.insert(idx) {
                    unsafe {
                        results.push((idx, self.inner.get_unchecked(idx)));
                    }
                }
            } else {
                break;
            }
        }

        results
    }
}
