//! # HiSlab
//!
//! A high-performance slab allocator using hierarchical bitmaps for O(1) operations.
//!
//! ## Example
//!
//! ```
//! use hislab::HiSlab;
//!
//! let mut slab = HiSlab::<i32>::new().unwrap();
//!
//! let idx = slab.insert(42);
//! assert_eq!(slab[idx], 42);
//!
//! let val = slab.remove(idx);
//! assert_eq!(val, Some(42));
//! assert!(slab.get(idx).is_none());
//! ```

use std::ops::{Index, IndexMut};

use memfd::MemfdOptions;
use memmap2::MmapMut;

use crate::bit_block::BitBlock;
use crate::bitmap_tree::BitmapTree;

mod bit_block;
mod bitmap_tree;
#[cfg(test)]
mod test;

// ============================================================================
// HiSlab
// ============================================================================

/// A slab allocator with O(1) insert and remove using hierarchical bitmaps.
///
/// `HiSlab` stores elements in a contiguous `Vec` and tracks free slots using
/// a 4-level bitmap hierarchy. This allows finding a free slot in constant time
/// regardless of fragmentation.
///
/// For tagging support, see [`TaggedHiSlab`].
pub struct HiSlab<T: 'static> {
    data: *mut T,
    pub(crate) tree: BitmapTree,
    mmap: MmapMut,
}

const SLAB_SIZE: u64 = 1024 * 1024 * 1024; // 1 GiB

pub fn alloc_huge_ref() -> Result<MmapMut, Box<dyn std::error::Error>> {
    // Try 2 MiB hugepages first (create + truncate + mmap must all succeed).
    // Any failure at any step falls back to a regular anonymous mapping.
    let hugetlb = (|| -> Result<MmapMut, Box<dyn std::error::Error>> {
        let fd = MemfdOptions::new()
            .hugetlb(Some(memfd::HugetlbSize::Huge2MB))
            .allow_sealing(true)
            .create("hislab-huge-ptr")?;
        fd.as_file().set_len(SLAB_SIZE)?;
        Ok(unsafe { MmapMut::map_mut(fd.as_file())? })
    })();

    match hugetlb {
        Ok(mmap) => Ok(mmap),
        Err(_) => Ok(MmapMut::map_anon(SLAB_SIZE as usize)?),
    }
}

impl<T: Default> HiSlab<T> {
    /// Creates a new empty `HiSlab`.
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let mmap = alloc_huge_ref()?;
        // 4. Initialisation (Crucial pour éviter l'UB)
        // On écrit la valeur par défaut à l'adresse de départ
        let ptr = mmap.as_ptr() as *mut T;
        unsafe {
            std::ptr::write(ptr, T::default());
        }

        Ok(Self {
            data: ptr,
            tree: BitmapTree::new(),
            mmap: mmap,
        })
    }
}

impl<T> HiSlab<T> {
    /// Inserts a value and returns its index.
    ///
    /// The returned index is stable and can be used to access the value
    /// until it is removed.
    #[inline(always)]
    pub fn insert(&mut self, val: T) -> u32 {
        // --- FAST PATH (0..512) ---
        if let Some(bit_idx) = self.tree.lvl1[0].find_first_free() {
            return self.finalize_insert(0, bit_idx, val);
        }

        // --- SLOW PATH ---
        let l1_block_idx = self.tree.find_free_block();

        self.tree.ensure_lvl1(l1_block_idx);

        let bit_idx = self.tree.lvl1[l1_block_idx]
            .find_first_free()
            .expect("Hierarchy out of sync");

        self.finalize_insert(l1_block_idx, bit_idx, val)
    }

    #[inline(always)]
    fn finalize_insert(&mut self, block_idx: usize, bit_idx: usize, val: T) -> u32 {
        let final_idx = (block_idx * 512 + bit_idx) as u32;

        unsafe {
            std::ptr::write(self.data.add(final_idx as usize), val);
        }

        self.tree.set_bit(final_idx);

        final_idx
    }

    /// Removes the element at the given index and returns it, or `None` if the slot is empty.
    ///
    /// The slot becomes available for future insertions.
    pub fn remove(&mut self, idx: u32) -> Option<T> {
        if !self.is_occupied(idx) {
            return None;
        }

        self.tree.clear_bit(idx);

        Some(unsafe { std::ptr::read(self.data.add(idx as usize)) })
    }

    /// Returns `true` if the slot at the given index is occupied.
    #[inline(always)]
    pub fn is_occupied(&self, idx: u32) -> bool {
        self.tree.is_set(idx)
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
    /// Iterates over all occupied slots with maximum performance.
    ///
    /// This is faster than using the iterator for simple operations
    /// as it avoids iterator overhead.
    pub fn for_each_occupied<F>(&self, mut f: F)
    where
        F: FnMut(u32, &T),
    {
        for (b_idx, block) in self.tree.lvl1.iter().enumerate() {
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
        let first_word = slab.tree.lvl1.first().map(|b| b.data[0]).unwrap_or(0);
        Self {
            slab: slab.data,
            lvl1: &slab.tree.lvl1,
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

            if let Some(block) = self.lvl1.get(self.block_idx) {
                self.current_word = block.data[self.word_idx];
            } else {
                return None;
            }
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
        let first_word = slab.tree.lvl1.first().map(|b| b.data[0]).unwrap_or(0);
        Self {
            data_ptr: slab.data,
            lvl1: &slab.tree.lvl1,
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

            if let Some(block) = self.lvl1.get(self.block_idx) {
                self.current_word = block.data[self.word_idx];
            } else {
                return None;
            }
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

            if let Some(block) = self.lvl1.get(self.block_idx) {
                self.current_word = block.data[self.word_idx];
            } else {
                return None;
            }
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
        let lvl1 = std::mem::take(&mut self.tree.lvl1);
        let _lvl2 = std::mem::take(&mut self.tree.lvl2);
        let _lvl3 = std::mem::take(&mut self.tree.lvl3);
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
        for (b_idx, block) in self.tree.lvl1.iter().enumerate() {
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
        /// Compte le nombre total d'éléments occupés
        #[inline]
        pub fn count_occupied(&self) -> usize {
            self.tree.lvl1.iter().map(|b| b.popcnt() as usize).sum()
        }

        /// Sélectionne un élément occupé aléatoirement.
        pub fn random_occupied<R: Rng>(&self, rng: &mut R) -> Option<(u32, &T)> {
            let block_counts: Vec<u32> = self.tree.lvl1.iter().map(|b| b.popcnt()).collect();
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

            let block = &self.tree.lvl1[block_idx];
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

        /// Sélectionne un élément occupé aléatoirement (version mutable).
        pub fn random_occupied_mut<R: Rng>(&mut self, rng: &mut R) -> Option<(u32, &mut T)> {
            let block_counts: Vec<u32> = self.tree.lvl1.iter().map(|b| b.popcnt()).collect();
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

            let block = &self.tree.lvl1[block_idx];
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

        /// Sélectionne N éléments occupés aléatoirement (avec remise possible).
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

        /// Sélectionne N éléments occupés aléatoirement SANS remise.
        pub fn random_occupied_unique<R: Rng>(&self, rng: &mut R, count: usize) -> Vec<(u32, &T)> {
            let total = self.count_occupied();
            if count >= total {
                return self.into_iter().collect();
            }

            use std::collections::HashSet;

            let mut selected_indices = HashSet::with_capacity(count);
            let mut results = Vec::with_capacity(count);

            let block_counts: Vec<u32> = self.tree.lvl1.iter().map(|b| b.popcnt()).collect();
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
            let mut block_idx = 0;
            for (i, &cnt) in block_counts.iter().enumerate() {
                if choice < cnt {
                    block_idx = i;
                    break;
                }
                choice -= cnt;
            }

            let block = &self.tree.lvl1[block_idx];
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
    tagged_tree: BitmapTree,
}

impl<T: Default> TaggedHiSlab<T> {
    /// Creates a new empty `TaggedHiSlab`.
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            inner: HiSlab::new()?,
            tagged_tree: BitmapTree::new(),
        })
    }
}

impl<T: 'static> TaggedHiSlab<T> {
    /// Inserts a value and returns its index. The element is not tagged.
    #[inline(always)]
    pub fn insert(&mut self, val: T) -> u32 {
        let idx = self.inner.insert(val);
        // Sécurité : effacer le tag si le slot réutilisé était taggé
        self.tagged_tree.clear_bit(idx);
        idx
    }

    /// Removes the element at the given index and returns it, or `None` if the slot is empty.
    ///
    /// Also clears the tagged flag if set.
    pub fn remove(&mut self, idx: u32) -> Option<T> {
        let result = self.inner.remove(idx);
        if result.is_some() {
            self.tagged_tree.clear_bit(idx);
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
    pub fn insert_tagged(&mut self, val: T) -> u32 {
        let idx = self.inner.insert(val);
        self.tagged_tree.set_bit(idx);
        idx
    }

    /// Tags an existing element at the given index.
    ///
    /// Returns `true` if the element was successfully tagged (it exists and wasn't already tagged).
    /// Returns `false` if the slot is empty or already tagged.
    #[inline]
    pub fn tag(&mut self, idx: u32) -> bool {
        if !self.inner.is_occupied(idx) {
            return false;
        }
        if self.tagged_tree.is_set(idx) {
            return false;
        }
        self.tagged_tree.set_bit(idx);
        true
    }

    /// Untags an element at the given index.
    ///
    /// Returns `true` if the element was successfully untagged.
    /// Returns `false` if the slot is empty or wasn't tagged.
    #[inline]
    pub fn untag(&mut self, idx: u32) -> bool {
        if !self.inner.is_occupied(idx) {
            return false;
        }
        if !self.tagged_tree.is_set(idx) {
            return false;
        }
        self.tagged_tree.clear_bit(idx);
        true
    }

    /// Returns `true` if the slot at the given index is tagged.
    #[inline(always)]
    pub fn is_tagged(&self, idx: u32) -> bool {
        self.tagged_tree.is_set(idx)
    }

    /// Iterates over all tagged elements with maximum performance.
    pub fn for_each_tagged<F>(&self, mut f: F)
    where
        F: FnMut(u32, &T),
    {
        self.tagged_tree.for_each_set(|idx| unsafe {
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

        for (b_idx, block) in self.tagged_tree.lvl1.iter().enumerate() {
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

        for (b_idx, block) in self.tagged_tree.lvl1.iter().enumerate() {
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
            self.tagged_tree.clear_bit(idx);
        }
    }

    /// Counts the number of tagged elements.
    #[cfg(feature = "rand")]
    #[inline]
    pub fn count_tagged(&self) -> usize {
        self.tagged_tree.count_set()
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
        let first_word = slab
            .tagged_tree
            .lvl1
            .first()
            .map(|b| b.data[0])
            .unwrap_or(0);
        Self {
            data: slab.inner.data,
            lvl1: &slab.tagged_tree.lvl1,
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

            if let Some(block) = self.lvl1.get(self.block_idx) {
                self.current_word = block.data[self.word_idx];
            } else {
                return None;
            }
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
        let first_word = slab
            .tagged_tree
            .lvl1
            .first()
            .map(|b| b.data[0])
            .unwrap_or(0);
        Self {
            data_ptr: slab.inner.data,
            lvl1: &slab.tagged_tree.lvl1,
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

            if let Some(block) = self.lvl1.get(self.block_idx) {
                self.current_word = block.data[self.word_idx];
            } else {
                return None;
            }
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
    /// Compte le nombre total d'éléments occupés.
    #[inline]
    pub fn count_occupied(&self) -> usize {
        self.inner.count_occupied()
    }

    /// Sélectionne un élément occupé aléatoirement.
    pub fn random_occupied<R: rand::Rng>(&self, rng: &mut R) -> Option<(u32, &T)> {
        self.inner.random_occupied(rng)
    }

    /// Sélectionne un élément occupé aléatoirement (version mutable).
    pub fn random_occupied_mut<R: rand::Rng>(&mut self, rng: &mut R) -> Option<(u32, &mut T)> {
        self.inner.random_occupied_mut(rng)
    }

    /// Sélectionne N éléments occupés aléatoirement (avec remise possible).
    pub fn random_occupied_many<R: rand::Rng>(&self, rng: &mut R, count: usize) -> Vec<(u32, &T)> {
        self.inner.random_occupied_many(rng, count)
    }

    /// Sélectionne N éléments occupés aléatoirement SANS remise.
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
        let idx = self.tagged_tree.random_set(rng)?;
        unsafe { Some((idx, self.inner.get_unchecked(idx))) }
    }

    /// Selects a random tagged element (mutable version).
    /// Returns None if no elements are tagged.
    pub fn random_tagged_mut<R: rand::Rng>(&mut self, rng: &mut R) -> Option<(u32, &mut T)> {
        let idx = self.tagged_tree.random_set(rng)?;
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
            self.tagged_tree.for_each_set(|idx| {
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
            if let Some(idx) = self.tagged_tree.random_set(rng) {
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
