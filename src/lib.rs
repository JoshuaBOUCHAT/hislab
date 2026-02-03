//! # HiSlab
//!
//! A high-performance slab allocator using hierarchical bitmaps for O(1) operations.
//!
//! ## Example
//!
//! ```
//! use hislab::HiSlab;
//!
//! let mut slab = HiSlab::new();
//!
//! let idx = slab.insert(42);
//! assert_eq!(slab[idx], 42);
//!
//! let val = slab.remove(idx);
//! assert_eq!(val, Some(42));
//! assert!(slab.get(idx).is_none());
//! ```

use std::ops::{Index, IndexMut};

use crate::bit_block::BitBlock;

mod bit_block;
#[cfg(test)]
mod test;

/// A slab allocator with O(1) insert and remove using hierarchical bitmaps.
///
/// `HiSlab` stores elements in a contiguous `Vec` and tracks free slots using
/// a 4-level bitmap hierarchy. This allows finding a free slot in constant time
/// regardless of fragmentation.
pub struct HiSlab<T> {
    data: Vec<T>,
    lvl1: Vec<BitBlock>,
    lvl2: Vec<BitBlock>,
    lvl3: Vec<BitBlock>,
    lvl4: u8, //actualy we only need 5 bit as each level cover 9 bits 3*9 =27, 32-27 = 5
}
impl<T> Default for HiSlab<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> HiSlab<T> {
    /// Creates a new empty `HiSlab`.
    pub fn new() -> Self {
        Self {
            data: Vec::default(),
            lvl1: vec![BitBlock::default()],
            lvl2: vec![BitBlock::default()],
            lvl3: vec![BitBlock::default()],
            lvl4: 0,
        }
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
        if let Some(bit_idx) = self.lvl1[0].find_first_free() {
            return self.finalize_insert(0, bit_idx, val);
        }

        // --- SLOW PATH ---
        let l1_block_idx = self.find_free_block_via_hierarchy();

        // Sécurité : on s'assure que le Vec lvl1 a assez de BitBlocks
        if l1_block_idx >= self.lvl1.len() {
            self.lvl1.resize_with(l1_block_idx + 1, BitBlock::default);
        }

        let bit_idx = self.lvl1[l1_block_idx]
            .find_first_free()
            .expect("Hierarchy out of sync");

        self.finalize_insert(l1_block_idx, bit_idx, val)
    }

    fn find_free_block_via_hierarchy(&mut self) -> usize {
        // 1. On commence par le sommet (lvl4)
        if self.lvl4 == 0xFF {
            // Ici, il faudrait agrandir lvl4 ou ajouter un lvl5.
            // Pour 2^18, on n'atteindra jamais ce cas.
            panic!("Slab overflow");
        }

        // Trouver quel BitBlock de lvl3 a de la place
        let l3_block_idx = (!self.lvl4).trailing_zeros() as usize;
        self.ensure_lvl3(l3_block_idx);

        // 2. Trouver quel BitBlock de lvl2 a de la place
        let l2_block_relative_idx = self.lvl3[l3_block_idx].find_first_free().unwrap();
        let l2_block_idx = (l3_block_idx * 512) + l2_block_relative_idx;
        self.ensure_lvl2(l2_block_idx);

        // 3. Trouver quel BitBlock de lvl1 a de la place
        let l1_block_relative_idx = self.lvl2[l2_block_idx].find_first_free().unwrap();
        let l1_block_idx = (l2_block_idx * 512) + l1_block_relative_idx;

        // Le resize de lvl1 est déjà géré dans ton `insert` avant l'appel à find_first_free
        l1_block_idx
    }

    // Fonctions utilitaires pour agrandir les couches de métadonnées
    fn ensure_lvl3(&mut self, idx: usize) {
        if idx >= self.lvl3.len() {
            self.lvl3.resize_with(idx + 1, BitBlock::default);
        }
    }

    fn ensure_lvl2(&mut self, idx: usize) {
        if idx >= self.lvl2.len() {
            self.lvl2.resize_with(idx + 1, BitBlock::default);
        }
    }

    #[inline(always)]
    fn finalize_insert(&mut self, block_idx: usize, bit_idx: usize, val: T) -> u32 {
        let final_idx = (block_idx * 512 + bit_idx) as u32;

        // Écriture de la donnée (ptr::write pour éviter de drop l'ancienne valeur non-initialisée)
        if (final_idx as usize) < self.data.len() {
            unsafe {
                std::ptr::write(self.data.as_mut_ptr().add(final_idx as usize), val);
            }
        } else {
            self.data.push(val);
        }

        // Propagation de l'occupation
        if self.lvl1[block_idx].set_bit_and_check_full(bit_idx) {
            self.propagate_full(block_idx);
        }

        final_idx
    }
    fn propagate_full(&mut self, l1_block_idx: usize) {
        // 1. Le bloc de niveau 1 à l'index `l1_block_idx` est plein.
        // Il correspond à un bit précis dans le niveau 2.
        // Puisque lvl2 est un Vec<BitBlock>, on cherche l'index du bloc et du bit.
        let l2_block_idx = l1_block_idx / 512;
        let l2_bit_idx = l1_block_idx % 512;

        // On met à jour le lvl2 et on regarde s'il devient plein à son tour
        if self.lvl2[l2_block_idx].set_bit_and_check_full(l2_bit_idx) {
            // 2. Si le bloc de lvl2 est plein, on propage au lvl3
            let l3_block_idx = l2_block_idx / 512;
            let l3_bit_idx = l2_block_idx % 512;

            if self.lvl3[l3_block_idx].set_bit_and_check_full(l3_bit_idx) {
                // 3. Enfin, on remonte au lvl4 (ton u8 / u64 de résumé final)
                // lvl4 est le résumé de tous les blocs du lvl3
                self.lvl4 |= 1 << l3_block_idx;
            }
        }
    }

    /// Removes the element at the given index and returns it, or `None` if the slot is empty.
    ///
    /// The slot becomes available for future insertions.
    pub fn remove(&mut self, idx: u32) -> Option<T> {
        if !self.is_occupied(idx) {
            return None;
        }

        let idx_usize = idx as usize;
        let l1_block_idx = idx_usize / 512;
        let l1_bit_idx = idx_usize % 512;

        // On libère le bit. Si le bloc était plein, on doit prévenir le parent.
        if self.lvl1[l1_block_idx].clear_bit_and_was_full(l1_bit_idx) {
            self.propagate_empty(l1_block_idx);
        }

        // Extraction de la valeur sans bouger les autres éléments
        Some(unsafe { std::ptr::read(self.data.as_ptr().add(idx_usize)) })
    }

    fn propagate_empty(&mut self, l1_block_idx: usize) {
        // 1. On remonte au Niveau 2
        let l2_block_idx = l1_block_idx / 512;
        let l2_bit_idx = l1_block_idx % 512;

        if self.lvl2[l2_block_idx].clear_bit_and_was_full(l2_bit_idx) {
            // 2. On remonte au Niveau 3
            let l3_block_idx = l2_block_idx / 512;
            let l3_bit_idx = l2_block_idx % 512;

            if self.lvl3[l3_block_idx].clear_bit_and_was_full(l3_bit_idx) {
                // 3. Enfin le niveau 4 (résumé final)
                self.lvl4 &= !(1 << l3_block_idx);
            }
        }
    }
    /// Returns `true` if the slot at the given index is occupied.
    #[inline(always)]
    pub fn is_occupied(&self, idx: u32) -> bool {
        let idx = idx as usize;
        let b_idx = idx >> 9; // / 512
        if b_idx >= self.lvl1.len() {
            return false;
        }
        let inner = (idx >> 6) & 0x7; // (idx % 512) / 64
        let bit = idx & 63; // idx % 64

        unsafe { (self.lvl1.get_unchecked(b_idx).data.get_unchecked(inner) & (1 << bit)) != 0 }
    }

    /// Returns a reference to the element at the given index, or `None` if empty.
    pub fn get(&self, idx: u32) -> Option<&T> {
        if self.is_occupied(idx) {
            // Ici on sait que c'est safe d'accéder à data
            Some(&self.data[idx as usize])
        } else {
            None
        }
    }

    /// Returns a mutable reference to the element at the given index, or `None` if empty.
    pub fn get_mut(&mut self, idx: u32) -> Option<&mut T> {
        if self.is_occupied(idx) {
            Some(&mut self.data[idx as usize])
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
        unsafe { self.data.get_unchecked(idx as usize) }
    }

    /// Returns a mutable reference without checking if the slot is occupied.
    ///
    /// # Safety
    /// The caller must ensure the index is valid and occupied.
    #[inline(always)]
    pub unsafe fn get_unchecked_mut(&mut self, idx: u32) -> &mut T {
        unsafe { self.data.get_unchecked_mut(idx as usize) }
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
        for (b_idx, block) in self.lvl1.iter().enumerate() {
            for (w_idx, &word) in block.data.iter().enumerate() {
                // OPTIMISATION : Si le mot est vide, on saute 64 éléments d'un coup
                if word == 0 {
                    continue;
                }

                let mut temp_word = word;
                let base_idx = (b_idx << 9) | (w_idx << 6); // (b * 512) + (w * 64)

                while temp_word != 0 {
                    // Trouver le prochain bit à 1
                    let bit = temp_word.trailing_zeros();
                    let final_idx = base_idx | (bit as usize);

                    unsafe {
                        // On sait que l'index est valide car le bit est à 1
                        f(final_idx as u32, self.data.get_unchecked(final_idx));
                    }

                    // On efface le bit le plus bas pour passer au suivant
                    // Utilise l'instruction BLSR sur x86
                    temp_word &= temp_word - 1;
                }
            }
        }
    }
}
pub struct SlabIter<'a, T> {
    slab: &'a Vec<T>,
    lvl1: &'a [BitBlock],
    block_idx: usize,
    word_idx: usize,
    current_word: u64,
}

impl<'a, T> SlabIter<'a, T> {
    fn new(slab: &'a HiSlab<T>) -> Self {
        let first_word = slab.lvl1.get(0).map(|b| b.data[0]).unwrap_or(0);
        Self {
            slab: &slab.data,
            lvl1: &slab.lvl1,
            block_idx: 0,
            word_idx: 0,
            current_word: first_word,
        }
    }
}

impl<'a, T> Iterator for SlabIter<'a, T> {
    type Item = (u32, &'a T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // 1. Si le mot actuel est vide, on cherche le prochain mot non vide
        while self.current_word == 0 {
            self.word_idx += 1;

            // Si on a fini les 8 mots du bloc, on passe au bloc suivant
            if self.word_idx >= 8 {
                self.word_idx = 0;
                self.block_idx += 1;
            }

            // Si on a fini tous les blocs, on s'arrête
            if let Some(block) = self.lvl1.get(self.block_idx) {
                self.current_word = block.data[self.word_idx];
            } else {
                return None;
            }
        }

        // 2. Extraire le prochain index du mot actuel
        let bit = self.current_word.trailing_zeros();
        let final_idx = (self.block_idx << 9) | (self.word_idx << 6) | (bit as usize);

        // On "éteint" le bit trouvé pour le prochain appel
        self.current_word &= self.current_word - 1;

        unsafe { Some((final_idx as u32, self.slab.get_unchecked(final_idx))) }
    }
}
impl<'a, T> IntoIterator for &'a HiSlab<T> {
    type Item = (u32, &'a T);
    type IntoIter = SlabIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        SlabIter::new(self)
    }
}

// Version Mutable
pub struct SlabIterMut<'a, T> {
    data_ptr: *mut T, // On utilise un pointeur car on ne peut pas stocker &mut Vec
    lvl1: &'a [BitBlock],
    block_idx: usize,
    word_idx: usize,
    current_word: u64,
    _marker: std::marker::PhantomData<&'a mut T>,
}

impl<'a, T> Iterator for SlabIterMut<'a, T> {
    type Item = (u32, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        // ... (Même logique de recherche de bit que SlabIter) ...
        let bit = self.current_word.trailing_zeros();
        let final_idx = (self.block_idx << 9) | (self.word_idx << 6) | (bit as usize);
        self.current_word &= self.current_word - 1;

        unsafe {
            // Transformation de l'index en référence mutable via le pointeur de base
            Some((final_idx as u32, &mut *self.data_ptr.add(final_idx)))
        }
    }
}
pub struct SlabIntoIter<T> {
    data: Vec<T>,
    lvl1: Vec<BitBlock>,
    block_idx: usize,
    word_idx: usize,
    current_word: u64,
}

impl<T> Iterator for SlabIntoIter<T> {
    type Item = (u32, T);
    fn next(&mut self) -> Option<Self::Item> {
        // Si le mot actuel est vide, on cherche le prochain mot non vide
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

        // On utilise std::ptr::read pour extraire la valeur par move
        let value = unsafe { std::ptr::read(self.data.as_ptr().add(final_idx)) };
        Some((final_idx as u32, value))
    }
}
impl<T> IntoIterator for HiSlab<T> {
    type Item = (u32, T);
    type IntoIter = SlabIntoIter<T>;

    fn into_iter(mut self) -> Self::IntoIter {
        let first_word = self.lvl1.get(0).map(|b| b.data[0]).unwrap_or(0);

        // Take ownership des champs avant que Drop ne run
        let data = std::mem::take(&mut self.data);
        let lvl1 = std::mem::take(&mut self.lvl1);

        // Empêcher Drop de run (les champs sont maintenant vides)
        std::mem::forget(self);

        SlabIntoIter {
            data,
            lvl1,
            block_idx: 0,
            word_idx: 0,
            current_word: first_word,
        }
    }
}

impl<T> Drop for HiSlab<T> {
    fn drop(&mut self) {
        // Drop seulement les éléments occupés
        for (b_idx, block) in self.lvl1.iter().enumerate() {
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
                        std::ptr::drop_in_place(self.data.as_mut_ptr().add(final_idx));
                    }
                    temp_word &= temp_word - 1;
                }
            }
        }
        // Éviter que Vec::drop ne redrop les éléments
        unsafe {
            self.data.set_len(0);
        }
    }
}
