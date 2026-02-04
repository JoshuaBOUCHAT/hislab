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
use crate::bitmap_tree::BitmapTree;

mod bit_block;
mod bitmap_tree;
#[cfg(test)]
mod test;

/// A slab allocator with O(1) insert and remove using hierarchical bitmaps.
///
/// `HiSlab` stores elements in a contiguous `Vec` and tracks free slots using
/// a 4-level bitmap hierarchy. This allows finding a free slot in constant time
/// regardless of fragmentation.
///
/// When the `tagged` feature is enabled, a second bitmap tree tracks "tagged"
/// elements, allowing O(1) random selection among tagged elements only.
pub struct HiSlab<T> {
    data: Vec<T>,
    pub(crate) tree: BitmapTree,
    #[cfg(feature = "tagged")]
    pub(crate) tagged_tree: BitmapTree,
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
            tree: BitmapTree::new(),
            #[cfg(feature = "tagged")]
            tagged_tree: BitmapTree::new(),
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
        if let Some(bit_idx) = self.tree.lvl1[0].find_first_free() {
            return self.finalize_insert(0, bit_idx, val);
        }

        // --- SLOW PATH ---
        let l1_block_idx = self.tree.find_free_block();

        // Sécurité : on s'assure que le Vec lvl1 a assez de BitBlocks
        self.tree.ensure_lvl1(l1_block_idx);

        let bit_idx = self.tree.lvl1[l1_block_idx]
            .find_first_free()
            .expect("Hierarchy out of sync");

        self.finalize_insert(l1_block_idx, bit_idx, val)
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

        // Propagation de l'occupation via BitmapTree
        self.tree.set_bit(final_idx);

        // insert normal = pas de tag (clear si jamais c'était taggé avant - normalement non car remove clear)
        #[cfg(feature = "tagged")]
        self.tagged_tree.clear_bit(final_idx);

        final_idx
    }

    /// Removes the element at the given index and returns it, or `None` if the slot is empty.
    ///
    /// The slot becomes available for future insertions.
    #[cfg(not(feature = "tagged"))]
    pub fn remove(&mut self, idx: u32) -> Option<T> {
        if !self.is_occupied(idx) {
            return None;
        }

        self.tree.clear_bit(idx);

        // Extraction de la valeur sans bouger les autres éléments
        Some(unsafe { std::ptr::read(self.data.as_ptr().add(idx as usize)) })
    }

    /// Removes the element at the given index and returns it, or `None` if the slot is empty.
    ///
    /// The slot becomes available for future insertions.
    /// Also clears the tagged flag if set.
    #[cfg(feature = "tagged")]
    pub fn remove(&mut self, idx: u32) -> Option<T> {
        if !self.is_occupied(idx) {
            return None;
        }

        self.tree.clear_bit(idx);
        self.tagged_tree.clear_bit(idx);

        // Extraction de la valeur sans bouger les autres éléments
        Some(unsafe { std::ptr::read(self.data.as_ptr().add(idx as usize)) })
    }

    /// Returns `true` if the slot at the given index is occupied.
    #[inline(always)]
    pub fn is_occupied(&self, idx: u32) -> bool {
        self.tree.is_set(idx)
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
        for (b_idx, block) in self.tree.lvl1.iter().enumerate() {
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
        let first_word = slab.tree.lvl1.first().map(|b| b.data[0]).unwrap_or(0);
        Self {
            slab: &slab.data,
            lvl1: &slab.tree.lvl1,
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
            data_ptr: slab.data.as_mut_ptr(),
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
        let first_word = self.tree.lvl1.first().map(|b| b.data[0]).unwrap_or(0);

        // Take ownership des champs avant que Drop ne run
        let data = std::mem::take(&mut self.data);
        let lvl1 = std::mem::take(&mut self.tree.lvl1);

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

impl<T> Drop for SlabIntoIter<T> {
    fn drop(&mut self) {
        // Consommer les éléments restants pour les drop correctement
        for _ in self.by_ref() {}
        // Éviter que Vec::drop ne redrop les éléments déjà moved
        unsafe {
            self.data.set_len(0);
        }
    }
}

impl<T> Drop for HiSlab<T> {
    fn drop(&mut self) {
        // Drop seulement les éléments occupés
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

// ============================================================================
// Random selection (feature "rand")
// ============================================================================

#[cfg(feature = "rand")]
mod random {
    use super::{BitBlock, HiSlab};
    use rand::Rng;

    impl BitBlock {
        /// Compte le nombre de bits à 1 (slots occupés)
        #[inline]
        pub fn popcnt(&self) -> u32 {
            self.data.iter().map(|w| w.count_ones()).sum()
        }
    }

    /// Trouve le n-ième bit à 1 dans un u64 (0-indexed)
    /// Utilise pdep si disponible (BMI2), sinon fallback
    #[inline]
    fn select_nth_bit_u64(word: u64, n: u32) -> usize {
        #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
        {
            // pdep place un 1 à la position du n-ième bit set
            // Exemple: word=0b1010, n=1 -> on veut le 2ème bit à 1 (index 3)
            // mask = 1 << n = 0b10
            // pdep(mask, word) = 0b1000 (le 2ème bit de word déplié)
            // trailing_zeros = 3
            use std::arch::x86_64::_pdep_u64;
            unsafe {
                let mask = 1u64 << n;
                let deposited = _pdep_u64(mask, word);
                deposited.trailing_zeros() as usize
            }
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
        {
            // Fallback: itérer sur les bits
            let mut remaining = n;
            let mut w = word;
            while w != 0 {
                let bit_pos = w.trailing_zeros();
                if remaining == 0 {
                    return bit_pos as usize;
                }
                remaining -= 1;
                w &= w - 1; // clear lowest set bit
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
        /// Retourne None si le slab est vide.
        ///
        /// L'algorithme descend la hiérarchie en utilisant popcnt pour
        /// faire un tirage pondéré à chaque niveau.
        pub fn random_occupied<R: Rng>(&self, rng: &mut R) -> Option<(u32, &T)> {
            // Étape 1: Compter les éléments par bloc lvl1 et faire un tirage pondéré
            let block_counts: Vec<u32> = self.tree.lvl1.iter().map(|b| b.popcnt()).collect();
            let total: u32 = block_counts.iter().sum();

            if total == 0 {
                return None;
            }

            // Tirage pour choisir quel bloc
            let mut choice = rng.gen_range(0..total);

            // Trouver le bloc correspondant
            let mut block_idx = 0;
            for (i, &cnt) in block_counts.iter().enumerate() {
                if choice < cnt {
                    block_idx = i;
                    break;
                }
                choice -= cnt;
            }

            // Étape 2: Dans le bloc choisi, faire un tirage parmi les 8 mots
            let block = &self.tree.lvl1[block_idx];
            let mut word_choice = choice; // Réutilise le reste du tirage

            let mut word_idx = 0;
            for (i, &word) in block.data.iter().enumerate() {
                let pop = word.count_ones();
                if word_choice < pop {
                    word_idx = i;
                    break;
                }
                word_choice -= pop;
            }

            // Étape 3: Dans le mot choisi, trouver le n-ième bit à 1
            let word = block.data[word_idx];
            let bit_pos = select_nth_bit_u64(word, word_choice);

            let final_idx = ((block_idx << 9) | (word_idx << 6) | bit_pos) as u32;

            unsafe { Some((final_idx, self.data.get_unchecked(final_idx as usize))) }
        }

        /// Sélectionne un élément occupé aléatoirement (version mutable).
        pub fn random_occupied_mut<R: Rng>(&mut self, rng: &mut R) -> Option<(u32, &mut T)> {
            // Même algorithme que random_occupied
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

            unsafe { Some((final_idx, self.data.get_unchecked_mut(final_idx as usize))) }
        }

        /// Sélectionne N éléments occupés aléatoirement (avec remise possible).
        /// Retourne un Vec de (index, &T).
        pub fn random_occupied_many<R: Rng>(
            &self,
            rng: &mut R,
            count: usize,
        ) -> Vec<(u32, &T)> {
            let mut results = Vec::with_capacity(count);
            for _ in 0..count {
                if let Some(item) = self.random_occupied(rng) {
                    results.push(item);
                } else {
                    break; // Slab vide
                }
            }
            results
        }

        /// Sélectionne N éléments occupés aléatoirement SANS remise.
        /// Retourne un Vec de (index, &T). Si count > nombre d'éléments, retourne tous les éléments.
        pub fn random_occupied_unique<R: Rng>(
            &self,
            rng: &mut R,
            count: usize,
        ) -> Vec<(u32, &T)> {
            let total = self.count_occupied();
            if count >= total {
                // Retourne tous les éléments
                return self.into_iter().collect();
            }

            // Fisher-Yates partiel: on tire count indices uniques
            // Pour être efficace, on utilise un HashSet si count est petit par rapport à total
            use std::collections::HashSet;

            let mut selected_indices = HashSet::with_capacity(count);
            let mut results = Vec::with_capacity(count);

            // Pré-calculer les cumuls pour éviter de recalculer popcnt à chaque tirage
            let block_counts: Vec<u32> = self.tree.lvl1.iter().map(|b| b.popcnt()).collect();
            let total_u32 = total as u32;

            while results.len() < count {
                let choice = rng.gen_range(0..total_u32);

                // Convertir choice en index réel
                let final_idx = self.choice_to_index(&block_counts, choice);

                if selected_indices.insert(final_idx) {
                    unsafe {
                        results.push((final_idx, self.data.get_unchecked(final_idx as usize)));
                    }
                }
            }

            results
        }

        /// Convertit un choix (0..total) en index réel dans le slab
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
// Tagged feature
// ============================================================================

#[cfg(feature = "tagged")]
impl<T> HiSlab<T> {
    /// Inserts a value with the tagged flag set and returns its index.
    ///
    /// The element will be tracked in both the main tree and the tagged tree.
    #[inline(always)]
    pub fn insert_tagged(&mut self, val: T) -> u32 {
        // --- FAST PATH (0..512) ---
        if let Some(bit_idx) = self.tree.lvl1[0].find_first_free() {
            return self.finalize_insert_tagged(0, bit_idx, val);
        }

        // --- SLOW PATH ---
        let l1_block_idx = self.tree.find_free_block();
        self.tree.ensure_lvl1(l1_block_idx);

        let bit_idx = self.tree.lvl1[l1_block_idx]
            .find_first_free()
            .expect("Hierarchy out of sync");

        self.finalize_insert_tagged(l1_block_idx, bit_idx, val)
    }

    #[inline(always)]
    fn finalize_insert_tagged(&mut self, block_idx: usize, bit_idx: usize, val: T) -> u32 {
        let final_idx = (block_idx * 512 + bit_idx) as u32;

        if (final_idx as usize) < self.data.len() {
            unsafe {
                std::ptr::write(self.data.as_mut_ptr().add(final_idx as usize), val);
            }
        } else {
            self.data.push(val);
        }

        // Set dans les deux arbres
        self.tree.set_bit(final_idx);
        self.tagged_tree.set_bit(final_idx);

        final_idx
    }

    /// Tags an existing element at the given index.
    ///
    /// Returns `true` if the element was successfully tagged (it exists and wasn't already tagged).
    /// Returns `false` if the slot is empty or already tagged.
    #[inline]
    pub fn tag(&mut self, idx: u32) -> bool {
        if !self.is_occupied(idx) {
            return false;
        }
        if self.is_tagged(idx) {
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
        if !self.is_occupied(idx) {
            return false;
        }
        if !self.is_tagged(idx) {
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
        self.tagged_tree.for_each_set(|idx| {
            unsafe {
                f(idx, self.data.get_unchecked(idx as usize));
            }
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
        // Collecter les indices à supprimer (on ne peut pas modifier pendant l'itération)
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

                    let val = unsafe { self.data.get_unchecked_mut(idx as usize) };
                    if !f(idx, val) {
                        to_remove.push(idx);
                    }

                    temp_word &= temp_word - 1;
                }
            }
        }

        // Supprimer les éléments marqués
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

                    let val = unsafe { self.data.get_unchecked_mut(idx as usize) };
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

/// Iterator over tagged elements (immutable).
#[cfg(feature = "tagged")]
pub struct TaggedIter<'a, T> {
    data: &'a Vec<T>,
    lvl1: &'a [BitBlock],
    block_idx: usize,
    word_idx: usize,
    current_word: u64,
}

#[cfg(feature = "tagged")]
impl<'a, T> TaggedIter<'a, T> {
    fn new(slab: &'a HiSlab<T>) -> Self {
        let first_word = slab.tagged_tree.lvl1.first().map(|b| b.data[0]).unwrap_or(0);
        Self {
            data: &slab.data,
            lvl1: &slab.tagged_tree.lvl1,
            block_idx: 0,
            word_idx: 0,
            current_word: first_word,
        }
    }
}

#[cfg(feature = "tagged")]
impl<'a, T> Iterator for TaggedIter<'a, T> {
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

        unsafe { Some((final_idx as u32, self.data.get_unchecked(final_idx))) }
    }
}

/// Iterator over tagged elements (mutable).
#[cfg(feature = "tagged")]
pub struct TaggedIterMut<'a, T> {
    data_ptr: *mut T,
    lvl1: &'a [BitBlock],
    block_idx: usize,
    word_idx: usize,
    current_word: u64,
    _marker: std::marker::PhantomData<&'a mut T>,
}

#[cfg(feature = "tagged")]
impl<'a, T> TaggedIterMut<'a, T> {
    fn new(slab: &'a mut HiSlab<T>) -> Self {
        let first_word = slab.tagged_tree.lvl1.first().map(|b| b.data[0]).unwrap_or(0);
        Self {
            data_ptr: slab.data.as_mut_ptr(),
            lvl1: &slab.tagged_tree.lvl1,
            block_idx: 0,
            word_idx: 0,
            current_word: first_word,
            _marker: std::marker::PhantomData,
        }
    }
}

#[cfg(feature = "tagged")]
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

// Tagged + Rand: random selection among tagged elements
#[cfg(all(feature = "tagged", feature = "rand"))]
impl<T> HiSlab<T> {
    /// Selects a random tagged element.
    /// Returns None if no elements are tagged.
    pub fn random_tagged<R: rand::Rng>(&self, rng: &mut R) -> Option<(u32, &T)> {
        let idx = self.tagged_tree.random_set(rng)?;
        unsafe { Some((idx, self.data.get_unchecked(idx as usize))) }
    }

    /// Selects a random tagged element (mutable version).
    /// Returns None if no elements are tagged.
    pub fn random_tagged_mut<R: rand::Rng>(&mut self, rng: &mut R) -> Option<(u32, &mut T)> {
        let idx = self.tagged_tree.random_set(rng)?;
        unsafe { Some((idx, self.data.get_unchecked_mut(idx as usize))) }
    }

    /// Selects N random tagged elements (with possible duplicates).
    pub fn random_tagged_many<R: rand::Rng>(
        &self,
        rng: &mut R,
        count: usize,
    ) -> Vec<(u32, &T)> {
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
    pub fn random_tagged_unique<R: rand::Rng>(
        &self,
        rng: &mut R,
        count: usize,
    ) -> Vec<(u32, &T)> {
        use std::collections::HashSet;

        let total = self.count_tagged();
        if count >= total {
            // Retourne tous les éléments taggés
            let mut results = Vec::with_capacity(total);
            // Collecter les indices d'abord
            let mut indices = Vec::with_capacity(total);
            self.tagged_tree.for_each_set(|idx| {
                indices.push(idx);
            });
            for idx in indices {
                unsafe {
                    results.push((idx, self.data.get_unchecked(idx as usize)));
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
                        results.push((idx, self.data.get_unchecked(idx as usize)));
                    }
                }
            } else {
                break;
            }
        }

        results
    }
}
