use crate::bit_block::BitBlock;

/// Arbre de bitmaps hiérarchique pour le suivi des slots occupés/libres.
/// Chaque niveau résume 512 entrées du niveau inférieur.
pub struct BitmapTree {
    pub(crate) lvl1: Vec<BitBlock>,
    pub(crate) lvl2: Vec<BitBlock>,
    pub(crate) lvl3: Vec<BitBlock>,
    pub(crate) lvl4: u8, // seulement 5 bits nécessaires (3*9=27, 32-27=5)
}

impl Default for BitmapTree {
    fn default() -> Self {
        Self::new()
    }
}

impl BitmapTree {
    pub fn new() -> Self {
        Self {
            lvl1: vec![BitBlock::default()],
            lvl2: vec![BitBlock::default()],
            lvl3: vec![BitBlock::default()],
            lvl4: 0,
        }
    }

    /// Vérifie si un bit est occupé
    #[inline(always)]
    pub fn is_set(&self, idx: u32) -> bool {
        let idx=idx as usize;
        let b_idx = idx >> 9; // / 512
        if b_idx >= self.lvl1.len() {
            return false;
        }
        self.lvl1[b_idx].is_set(idx&0x1FF)
    }

    /// Trouve le premier slot libre dans toute la hiérarchie, croît si nécessaire,
    /// et retourne son index plat. Ne réserve PAS le slot (n'écrit pas le bit).
    #[inline(always)]
    pub fn find_free_idx(&mut self) -> u32 {
        // Fast path : premier bloc lvl1 (slots 0..512)
        if let Some(bit_idx) = self.lvl1[0].find_first_free() {
            return bit_idx as u32;
        }
        // Slow path : descente hiérarchique avec croissance automatique
        let block_idx = self.find_free_block();
        self.ensure_lvl1(block_idx);
        let bit_idx = self.lvl1[block_idx]
            .find_first_free()
            .expect("Hierarchy out of sync");
        (block_idx * 512 + bit_idx) as u32
    }

    /// Trouve le premier slot libre, le réserve (set le bit) et retourne son index plat.
    #[inline(always)]
    pub fn reserve_free_idx(&mut self) -> u32 {
        let idx = self.find_free_idx();
        self.set_bit(idx);
        idx
    }

    /// Trouve le premier bloc libre via la hiérarchie (slow path)
    pub fn find_free_block(&mut self) -> usize {
        if self.lvl4 == 0xFF {
            panic!("BitmapTree overflow");
        }

        let l3_block_idx = (!self.lvl4).trailing_zeros() as usize;
        self.ensure_lvl3(l3_block_idx);

        let l2_block_relative_idx = self.lvl3[l3_block_idx].find_first_free().unwrap();
        let l2_block_idx = (l3_block_idx * 512) + l2_block_relative_idx;
        self.ensure_lvl2(l2_block_idx);

        let l1_block_relative_idx = self.lvl2[l2_block_idx].find_first_free().unwrap();
        (l2_block_idx * 512) + l1_block_relative_idx
    }

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

    pub fn ensure_lvl1(&mut self, idx: usize) {
        if idx >= self.lvl1.len() {
            self.lvl1.resize_with(idx + 1, BitBlock::default);
        }
    }

    /// Set un bit à l'index donné et propage la fullness si nécessaire
    #[inline(always)]
    pub fn set_bit(&mut self, idx: u32) {
        let idx_usize = idx as usize;
        let block_idx = idx_usize / 512;
        let bit_idx = idx_usize % 512;

        self.ensure_lvl1(block_idx);

        if self.lvl1[block_idx].set_bit_and_check_full(bit_idx) {
            self.propagate_full(block_idx);
        }
    }

    /// Clear un bit à l'index donné et propage l'emptiness si nécessaire
    #[inline(always)]
    pub fn clear_bit(&mut self, idx: u32) {
        let idx_usize = idx as usize;
        let block_idx = idx_usize / 512;
        let bit_idx = idx_usize % 512;

        if block_idx >= self.lvl1.len() {
            return;
        }

        if self.lvl1[block_idx].clear_bit_and_was_full(bit_idx) {
            self.propagate_empty(block_idx);
        }
    }

    fn propagate_full(&mut self, l1_block_idx: usize) {
        let l2_block_idx = l1_block_idx / 512;
        let l2_bit_idx = l1_block_idx % 512;

        self.ensure_lvl2(l2_block_idx);

        if self.lvl2[l2_block_idx].set_bit_and_check_full(l2_bit_idx) {
            let l3_block_idx = l2_block_idx / 512;
            let l3_bit_idx = l2_block_idx % 512;

            self.ensure_lvl3(l3_block_idx);

            if self.lvl3[l3_block_idx].set_bit_and_check_full(l3_bit_idx) {
                self.lvl4 |= 1 << l3_block_idx;
            }
        }
    }

    fn propagate_empty(&mut self, l1_block_idx: usize) {
        let l2_block_idx = l1_block_idx / 512;
        let l2_bit_idx = l1_block_idx % 512;

        if l2_block_idx >= self.lvl2.len() {
            return;
        }

        if self.lvl2[l2_block_idx].clear_bit_and_was_full(l2_bit_idx) {
            let l3_block_idx = l2_block_idx / 512;
            let l3_bit_idx = l2_block_idx % 512;

            if l3_block_idx >= self.lvl3.len() {
                return;
            }

            if self.lvl3[l3_block_idx].clear_bit_and_was_full(l3_bit_idx) {
                self.lvl4 &= !(1 << l3_block_idx);
            }
        }
    }

    /// Itère sur tous les bits à 1 avec un callback
    #[allow(dead_code)]
    pub fn for_each_set<F>(&self, mut f: F)
    where
        F: FnMut(u32),
    {
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
                    f(final_idx as u32);
                    temp_word &= temp_word - 1;
                }
            }
        }
    }
}

#[cfg(feature = "rand")]
impl BitmapTree {
    /// Compte le nombre de bits à 1
    pub fn count_set(&self) -> usize {
        self.lvl1.iter().map(|b| b.popcnt() as usize).sum()
    }

    /// Sélectionne un bit à 1 aléatoirement
    pub fn random_set<R: rand::Rng>(&self, rng: &mut R) -> Option<u32> {
        let block_counts: Vec<u32> = self.lvl1.iter().map(|b| b.popcnt()).collect();
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

        let block = &self.lvl1[block_idx];
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

        Some(((block_idx << 9) | (word_idx << 6) | bit_pos) as u32)
    }
}

#[cfg(feature = "rand")]
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
