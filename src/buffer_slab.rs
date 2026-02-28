//! # BufferSlab
//!
//! Pool d'allocation de buffers de taille fixe avec tracking O(1) via bitmap hiérarchique.
//!
//! ## Design
//!
//! - `!Send + !Sync` via le raw ptr — single-thread ou wrapping explicite
//! - `allocate(&self)` — plusieurs guards simultanés possible
//! - `AutoGuard` libère automatiquement via `Drop` — pas de `free` manuel
//! - Espace virtuel pré-réservé via `mmap` anonyme, pages initiales pré-faultées via
//!   `madvise(MADV_POPULATE_WRITE)` — le ptr `storage` ne change jamais.
//!
//! ## Exemple
//!
//! ```
//! use hislab::buffer_slab::BufferSlab;
//!
//! let pool = BufferSlab::<4096>::new(4, 128).unwrap();
//!
//! let mut g0 = pool.allocate();
//! let mut g1 = pool.allocate(); // plusieurs guards simultanés
//! g0.as_slice_mut().fill(0xAB);
//! assert_eq!(g0.index(), 0);
//! assert_eq!(g1.index(), 1);
//! // drop(g0) et drop(g1) libèrent automatiquement
//! ```

use std::cell::UnsafeCell;

use memmap2::MmapMut;

use crate::bitmap_tree::BitmapTree;

// ============================================================================
// BufferSlab
// ============================================================================

/// Pool de buffers de taille fixe `N` octets.
///
/// L'espace virtuel est réservé entièrement à la construction via `mmap` anonyme.
/// Les pages couvrant `initial_capacity` slots sont pré-faultées immédiatement via
/// `madvise(MADV_POPULATE_WRITE)`. Les pages au-delà sont committées à la demande.
///
/// `!Send + !Sync` par construction (champ `*mut u8`) — wrapping dans un
/// `Mutex` ou `RwLock` est la responsabilité du caller si nécessaire.
pub struct BufferSlab<const N: usize> {
    /// Ptr stable vers le début du stockage — jamais déplacé.
    storage: *mut u8,
    /// Limite max de slots — détermine la taille de la réservation virtuelle.
    virtual_capacity: u32,
    /// Bitmap hiérarchique de tracking. Mutable via &self grâce à UnsafeCell.
    /// Sound car !Sync interdit tout accès concurrent.
    tree: UnsafeCell<BitmapTree>,
    /// Garde le mapping mmap en vie. Jamais modifié après construction.
    _mmap: MmapMut,
}

// ============================================================================
// AutoGuard
// ============================================================================

/// Guard RAII sur un buffer alloué dans un [`BufferSlab`].
///
/// Libère automatiquement le slot à la destruction via `Drop`.
/// Tient `&'a BufferSlab` — le borrow checker garantit que la slab
/// outlive le guard.
///
/// Plusieurs guards peuvent coexister simultanément.
pub struct AutoGuard<'a, const N: usize> {
    ptr: *mut u8,
    idx: u32,
    slab: &'a BufferSlab<N>,
}

// ============================================================================
// Impl BufferSlab
// ============================================================================

impl<const N: usize> BufferSlab<N> {
    /// Crée un pool avec `virtual_capacity` slots réservés en espace virtuel.
    ///
    /// Les pages couvrant les `initial_capacity` premiers slots sont pré-faultées
    /// immédiatement via `madvise(MADV_POPULATE_WRITE)` — zéro page fault lors des
    /// premières allocations. Les pages au-delà sont committées à la demande par le kernel.
    ///
    /// Le ptr de stockage ne se déplace jamais — les index retournés par `allocate`
    /// sont stables pour toute la durée de vie du pool.
    ///
    /// # Panics
    ///
    /// - Si `initial_capacity > virtual_capacity`
    /// - Si `virtual_capacity * N` dépasse `usize::MAX` — combinaison de paramètres
    ///   invalide, impossible de réserver l'espace virtuel demandé
    ///
    /// # Errors
    ///
    /// Retourne une erreur si `mmap` échoue (OOM système).
    pub fn new(initial_capacity: u32, virtual_capacity: u32) -> Result<Self, std::io::Error> {
        assert!(N > 0, "N doit être > 0");
        assert!(
            initial_capacity <= virtual_capacity,
            "initial_capacity doit être <= virtual_capacity"
        );

        // Panic intentionnel : virtual_capacity * N > usize::MAX est un bug appelant,
        // pas une condition récupérable à runtime.
        let virtual_bytes = (virtual_capacity as usize)
            .checked_mul(N)
            .expect("virtual_capacity * N dépasse usize::MAX");

        let mmap = MmapMut::map_anon(virtual_bytes)?;
        let storage = mmap.as_ptr() as *mut u8;

        // Pré-faulter les pages initiales : évite les page faults sur les premières
        // allocations. MADV_POPULATE_WRITE demande au kernel de résoudre les page faults
        // en écriture immédiatement (Linux 5.14+). Best-effort : on ignore les erreurs
        // (noyau trop ancien ou madvise non supporté).
        if initial_capacity > 0 {
            let initial_bytes = initial_capacity as usize * N;
            unsafe {
                libc::madvise(
                    storage as *mut libc::c_void,
                    initial_bytes,
                    libc::MADV_POPULATE_WRITE|libc::MADV_HUGEPAGE,
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

    /// Capacité virtuelle maximale (nombre de slots total réservé).
    #[inline]
    pub fn virtual_capacity(&self) -> u32 {
        self.virtual_capacity
    }

    /// Alloue un buffer libre et retourne son guard RAII.
    ///
    /// Panics si `virtual_capacity` est atteinte (pool épuisé).
    /// O(1) via le bitmap hiérarchique.
    #[inline]
    pub fn allocate(&self) -> AutoGuard<'_, N> {
        self.try_allocate()
            .expect("BufferSlab: virtual_capacity épuisée")
    }

    /// Tente d'allouer un buffer libre. Retourne `None` si `virtual_capacity` est atteinte.
    ///
    /// O(1) via le bitmap hiérarchique.
    #[inline]
    pub fn try_allocate(&self) -> Option<AutoGuard<'_, N>> {
        let tree = unsafe { &mut *self.tree.get() };
        let idx = tree.find_free_idx();
        if idx >= self.virtual_capacity {
            return None;
        }
        tree.set_bit(idx);
        let ptr = unsafe { self.storage.add(idx as usize * N) };
        Some(AutoGuard { ptr, idx, slab: self })
    }

    /// Libère un slot par index. Appelé par `Drop` de [`AutoGuard`].
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
    const fn is_ptr_power_of_two()->bool{
        N.is_power_of_two()
    }


    #[inline(always)]
    unsafe fn hint_aligned(&self) {
        // 1. On vérifie à la compilation si N est une puissance de 2
        
        // 2. LLVM ne verra le hint QUE si c'est vrai
        if Self::is_ptr_power_of_two() {
            // Utilise l'intrinsèque de LLVM pour l'alignement
            unsafe{std::hint::assert_unchecked(self.ptr as usize & (N - 1) == 0);}
        }
    }

    /// Slice immutable sur les `N` octets du buffer.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            self.hint_aligned();
            std::slice::from_raw_parts(self.ptr, N)
        }
    }

    /// Slice mutable sur les `N` octets du buffer.
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        unsafe {
            self.hint_aligned();
            std::slice::from_raw_parts_mut(self.ptr, N)
        }
    }

    /// Index du buffer dans le pool.
    #[inline]
    pub fn index(&self) -> u32 {
        self.idx
    }

    /// Pointeur brut vers le début du buffer.
    ///
    /// Valide tant que le guard est en vie.
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
        // Allouer au-delà de initial_capacity fonctionne (pages on-demand)
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
