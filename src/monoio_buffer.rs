//! # MonoioSlabBuffer
//!
//! Intégration de [`BufferSlab`] avec le runtime monoio via `Rc<BufferSlab<N>>`.
//!
//! ## Problème résolu
//!
//! Les traits [`monoio::buf::IoBuf`] et [`monoio::buf::IoBufMut`] exigent `'static`.
//! L'existant [`crate::buffer_slab::AutoGuard`] tient `&'a BufferSlab<N>`, ce qui
//! empêche son utilisation directe dans les opérations I/O asynchrones de monoio.
//!
//! ## Solution
//!
//! [`MonoioSlabBuffer`] enrobe le pool dans un `Rc`. [`MonoBuffer`] possède un clone
//! de ce `Rc` au lieu d'une référence empruntée, le rendant `'static` tout en gardant
//! la libération automatique via `Drop`.
//!
//! ## Contraintes
//!
//! - `!Send + !Sync` — usage single-thread uniquement (compatible monoio).
//! - `Rc` (non `Arc`) : monoio étant mono-thread, le surcoût atomique d'`Arc` est inutile.
//!
//! ## Exemple
//!
//! ```no_run
//! # #[cfg(feature = "monoio")]
//! # {
//! use std::rc::Rc;
//! use hislab::monoio_buffer::MonoioSlabBuffer;
//!
//! let pool = Rc::new(MonoioSlabBuffer::<4096>::new(4, 128).unwrap());
//!
//! // buf est 'static, peut être passé à monoio::fs::File::read ou write
//! let buf = pool.allocate();
//! assert_eq!(buf.index(), 0);
//! // drop(buf) libère le slot automatiquement
//! # }
//! ```

use std::rc::Rc;

use monoio::buf::{IoBuf, IoBufMut};

use crate::buffer_slab::BufferSlab;

// ============================================================================
// MonoioSlabBuffer
// ============================================================================

/// [`BufferSlab`] wrappé dans un `Rc` pour une utilisation avec monoio.
///
/// Distribue des [`MonoBuffer`] qui sont `'static` et implémentent
/// [`IoBuf`]/[`IoBufMut`], permettant leur passage aux opérations I/O
/// asynchrones de monoio sans durée de vie empruntée.
pub struct MonoioSlabBuffer<const N: usize>(Rc<BufferSlab<N>>);

impl<const N: usize> MonoioSlabBuffer<N> {
    /// Crée un nouveau pool.
    ///
    /// Voir [`BufferSlab::new`] pour la sémantique des paramètres.
    pub fn new(initial_capacity: u32, virtual_capacity: u32) -> Result<Self, std::io::Error> {
        Ok(Self(Rc::new(BufferSlab::new(
            initial_capacity,
            virtual_capacity,
        )?)))
    }

    /// Alloue un buffer. Panique si le pool est épuisé.
    #[inline]
    pub fn allocate(&self) -> MonoBuffer<N> {
        self.try_allocate()
            .expect("MonoioSlabBuffer: virtual_capacity épuisée")
    }

    /// Tente d'allouer un buffer. Retourne `None` si le pool est épuisé.
    #[inline]
    pub fn try_allocate(&self) -> Option<MonoBuffer<N>> {
        let (ptr, idx) = self.0.alloc_raw()?;
        Some(MonoBuffer {
            ptr,
            idx,
            init: 0,
            slab: Rc::clone(&self.0),
        })
    }
}

// ============================================================================
// MonoBuffer
// ============================================================================

/// Guard RAII `'static` sur un buffer alloué dans un [`MonoioSlabBuffer`].
///
/// Contrairement à [`crate::buffer_slab::AutoGuard`], `MonoBuffer` est `'static`
/// car il **possède** un `Rc<BufferSlab<N>>` plutôt qu'une référence empruntée.
/// Cela permet de le passer directement aux opérations I/O de monoio.
///
/// Le slot est libéré automatiquement via `Drop`.
pub struct MonoBuffer<const N: usize> {
    ptr: *mut u8,
    idx: u32,
    /// Nombre de bytes initialisés (semantique IoBuf/IoBufMut).
    init: usize,
    slab: Rc<BufferSlab<N>>,
}

impl<const N: usize> MonoBuffer<N> {
    /// Index du buffer dans le pool.
    #[inline]
    pub fn index(&self) -> u32 {
        self.idx
    }

    /// Slice sur les bytes initialisés (0..`bytes_init()`).
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.init) }
    }

    /// Slice mutable sur la totalité des `N` bytes du buffer.
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, N) }
    }

    /// Pointeur brut vers le début du buffer.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Pointeur brut mutable vers le début du buffer.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    /// Nombre de bytes initialisés.
    #[inline]
    pub fn initialized_len(&self) -> usize {
        self.init
    }
}

impl<const N: usize> Drop for MonoBuffer<N> {
    #[inline]
    fn drop(&mut self) {
        self.slab.free_at(self.idx);
    }
}

// ============================================================================
// IoBuf / IoBufMut
// ============================================================================

/// # Safety
///
/// `ptr` pointe vers `N` bytes valides pour toute la durée de vie du `MonoBuffer`.
/// `bytes_init()` retourne toujours une valeur ≤ `N`, correspondant aux bytes
/// effectivement initialisés via `set_init`.
unsafe impl<const N: usize> IoBuf for MonoBuffer<N> {
    #[inline]
    fn read_ptr(&self) -> *const u8 {
        self.ptr
    }

    #[inline]
    fn bytes_init(&self) -> usize {
        self.init
    }
}

/// # Safety
///
/// `write_ptr()` retourne un pointeur valide vers `bytes_total()` bytes accessibles
/// en écriture. `set_init` ne peut qu'augmenter le compteur (invariant IoBufMut).
unsafe impl<const N: usize> IoBufMut for MonoBuffer<N> {
    #[inline]
    fn write_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    #[inline]
    fn bytes_total(&mut self) -> usize {
        N
    }

    #[inline]
    unsafe fn set_init(&mut self, pos: usize) {
        // Invariant : set_init ne peut qu'avancer le curseur d'initialisation.
        if pos > self.init {
            self.init = pos;
        }
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
        let pool = MonoioSlabBuffer::<4096>::new(4, 8).unwrap();

        let b0 = pool.allocate();
        let b1 = pool.allocate();
        assert_eq!(b0.index(), 0);
        assert_eq!(b1.index(), 1);
        assert_eq!(b0.initialized_len(), 0);

        drop(b0);
        let b0b = pool.allocate();
        assert_eq!(b0b.index(), 0); // slot réutilisé
    }

    #[test]
    fn test_exhausted_returns_none() {
        let pool = MonoioSlabBuffer::<64>::new(2, 2).unwrap();
        let _b0 = pool.allocate();
        let _b1 = pool.allocate();
        assert!(pool.try_allocate().is_none());
    }

    #[test]
    fn test_set_init_advances_only() {
        let pool = MonoioSlabBuffer::<64>::new(1, 4).unwrap();
        let mut buf = pool.allocate();

        unsafe { buf.set_init(32) };
        assert_eq!(buf.initialized_len(), 32);

        // set_init ne peut pas reculer
        unsafe { buf.set_init(10) };
        assert_eq!(buf.initialized_len(), 32);

        unsafe { buf.set_init(64) };
        assert_eq!(buf.initialized_len(), 64);
    }

    #[test]
    fn test_iobuf_ptrs() {
        let pool = MonoioSlabBuffer::<128>::new(1, 4).unwrap();
        let mut buf = pool.allocate();

        assert_eq!(buf.read_ptr(), buf.as_ptr());
        assert_eq!(buf.write_ptr(), buf.as_mut_ptr());
        assert_eq!(buf.bytes_total(), 128);
        assert_eq!(buf.bytes_init(), 0);
    }

    #[test]
    fn test_rc_clone_keeps_pool_alive() {
        let b: MonoBuffer<64>;
        {
            let pool = MonoioSlabBuffer::<64>::new(1, 4).unwrap();
            b = pool.allocate();
            // pool est droppé ici, mais b tient un Rc donc la slab survit
        }
        // b est encore valide — pas de use-after-free
        assert_eq!(b.index(), 0);
        drop(b); // free_at appelé correctement via le Rc survivant
    }
}
