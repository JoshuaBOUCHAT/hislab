use std::time::Instant;

use crate::HiSlab;
use crate::TaggedHiSlab;

#[cfg(feature = "rand")]
use rand::SeedableRng;

#[test]
fn test_basic_insert_delete() {
    let count = (1 << 18) + 1; // 262,144 éléments
    let slab = HiSlab::<u8>::new(0, 1 << 19).expect("can't allocate");

    println!("🚀 Insertion de {} éléments...", count);

    let start = Instant::now();
    for i in 0..count {
        let idx = slab.insert((i % 255) as u8);

        if idx != i as u32 {
            panic!("Erreur d'indexage à {}: reçu {}", i, idx);
        }
    }
    let duration = start.elapsed();

    println!("✅ Insertion terminée en : {:?}", duration);
    println!("⏱️  Moyenne par insertion : {:?}", duration / count as u32);

    println!("--- État de la hiérarchie ---");
    println!("Lvl4 (résumé) : {:08b}", unsafe { &*slab.tree.get() }.lvl4);

    println!("\n🗑️ Suppression de tous les éléments...");
    let start_remove = Instant::now();
    for i in 0..count {
        slab.remove(i as u32);
    }
    let duration_remove = start_remove.elapsed();

    println!("✅ Suppression terminée en : {:?}", duration_remove);

    assert_eq!(
        unsafe { &*slab.tree.get() }.lvl4,
        0,
        "La hiérarchie n'est pas revenue à zéro !"
    );
    println!("\n✨ Test réussi : La structure est cohérente et rapide.");
}

#[test]
fn test_stress_lifecycle() {
    let slab = HiSlab::<u32>::new(0, 65536).expect("can't allocate");
    let iterations = 10_000;
    let mut indices = Vec::with_capacity(iterations);

    println!("🏗️ Phase 1 : Insertion massive...");
    for i in 0..iterations {
        let idx = slab.insert(i as u32 * 10);
        indices.push(idx);
    }

    for (i, &idx) in indices.iter().enumerate() {
        assert_eq!(slab.get(idx), Some(&(i as u32 * 10)));
    }

    println!("🗑️ Phase 2 : Fragmentation (suppression d'un élément sur deux)...");
    for i in (0..iterations).step_by(2) {
        let idx = indices[i];
        slab.remove(idx);
        assert!(slab.get(idx).is_none(), "L'index {} devrait être vide", idx);
    }

    println!("🧐 Phase 3 : Vérification de la persistance des autres...");
    for i in (1..iterations).step_by(2) {
        let idx = indices[i];
        assert_eq!(slab.get(idx), Some(&(i as u32 * 10)));
    }

    println!("♻️ Phase 4 : Ré-insertion dans les trous...");
    for _ in 0..1000 {
        let new_val = 999_999;
        let idx = slab.insert(new_val);
        assert!(
            idx % 2 == 0,
            "L'insertion devrait combler les trous (index pair), reçu {}",
            idx
        );
        assert_eq!(slab[idx], new_val);
    }

    println!("🎯 Phase 5 : Test des accès Panic/Unchecked...");
    let valid_idx = indices[1];

    let _ = slab[valid_idx];

    unsafe {
        assert_eq!(*slab.get_unchecked(valid_idx), 1 * 10);
    }

    println!(
        "✨ Test réussi ! Cohérence maintenue après {} itérations.",
        iterations
    );
}

#[test]
fn test_iterator_order_and_stability() {
    let slab = HiSlab::<&str>::new(0, 65536).expect("can't allocate");

    slab.insert("Rust"); // Index 0
    slab.insert("is"); // Index 1
    slab.insert("fast"); // Index 2

    println!("Test de l'ordre simple...");
    let mut iter = (&slab).into_iter();

    assert_eq!(iter.next(), Some((0, &"Rust")));
    assert_eq!(iter.next(), Some((1, &"is")));
    assert_eq!(iter.next(), Some((2, &"fast")));
    assert_eq!(iter.next(), None);

    let slab2 = HiSlab::<&str>::new(0, 65536).expect("can't allocate");
    slab2.insert("A"); // 0
    let idx_b = slab2.insert("B"); // 1
    slab2.insert("C"); // 2

    slab2.remove(idx_b);

    println!("Test apres suppression (doit sauter l'index 1)...");
    let mut iter2 = (&slab2).into_iter();
    assert_eq!(iter2.next(), Some((0, &"A")));
    assert_eq!(iter2.next(), Some((2, &"C")));
    assert_eq!(iter2.next(), None);

    slab2.insert("D");

    println!("Test apres re-insertion (D doit apparaitre entre A et C)...");
    let mut iter3 = (&slab2).into_iter();
    assert_eq!(iter3.next(), Some((0, &"A")));
    assert_eq!(iter3.next(), Some((1, &"D")));
    assert_eq!(iter3.next(), Some((2, &"C")));
}

#[test]
fn test_iter_mut() {
    let mut slab = HiSlab::<i32>::new(0, 65536).expect("can't allocate");

    slab.insert(10); // 0
    slab.insert(20); // 1
    slab.insert(30); // 2
    slab.remove(1); // trou à 1

    for (idx, val) in &mut slab {
        *val += idx as i32;
    }

    assert_eq!(slab.get(0), Some(&10)); // 10 + 0
    assert_eq!(slab.get(1), None); // supprimé
    assert_eq!(slab.get(2), Some(&32)); // 30 + 2
}

#[test]
fn test_into_iter_partial_consumption() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

    #[derive(Default)]
    #[allow(dead_code)]
    struct DropCounter(Arc<()>);
    impl Drop for DropCounter {
        fn drop(&mut self) {
            DROP_COUNT.fetch_add(1, Ordering::SeqCst);
        }
    }

    DROP_COUNT.store(0, Ordering::SeqCst);

    let slab = HiSlab::<DropCounter>::new(0, 65536).expect("can't allocate");
    for _ in 0..10 {
        slab.insert(DropCounter(Arc::new(())));
    }

    let mut iter = slab.into_iter();
    let _ = iter.next();
    let _ = iter.next();
    let _ = iter.next();

    drop(iter);

    assert_eq!(
        DROP_COUNT.load(Ordering::SeqCst),
        10,
        "Fuite mémoire détectée!"
    );
}

#[test]
fn test_empty_slab_iteration() {
    let slab = HiSlab::<i32>::new(0, 65536).expect("can't allocate");

    let mut count = 0;
    for _ in &slab {
        count += 1;
    }
    assert_eq!(count, 0);

    let slab2 = HiSlab::<i32>::new(0, 65536).expect("can't allocate");
    let collected: Vec<_> = slab2.into_iter().collect();
    assert!(collected.is_empty());
}

// ============================================================================
// Tests pour la feature "rand"
// ============================================================================

#[cfg(feature = "rand")]
#[test]
fn test_random_occupied_basic() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let slab = HiSlab::<i32>::new(0, 65536).expect("can't allocate");

    assert!(slab.random_occupied(&mut rng).is_none());

    for i in 0..100 {
        slab.insert(i * 10);
    }

    for _ in 0..50 {
        let (idx, val) = slab.random_occupied(&mut rng).unwrap();
        assert_eq!(slab.get(idx), Some(val));
        assert_eq!(*val, idx as i32 * 10);
    }
}

#[cfg(feature = "rand")]
#[test]
fn test_random_occupied_with_holes() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(123);
    let slab = HiSlab::<i32>::new(0, 65536).expect("can't allocate");

    for i in 0..100 {
        slab.insert(i);
    }

    for i in (0..100).step_by(2) {
        slab.remove(i);
    }

    for _ in 0..100 {
        let (idx, val) = slab.random_occupied(&mut rng).unwrap();
        assert!(slab.is_occupied(idx));
        assert_eq!(idx % 2, 1, "Devrait retourner un index impair");
        assert_eq!(*val, idx as i32);
    }
}

#[cfg(feature = "rand")]
#[test]
fn test_random_occupied_many() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(456);
    let slab = HiSlab::<i32>::new(0, 65536).expect("can't allocate");

    for i in 0..1000 {
        slab.insert(i);
    }

    let results = slab.random_occupied_many(&mut rng, 50);
    assert_eq!(results.len(), 50);

    for (idx, val) in results {
        assert!(slab.is_occupied(idx));
        assert_eq!(*val, idx as i32);
    }
}

#[cfg(feature = "rand")]
#[test]
fn test_random_occupied_unique() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(789);
    let slab = HiSlab::<i32>::new(0, 65536).expect("can't allocate");

    for i in 0..100 {
        slab.insert(i);
    }

    let results = slab.random_occupied_unique(&mut rng, 20);
    assert_eq!(results.len(), 20);

    let mut indices: Vec<u32> = results.iter().map(|(idx, _)| *idx).collect();
    indices.sort();
    indices.dedup();
    assert_eq!(indices.len(), 20, "Les indices doivent être uniques");
}

#[cfg(feature = "rand")]
#[test]
fn test_random_occupied_unique_overflow() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(999);
    let slab = HiSlab::<i32>::new(0, 65536).expect("can't allocate");

    for i in 0..10 {
        slab.insert(i);
    }

    let results = slab.random_occupied_unique(&mut rng, 100);
    assert_eq!(results.len(), 10);
}

#[cfg(feature = "rand")]
#[test]
fn test_count_occupied() {
    let slab = HiSlab::<i32>::new(0, 65536).expect("can't allocate");

    assert_eq!(slab.count_occupied(), 0);

    for i in 0..500 {
        slab.insert(i);
    }
    assert_eq!(slab.count_occupied(), 500);

    for i in (0..500).step_by(5) {
        slab.remove(i);
    }
    assert_eq!(slab.count_occupied(), 400); // 500 - 100
}

// ============================================================================
// Tests pour TaggedHiSlab
// ============================================================================

#[test]
fn test_insert_tagged_basic() {
    let slab = TaggedHiSlab::<i32>::new(0, 65536).expect("can't allocate");

    let idx1 = slab.insert(100);
    assert!(!slab.is_tagged(idx1));

    let idx2 = slab.insert_tagged(200);
    assert!(slab.is_tagged(idx2));

    assert!(slab.is_occupied(idx1));
    assert!(slab.is_occupied(idx2));

    assert_eq!(slab.get(idx1), Some(&100));
    assert_eq!(slab.get(idx2), Some(&200));
}

#[test]
fn test_tag_untag() {
    let slab = TaggedHiSlab::<i32>::new(0, 65536).expect("can't allocate");

    let idx = slab.insert(42);
    assert!(!slab.is_tagged(idx));

    assert!(slab.tag(idx));
    assert!(slab.is_tagged(idx));

    assert!(!slab.tag(idx));

    assert!(slab.untag(idx));
    assert!(!slab.is_tagged(idx));

    assert!(!slab.untag(idx));
}

#[test]
fn test_remove_clears_tag() {
    let slab = TaggedHiSlab::<i32>::new(0, 65536).expect("can't allocate");

    let idx = slab.insert_tagged(42);
    assert!(slab.is_tagged(idx));

    slab.remove(idx);
    assert!(!slab.is_occupied(idx));
    assert!(!slab.is_tagged(idx));
}

#[test]
fn test_insert_after_tagged_remove() {
    let slab = TaggedHiSlab::<i32>::new(0, 65536).expect("can't allocate");

    let idx1 = slab.insert_tagged(100);
    slab.remove(idx1);

    let idx2 = slab.insert(200);
    assert_eq!(idx1, idx2);

    assert!(!slab.is_tagged(idx2));
}

#[test]
fn test_for_each_tagged() {
    let slab = TaggedHiSlab::<i32>::new(0, 65536).expect("can't allocate");

    for i in 0..10 {
        let idx = slab.insert(i * 10);
        if i % 2 == 0 {
            slab.tag(idx);
        }
    }

    let mut tagged_values = Vec::new();
    slab.for_each_tagged(|idx, val| {
        tagged_values.push((idx, *val));
    });

    assert_eq!(tagged_values.len(), 5);
    for (idx, val) in tagged_values {
        assert_eq!(idx % 2, 0);
        assert_eq!(val, idx as i32 * 10);
    }
}

#[cfg(feature = "rand")]
#[test]
fn test_random_tagged() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let slab = TaggedHiSlab::<i32>::new(0, 65536).expect("can't allocate");

    assert!(slab.random_tagged(&mut rng).is_none());

    for i in 0..100 {
        if i % 2 == 1 {
            slab.insert_tagged(i);
        } else {
            slab.insert(i);
        }
    }

    for _ in 0..50 {
        let (idx, val) = slab.random_tagged(&mut rng).unwrap();
        assert!(slab.is_tagged(idx));
        assert_eq!(idx % 2, 1);
        assert_eq!(*val, idx as i32);
    }
}

#[cfg(feature = "rand")]
#[test]
fn test_count_tagged() {
    let slab = TaggedHiSlab::<i32>::new(0, 65536).expect("can't allocate");

    assert_eq!(slab.count_tagged(), 0);

    for i in 0..100 {
        if i % 3 == 0 {
            slab.insert_tagged(i);
        } else {
            slab.insert(i);
        }
    }

    assert_eq!(slab.count_tagged(), 34);

    slab.untag(0);
    slab.untag(3);
    assert_eq!(slab.count_tagged(), 32);
}

#[cfg(feature = "rand")]
#[test]
fn test_random_tagged_unique() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(789);
    let slab = TaggedHiSlab::<i32>::new(0, 65536).expect("can't allocate");

    for i in 0..50 {
        slab.insert_tagged(i);
    }
    for i in 50..100 {
        slab.insert(i);
    }

    let results = slab.random_tagged_unique(&mut rng, 20);
    assert_eq!(results.len(), 20);

    let mut indices: Vec<u32> = results.iter().map(|(idx, _)| *idx).collect();
    indices.sort();
    indices.dedup();
    assert_eq!(indices.len(), 20);

    for (idx, _) in results {
        assert!(slab.is_tagged(idx));
        assert!(idx < 50);
    }
}

#[test]
fn test_iter_tagged() {
    let slab = TaggedHiSlab::<i32>::new(0, 65536).expect("can't allocate");

    for i in 0..20 {
        if i % 3 == 0 {
            slab.insert_tagged(i * 10);
        } else {
            slab.insert(i * 10);
        }
    }

    let tagged: Vec<_> = slab.iter_tagged().collect();
    assert_eq!(tagged.len(), 7);

    for (idx, val) in tagged {
        assert_eq!(idx % 3, 0);
        assert_eq!(*val, idx as i32 * 10);
    }
}

#[test]
fn test_iter_tagged_mut() {
    let mut slab = TaggedHiSlab::<i32>::new(0, 65536).expect("can't allocate");

    for i in 0..10 {
        if i % 2 == 0 {
            slab.insert_tagged(i);
        } else {
            slab.insert(i);
        }
    }

    for (_, val) in slab.iter_tagged_mut() {
        *val *= 2;
    }

    assert_eq!(slab.get(0), Some(&0)); // 0 * 2 = 0
    assert_eq!(slab.get(1), Some(&1)); // impair, inchangé
    assert_eq!(slab.get(2), Some(&4)); // 2 * 2 = 4
    assert_eq!(slab.get(3), Some(&3)); // impair, inchangé
    assert_eq!(slab.get(4), Some(&8)); // 4 * 2 = 8
}

#[test]
fn test_retain_tagged() {
    let mut slab = TaggedHiSlab::<i32>::new(0, 65536).expect("can't allocate");

    for i in 0..20 {
        slab.insert_tagged(i);
    }

    slab.retain_tagged(|_idx, val| *val % 5 == 0);

    assert_eq!(slab.get(0), Some(&0));
    assert_eq!(slab.get(1), None);
    assert_eq!(slab.get(2), None);
    assert_eq!(slab.get(5), Some(&5));
    assert_eq!(slab.get(10), Some(&10));
    assert_eq!(slab.get(15), Some(&15));

    let count: usize = slab.iter_tagged().count();
    assert_eq!(count, 4);
}

#[test]
fn test_retain_tagged_ttl_simulation() {
    #[derive(Default)]
    struct Entity {
        _value: i32,
        ttl: u32,
    }

    let mut slab = TaggedHiSlab::<Entity>::new(0, 65536).expect("can't allocate");

    for i in 0..10 {
        slab.insert_tagged(Entity {
            _value: i,
            ttl: i as u32,
        });
    }

    slab.retain_tagged(|_idx, entity| {
        if entity.ttl == 0 {
            false
        } else {
            entity.ttl -= 1;
            true
        }
    });

    assert!(slab.get(0).is_none());

    assert_eq!(slab.get(1).unwrap().ttl, 0);
    assert_eq!(slab.get(5).unwrap().ttl, 4);
}

#[test]
fn test_retain_tag_only() {
    let mut slab = TaggedHiSlab::<i32>::new(0, 65536).expect("can't allocate");

    for i in 0..10 {
        slab.insert_tagged(i);
    }

    slab.retain_tag(|_idx, val| *val % 2 == 0);

    for i in 0..10 {
        assert!(slab.is_occupied(i));
    }

    for i in 0..10 {
        if i % 2 == 0 {
            assert!(slab.is_tagged(i), "L'index {} devrait être taggé", i);
        } else {
            assert!(
                !slab.is_tagged(i),
                "L'index {} ne devrait pas être taggé",
                i
            );
        }
    }
}
