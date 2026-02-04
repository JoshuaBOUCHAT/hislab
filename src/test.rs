use std::time::Instant;

use crate::HiSlab;

#[cfg(feature = "rand")]
use rand::SeedableRng;
#[test]
fn test_basic_insert_delete() {
    let count = (1 << 18) + 1; // 262,144 éléments
    let mut slab = HiSlab::<u8>::new();

    println!("🚀 Insertion de {} éléments...", count);

    // 1. Benchmark Insertion
    let start = Instant::now();
    for i in 0..count {
        let idx = slab.insert((i % 255) as u8);

        // Vérification de sécurité optionnelle (à retirer pour le bench pur)
        if idx != i as u32 {
            panic!("Erreur d'indexage à {}: reçu {}", i, idx);
        }
    }
    let duration = start.elapsed();

    println!("✅ Insertion terminée en : {:?}", duration);
    println!("⏱️  Moyenne par insertion : {:?}", duration / count as u32);

    // 2. Vérification de la hiérarchie
    // À 2^18, le lvl1 doit avoir 512 blocs pleins, le lvl2 doit avoir 1 bloc plein.
    println!("--- État de la hiérarchie ---");
    println!("Lvl4 (résumé) : {:08b}", slab.tree.lvl4);

    // 3. Benchmark Suppression
    println!("\n🗑️ Suppression de tous les éléments...");
    let start_remove = Instant::now();
    for i in 0..count {
        slab.remove(i as u32);
    }
    let duration_remove = start_remove.elapsed();

    println!("✅ Suppression terminée en : {:?}", duration_remove);

    // 4. Vérification finale : Tout doit être à zéro
    assert_eq!(slab.tree.lvl4, 0, "La hiérarchie n'est pas revenue à zéro !");
    println!("\n✨ Test réussi : La structure est cohérente et rapide.");
}

#[test]
fn test_stress_lifecycle() {
    let mut slab = HiSlab::<u32>::new();
    let iterations = 10_000;
    let mut indices = Vec::with_capacity(iterations);

    println!("🏗️ Phase 1 : Insertion massive...");
    for i in 0..iterations {
        let idx = slab.insert(i as u32 * 10);
        indices.push(idx);
    }

    // Vérification initiale
    for (i, &idx) in indices.iter().enumerate() {
        assert_eq!(slab.get(idx), Some(&(i as u32 * 10)));
    }

    println!("🗑️ Phase 2 : Fragmentation (suppression d'un élément sur deux)...");
    for i in (0..iterations).step_by(2) {
        let idx = indices[i];
        slab.remove(idx);
        // Vérification que le get renvoie bien None maintenant
        assert!(slab.get(idx).is_none(), "L'index {} devrait être vide", idx);
    }

    println!("🧐 Phase 3 : Vérification de la persistance des autres...");
    for i in (1..iterations).step_by(2) {
        let idx = indices[i];
        assert_eq!(slab.get(idx), Some(&(i as u32 * 10)));
    }

    println!("♻️ Phase 4 : Ré-insertion dans les trous...");
    // On ré-insère 1000 éléments, ils devraient combler les premiers index libres (0, 2, 4...)
    for _ in 0..1000 {
        let new_val = 999_999;
        let idx = slab.insert(new_val);
        // Comme on remplit dans l'ordre, les premiers index pairs devraient revenir
        assert!(
            idx % 2 == 0,
            "L'insertion devrait combler les trous (index pair), reçu {}",
            idx
        );
        assert_eq!(slab[idx], new_val);
    }

    println!("🎯 Phase 5 : Test des accès Panic/Unchecked...");
    let valid_idx = indices[1];

    // Doit fonctionner
    let _ = slab[valid_idx];

    unsafe {
        // Accès direct sans check (extrêmement rapide)
        assert_eq!(*slab.get_unchecked(valid_idx), 1 * 10);
    }

    println!(
        "✨ Test réussi ! Cohérence maintenue après {} itérations.",
        iterations
    );
}
#[test]
fn test_iterator_order_and_stability() {
    let mut slab = HiSlab::<&str>::new();

    // 1. Insertion séquentielle
    slab.insert("Rust"); // Index 0
    slab.insert("is"); // Index 1
    slab.insert("fast"); // Index 2

    println!("Test de l'ordre simple...");
    // Utilise l'itérateur par référence (&slab)
    let mut iter = (&slab).into_iter();

    assert_eq!(iter.next(), Some((0, &"Rust")));
    assert_eq!(iter.next(), Some((1, &"is")));
    assert_eq!(iter.next(), Some((2, &"fast")));
    assert_eq!(iter.next(), None);

    // 2. Test de stabilité après un "trou"
    let mut slab2 = HiSlab::<&str>::new();
    slab2.insert("A"); // 0
    let idx_b = slab2.insert("B"); // 1
    slab2.insert("C"); // 2

    slab2.remove(idx_b); // On crée un trou à l'index 1

    println!("Test apres suppression (doit sauter l'index 1)...");
    let mut iter2 = (&slab2).into_iter();
    assert_eq!(iter2.next(), Some((0, &"A")));
    assert_eq!(iter2.next(), Some((2, &"C")));
    assert_eq!(iter2.next(), None);

    // 3. Test de ré-insertion (L'ordre reste croissant par index)
    slab2.insert("D"); // Devrait prendre l'index 1 (le premier libre)

    println!("Test apres re-insertion (D doit apparaitre entre A et C)...");
    let mut iter3 = (&slab2).into_iter();
    assert_eq!(iter3.next(), Some((0, &"A")));
    assert_eq!(iter3.next(), Some((1, &"D"))); // Stable : index 1 vient avant index 2
    assert_eq!(iter3.next(), Some((2, &"C")));
}

#[test]
fn test_iter_mut() {
    let mut slab = HiSlab::<i32>::new();

    slab.insert(10); // 0
    slab.insert(20); // 1
    slab.insert(30); // 2
    slab.remove(1);  // trou à 1

    // Test de l'itérateur mutable
    for (idx, val) in &mut slab {
        *val += idx as i32;
    }

    assert_eq!(slab.get(0), Some(&10)); // 10 + 0
    assert_eq!(slab.get(1), None);       // supprimé
    assert_eq!(slab.get(2), Some(&32)); // 30 + 2
}

#[test]
fn test_into_iter_partial_consumption() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    // Compteur de drops
    static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

    #[allow(dead_code)]
    struct DropCounter(Arc<()>);
    impl Drop for DropCounter {
        fn drop(&mut self) {
            DROP_COUNT.fetch_add(1, Ordering::SeqCst);
        }
    }

    DROP_COUNT.store(0, Ordering::SeqCst);

    let mut slab = HiSlab::<DropCounter>::new();
    for _ in 0..10 {
        slab.insert(DropCounter(Arc::new(())));
    }

    // Consommation partielle : on ne prend que 3 éléments
    let mut iter = slab.into_iter();
    let _ = iter.next();
    let _ = iter.next();
    let _ = iter.next();
    // iter est droppé ici avec 7 éléments restants

    drop(iter);

    // Tous les 10 éléments doivent avoir été droppés
    assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 10, "Fuite mémoire détectée!");
}

#[test]
fn test_empty_slab_iteration() {
    let slab = HiSlab::<i32>::new();

    // Itérateur sur slab vide
    let mut count = 0;
    for _ in &slab {
        count += 1;
    }
    assert_eq!(count, 0);

    // into_iter sur slab vide
    let slab2 = HiSlab::<i32>::new();
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
    let mut slab = HiSlab::<i32>::new();

    // Slab vide -> None
    assert!(slab.random_occupied(&mut rng).is_none());

    // Ajouter des éléments
    for i in 0..100 {
        slab.insert(i * 10);
    }

    // Doit retourner un élément valide
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
    let mut slab = HiSlab::<i32>::new();

    // Insérer 100 éléments
    for i in 0..100 {
        slab.insert(i);
    }

    // Supprimer les pairs (créer des trous)
    for i in (0..100).step_by(2) {
        slab.remove(i);
    }

    // Vérifier que random_occupied ne retourne que des éléments valides (impairs)
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
    let mut slab = HiSlab::<i32>::new();

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
    let mut slab = HiSlab::<i32>::new();

    for i in 0..100 {
        slab.insert(i);
    }

    // Demander 20 éléments uniques
    let results = slab.random_occupied_unique(&mut rng, 20);
    assert_eq!(results.len(), 20);

    // Vérifier qu'ils sont tous différents
    let mut indices: Vec<u32> = results.iter().map(|(idx, _)| *idx).collect();
    indices.sort();
    indices.dedup();
    assert_eq!(indices.len(), 20, "Les indices doivent être uniques");
}

#[cfg(feature = "rand")]
#[test]
fn test_random_occupied_unique_overflow() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(999);
    let mut slab = HiSlab::<i32>::new();

    for i in 0..10 {
        slab.insert(i);
    }

    // Demander plus que disponible -> retourne tout
    let results = slab.random_occupied_unique(&mut rng, 100);
    assert_eq!(results.len(), 10);
}

#[cfg(feature = "rand")]
#[test]
fn test_count_occupied() {
    let mut slab = HiSlab::<i32>::new();

    assert_eq!(slab.count_occupied(), 0);

    for i in 0..500 {
        slab.insert(i);
    }
    assert_eq!(slab.count_occupied(), 500);

    // Supprimer quelques éléments
    for i in (0..500).step_by(5) {
        slab.remove(i);
    }
    assert_eq!(slab.count_occupied(), 400); // 500 - 100
}

// ============================================================================
// Tests pour la feature "tagged"
// ============================================================================

#[cfg(feature = "tagged")]
#[test]
fn test_insert_tagged_basic() {
    let mut slab = HiSlab::<i32>::new();

    // Insert normal = pas taggé
    let idx1 = slab.insert(100);
    assert!(!slab.is_tagged(idx1));

    // Insert tagged = taggé
    let idx2 = slab.insert_tagged(200);
    assert!(slab.is_tagged(idx2));

    // Les deux sont occupés
    assert!(slab.is_occupied(idx1));
    assert!(slab.is_occupied(idx2));

    // Valeurs correctes
    assert_eq!(slab.get(idx1), Some(&100));
    assert_eq!(slab.get(idx2), Some(&200));
}

#[cfg(feature = "tagged")]
#[test]
fn test_tag_untag() {
    let mut slab = HiSlab::<i32>::new();

    let idx = slab.insert(42);
    assert!(!slab.is_tagged(idx));

    // Tag
    assert!(slab.tag(idx));
    assert!(slab.is_tagged(idx));

    // Re-tag = false (déjà taggé)
    assert!(!slab.tag(idx));

    // Untag
    assert!(slab.untag(idx));
    assert!(!slab.is_tagged(idx));

    // Re-untag = false (pas taggé)
    assert!(!slab.untag(idx));
}

#[cfg(feature = "tagged")]
#[test]
fn test_remove_clears_tag() {
    let mut slab = HiSlab::<i32>::new();

    let idx = slab.insert_tagged(42);
    assert!(slab.is_tagged(idx));

    // Remove doit effacer le tag
    slab.remove(idx);
    assert!(!slab.is_occupied(idx));
    assert!(!slab.is_tagged(idx));
}

#[cfg(feature = "tagged")]
#[test]
fn test_insert_after_tagged_remove() {
    let mut slab = HiSlab::<i32>::new();

    // Insert tagged puis remove
    let idx1 = slab.insert_tagged(100);
    slab.remove(idx1);

    // Le prochain insert (non-taggé) devrait réutiliser l'index
    let idx2 = slab.insert(200);
    assert_eq!(idx1, idx2);

    // Ne doit PAS être taggé
    assert!(!slab.is_tagged(idx2));
}

#[cfg(feature = "tagged")]
#[test]
fn test_for_each_tagged() {
    let mut slab = HiSlab::<i32>::new();

    // Insérer 10 éléments, tagger les pairs
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

#[cfg(all(feature = "tagged", feature = "rand"))]
#[test]
fn test_random_tagged() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut slab = HiSlab::<i32>::new();

    // Slab vide -> None
    assert!(slab.random_tagged(&mut rng).is_none());

    // Insérer des éléments, tagger uniquement les impairs
    for i in 0..100 {
        if i % 2 == 1 {
            slab.insert_tagged(i);
        } else {
            slab.insert(i);
        }
    }

    // Random tagged doit toujours retourner un impair
    for _ in 0..50 {
        let (idx, val) = slab.random_tagged(&mut rng).unwrap();
        assert!(slab.is_tagged(idx));
        assert_eq!(idx % 2, 1);
        assert_eq!(*val, idx as i32);
    }
}

#[cfg(all(feature = "tagged", feature = "rand"))]
#[test]
fn test_count_tagged() {
    let mut slab = HiSlab::<i32>::new();

    assert_eq!(slab.count_tagged(), 0);

    for i in 0..100 {
        if i % 3 == 0 {
            slab.insert_tagged(i);
        } else {
            slab.insert(i);
        }
    }

    // 0, 3, 6, 9, ..., 99 = 34 éléments
    assert_eq!(slab.count_tagged(), 34);

    // Untag quelques-uns
    slab.untag(0);
    slab.untag(3);
    assert_eq!(slab.count_tagged(), 32);
}

#[cfg(all(feature = "tagged", feature = "rand"))]
#[test]
fn test_random_tagged_unique() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(789);
    let mut slab = HiSlab::<i32>::new();

    for i in 0..50 {
        slab.insert_tagged(i);
    }
    for i in 50..100 {
        slab.insert(i);
    }

    // Demander 20 éléments taggés uniques
    let results = slab.random_tagged_unique(&mut rng, 20);
    assert_eq!(results.len(), 20);

    // Vérifier qu'ils sont tous taggés et différents
    let mut indices: Vec<u32> = results.iter().map(|(idx, _)| *idx).collect();
    indices.sort();
    indices.dedup();
    assert_eq!(indices.len(), 20);

    for (idx, _) in results {
        assert!(slab.is_tagged(idx));
        assert!(idx < 50);
    }
}

#[cfg(feature = "tagged")]
#[test]
fn test_iter_tagged() {
    let mut slab = HiSlab::<i32>::new();

    // Insérer 20 éléments, tagger ceux divisibles par 3
    for i in 0..20 {
        if i % 3 == 0 {
            slab.insert_tagged(i * 10);
        } else {
            slab.insert(i * 10);
        }
    }

    // iter_tagged doit retourner: 0, 3, 6, 9, 12, 15, 18 (7 éléments)
    let tagged: Vec<_> = slab.iter_tagged().collect();
    assert_eq!(tagged.len(), 7);

    for (idx, val) in tagged {
        assert_eq!(idx % 3, 0);
        assert_eq!(*val, idx as i32 * 10);
    }
}

#[cfg(feature = "tagged")]
#[test]
fn test_iter_tagged_mut() {
    let mut slab = HiSlab::<i32>::new();

    for i in 0..10 {
        if i % 2 == 0 {
            slab.insert_tagged(i);
        } else {
            slab.insert(i);
        }
    }

    // Modifier tous les taggés: doubler leur valeur
    for (_, val) in slab.iter_tagged_mut() {
        *val *= 2;
    }

    // Vérifier: les pairs (taggés) sont doublés, les impairs inchangés
    assert_eq!(slab.get(0), Some(&0));   // 0 * 2 = 0
    assert_eq!(slab.get(1), Some(&1));   // impair, inchangé
    assert_eq!(slab.get(2), Some(&4));   // 2 * 2 = 4
    assert_eq!(slab.get(3), Some(&3));   // impair, inchangé
    assert_eq!(slab.get(4), Some(&8));   // 4 * 2 = 8
}

#[cfg(feature = "tagged")]
#[test]
fn test_retain_tagged() {
    let mut slab = HiSlab::<i32>::new();

    // Insérer 20 éléments, tous taggés
    for i in 0..20 {
        slab.insert_tagged(i);
    }

    // Garder uniquement les multiples de 5 (0, 5, 10, 15)
    slab.retain_tagged(|_idx, val| *val % 5 == 0);

    // Les autres doivent avoir été supprimés
    assert_eq!(slab.get(0), Some(&0));
    assert_eq!(slab.get(1), None);  // supprimé
    assert_eq!(slab.get(2), None);  // supprimé
    assert_eq!(slab.get(5), Some(&5));
    assert_eq!(slab.get(10), Some(&10));
    assert_eq!(slab.get(15), Some(&15));

    // Seulement 4 éléments restants (taggés)
    let count: usize = slab.iter_tagged().count();
    assert_eq!(count, 4);
}

#[cfg(feature = "tagged")]
#[test]
fn test_retain_tagged_ttl_simulation() {
    // Simule un système de TTL
    struct Entity {
        value: i32,
        ttl: u32,
    }

    let mut slab = HiSlab::<Entity>::new();

    // Créer des entités avec différents TTL
    for i in 0..10 {
        slab.insert_tagged(Entity { value: i, ttl: i as u32 });
    }

    // Simuler le passage du temps: décrémenter TTL et supprimer les expirés
    slab.retain_tagged(|_idx, entity| {
        if entity.ttl == 0 {
            false // Expiré, supprimer
        } else {
            entity.ttl -= 1;
            true // Garder
        }
    });

    // L'entité 0 (TTL=0) doit être supprimée
    assert!(slab.get(0).is_none());

    // Les autres doivent exister avec TTL décrémenté
    assert_eq!(slab.get(1).unwrap().ttl, 0);  // était 1, maintenant 0
    assert_eq!(slab.get(5).unwrap().ttl, 4);  // était 5, maintenant 4
}

#[cfg(feature = "tagged")]
#[test]
fn test_retain_tag_only() {
    let mut slab = HiSlab::<i32>::new();

    // Insérer 10 éléments taggés
    for i in 0..10 {
        slab.insert_tagged(i);
    }

    // Détagger les impairs (mais ne pas les supprimer)
    slab.retain_tag(|_idx, val| *val % 2 == 0);

    // Tous les éléments existent encore
    for i in 0..10 {
        assert!(slab.is_occupied(i));
    }

    // Mais seuls les pairs sont taggés
    for i in 0..10 {
        if i % 2 == 0 {
            assert!(slab.is_tagged(i), "L'index {} devrait être taggé", i);
        } else {
            assert!(!slab.is_tagged(i), "L'index {} ne devrait pas être taggé", i);
        }
    }
}
