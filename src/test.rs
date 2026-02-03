use std::time::Instant;

use crate::HiSlab;
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
    println!("Lvl4 (résumé) : {:08b}", slab.lvl4);

    // 3. Benchmark Suppression
    println!("\n🗑️ Suppression de tous les éléments...");
    let start_remove = Instant::now();
    for i in 0..count {
        slab.remove(i as u32);
    }
    let duration_remove = start_remove.elapsed();

    println!("✅ Suppression terminée en : {:?}", duration_remove);

    // 4. Vérification finale : Tout doit être à zéro
    assert_eq!(slab.lvl4, 0, "La hiérarchie n'est pas revenue à zéro !");
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
