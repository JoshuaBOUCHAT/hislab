# HiSlab

A high-performance slab allocator using hierarchical bitmaps for O(1) slot allocation and deallocation. Backed by anonymous `mmap` with demand paging.

## Features

- **O(1) insert and remove** — hierarchical bitmap finds a free slot in constant time regardless of fragmentation
- **mmap-backed storage** — virtual memory reserved upfront, physical pages committed on demand; pointer never moves
- **Huge page support** — initial capacity pre-faulted with `MADV_POPULATE_WRITE | MADV_HUGEPAGE`
- **Stable `u32` indices** — inserted elements keep their index until removed
- **SIMD optimized** — bitmap operations use AVX-512, AVX2, NEON, or scalar at compile time
- **Tagging** — `TaggedHiSlab` maintains a second bitmap to track a subset of elements
- **Buffer pool** — `BufferSlab<N>` for fixed-size raw byte buffers with RAII guards
- **Random sampling** — uniform random selection over occupied/tagged slots (feature `rand`)

## Crates

```toml
[dependencies]
hislab = "0.2"

# Optional: random sampling
hislab = { version = "0.2", features = ["rand"] }
```

## HiSlab

```rust
use hislab::HiSlab;

// Reserve virtual space for 65 536 slots, pre-fault the first 1 024
let mut slab = HiSlab::<i32>::new(1024, 65536).unwrap();

// insert() returns a stable u32 index
let idx = slab.insert(42);
assert_eq!(slab[idx], 42);

// remove() returns the value
let val = slab.remove(idx);
assert_eq!(val, Some(42));
assert!(slab.get(idx).is_none());

// Freed slots are reused
let idx2 = slab.insert(99);
assert_eq!(idx2, idx);
```

### `new(initial_capacity, virtual_capacity)`

| Parameter | Description |
|---|---|
| `initial_capacity` | Slots pre-faulted immediately (zero page faults on first accesses) |
| `virtual_capacity` | Maximum number of slots (hard cap — `insert` panics if exceeded) |

Both are `u32`. `initial_capacity` must be `<= virtual_capacity`.

### Core methods

| Method | Description |
|---|---|
| `insert(val) -> u32` | Insert a value, returns its stable index |
| `remove(idx) -> Option<T>` | Remove and return the value, or `None` if empty |
| `get(idx) -> Option<&T>` | Borrow the value at index |
| `get_mut(idx) -> Option<&mut T>` | Mutably borrow the value at index |
| `is_occupied(idx) -> bool` | Check if a slot is occupied |
| `slab[idx]` | Panics if the slot is empty |

### Iteration

```rust
// Shared reference
for (idx, val) in &slab {
    println!("{idx}: {val}");
}

// Mutable reference
for (idx, val) in &mut slab {
    *val += 1;
}

// Consuming (moves values out)
for (idx, val) in slab {
    println!("{idx}: {val}");
}

// Callback — avoids iterator overhead
slab.for_each_occupied(|idx, val| {
    println!("{idx}: {val}");
});
```

### Unsafe fast access

```rust
// Caller guarantees the slot is occupied
unsafe {
    let val = slab.get_unchecked(idx);
    let val_mut = slab.get_unchecked_mut(idx);
}
```

---

## TaggedHiSlab

`TaggedHiSlab<T>` wraps a `HiSlab` and maintains a second bitmap to mark a subset of elements as *tagged*. All base slab operations are identical.

```rust
use hislab::TaggedHiSlab;

let mut slab = TaggedHiSlab::<u32>::new(0, 1024).unwrap();

let a = slab.insert(1);       // not tagged
let b = slab.insert_tagged(2); // inserted and tagged in one call

slab.tag(a);         // tag an existing element (returns false if already tagged)
slab.untag(b);       // remove tag
assert!(slab.is_tagged(a));
assert!(!slab.is_tagged(b));
```

### Tagging methods

| Method | Description |
|---|---|
| `insert_tagged(val) -> u32` | Insert and tag in one step |
| `tag(idx) -> bool` | Tag an existing element |
| `untag(idx) -> bool` | Remove tag from an element |
| `is_tagged(idx) -> bool` | Check if an element is tagged |

### Iterating tagged elements

```rust
// Callback — fastest
slab.for_each_tagged(|idx, val| { /* … */ });

// Iterator
for (idx, val) in slab.iter_tagged() { /* … */ }
for (idx, val) in slab.iter_tagged_mut() { /* … */ }
```

### Bulk operations on tagged elements

```rust
// Remove tagged elements that fail the predicate
slab.retain_tagged(|idx, val| val.ttl > 0);

// Untag (but keep in slab) elements that fail the predicate
slab.retain_tag(|idx, val| val.is_active());
```

---

## BufferSlab

`BufferSlab<N>` is a pool of fixed-size raw byte buffers. Slots are managed through RAII `AutoGuard`s that free automatically on drop.

```rust
use hislab::buffer_slab::BufferSlab;

// Pre-fault 4 slots, reserve space for 128
let pool = BufferSlab::<4096>::new(4, 128).unwrap();

let mut g0 = pool.allocate();          // panics if pool is full
let g1 = pool.try_allocate().unwrap(); // returns None if pool is full

g0.as_slice_mut().fill(0xAB);
assert_eq!(g0.index(), 0);
assert_eq!(g1.index(), 1);

// Drop releases the slot back to the pool
drop(g0);
drop(g1);
```

### `AutoGuard` API

| Method | Description |
|---|---|
| `as_slice() -> &[u8]` | Read-only view of the N bytes |
| `as_slice_mut() -> &mut [u8]` | Mutable view of the N bytes |
| `as_ptr() -> *mut u8` | Raw pointer (valid while guard is alive) |
| `index() -> u32` | Slot index in the pool |

Multiple guards can coexist simultaneously. `BufferSlab` is `!Send + !Sync`.

---

## Random sampling (feature `rand`)

Available on both `HiSlab` and `TaggedHiSlab`.

```rust
use hislab::HiSlab;
use rand::thread_rng;

let mut rng = thread_rng();

// Pick one random occupied element
if let Some((idx, val)) = slab.random_occupied(&mut rng) { /* … */ }
if let Some((idx, val)) = slab.random_occupied_mut(&mut rng) { /* … */ }

// Pick N elements with possible duplicates
let sample = slab.random_occupied_many(&mut rng, 10);

// Pick N unique elements
let unique = slab.random_occupied_unique(&mut rng, 5);

// Count occupied slots
let n = slab.count_occupied();
```

`TaggedHiSlab` adds the same methods for the tagged subset:
`random_tagged`, `random_tagged_mut`, `random_tagged_many`, `random_tagged_unique`, `count_tagged`.

---

## Architecture

HiSlab uses a 4-level hierarchical bitmap:

```
lvl4 (u8)       → 8 bits summarize lvl3
lvl3 (BitBlock) → 512 bits summarize lvl2
lvl2 (BitBlock) → 512 bits summarize lvl1
lvl1 (BitBlock) → 512 bits track actual data slots
```

Each `BitBlock` is 512 bits (8 × u64), aligned to 64 bytes. Insert descends the tree to find a free slot; fullness/emptiness propagates upward.

## SIMD support

Bitmap operations select exactly one implementation path at compile time — no runtime dispatch overhead.

| Architecture | Feature | Used for |
|---|---|---|
| x86_64 | AVX-512 | 512-bit comparisons |
| x86_64 | AVX2 | 256-bit comparisons |
| aarch64 | NEON | 128-bit comparisons |
| Fallback | Scalar | Loop over 8 × u64 |

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
