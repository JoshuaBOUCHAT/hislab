# HiSlab

A high-performance slab allocator using hierarchical bitmaps for O(1) slot allocation and deallocation.

## Features

- **O(1) insert and remove** - Hierarchical bitmap allows constant-time free slot lookup
- **SIMD optimized** - Automatic use of AVX-512, AVX2, or NEON when available
- **Cache-friendly** - 64-byte aligned bitmap blocks fit exactly in a cache line
- **Stable indices** - Inserted elements keep their index until removed
- **Efficient iteration** - Skip empty regions using bitmap scanning

## Usage

```rust
use hislab::HiSlab;

let mut slab = HiSlab::new();

// Insert returns a stable index
let idx = slab.insert("hello");
assert_eq!(slab[idx], "hello");

// Remove frees the slot for reuse
slab.remove(idx);
assert!(slab.get(idx).is_none());

// New inserts reuse freed slots
let idx2 = slab.insert("world");
assert_eq!(idx2, idx); // Same slot reused
```

## Iteration

```rust
use hislab::HiSlab;

let mut slab = HiSlab::new();
slab.insert("a");
slab.insert("b");
slab.insert("c");

// Iterate over occupied slots
for (idx, value) in &slab {
    println!("{}: {}", idx, value);
}

// Or use for_each_occupied for maximum performance
slab.for_each_occupied(|idx, value| {
    println!("{}: {}", idx, value);
});
```

## Architecture

HiSlab uses a 4-level hierarchical bitmap:

```
lvl4 (u8)      →  8 bits summarize lvl3
lvl3 (BitBlock) →  512 bits summarize lvl2
lvl2 (BitBlock) →  512 bits summarize lvl1
lvl1 (BitBlock) →  512 bits track actual slots
```

Each `BitBlock` is 512 bits (8 × u64), aligned to 64 bytes for cache efficiency.

**Insert**: Descend the hierarchy to find a free slot in O(1).
**Remove**: Clear the bit and propagate "not full" status upward.

## SIMD Support

Bitmap operations automatically use the best available instruction set:

| Architecture | Feature | Used for |
|--------------|---------|----------|
| x86_64 | AVX-512 | 512-bit comparisons |
| x86_64 | AVX2 | 256-bit comparisons |
| aarch64 | NEON | 128-bit comparisons |
| Fallback | Scalar | Loop over 8 × u64 |

No runtime feature detection - the optimal path is selected at compile time.

## Performance

Typical performance on modern x86_64 with AVX2:

| Operation | Time |
|-----------|------|
| Insert | ~15-25 ns |
| Remove | ~10-15 ns |
| Get | ~2-5 ns |

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
