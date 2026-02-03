# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
cargo build           # Build the library
cargo test            # Run all tests
cargo test <name>     # Run a specific test (e.g., cargo test test_basic_insert_delete)
cargo check           # Fast compilation check
```

## Architecture

HiSlab is a high-performance slab allocator using a hierarchical bitmap structure for O(1) slot allocation/deallocation.

### Core Structure

**HiSlab<T>** (`lib.rs`) - Main container storing:
- `data: Vec<T>` - Actual stored elements
- `lvl1-4` - Hierarchical bitmap levels tracking free/occupied slots

**BitBlock** (`bit_block.rs`) - 512-bit bitmap (8×u64) with SIMD-optimized operations:
- `is_full()` - Check if all bits set (AVX-512/AVX2/NEON/scalar)
- `find_first_free()` - Find first zero bit (AVX-512/AVX2/scalar)
- `set_bit_and_check_full()` / `clear_bit_and_was_full()` - Atomic-style bit ops with fullness tracking

### Hierarchy Design

Each level summarizes 512 entries from the level below:
- **lvl1**: Each BitBlock tracks 512 data slots
- **lvl2**: Each bit tracks one lvl1 BitBlock (512 blocks = 262K slots)
- **lvl3**: Each bit tracks one lvl2 BitBlock
- **lvl4**: u8 summary of lvl3 (only needs 5 bits for 2^32 capacity)

Insert finds free slot by descending hierarchy; fullness propagates upward on insert, emptiness propagates upward on remove.

### SIMD Conditional Compilation

BitBlock uses `#[cfg]` attributes with `not()` guards to select exactly one implementation path at compile time (AVX-512 > AVX2 > NEON > scalar fallback).
