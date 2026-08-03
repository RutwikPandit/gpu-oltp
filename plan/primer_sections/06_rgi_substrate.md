# Section 06 — The RGI Substrate As This Project Uses It

> **STANDING RULE (read this before touching anything):**
> **RGI source is NEVER modified by this project.** The RobustGPUIndexing
> library at `c:\Users\rutwi\OneDrive\Documents\CMU\Research_spring26\Research_spring26\RobustGPUIndexing\`
> is an upstream dependency owned by its author (Hyoungjoo Kim, CMU). Every
> finding that suggests an RGI-side improvement (e.g., an inline-two-slice
> node layout, a mid-kernel drain entry point) is written up as a separate
> one-page proposal for the RGI author — it is **never** implemented by
> editing RGI headers. All adaptation happens in this project's wrapper
> layer (`gpu_oltp\engine\*.cu`), which composes RGI's **public** host and
> device APIs only. WP1's contract section, WP6's contract section, and the
> header comment of `rgi_persist_engine.cu` (line 17: "RGI source is NOT
> modified") all restate this rule; it is load-bearing for the
> collaboration, not a style preference.

---

## Table of contents

1. [Why this section exists](#why-this-section-exists)
2. [Orientation: the exact instantiation and the mental model](#orientation)
3. [File: include/nodes.hpp — the tile-size dispatch](#file-nodeshpp)
4. [File: include/hashtable_node_subwarp.hpp — the 128-byte node](#file-hashtable_node_subwarphpp)
5. [File: include/suffix_node_subwarp.hpp — the suffix chain](#file-suffix_node_subwarphpp)
6. [File: include/compute_hash.hpp — hashing](#file-compute_hashhpp)
7. [File: include/gpu_chainhashtable.hpp — the table itself](#file-gpu_chainhashtablehpp)
8. [File: include/kernels.hpp — batch_kernel and the device_func contract](#file-kernelshpp)
9. [File: include/simple_slab_alloc.hpp — the slab allocator](#file-simple_slab_allochpp)
10. [File: include/simple_debra_reclaim.hpp — DEBRA reclamation](#file-simple_debra_reclaimhpp)
11. [The exact API surface this project calls](#the-exact-api-surface-this-project-calls)
12. [Assumptions the project makes about RGI](#assumptions-the-project-makes-about-rgi)
13. [Safe-extension guide (WP1 / WP5 / WP6)](#safe-extension-guide)

All file paths below are relative to
`RobustGPUIndexing\include\` unless prefixed with `gpu_oltp\`. Every line
number cites the file state read on 2026-06-12; if RGI is updated upstream,
re-verify the citations before relying on them (and see assumption A12 on
pinning the validated commit hash).

Companion references (read, do not copy from):
- `RobustGPUIndexing\docs\PROJECT_UNDERSTANDING.md` — the RGI author's own
  description of the library.
- `gpu_oltp\comp_arch_db_explainer\FULL_PROJECT_EXPLAINER.md` Part III §13
  (lines 527–593) — this project's narrative account of what was inherited
  from RGI; §13 describes the **warp** node layout in its diagram, which is
  NOT the layout this project's instantiation uses in memory (see the
  pitfall in the node-file section below).

---

## Why this section exists

Future subagents will build **above** RGI: new wrapper entry points, the
WP1 small-key fast path, the WP5 persistent-kernel v2, the WP6 enumeration
kernel. All of these either call RGI's public device API from kernels this
project owns, or — in exactly one sanctioned case (WP6) — read RGI's
in-memory node layout directly without calling RGI code. Both activities
are safe only if the agent knows precisely:

1. **What RGI guarantees** (and at which line of which header the guarantee
   is implemented), so wrapper code does not silently depend on behavior
   RGI never promised.
2. **What this project already assumes about RGI**, so a change in either
   codebase that breaks an assumption is recognized as a breaking change
   rather than discovered as a heisenbug.
3. **The do-not-modify rule**, restated above. The single most damaging
   thing a subagent can do in this repository pair is "fix" something
   inside `RobustGPUIndexing\include\` — it would fork the substrate,
   invalidate every published measurement's provenance, and violate the
   working agreement with the library's author. If RGI appears to need a
   change, the correct outputs are (a) a workaround in the wrapper, and
   (b) a written upstream proposal. Nothing else.

### What a subagent may safely assume without re-reading RGI

- The instantiation is fixed:
  `GpuHashtable::gpu_chainhashtable<simple_slab_allocator<128>, simple_debra_reclaimer<>, 16>`
  (declared at `gpu_oltp\engine\rgi_oltp_engine.cu:38–40` and
  `gpu_oltp\engine\rgi_persist_engine.cu:49–51`). Tile size 16 selects the
  `_subwarp` node variants. Nothing in this project instantiates the
  Masstree, cuckoo, or extendible-hash structures, the warp (tile 32)
  variants, the bump allocator, or the dynamic stack.
- Keys are arrays of `uint32_t` "slices"; this project stores every SQL
  `bigint` as **two** slices (`max_key_length = 2`,
  `d_key_lengths = nullptr`), so **every key currently takes the suffix
  path** (one extra dependent 128 B load per find, one extra slab
  allocation per insert, ~134 B per entry). WP1 changes this to
  per-request lengths; until it lands, assume two slices everywhere.
- Values are `uint32_t`, and `0xFFFFFFFF` is reserved (RGI's
  `invalid_value` not-found sentinel). The 32-bit value width is
  structural, not stylistic — see assumption A3.
- `cooperative_insert(..., update_if_exists = true)` **cannot fail**. The
  engine's transactional atomicity (validate-then-apply) is built on this;
  see assumption A4 before changing anything near `rgi_stage_commit`.
- Erase requires DEBRA reclamation, and DEBRA only frees memory when its
  drain runs — which, in stock RGI, happens at **kernel exit**. A
  persistent kernel never exits; that is why v1 of the persistent engine
  supports INSERT + FIND only (assumption A8, WP5).
- The launch configuration (block size 128, 8 blocks/SM ceiling) is fixed
  by RGI constants, and the project's `-maxrregcount=64` build flag makes
  the occupancy ceiling land exactly on Ada's 64 K-register file. Changing
  build flags or RGI's constants silently changes occupancy (assumption
  A7).

Everything else — what each template flag means, where the suffix branch
is, what the lock protocol orders — is documented file by file below.

---

## Orientation

### The instantiation, spelled out

```cpp
// gpu_oltp\engine\rgi_oltp_engine.cu:38-40 (identical in rgi_persist_engine.cu:49-51)
using slab_t  = simple_slab_allocator<128>;                       // 128-byte slabs
using debra_t = simple_debra_reclaimer<>;                         // 32768 B limbo buffer per block
using table_t = GpuHashtable::gpu_chainhashtable<slab_t, debra_t, 16>;
```

The three template arguments of `gpu_chainhashtable<Allocator, Reclaimer,
tile_size>` (`gpu_chainhashtable.hpp:43–46`):

| Parameter   | This project's choice          | What it controls |
|-------------|--------------------------------|------------------|
| `Allocator` | `simple_slab_allocator<128>`   | Where chain-overflow nodes and suffix nodes live; defines the 32-bit slab-index pointer convention. |
| `Reclaimer` | `simple_debra_reclaimer<>`     | Safe memory reclamation for erased nodes; also (via `block_size_ = 128`) fixes the batch-kernel block size. |
| `tile_size` | `16`                           | Cooperative-group tile width; `static_assert(tile_size == 32 \|\| tile_size == 16)` at `gpu_chainhashtable.hpp:53`. 16 selects `hashtable_node_subwarp` / `suffix_node_subwarp` via `nodes.hpp`. |

### The one-paragraph mental model

The table is a fixed array of 128-byte **bucket head nodes** in plain
`cudaMalloc` memory (`d_table_`), plus a slab pool from which 128-byte
**overflow nodes** (chain continuation) and 128-byte **suffix nodes**
(full-key + value storage for multi-slice keys) are allocated. A node
holds up to 15 `(key, value)` entries plus one metadata word and one
next-pointer word. All node operations are **tile-cooperative**: 16 lanes
each own one 8-byte element of the node, so loading a node is one
coalesced 128 B transaction and searching it is one ballot. Host-side
batch APIs (`find` / `insert` / `erase` / `mixed_batch`) wrap the device
API in `batch_kernel`, which assigns one request per lane and drains them
through a ballot work queue. Writers lock the bucket head (one `fetch_or`
by one lane); readers can be latch-free if the `concurrent` template flag
says so. Erase retires nodes into DEBRA limbo bags; the bags drain when
epochs advance and, finally, at kernel exit.

### Vocabulary used throughout this section

| Term | Meaning | Defined at |
|------|---------|-----------|
| **slice** | one `uint32_t` unit of a key (`key_slice_type`, `gpu_chainhashtable.hpp:49`) | — |
| **tile** | a `cooperative_groups::thread_block_tile<16>`; the unit of cooperation | `kernels.hpp:46` |
| **lane** | one thread within a tile; `tile.thread_rank()` ∈ [0, 16) | — |
| **node** | 128 B = 16 lanes × 8 B `{key, value}` element | `hashtable_node_subwarp.hpp:29–34` |
| **head node** | the node embedded in the bucket array `d_table_` at `bucket_index` | `hashtable_node_subwarp.hpp:48` (head bit) |
| **aux node** | a chain-overflow node living in a slab | `gpu_chainhashtable.hpp:243–252` |
| **suffix node** | slab node holding `{key_length, value, slices...}` for a multi-slice key | `suffix_node_subwarp.hpp` |
| **more_key** | runtime flag `key_length > 1`: this key uses the suffix path | `gpu_chainhashtable.hpp:164` |
| **hash tag** | second independent 32-bit hash stored in the node's key field when `use_hash_tag && more_key` | `gpu_chainhashtable.hpp:165–169` |
| **slab index** | 32-bit pointer = slab ordinal within the pool; `address(p) = pool + p*128` | `simple_slab_alloc.hpp:132–134` |
| **ballot queue** | `tile.ballot(task_exists)` → `__ffs` pick → `shfl` broadcast → cooperative exec | `kernels.hpp:63–72` |
| **limbo bag** | per-block shared-memory list of retired slab indices awaiting an epoch | `simple_debra_reclaim.hpp:296–303` |
| `DEVICE_QUALIFIER` | `__device__ __forceinline__` | `macros.hpp:20` |

### Where this project touches RGI, at a glance

```
                gpu_oltp (this project)                      RobustGPUIndexing (NEVER modified)
  +--------------------------------------------+   +--------------------------------------------+
  | engine/rgi_oltp_engine.cu                  |   | gpu_chainhashtable.hpp                     |
  |   table->insert/find/erase  --------------------> host batch APIs -> kernels.hpp            |
  | engine/rgi_persist_engine.cu               |   |   batch_kernel (ballot queue)              |
  |   own persistent kernel calling            |   |   device_func structs                      |
  |   table.cooperative_insert/find  ---------------> device APIs (cooperative_*)               |
  | engine/rgi_sweep.cu, rgi_prof*.cu          |   | hashtable_node_subwarp.hpp (node + lock)   |
  |   host batch APIs for benchmarks           |   | suffix_node_subwarp.hpp (suffix chain)     |
  | [future WP6] enumeration kernel            |   | compute_hash.hpp (hashing)                 |
  |   reads node layout directly (no RGI call; |   | simple_slab_alloc.hpp (slab pool)          |
  |   mirrors validate_nodes_task decoding) ---+--> | simple_debra_reclaim.hpp (DEBRA)           |
  +--------------------------------------------+   +--------------------------------------------+
```

---

## File: nodes.hpp

**Path:** `RobustGPUIndexing\include\nodes.hpp` (39 lines)

**Purpose:** a single dispatch header that maps the tile size to the
concrete node implementation. There are two implementations of every node
type: a `_warp` variant (tile 32, one `uint32_t` per lane) and a
`_subwarp` variant (tile 16, one 8-byte `{key,value}` element per lane).
Both occupy 128 bytes; they differ in lane-to-byte mapping.

**Walkthrough:**

- Lines 18–23 include all six node headers (masstree / hashtable / suffix
  × warp / subwarp).
- Lines 25–28: `masstree_node<tile_type, allocator_type>` =
  `std::conditional_t<tile_type::size() == 32, masstree_node_warp<…>,
  masstree_node_subwarp<…>>`. Unused by this project.
- Lines 30–33: `hashtable_node<tile_type, allocator_type>` — same
  conditional. **With this project's tile 16, this resolves to
  `hashtable_node_subwarp`.** This alias is what `gpu_chainhashtable.hpp`
  names `node_type` inside every device function (e.g.,
  `gpu_chainhashtable.hpp:160`).
- Lines 35–38: `suffix_node<tile_type, allocator_type>` — resolves to
  `suffix_node_subwarp` for this project (named `suffix_type` at
  `gpu_chainhashtable.hpp:161`).

**Pitfalls:**

- The dispatch is on `tile_type::size()`, a compile-time property of the
  cooperative-groups tile, not on the table's `tile_size` parameter
  directly. Any project-owned kernel (persistent engine, WP6 enumeration)
  that builds its own tile **must** use `cg::tiled_partition<16>` to get
  the same node types the table was built with. Partitioning at 32 would
  compile against the warp layout and read garbage. The persistent kernel
  does this correctly with `#define TILE 16` at
  `rgi_persist_engine.cu:55,94`.
- Documentation drift hazard: prose elsewhere in this repo (explainer §13,
  `FULL_PROJECT_EXPLAINER.md:531–535`) draws the **warp** layout
  (lanes 0–14 keys, lane 15 metadata, lanes 16–30 values, lane 31 next).
  That is correct for `hashtable_node_warp` only. The bytes in memory for
  this project's instantiation follow the **subwarp interleaved** layout
  documented in the next section. WP6's enumeration kernel must use the
  subwarp layout.

---

## File: hashtable_node_subwarp.hpp

**Path:** `RobustGPUIndexing\include\hashtable_node_subwarp.hpp` (405 lines)

**Purpose:** the 128-byte hash-table node for tile size 16 — the data
structure unit of everything this project stores. Implements: the lane
layout, the metadata word, tile-cooperative load/store with selectable
atomicity and ordering, the per-bucket lock protocol, ballot-based key
matching, and the in-register mutators (`insert`, `update`, `erase`,
`merge`).

### Layout

A node is 16 elements of `struct __align__(8) elem_type { key_type key;
value_type value; }` (lines 29–32) — 16 lanes × 8 B = 128 B. `node_width
= 16`, `capacity = 15` (lines 34–35); a `static_assert` pins the tile
width to the node width (line 36). The authoritative layout comment is at
lines 364–375.

```
            ONE 128-BYTE NODE, AS 16 LANES x 8 BYTES (hashtable_node_subwarp)

 lane:        0          1          2                  14            15
          +----------+----------+----------+  ...  +----------+----------------+
 .key     |  key[0]  |  key[1]  |  key[2]  |       |  key[14] |  METADATA (32b)|
 .value   |  val[0]  |  val[1]  |  val[2]  |       |  val[14] |  NEXT INDEX    |
          +----------+----------+----------+  ...  +----------+----------------+
           <-- entry slots 0..14: up to 15 (key,value) pairs -->  <- lane 15 ->

 IN LINEAR MEMORY (32 consecutive uint32 words, as the WP6 enumeration
 kernel or any raw reader will see them — INTERLEAVED key/value pairs):

   word: 0    1    2    3    4    5   ...  28    29    30        31
        k[0] v[0] k[1] v[1] k[2] v[2] ... k[14] v[14] METADATA  NEXT

 metadata_lane_ = 15 (the .key of lane 15), next_ptr_lane_ = 15 (the
 .value of lane 15) — lines 380-381.
```

Each entry slot's **key field** holds either a real key slice (single-slice
key) or a **hash tag** (multi-slice key, `use_hash_tag = true`); its
**value field** holds either the inline value or a **suffix-node slab
index**. Which interpretation applies is recorded per-slot in the metadata
word's suffix bits.

### The metadata word (32 bits, lane 15's key field)

Defined by constants at lines 380–401, documented in the comment block at
lines 367–375:

```
 bit:  31 30 29 | 28 27 26 25 24 23 | 22 ........ 9  8 | 7  | 6  | 5  | 4  | 3  2  1  0
      +---------+-------------------+------------------+----+----+----+----+-------------+
      | (empty) |   local_depth:6   | suffix bits 14..0| GB | HD | NX | LK |  num_keys:4 |
      +---------+-------------------+------------------+----+----+----+----+-------------+
        unused    extendible-hash     suffix_bits_       |    |    |    |     entry count
        (3 bits)  only; always 0      offset_ = 8        |    |    |    |     (0..15),
                  for the chain HT    (lines 393-395):   |    |    |    |     offset 0,
                  (lines 396-398)     bit (8+i) == 1     |    |    |    |     4 bits
                                      means slot i's     |    |    |    |     (382-384)
                                      value is a SUFFIX  |    |    |    |
                                      NODE INDEX         |    |    |    +-- is_locked (bit 4,
                                                         |    |    |        lines 385-386)
                                                         |    |    +------- has_next  (bit 5,
                                                         |    |             lines 387-388)
                                                         |    +------------ is_head   (bit 6,
                                                         |                  lines 389-390)
                                                         +----------------- is_garbage(bit 7,
                                                                            lines 391-392)
```

Two deliberate design facts encoded by `static_assert`s:

- `num_keys_offset_ == 0` (line 400) makes `metadata_++` equivalent to
  `num_keys++` (used by `insert` at line 251) and `metadata_--` to
  `num_keys--` (used by `erase` at line 289), as the comment at line 400
  says — valid only while the 4-bit field does not over/underflow, which
  the `assert(!is_full())` / `assert(location < num_keys())` guards
  enforce in debug builds.
- `max_num_keys_ = node_width - 1 = 15 == capacity` (lines 399, 401).
- `is_garbage` and `is_locked` are meaningful only on head nodes
  (comment line 376); `local_depth` belongs to the extendible hash table
  and is always zero in this project's chain table.

### Construction and initialization

- **Constructors** (lines 37–40): both take `(tile, allocator)` by
  reference and store references (`tile_`, `allocator_` members, lines
  361–362); the second additionally takes the node's `node_index_`. A node
  object is *register-resident state plus identity* — constructing one
  performs **no memory access**. Loading is always a separate explicit
  call. Consequence: the copy-assignment operator (lines 305–311) copies
  `node_index_`, `lane_elem_`, `metadata_` but **not** the tile/allocator
  references (it cannot; they are references). Assigning across different
  tiles is meaningless — never do it.
- **`initialize_empty(bool is_head, size_type local_depth = 0, bool
  is_locked = false)`** (lines 41–52): zeroes the lane element, builds a
  metadata word with `num_keys = 0`, `has_next = false`,
  `is_garbage = false` (lines 43–47), ORs in the head bit (line 48) and
  optionally the lock bit (line 49) and local depth (line 50), then calls
  `write_metadata_to_registers()` (line 51) so lane 15's register copy
  matches. Used by `initialize_bucket`
  (`gpu_chainhashtable.hpp:526–534`, head = true) and by fresh overflow
  nodes in `cooperative_insert` (`gpu_chainhashtable.hpp:246`,
  head = false).

### Load/store: the `<atomic, acquire/release>` template discipline

Six entry points (lines 54–94) funnel into `do_load` / `do_store`:

| Member | Source/target address | Lines |
|--------|----------------------|-------|
| `load_from_array<atomic, acquire=true>(table_ptr)` | `table_ptr + 2*16*node_index_` — head node in the bucket array | 54–58 |
| `load_from_allocator<atomic, acquire=true>()` | `allocator_.address(node_index_)` — aux node in the slab pool | 59–63 |
| `store_to_array<atomic, release=true>(table_ptr)` | bucket array | 72–76 |
| `store_to_allocator<atomic, release=true>()` | slab pool | 77–81 |
| `store_head_to_array_aux_to_allocator<atomic, release=true>(table_ptr)` | picks the destination at runtime by `is_head()` (lines 84–86) — the workhorse store used by insert/erase, which may be holding either a head or an aux node when they mutate | 82–88 |
| `do_load<atomic, acquire>` / `do_store<atomic, release>` | per-lane 64-bit element transfer | 64–71 / 89–94 |

`do_load` (lines 64–71), line by line:

```cpp
66:  if constexpr (atomic) { tile_.sync(); }       // close any preceding divergence so the
                                                   // 16 lane loads form one coalesced window
67:  auto elem = utils::memory::load<elem_unsigned_type, atomic, acquire>(node_ptr + tile_.thread_rank());
                                                   // each lane loads ITS 8-byte element;
                                                   // utils.hpp:69-83: atomic=true ->
                                                   // cuda::atomic_ref<uint64_t>.load(acquire or
                                                   // relaxed); atomic=false -> plain *ptr
68:  lane_elem_ = *reinterpret_cast<elem_type*>(&elem);   // reinterpret the 64b word as {key,value}
69:  if constexpr (atomic) { tile_.sync(); }       // all lanes hold their element before any
                                                   // lane shuffles metadata out of lane 15
70:  read_metadata_from_registers();               // metadata_ = shfl(lane 15's key) (lines 96-98)
```

`do_store` (lines 89–94) is the mirror image: optional `tile_.sync()`,
per-lane `utils::memory::store<uint64_t, atomic, release>` (utils.hpp:85–99),
optional `tile_.sync()`. Note that store does **not** call
`write_metadata_to_registers()` — every mutator already did (e.g., lines
252, 284, 302); the caller is responsible for register/metadata coherence
before storing.

**The memory-ordering discipline, precisely:**

- `atomic = false` compiles to plain loads/stores (utils.hpp:81, 97).
  No ordering, no atomicity. Legal only when the caller has otherwise
  excluded concurrent writers (reader under worker serialization) or
  established ordering through a prior acquire (see next bullet).
- `atomic = true, acquire = true` (the default for loads) compiles to
  `cuda::atomic_ref<uint64_t, thread_scope_device>.load(memory_order_acquire)`
  per 8-byte element (utils.hpp:72–74). There is **no** 128-byte single-copy
  atomicity claim: atomicity is per element. Cross-element consistency of a
  node read concurrent with a writer is delivered by the **publication
  order** the writers obey (see the protocol summary below), not by the
  load itself.
- `atomic = true, release = true` for stores compiles to per-element
  `.store(memory_order_release)` (utils.hpp:89–91).
- The lock acquisition/release supply the heavyweight fences (next
  subsection), allowing most loads/stores *inside* a critical section to
  be the cheap `<false>` non-atomic flavor. RGI exploits this everywhere
  and leaves comments saying so, e.g. `gpu_chainhashtable.hpp:218`
  ("use weak load here b/c the first load did memory_order_acquire") and
  `:398` ("future unlock will do memory_order_release").

### The lock protocol (lines 172–230)

Two overload families: one addressing head nodes embedded in the bucket
array (`table_ptr, bucket_index` — lines 172–200, the only family the
chain hashtable uses), one addressing head nodes in the slab pool
(`head_index, allocator` — lines 202–230, used by the extendible hash
table; dead code for this project).

`try_lock(table_ptr, bucket_index, tile)` (lines 172–183), line by line:

```cpp
174:  auto bucket_ptr = reinterpret_cast<elem_type*>(table_ptr + 2*16*bucket_index);
                       // address of the head node's element array
175:  if (tile.thread_rank() == metadata_lane_) {          // ONLY lane 15 touches memory
176:    cuda::atomic_ref<key_type, cuda::thread_scope_device> metadata_ref(bucket_ptr[15].key);
177:    old = metadata_ref.fetch_or(lock_bit_mask_, cuda::memory_order_relaxed);
                       // RELAXED RMW: sets bit 4 unconditionally, returns prior word
180:  bool is_locked = (tile.shfl(old, metadata_lane_) & lock_bit_mask_) == 0;
                       // every lane learns whether bit 4 was previously clear
181:  if (is_locked) { cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_device); }
                       // ONLY on success: an explicit acquire fence, device scope
182:  return is_locked;
```

The ordering recipe: **relaxed `fetch_or` + conditional acquire fence on
success**. The relaxed RMW is sufficient to win the lock (atomicity of the
RMW decides the race); the acquire fence then prevents any subsequent read
of the bucket/chain from being reordered before the lock acquisition, so
the winner observes everything the previous lock holder published. Note
that the fence executes in **all 16 lanes** (line 181 is unconditional on
lane), which is required because every lane will subsequently read node
memory.

`lock` (lines 184–185) is `while (!try_lock(...));` — a plain spin with no
backoff. Bucket locks are expected to be uncontended at OLTP bucket
counts; a future contention study would observe this spin.

`unlock<release = true>` (lines 187–200):

```cpp
190:  auto bucket_ptr = ...;                                // same addressing
191:  if (tile.thread_rank() == metadata_lane_) {           // only lane 15
193:    cuda::atomic_ref<key_type, ...> metadata_ref(bucket_ptr[15].key);
194:    metadata_ref.fetch_and(~lock_bit_mask_, cuda::memory_order_release);
                       // clears bit 4 with RELEASE: publishes every prior
                       // in-critical-section store to the next acquirer
197:  } else /* release == false */ {
197:    bucket_ptr[metadata_lane_].key &= ~lock_bit_mask_;   // non-atomic clear, NO ordering
```

Why `fetch_and` on memory rather than rewriting lane 15's register copy:
the in-register `metadata_` may have been mutated during the critical
section (e.g., `num_keys` incremented) and already stored; the unlock must
only clear the lock bit of whatever word is **now** in memory — comment at
line 189 ("only using the pointer, not load the entire register"). The
`release = false` variant exists for callers that have not mutated
anything and need no publication; this project's wrapper never reaches it
(the chain table's own code always uses the default).

**Protocol summary as the writers obey it** (assembled from
`gpu_chainhashtable.hpp`, detailed in that file's section):

1. Writer locks the bucket head: relaxed `fetch_or` + acquire fence.
2. Writer's **first** node load is `load_from_array<true>` (acquire);
   subsequent chain loads inside the section are `<false>` (non-atomic) —
   ordering is inherited.
3. New nodes are fully written **before** being linked
   (`gpu_chainhashtable.hpp:248–251`, "write order: new_node -> node").
4. The linking store of the head uses `<true>` (per-element release) when
   it publishes a new reachable node (`gpu_chainhashtable.hpp:256`), or
   `<false>` when the unlock's release will cover it
   (`gpu_chainhashtable.hpp:228, 297, 398`).
5. Unlock: `fetch_and` release.

A concurrent reader (`concurrent = true` find) therefore observes either
the pre-publication or post-publication state of each 8-byte element; the
write order guarantees a new node's contents are globally visible before
any pointer to it is.

### Reading and matching

- `read_metadata_from_registers()` (lines 96–98): `metadata_ =
  tile_.shfl(lane_elem_.key, 15)` — every lane gets a coherent copy of the
  metadata word.
- `write_metadata_to_registers()` (lines 99–103): lane 15 writes
  `metadata_` back into its `lane_elem_.key`. All lanes maintain
  `metadata_` redundantly; only lane 15's element copy reaches memory.
- `get_key_from_location(location)` / `get_value_from_location(location)`
  (lines 105–110): `shfl` from the slot's lane. This is how the traversal
  extracts a suffix-node index (`gpu_chainhashtable.hpp:328`) or the final
  inline value (`gpu_chainhashtable.hpp:185`).
- `is_valid_lane()` (lines 111–113): `thread_rank() < num_keys()` — the
  per-lane occupancy predicate behind every ballot.
- `num_keys` / `set_num_keys` (lines 115–122), `is_full` (lines 123–125,
  `num_keys() == 15`), `is_mergeable(next)` (lines 126–128, combined
  count ≤ 15).
- Suffix-bit accessors (lines 129–136): bit `(8 + location)` of metadata;
  `get_suffix_of_location` is also what `validate_nodes_task` and the
  future WP6 enumeration use to decide whether a value is data or a
  pointer.
- Chain accessors: `has_next` (137–139, bit 5), `set_has_next` (140–143,
  sets bit and syncs registers), `get_next_index` (144–146, `shfl` of lane
  15's **value**), `set_next_index` (147–149, lane 15 writes its value).
- `is_head` (150–152, bit 6), `is_garbage`/`make_garbage` (153–159,
  bit 7 — used by the extendible table; the chain table never sets it),
  `get/set_local_depth` (160–167, extendible only), `get_node_index`
  (169).

`match_key_in_node(key, more_key)` (lines 232–236) — the heart of lookup:

```cpp
233:  return tile_.ballot(is_valid_lane() &&                       // slot is occupied
234:                      lane_elem_.key == key &&                 // slice or hash tag matches
235:                      get_suffix_of_location(tile_.thread_rank()) == more_key);
                          // and the slot's KIND (inline vs suffix) matches the probe's kind
```

One ballot compares all 15 slots simultaneously and returns a 16-bit mask
of candidates. The third conjunct is essential to WP1's correctness
argument: a single-slice key (`more_key = false`, real slice in the key
field, suffix bit 0) can never collide with a hash-tag slot
(`more_key = true`, suffix bit 1) even if the 32-bit values coincide —
the suffix bit disambiguates. `match_key_value_in_node` (lines 237–242)
additionally matches the value; unused by the chain table's current paths.

### Mutators (operate on registers; caller stores afterward)

- **`insert(key, value, more_key)`** (lines 244–253):
  `assert(!is_full())` (245); the new slot is `location = num_keys()`
  (246); the owning lane writes `{key, value}` (247–249);
  `set_suffix_of_location(location, more_key)` (250); `metadata_++`
  increments the count (251, valid by the offset-0 static_assert);
  registers synced (252). Nothing reaches memory until the caller's
  `store_*`.
- **`update(location, value)`** (lines 255–260): the owning lane
  overwrites its value field. Used by `cooperative_insert`'s
  update-if-exists branch for single-slice keys
  (`gpu_chainhashtable.hpp:227`).
- **`merge(next_node)`** (lines 262–285), used only by the erase-with-merge
  traversal: `assert(is_mergeable(...))` (263); shift the next node's
  elements **up** by `num_keys()` via `shfl_up` and adopt them in lanes
  `[num_keys(), new_num_keys)` (266–272); recompute the suffix bits with a
  ballot of each lane's (possibly adopted) suffix bit (274–276); adopt the
  next node's next pointer (278–280) and its `has_next` bit (282–283);
  sync registers (284). After a merge, the absorbed node is retired by the
  caller (`gpu_chainhashtable.hpp:409`).
- **`erase(location)`** (lines 287–303): `metadata_--` (289); every lane
  at or above `location` adopts its right neighbor's element and suffix
  bit via `shfl_down(…, 1)` (290–298) — a compaction, so slot order is
  dense but **unstable** (slot indices shift on erase; never cache a
  location across mutations); suffix bits recomputed by ballot (299–301);
  registers synced (302).

### Debug printer

`print()` (lines 313–356) dumps `node[index]: {head ld(depth) count
(key value s|$)... locked|free next(i)|nullnext}` where `s` marks
suffix-backed slots and `$` inline slots (line 336), then recursively
prints each referenced suffix node (lines 347–355). Reachable from host
via `table.print()` (`gpu_chainhashtable.hpp:458–461`). Output volume is
proportional to table size — usable only on toy tables.

### Pitfalls (node)

1. **The in-memory word order is interleaved** `k v k v … meta next`, not
   the explainer's warp diagram. Any raw reader (WP6) must compute: slot
   `i` key at word `2i`, value at word `2i+1`, metadata at word 30, next
   at word 31.
2. **Slot indices are unstable across `erase`** (compaction). Code that
   walks `to_check` ballots while erasing must re-match after mutation;
   RGI's own traversals only erase once per request, after which they
   return.
3. **The metadata register copy and memory can diverge** between a mutator
   and the next `store_*`. Always store through the node object that
   performed the mutation; never re-load and merge by hand.
4. **`try_lock`'s fetch_or is unconditional** — a failed attempt still
   wrote the (already-set) bit. This is harmless but means the lock word
   is RMW-hot under contention; relevant when interpreting profiler
   atomics counters.
5. **`unlock` does not check ownership.** Unlocking a bucket the tile does
   not hold corrupts the protocol silently. RGI's call sites are strictly
   balanced (every path from `lock` reaches exactly one `unlock`:
   `gpu_chainhashtable.hpp:231, 257, 301, 305`); project-owned kernels
   that ever take these locks (none do today, and WP6 must not) carry the
   same obligation.
6. **`local_depth`/`is_garbage` look meaningful but are not** in the chain
   table — do not interpret them in enumeration output.

---

## File: suffix_node_subwarp.hpp

**Path:** `RobustGPUIndexing\include\suffix_node_subwarp.hpp` (573 lines —
skimmed for the paths this project exercises; documented to the depth the
project depends on)

**Purpose:** stores the **full key slices and the value** of any entry
whose key is longer than one slice. With this project's 2-slice keys and
`use_hash_tag = true`, every entry today owns exactly one suffix node:
the bucket node stores only a 32-bit tag, and the suffix node is the only
place the actual key and the actual value exist.

### Layout

Authoritative comment at lines 562–572. Element is `{slice_type first,
second}` (lines 26–28), 16 lanes × 8 B = 128 B, but unlike the hashtable
node the two halves are **transposed**: logical slice slot `i` (0…30)
lives in lane `i % 16`, field `.first` if `i < 16`, field `.second`
otherwise. Lane 15's `.second` is the next-pointer (`next_lane_ = 15`,
line 571), so a node carries up to `node_max_len_ = 31` logical slots
(line 572). In the **head** node, slot 0 is the key length and slot 1 is
the value (`head_node_length_lane_ = 0`, `head_node_value_lane_ = 1`,
lines 569–570).

```
        SUFFIX HEAD NODE (this project: key_length = 2, one node suffices)

 lane:        0           1           2           3        4..14     15
          +-----------+-----------+-----------+-----------+-------+-----------+
 .first   | length=2  |  VALUE    | slice[0]  | slice[1]  | unused| unused    |
          | (slot 0)  | (slot 1)  | (slot 2)  | (slot 3)  |       | (slot 15) |
          +-----------+-----------+-----------+-----------+-------+-----------+
 .second  | unused    | unused    | unused    | unused    | unused| NEXT idx  |
          | (slot 16) | (slot 17) | (slot 18) | (slot 19) |       | (unused   |
          +-----------+-----------+-----------+-----------+-------+  here)    |

 Logical slot map: slot i -> lane (i % 16), .first if i < 16 else .second.
 Head node burns slots 0..1 on {length, value}; key slices start at slot 2.
 Capacity per node: 31 logical slots; keys longer than 29 slices chain
 through lane15.second to continuation nodes (none occur in this project).

        THE CHAIN A 2-SLICE KEY CREATES (the "suffix path" — every key today)

   bucket head node, slot i             slab pool
   +--------------------------+
   | key field  = HASH TAG    |        +--------------------------------+
   |   (PRIME1 hash of key)   |        | suffix node (128 B slab)       |
   | value field = slab index-+------> | slot0=2 slot1=value            |
   | suffix bit i = 1         |        | slot2=key_lo slot3=key_hi      |
   +--------------------------+        +--------------------------------+
      one extra DEPENDENT 128 B load per find; one extra slab
      allocation per insert; ~134 B total per 16 B key/value pair
```

`get_num_nodes()` (lines 75–79) returns `((length + 2) + 30) / 31` —
for this project's `length = 2`: exactly 1.

### Atomicity contract (lines 38–40 — read this comment verbatim)

> "ALL suffix loads/stores in this file are done as non-atomic, because
> suffix loads are done with the pointer in tree/bucket node, which is
> loaded with memory_order_acquire; suffix stores are protected by
> tree/bucket node's locks, which includes threadfence with
> memory_order_release."

I.e., the suffix node piggybacks entirely on the bucket-node ordering: a
reader can only learn a suffix index from a bucket-node load that carried
acquire semantics (or was lock-protected), and that acquire ordered the
suffix node's prior contents. Consequence for this project: **any new code
that obtains a suffix index by other means** (WP6's enumeration reading
raw words) must run at a moment when no writer is active — there is no
per-suffix-node synchronization to fall back on. WP6's "enumeration runs
only while no mutation is in flight" contract is what makes this legal.

### Members the project's paths exercise

- **`load_head()` / `store_head()`** (lines 41–49): per-lane non-atomic
  64-bit load/store of the head node at `allocator_.address(node_index_)`.
  Called from traversal (`gpu_chainhashtable.hpp:330`), update-in-place
  (`:224`), and creation (`:241`).
- **`get_next()`** (55–57): `shfl` lane 15 `.second`.
- **`get_key_length()` / `set_length`** (58–65): slot 0.
- **`get_value()` / `update_value(value)`** (66–73): slot 1. `get_value`
  is the terminal read of every successful multi-slice find
  (`gpu_chainhashtable.hpp:182`); `update_value` + `store_head` is the
  whole update-if-exists write for multi-slice keys
  (`gpu_chainhashtable.hpp:223–224`).
- **`streq(key, key_length)`** (lines 81–110): equality compare against a
  probe key. Line 82 short-circuits on length mismatch — this is why a
  key stored with length 2 can never be matched by a length-1 probe (and
  vice versa), the silent-miss hazard WP1's erase rule must respect. The
  `key -= 2; key_length += 2; skip_elems = 2` adjustment (lines 84–87)
  aligns the probe array with the stored layout (slots 0–1 are
  length/value, skipped via the `skip_elems <= thread_rank()` predicate,
  line 91). Each lane compares its `.first` (lines 91–94) and `.second`
  (95–97) slots; one ballot detects any mismatch (98–99); ≤ 31-slot keys
  finish in one iteration (line 100); longer keys follow `next` with
  non-atomic loads (103–107). For this project: 2-slice keys compare in
  the head node, one ballot, done.
- **`create_from(key, key_length, value)`** (lines 313–354): builds the
  node(s) in registers and stores continuation nodes; lane 0 takes
  `length`, lane 1 takes `value` (317–318), the same `key -= 2` trick
  places `key[0..]` at slots 2+ (320–322); continuation nodes are
  allocated (336) and **stored before** the head is published — the head
  itself is only put in `lane_elem_` (345) and reaches memory when the
  caller invokes `store_head()` (`gpu_chainhashtable.hpp:241`), which is
  in turn ordered before the bucket-node store that publishes the suffix
  index (`gpu_chainhashtable.hpp:254–256`). Publication order is
  therefore: suffix body → suffix head → bucket slot → (lock release).
- **`retire(reclaimer)`** (lines 487–499): retires the head slab index,
  then walks `get_next()` (using `fetch`-style raw loads at line 494)
  retiring each continuation node. Called by `cooperative_erase` after a
  successful multi-slice erase (`gpu_chainhashtable.hpp:299`). For this
  project: one retire per erased key.
- **`fetch_value_only(suffix_index, allocator)` /
  `fetch_length_only`** (lines 501–509): single-thread, single-word
  reads of slot 1 / slot 0 (address arithmetic `ptr + 2*lane` because of
  the interleaved element layout). Not used by the chain table's hot
  paths, but the cheapest primitive for WP6's enumeration to fetch values
  without a full 128 B cooperative load.
- **`compute_polynomial` / `compute_polynomialx2`** (lines 187–311):
  recompute the rolling hash of a **stored** key (used by the extendible
  table's rehash and by `compute_hash_suffix`, `compute_hash.hpp:138–172`);
  not on this project's paths but they document that the hash of a stored
  key is recoverable without materializing the slices.
- **`strcmp`** (112–185), **`flush(key_buffer)`** (356–381),
  **`move_from`** (383–485): ordered-compare, key materialization, and
  prefix-strip moves for the Masstree; `flush` is the member WP6 would use
  to materialize full key slices for entries whose suffix exceeds one node
  (not needed while keys are ≤ 2 slices, since slots 2–3 of the head are
  sufficient).
- **`print()`** (518–554): debug dump, reached from the hashtable node's
  printer.

### Pitfalls (suffix)

1. **The value lives only here** for multi-slice keys. The bucket node's
   value field is a slab index, not data. Reading bucket values without
   checking the suffix bit yields pointers-as-values — the exact bug class
   WP6's "mirror `validate_nodes_task`" instruction exists to prevent.
2. **The transposed layout** differs from the hashtable node: logical
   slots wrap at 16 into `.second`, and word addressing for raw reads is
   `word = 2*(i % 16) + (i < 16 ? 0 : 1)`.
3. **No internal synchronization** — see the atomicity contract above.
4. **`streq` requires the probe pointer to be tile-uniform** (it is
   `shfl`-broadcast by the device_funcs before the call); passing
   divergent pointers deadlocks the ballots.

---

## File: compute_hash.hpp

**Path:** `RobustGPUIndexing\include\compute_hash.hpp` (175 lines)

**Purpose:** tile-cooperative polynomial rolling hashes. Decides which
bucket a key maps to and (for multi-slice keys) the 32-bit tag stored in
the node. WP1's correctness argument rests on the precise formulas here.

**Constants** (lines 26–28): `PRIME0 = 0x9e3779b1`, `PRIME1 = 0x01000193`,
`PRIME2 = 0xfffffffb`. The chain table uses PRIME0 for the bucket hash and
PRIME1 for the tag (`gpu_chainhashtable.hpp:166, 171`).

**`finalize(x)`** (lines 30–38): the murmur3 finalizer (xor-shift /
multiply avalanche). Applied to every hash before use.

**`compute_hash<prime0>(input, length, tile)`** (lines 40–74), the
single-hash path (used when `use_hash_tag == false` or `key_length == 1`):

1. Lines 46–52: build per-lane exponents `[1, p, p², …, p^15]` by a
   log-step prefix product over shuffles.
2. Lines 54–65: each lane multiplies its slice by its exponent and
   accumulates; keys longer than the tile stride the input with the
   exponent scaled by `p^16` per round (line 64). For a 1-slice key, lane
   0 contributes `slice * 1` and everyone else 0.
3. Lines 67–69: tree-reduce via `shfl_down`.
4. Line 70: `hash = ((hash * p) + original_length) * p` — **the key length
   is folded into the hash**. Probing the same bytes with a different
   declared length yields a different hash and a different bucket.
5. Lines 72–73: finalize; broadcast lane 0's result.

**`compute_hash_slice<prime0>(slice)`** (lines 76–81): scalar shortcut for
`length == 1`: `finalize(((slice * p) + 1) * p)`. Verifiably identical to
`compute_hash` at length 1 (sum = `slice`, then step 4 with
`original_length = 1`). This identity is the formal footing for WP1's
claim that a key stored via the cooperative path and probed via any
future single-slice shortcut hashes identically.

**`compute_hashx2<prime0, prime1>(input, length, tile)`** (lines 83–126):
computes **both** hashes in one pass — this is what every multi-slice
operation calls (`gpu_chainhashtable.hpp:166, 205, 273`). Same structure
as `compute_hash` with two exponent/accumulator sets; one subtlety worth
recording because it looks like a bug and is not: in the reduction (lines
116–119), `hash` reduces with `shfl_down` (sum lands in lane 0) while
`hash1` reduces with `shfl_up` (sum lands in lane `tile_size-1`); line 122
then moves `hash1` into lane 15's `hash` register, a single `finalize`
call avalanches both lanes' values (line 124), and line 125 returns
`uint2{ shfl(hash,0), shfl(hash,15) }` = `{PRIME0-hash, PRIME1-hash}`.
The caller uses `.x % num_buckets_` as the bucket and `.y` as the stored
tag (`gpu_chainhashtable.hpp:167–168`).

`compute_hashx2_slice` (128–136) and the `*_suffix` variants (138–172)
serve the extendible table and rehash paths; not on this project's paths.

**Pitfalls (hash):**

1. **Length is part of the hash** (line 70/120–121). Store/probe length
   disagreement = silent miss in a different bucket. This is WP1's #1
   stated risk and applies equally to erase.
2. **Tag and bucket are independent hashes** of the *same* slices —
   a tag match is necessary but not sufficient; `streq` against the suffix
   node is always the final word (`gpu_chainhashtable.hpp:332`). False
   tag matches cost one extra suffix load, not correctness.
3. These functions execute **ballot/shfl over the full tile** — they must
   be called by all 16 lanes convergently (guaranteed inside RGI's device
   API; a project kernel calling `cooperative_*` under divergence would
   hang).

---

## File: gpu_chainhashtable.hpp

**Path:** `RobustGPUIndexing\include\gpu_chainhashtable.hpp` (572 lines)

**Purpose:** the chained hash table this project instantiates — the only
RGI index structure in use. Owns the bucket array (`d_table_`), composes
the allocator and reclaimer device instances, and provides three layers of
API: host batch entry points (`find`/`insert`/`erase`/`mixed_batch`),
device cooperative operations (`cooperative_find`/`insert`/`erase` plus
the private traversal helpers), and debug/introspection tasks
(`traverse_nodes`, `print`, `validate`, `print_memory_use`).

### Types and constants (lines 47–65)

| Symbol | Definition | Line | Consequence |
|--------|-----------|------|-------------|
| `size_type`, `elem_type`, `key_slice_type`, `value_type` | all `uint32_t` | 47–50 | keys are arrays of u32 slices; values are u32 — see assumption A3 |
| `table_ptr_type` | `uint64_t` | 51 | unused by this project's paths |
| `tile_size_` | template param (16 here) | 52 | propagated to `batch_kernel` and tile partitioning |
| `bucket_size` | 32 elements | 54 | a bucket head is 32 u32 = 128 B |
| `bucket_bytes` | `4 * 32 = 128` | 55 | `d_table_` stride; also the unit `validate` uses for space accounting (line 506) |
| `invalid_value` | `0xFFFFFFFF` | 57 | the not-found sentinel `cooperative_find` returns (line 189); mirrored as `RGI_INVALID` at `rgi_oltp_engine.cu:43` |
| allocator/reclaimer context aliases | lines 59–65 | — | `device_allocator_context_type` / `device_reclaimer_context_type` are the per-kernel views; the persistent engine re-derives them at `rgi_persist_engine.cu:52–53` |

### Constructors, ownership, destructor (lines 67–96)

- **Default constructor deleted** (line 67).
- **Constructor 1** `(host_allocator, host_reclaimer, num_buckets)` (lines
  68–75): captures the **device instances** of the host allocator and
  reclaimer (`get_device_instance()`, lines 71–72) — the table does NOT
  own the pool or the epoch arrays, only references them — sets
  `num_buckets_`, and calls `allocate()`. Not used by this project.
- **Constructor 2** `(host_allocator, host_reclaimer, num_elements,
  fill_factor)` (lines 76–84): computes
  `num_buckets_ = max(num_elements / fill_factor / 15, 1UL)` (line 82).
  The `15` is the node capacity (`capacity = 15`,
  `hashtable_node_subwarp.hpp:35`): a fill factor of 1.0 sizes one bucket
  node slot per element. **This is the constructor the engine calls**
  (`rgi_oltp_engine.cu:84–85`, with `fill_factor = 2.0` from `rgi_create`,
  i.e., buckets sized for 2× over-occupancy → chains expected;
  `rgi_persist_engine.cu:258` likewise).
- **Copy constructor** (lines 87–92): shallow — copies `d_table_`,
  `num_buckets_`, allocator/reclaimer instances, and sets
  `is_owner_ = false`. Copy **assignment** is deleted (line 86). This is
  the mechanism that makes "pass the table **by value** into a kernel"
  safe and idiomatic: the kernel parameter is a non-owning shallow copy.
  RGI's own kernels do it (`kernels.hpp:38`, `:597`) and so does the
  persistent engine (`rgi_persist_engine.cu:83, 205–207`).
- **Destructor** (lines 94–96): `deallocate()` → `cudaFree(d_table_)`
  only if `is_owner_` (lines 542–546). The engine therefore must keep the
  host-side `table_t` object alive for the lifetime of any kernel using a
  copy of it — `rgi_destroy` (`rgi_oltp_engine.cu:312–316`) deletes the
  table before the allocator and reclaimer, which is the correct order
  (the table holds instances of both).

### Allocation and initialization (lines 525–553)

- `allocate()` (536–540): `is_owner_ = true`, one `cudaMalloc` of
  `128 * num_buckets_` bytes, then `initialize()`.
- `initialize()` (548–553): launches
  `kernels::GpuHashtable::initialize_kernel<16><<<num_buckets_, 16>>>`
  (one block per bucket, one tile per block) and synchronizes.
- `initialize_bucket(bucket_index, tile, allocator)` (526–534), called by
  that kernel (`kernels.hpp:412–420`): constructs the head node,
  `initialize_empty(true)` (head bit set, unlocked, zero keys), and
  `store_to_array<false>` — non-atomic store is fine because the kernel
  ends (and the `cudaDeviceSynchronize` at line 552 orders it) before any
  operation can run.

### Host batch APIs (lines 98–152)

All four share one shape: construct the matching `device_func` struct from
`kernels.hpp`, then `kernels::launch_batch_kernel(*this, func, n, stream)`.
The comment at line 99 states the key-length convention: **"if
key_lengths == NULL, we use max_key_length as a fixed length."** This
project passes `nullptr` everywhere today; WP1's whole mechanism is to
start passing a real per-request `d_key_lengths` array — no RGI change
needed, the plumbing already exists (see the device_func `load` methods,
e.g. `kernels.hpp:445`).

- **`find<concurrent = false, use_hash_tag = true>(keys, max_key_length,
  key_lengths, values, num_keys, stream = 0)`** (lines 100–111).
  - `concurrent` (template): `false` → traversal uses **non-atomic**
    loads (`load_from_array<false>` / `load_from_allocator<false>`); safe
    only when no writer can overlap the launch. `true` → per-element
    acquire loads, safe against concurrent writers. **The engine currently
    calls `find<false, true>`** (`rgi_oltp_engine.cu:74`) and relies on
    worker-level serialization — assumption A5. The persistent kernel and
    `mixed_batch` both use `concurrent = true` because batches there mix
    reads and writes in flight.
  - `use_hash_tag` (template): `true` → multi-slice keys store a PRIME1
    tag in the node and all slices in the suffix; `false` → the node
    stores the first slice and the suffix holds the remainder
    (`suffix_offset` of 1 vs 0, lines 239, 331). The project always uses
    `true` (every call site).
  - Runtime args: `keys` = device array, row-major
    `max_key_length` slices per request; `key_lengths` = per-request
    lengths or `nullptr`; `values` = output array, `invalid_value` for
    misses; `stream` = CUDA stream (engine uses the default stream).
  - Asynchronous: no sync inside; the engine syncs explicitly
    (`rgi_oltp_engine.cu:75`).
- **`insert<use_hash_tag = true>(keys, max_key_length, key_lengths,
  values, num_keys, stream = 0, update_if_exists = false)`** (lines
  113–124). `update_if_exists` is a **runtime** parameter carried inside
  the device_func (line 122). Note the host API discards per-request
  results — `insert_device_func::store` is a no-op (`kernels.hpp:456`);
  only `mixed_batch` surfaces insert/erase results. The engine calls
  `insert<true>(d_keys, 2, nullptr, d_vals, n, 0, true)`
  (`rgi_oltp_engine.cu:68`) — always overwrite mode; see fact A4.
- **`erase<use_hash_tag = true, do_merge = true>(keys, max_key_length,
  key_lengths, num_keys, stream = 0)`** (lines 126–136). `do_merge`
  selects the merging traversal (`coop_traverse_until_found_merge`) which
  opportunistically compacts chains while walking and retires absorbed
  nodes. The engine calls `erase<true, true>` (`rgi_oltp_engine.cu:150,
  204`). Erase results (hit/miss) are likewise dropped by the host API —
  the engine treats delete-of-absent as a no-op by design
  (`rgi_oltp_engine.cu:260`).
- **`mixed_batch<use_hash_tag = true, erase_do_merge = true>
  (request_types, keys, max_key_length, key_lengths, values, results,
  num_requests, stream = 0, insert_update_if_exists = false)`** (lines
  138–152): one launch processing a heterogeneous batch tagged by
  `kernels::request_type` (`request_type_insert = 0`,
  `request_type_erase = 1`, `request_type_find = 2`, `kernels.hpp:28–32`);
  per-request bool results for insert/erase land in `results`, find values
  in `values`. **Not currently called by the engine** (the wrapper batches
  homogeneously), but it is the natural host-API landing zone for a future
  coalescer that interleaves op types, and its device_func is the
  template the persistent kernel's dispatch mimics.

### Device API: `cooperative_find` (lines 155–190), line by line

Signature (155–159): `template <bool concurrent, bool use_hash_tag,
typename tile_type> value_type cooperative_find(const key_slice_type* key,
size_type key_length, const tile_type& tile,
device_allocator_context_type& allocator)`. All 16 lanes must call
convergently with tile-uniform arguments (the device_funcs guarantee this
by `shfl`-broadcasting before the call, `kernels.hpp:486–488`).

```cpp
162:  key_slice_type first_slice;        // what we will match against node key fields
163:  size_type bucket_index;
164:  const bool more_key = (key_length > 1);          // THE SUFFIX-PATH SWITCH (runtime!)
165:  if (use_hash_tag && more_key) {                  // ── multi-slice + tag mode ──
166:    auto hash = utils::compute_hashx2<PRIME0, PRIME1>(key, key_length, tile);
167:    bucket_index = hash.x;                         // PRIME0 hash -> bucket
168:    first_slice = hash.y;                          // PRIME1 hash -> the TAG matched in-node
169:  }
170:  else {                                           // ── single-slice (or no-tag) mode ──
171:    bucket_index = utils::compute_hash<PRIME0>(key, key_length, tile);
172:    first_slice = key[0];                          // the REAL first slice is matched in-node
173:  }
174:  bucket_index %= num_buckets_;
175:  suffix_type suffix_if_found(tile, allocator);    // empty receptacle (no memory touched)
176:  auto node = node_type(bucket_index, tile, allocator);
177:  node.template load_from_array<concurrent>(d_table_);
       // concurrent=true: 16 acquire loads; false: plain loads (engine's current find path)
178:  int location_if_found = coop_traverse_until_found<concurrent, use_hash_tag>(
179:      node, first_slice, more_key, key, key_length, suffix_if_found, tile, allocator);
180:  if (location_if_found >= 0) {                    // found
181:    if (more_key) {
182:      return suffix_if_found.get_value();          // SUFFIX PATH: value lives in the suffix
183:    }
184:    else {
185:      return node.get_value_from_location(location_if_found);  // FAST PATH: inline value
186:    }
187:  }
189:  return invalid_value;                            // 0xFFFFFFFF == miss
```

Where the suffix branch occurs, exactly: the **mode split** is lines
165–173 (tag vs raw slice); the **per-candidate suffix dereference** is
inside the traversal helper (lines 326–338, below); the **result split**
is lines 181–186. With this project's current calls (`key_length = 2`),
`more_key` is always true: every find computes two hashes, matches tags,
loads at least one suffix node (a dependent 128 B load), and `streq`s it.
When WP1 sends `key_length = 1` for small keys, `more_key` is false:
one hash, raw slice match at line 172, inline value at line 185 — no
suffix node exists at all. That is the entire fast path; no other code
changes.

`cooperative_find` takes **no lock and no reclaimer** — finds are
latch-free. Its safety against concurrent writers is exactly the
`concurrent` flag (assumption A5) plus the writers' publication order.

### Device API: `cooperative_insert` (lines 192–259), line by line

Signature (192–198): returns `bool`; takes the value and the runtime
`update_if_exists` flag; needs the allocator but **no reclaimer** (insert
never frees memory — which is why `insert_device_func::reclaim_required =
false`, `kernels.hpp:427`).

```cpp
201:-213:  // identical hash/tag/bucket computation as find (lines 201-213 mirror 162-174)
214:  node_type::lock(d_table_, bucket_index, tile);   // SPIN until bucket head lock acquired
                                                       // (fetch_or relaxed + acquire fence)
215:  suffix_type suffix_if_found(tile, allocator);
216:  auto node = node_type(bucket_index, tile, allocator);
217:  node.template load_from_array<true>(d_table_);   // FIRST load after lock: acquire
218:  int location_if_found = coop_traverse_until_found<false, use_hash_tag>(
       // concurrent=false here ON PURPOSE: the comment says "use weak load here b/c the
       // first load did memory_order_acquire" — inside the lock, plain loads suffice
220:  if (location_if_found >= 0) {                    // ── KEY ALREADY EXISTS ──
221:    if (update_if_exists) {
222:      if (more_key) {
223:        suffix_if_found.update_value(value);       // overwrite slot 1 of the suffix head
224:        suffix_if_found.store_head();              //   (non-atomic; unlock will release)
225:      }
226:      else {
227:        node.update(location_if_found, value);     // overwrite inline value in-register
228:        node.template store_head_to_array_aux_to_allocator<false>(d_table_);
229:      }                                            //   (non-atomic; unlock will release)
230:    }
231:    node_type::unlock(d_table_, bucket_index, tile);  // fetch_and release publishes all
232:    return update_if_exists;                       // *** RETURN SEMANTICS — SEE BELOW ***
233:  }
234:  // ── KEY DOES NOT EXIST: FRESH INSERT ──
235:  value_type to_insert = value;
236:  if (more_key) {                                  // SUFFIX PATH: build the suffix first
237:    to_insert = allocator.allocate(tile);          // +1 slab allocation (the WP1 cost)
238:    auto suffix = suffix_type(to_insert, tile, allocator);
239:    static constexpr uint32_t suffix_offset = use_hash_tag ? 0 : 1;
                                                       // tag mode stores ALL slices in suffix
240:    suffix.create_from(key + suffix_offset, key_length - suffix_offset, value);
241:    suffix.store_head();                           // suffix fully written BEFORE linked
242:  }
243:  if (node.is_full()) {                            // ── traversal left us on a FULL tail ──
244:    auto next_index = allocator.allocate(tile);    // new overflow (aux) node
245:    auto new_node = node_type(next_index, tile, allocator);
246:    new_node.initialize_empty(false);              // not a head; unlocked
247:    new_node.insert(first_slice, to_insert, more_key);
248:    // write order: new_node -> node   (RGI's own comment, line 248)
249:    new_node.template store_to_allocator<false>(); // body first...
250:    node.set_next_index(next_index);               // ...then link it from the tail
251:    node.set_has_next();
252:  }
253:  else {
254:    node.insert(first_slice, to_insert, more_key); // in-register append to current node
255:  }
256:  node.template store_head_to_array_aux_to_allocator<true>(d_table_);
       // THE PUBLISHING STORE uses <true> (per-element release): the new entry/link becomes
       // visible to LATCH-FREE readers in a safe order even before the unlock
257:  node_type::unlock(d_table_, bucket_index, tile);
258:  return true;                                     // fresh insert always succeeds
```

**Fact to engrave (return semantics, lines 232 and 258):**
`cooperative_insert` returns **`update_if_exists` when the key already
exists** and **`true` on a fresh insert**. Therefore with
`update_if_exists = true` the function **returns true on every path — it
cannot fail.** (With `update_if_exists = false` a duplicate returns
`false` and mutates nothing.) The engine's transactional commit is built
directly on this: `rgi_stage_commit` (`rgi_oltp_engine.cu:228–286`)
validates every error condition *before* mutating anything (duplicate scan
+ batched find), then applies deletes (no-op if absent) and
inserts/updates via `update_if_exists = true` — "no expected failure
path" (comment at line 269: "insert with update_if_exists never fails
(validated)"). **If RGI ever changed this return contract or made
overwrite-mode fallible (e.g., allocation failure surfaced as `false`),
the engine's atomicity argument collapses** — see assumption A4.

Note also what fresh insert does *not* do: it never returns a failure for
pool exhaustion — `allocator.allocate` spins until it finds a slab (see
the slab section). Out-of-memory manifests as a hang, not an error.

The traversal helper leaves `node` positioned on the **last node of the
chain** when the key is absent (the loop at 322–352 only exits via
`!has_next()` at line 348), so lines 243–255 always append at the tail.
The "is_full" overflow allocation order (body → link, lines 248–251, then
the release-store of the tail at 256) is the publication order latch-free
readers depend on.

### Device API: `cooperative_erase` (lines 261–307), line by line

Signature (261–266): returns `bool` (true = key existed and was erased);
takes both the allocator **and the reclaimer** — erase is the only basic
op that frees memory (`erase_device_func::reclaim_required = true`,
`kernels.hpp:503`).

```cpp
269:-281:  // identical hash/tag/bucket computation (mirror of find/insert)
282:  node_type::lock(d_table_, bucket_index, tile);
283:  int location_if_found;
284:  suffix_type suffix_if_found(tile, allocator);
285:  auto node = node_type(bucket_index, tile, allocator);
286:  node.template load_from_array<true>(d_table_);   // acquire under lock, as in insert
287:  if constexpr (do_merge) {
288:    location_if_found = coop_traverse_until_found_merge<use_hash_tag>(
289:      node, first_slice, more_key, key, key_length, suffix_if_found, tile, allocator, reclaimer);
       // merging traversal: compacts the chain as a side effect, retires absorbed nodes
290:  }
291:  else {
292:    location_if_found = coop_traverse_until_found<false, use_hash_tag>(
       // again weak loads under the lock ("first load did memory_order_acquire", line 292)
293:  }
295:  if (location_if_found >= 0) {                    // ── EXISTS ──
296:    node.erase(location_if_found);                 // compact the node in-register
297:    node.template store_head_to_array_aux_to_allocator<false>(d_table_);
       // <false>: the unlock's release fence will publish; an erase makes entries DISAPPEAR,
       // so unlike insert there is no new-node publication race to order against
298:    if (more_key) {
299:      suffix_if_found.retire(reclaimer);           // SUFFIX PATH: park the suffix node(s)
       // in the limbo bag — NOT freed yet; freed only after the epoch advances (DEBRA)
300:    }
301:    node_type::unlock(d_table_, bucket_index, tile);
302:    return true;
303:  }
305:  node_type::unlock(d_table_, bucket_index, tile);  // ── NOT EXISTS ──
306:  return false;
```

Where the suffix branch occurs in erase: candidate verification inside the
traversal (same lines 326–338 / 374–387 as find), and the retire at lines
298–300. The bucket-node entry (tag + suffix index) is removed
immediately under the lock; the 128 B suffix slab is reclaimed lazily via
DEBRA. A `concurrent = true` reader racing this erase may still hold the
suffix index it read before the entry vanished — that is exactly the
use-after-free DEBRA prevents, and why erase **requires** a functioning
reclaimer (assumption A8): with reclamation broken, the slab could be
reused for a new suffix while the old reader dereferences it.

### Private traversal helpers

**`coop_traverse_until_found<concurrent, use_hash_tag>`** (lines 311–355)
— shared by find (with its caller's `concurrent`) and by insert/erase
(with `concurrent = false` under the lock). Line by line:

```cpp
322:  while (true) {
323:    uint32_t to_check = node.match_key_in_node(first_slice, more_key);
        // ONE ballot: bitmask of slots whose key field matches the tag/slice AND whose
        // suffix bit equals more_key (kind must match — hashtable_node_subwarp.hpp:232-236)
324:    if (more_key) {                                // ── SUFFIX VERIFICATION LOOP ──
326:      while (to_check != 0) {
327:        auto cur_location = __ffs(to_check) - 1;   // lowest candidate slot
328:        auto suffix_index = node.get_value_from_location(cur_location);
        // the value field IS the suffix slab index for suffix-marked slots
329:        auto suffix = suffix_type(suffix_index, tile, allocator);
330:        suffix.load_head();                        // THE DEPENDENT 128 B LOAD (WP1's target)
331:        static constexpr uint32_t suffix_offset = use_hash_tag ? 0 : 1;
332:        if (suffix.streq(key + suffix_offset, key_length - suffix_offset)) {
333:          // found
334:          suffix_if_found = suffix;                // hand the loaded suffix to the caller
335:          return cur_location;                     //   (so find/update need not reload it)
336:        }
337:        to_check &= ~(1u << cur_location);         // clear and try the next candidate
338:      }                                            //   (false tag match — rare)
339:    }
340:    else {                                         // ── SINGLE-SLICE: MATCH IS FINAL ──
342:      if (to_check != 0) {
344:        return __ffs(to_check) - 1;                // raw 32-bit compare already proved it
345:      }
346:    }
348:    if (!node.has_next()) { break; }               // end of chain
349:    auto next_index = node.get_next_index();
350:    node = node_type(next_index, tile, allocator); // move to the overflow node
351:    node.template load_from_allocator<concurrent>();
        // chain nodes live in the slab pool; load atomicity again governed by <concurrent>
352:  }
354:  return -1;                                       // not found; `node` rests on the TAIL
```

**`coop_traverse_until_found_merge<use_hash_tag>`** (lines 357–417) — the
erase-path variant. Differences from the plain walker:

- Carries `current_node_store_deferred` (line 369): when the current node
  absorbed its successor, the merged node must be written back — but the
  write is deferred so that a node that merges multiple successors is
  stored once (lines 397–400; the store is `<false>` with the comment
  "future unlock will do memory_order_release", line 398).
- Found-paths return **without** flushing a deferred store; RGI marks this
  with the comments "current_node_store_deferred: USER SHOULD STORE the
  node returned" (lines 383, 394). The user is `cooperative_erase`, whose
  line 297 store satisfies the obligation (it always stores the returned
  node after `node.erase`). Any future caller of this helper must honor
  the same contract or merged chains lose entries.
- The merge step (lines 402–413): peek the next node with a weak load
  (line 405 — "first load after lock did memory_order_acquire"); if
  `node.is_mergeable(next_node)` (combined ≤ 15 entries), absorb it
  (`node.merge(next_node)`, line 407), mark the store deferred, and
  **retire the absorbed node's slab index** (line 409,
  `reclaimer.retire(next_index, tile)`); otherwise advance (`node =
  next_node`, line 412). Merging only ever absorbs *aux* nodes —
  the loop starts at the head and `merge` pulls the successor *into* the
  current node, so head nodes are never retired (heads live in
  `d_table_`, not the slab pool, and could not be retired anyway).

### Debug/introspection surface (lines 419–523)

- **`cooperative_traverse_nodes(task, tile)`** (lines 421–438): the
  device-side walker — single tile, sequential, intentionally inefficient
  ("debug-purpose", line 423). For each bucket: construct + weak-load the
  head (lines 428–429), `task.exec(node, bucket_index, tile, allocator)`
  (line 430 — `head_index >= 0` signals "this is a head"), then follow
  `has_next` calling `task.exec(node, -1, ...)` for each aux node (lines
  431–436). **No locks taken, weak loads only** — the walk is safe only
  on a quiescent table.
- **`traverse_nodes(task)`** (lines 440–445): host wrapper; launches
  `traverse_nodes_kernel<16><<<1, 16>>>` (`kernels.hpp:596–606` asserts
  the single-block, tile-wide launch shape) and synchronizes. The task is
  passed **by value** to the kernel and its `fini` prints results from the
  device — host code cannot read the task's fields back.
- **Task protocol** (seen in both tasks): `init(tile)` once, `exec(node,
  head_index, tile, allocator)` per node, `fini(tile)` once
  (`kernels.hpp:603–605`).
- **`print_nodes_task` / `print()`** (lines 447–461): dumps every node via
  the node/suffix printers.
- **`print_memory_use()`** (lines 463–469): bucket-array bytes vs total
  device memory. Does not count slab usage — use the allocator's
  `print_stats()` for that.
- **`validate_nodes_task` / `validate()`** (lines 471–523) — **the layout
  oracle for WP6.** Per node: on a new bucket (`head_index >= 0`) it
  closes out per-bucket maxima (lines 477–482); reads `node.num_keys()`
  (line 483); **for each slot reads the suffix bit and, when set, loads
  the suffix head to count its chain length** (lines 484–492 —
  `get_suffix_of_location(i)` → `get_value_from_location(i)` →
  `suffix.load_head()` → `get_num_nodes()`); accumulates entries, head
  vs aux node counts (lines 493–497). `fini` (lines 499–515) prints
  entries, per-bucket maxima/averages, head/aux/suffix node counts, the
  fill factor against the ×15 capacity (line 505), and **total space:
  `(heads + aux + suffix) * 128 B` and bytes/entry** (lines 506–513) —
  this printout is the measured source of the "~134 B/entry" figure WP1
  cites, and WP1 task 2 explicitly calls `validate()` before/after.
  This task's field-decoding sequence is the **reference implementation**
  the WP6 enumeration kernel must mirror exactly (WP6 risk note,
  `WP6_gpu_scan_and_aggregate_pushdown.md:103–106`).

### Friends (lines 561–567)

`initialize_kernel` and `batch_kernel` are friends so they can reach the
private `allocator_`, `reclaimer_`, and `initialize_bucket`. **Note what
is *not* a friend: project-owned kernels.** The persistent engine cannot
read `table.allocator_`; that is why it threads the allocator's device
instance in as a separate kernel parameter
(`rgi_persist_engine.cu:83, 95, 206`) and builds its own
`device_allocator_context`. Any future project kernel (WP6) must do the
same — needing `allocator_` is not a reason to touch RGI.

### Pitfalls (table)

1. **`cooperative_find` returns `invalid_value` (0xFFFFFFFF) for miss** —
   inserting `0xFFFFFFFF` as a real value makes it indistinguishable from
   a miss. The engine truncates 64-bit values to u32 and reserves this
   sentinel (`rgi_oltp_engine.cu:9, 43`); row-ids must stay below it.
2. **Erase with the wrong `key_length` is a silent no-op-miss**, because
   the length is folded into both hashes and into `streq`. WP1's erase
   rule (same representation rule on every path) is correctness-critical.
3. **`insert<…>` host API discards per-key results.** If a caller needs
   duplicate detection from the index itself, it must pre-`find`
   (the engine's `rgi_flush_unique` / commit validation does exactly
   this) or move to `mixed_batch`, whose results array surfaces them.
4. **The table object on the host is the owner.** Passing it by value is
   fine (shallow copy); letting the owner be destroyed while device work
   using a copy is in flight frees `d_table_` under the kernel.
5. **`traverse_nodes`/`validate`/`print` are single-tile and unlocked** —
   quiescent use only; on a mutating table they may read torn chains.
6. **The fill-factor constructor divides by 15** — sizing intuition from
   open-addressing tables ("fill factor < 1") does not transfer; here
   fill factor ≈ average entries per 15-slot bucket node.

<!-- CONTINUED -->
