# NPB-Rust-Idiomatic
An opinionated rust implementation of the NAS Parallel Benchmarks(NPB) focused on idiomatic rust.

All timing is referenced against the NPB-Rust results, so that we are comparing rust to rust.

References:
- [NAS Parallel Benchmarks](https://www.nas.nasa.gov/software/npb.html)
- [NPB-Rust](https://github.com/GMAP/NPB-Rust)
---

## Embarrassingly Parallel
### Lessons learned
Utilizing a lazy algorithm over an eager algorithm, greatly improved performance. The lazy version exposed an iterator to generate random numbers as needed, whereas the eager algorithm would generate a vec of random numbers at once, causing frequent reallocations and a larger memory footprint.

*TODO*: Class D verification for the x summation fails just barely, I think this is some accumulated error from doing billions of math operations, but should be investigated as the spec does not say that Class D problems have a more relaxed verifiaction standard.

#### Ep Results (All Times in seconds)
|  | Class A | Class B | Class C |
| -------- | -------- | -------- | -------- |
|Serial Lazy| 1.46 | 5.75 | 22.88 |
|Parallel Lazy| 0.14 | 0.49 | 1.95 |
|Serial Reference| 9.92 | 39.13 | 156.33 |
|Parallel Reference| 0.75 | 2.95 | 11.84 |
---

## Conjugate Gradient
### Lessons learned
The vast majority of time is used in `CompressedSparseMatrix::multiply()` function. Specifically when indexing vector with the `col_idx` from the sparse matrix. This indexing is "random" and by design of this kernel causes frequent cache misses that eat up the majority of time used.

Additionally it was found that for some reason the random numbers used to populate the sparse matrix do not start at the seed listed in the spec, instead you have to advance one random number, and start from there.

The biggest successful speed up came from storing all the sparse matrix values in a single continious vector as opposed to a separate vector per row. This reduced Class B run time by a few seconds. It makes sense that the cpu is able to grab data faster since its all in the same spot vs jumping around in memory.

*TODO*: Can possibly optimize slightly further by somehow getting CompressedSparseMatrix::data to be stored as 12 bytes as opposed to now where it is padded to 16 bytes. This means for every read, we read 4 bytes of data that are not useful.

#### Cg Results (All Times in seconds)
|  |Class A | Class B | Class C |
| -------- | -------- | -------- | -------- |
|Serial| 0.31 | 14.3 | 41.33 |
|Parallel| 0.13 | 2.53 | 5.79 |
|Serial Reference| 0.31 | 13.9 | 39.21 |
|Parallel Reference| 0.25 | 2.74 | 5.23 |
