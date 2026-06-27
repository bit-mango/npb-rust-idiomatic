# NPB-Rust-Idiomatic
An opinionated rust implementation of the NAS Parallel Benchmarks(NPB) focused on idiomatic rust.

All timing is referenced against the NPB-Rust results, so that we are comparing rust to rust.

References:
- [NAS Parallel Benchmarks](https://www.nas.nasa.gov/software/npb.html)
- [NPB-Rust](https://github.com/GMAP/NPB-Rust)
---
## Embarrassingly Parallel
### Lessons learned
Utilizing a lazy algorithm over an eager algorithm, greatly improved performance.
TODO add part talking about lazy vs eager.
README: you can show the actual line of their code that causes it, explain the allocation count precisely (65536 tasks × ~1MB buffer for Class C), and contrast it with your zero-allocation design. 

Ep Results All Times in seconds
||Class A | Class B | Class C |
| -------- | -------- | -------- | -------- |
|Serial Eager| 0 | 0 | 0|
|Serial Lazy| 1.46 | 5.75 | 22.88|
|Parallel Lazy| 0.14 | 0.49 | 1.95|
|Serial Reference| 0 | 0 | 0|
|Parallel Reference| 0 | 0 | 0|
