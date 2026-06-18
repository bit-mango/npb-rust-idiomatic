use common::LazyRanddp;
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("Generate 1M psuedo random double precision numbers", |b| {
        b.iter_with_setup(
            || {
                let seed = 271828182845904523;
                let lazy_randdp = LazyRanddp::new(seed, 1_000_000, 5_u64.pow(13));
                lazy_randdp
            },
            |lazy_randdp: LazyRanddp| {
                // Generate 1_000_000 random numbers.
                let mut sum = 0.0;
                for rand in lazy_randdp {
                    sum += rand;
                }
                black_box(sum); // So compiler doesn't optimize this away.
            },
        )
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
