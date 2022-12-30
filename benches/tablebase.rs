mod common;

use common::Perf;
use criterion::{criterion_group, criterion_main, Criterion};

use fairy::piece::{Type::*, *};
use fairy::player::Player::*;
use fairy::tablebase::*;

fn run<const W: i8, const H: i8>() -> Tablebase<W, H> {
    let mut tablebase = Tablebase::default();
    let kk = [Piece::new(White, King), Piece::new(Black, King)];
    let kqk = [
        Piece::new(White, King),
        Piece::new(White, Queen),
        Piece::new(Black, King),
    ];
    let krk = [
        Piece::new(White, King),
        Piece::new(White, Rook),
        Piece::new(Black, King),
    ];
    generate_tablebase(&mut tablebase, &kk);
    generate_tablebase(&mut tablebase, &kqk);
    generate_tablebase(&mut tablebase, &krk);
    tablebase
}

fn tb(c: &mut Criterion<Perf>) {
    c.bench_function("kk/kqk/krk", |b| b.iter(run::<5, 5>));
}

criterion_group!(
    name = benches;
    config = Criterion::default().with_measurement(Perf);
    targets = tb
);
criterion_main!(benches);
