mod common;

use common::Perf;
use criterion::{criterion_group, criterion_main, Criterion};

use fairy::piece::{Type::*, *};
use fairy::player::Player::*;
use fairy::tablebase::*;

fn run<const W: i8, const H: i8>() -> Tablebase<W, H> {
    let kk = PieceSet::new(&[Piece::new(White, King), Piece::new(Black, King)]);
    let kqk = PieceSet::new(&[
        Piece::new(White, King),
        Piece::new(White, Queen),
        Piece::new(Black, King),
    ]);
    let krk = PieceSet::new(&[
        Piece::new(White, King),
        Piece::new(White, Rook),
        Piece::new(Black, King),
    ]);
    generate_tablebase(&[kk, kqk, krk])
}

fn run_parallel<const W: i8, const H: i8>() -> Tablebase<W, H> {
    let kk = PieceSet::new(&[Piece::new(White, King), Piece::new(Black, King)]);
    let kqk = PieceSet::new(&[
        Piece::new(White, King),
        Piece::new(White, Queen),
        Piece::new(Black, King),
    ]);
    let krk = PieceSet::new(&[
        Piece::new(White, King),
        Piece::new(White, Rook),
        Piece::new(Black, King),
    ]);
    generate_tablebase_parallel(&[kk, kqk, krk], Some(2))
}

fn tb(c: &mut Criterion<Perf>) {
    c.bench_function("kk/kqk/krk", |b| b.iter(run::<5, 5>));
    c.bench_function("kk/kqk/krk parallel", |b| b.iter(run_parallel::<5, 5>));
}

criterion_group!(
    name = benches;
    config = Criterion::default().with_measurement(Perf);
    targets = tb
);
criterion_main!(benches);
