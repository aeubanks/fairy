mod common;

use common::{fairy_criterion, FairyCriterion};
use criterion::{criterion_group, criterion_main};

use fairy::piece::{Type::*, *};
use fairy::player::Player::*;
use fairy::tablebase::*;

const WK: Piece = Piece::new(White, King);
const WQ: Piece = Piece::new(White, Queen);
const WP: Piece = Piece::new(White, Pawn);
const BK: Piece = Piece::new(Black, King);
const BQ: Piece = Piece::new(Black, Queen);
const BR: Piece = Piece::new(Black, Rook);

fn piece_sets(sets: &[&[Piece]]) -> Vec<PieceSet> {
    sets.iter().map(|&ps| PieceSet::new(ps)).collect::<Vec<_>>()
}

fn run<const W: i8, const H: i8>(sets: &[&[Piece]]) -> Tablebase<W, H> {
    generate_tablebase(&piece_sets(sets))
}

fn run_parallel<const W: i8, const H: i8>(sets: &[&[Piece]]) -> Tablebase<W, H> {
    generate_tablebase_parallel(&piece_sets(sets), Some(2))
}

fn tb(c: &mut FairyCriterion) {
    c.bench_function("kqk/krk", |b| {
        b.iter(|| run::<5, 5>(&[&[WK, WQ, BK], &[WK, BK, BQ]]))
    });
    c.bench_function("kpk", |b| b.iter(|| run::<5, 5>(&[&[WK, WP, BK]])));
    c.bench_function("kqkr", |b| b.iter(|| run::<5, 5>(&[&[WK, WQ, BK, BR]])));
    c.bench_function("kqk/krk parallel", |b| {
        b.iter(|| run_parallel::<5, 5>(&[&[WK, WQ, BK], &[WK, BK, BQ]]))
    });
}

criterion_group!(
    name = benches;
    config = fairy_criterion();
    targets = tb
);
criterion_main!(benches);
