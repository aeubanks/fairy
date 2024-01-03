mod common;

use common::{fairy_criterion, FairyCriterion};
use criterion::{black_box, criterion_group, criterion_main};
use fairy::board::presets;
use fairy::perft::{fen, perft, perft_all, Position};
use fairy::player::Player::*;

fn run_perft_all(c: &mut FairyCriterion) {
    let pos1 = Position {
        board: presets::classical(),
        player: White,
    };
    let pos2 = fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -");
    c.bench_function("perft_classical1", |b| {
        b.iter(|| perft(black_box(&pos1), 4))
    });
    c.bench_function("perft_classical2", |b| {
        b.iter(|| perft(black_box(&pos2), 3))
    });
    c.bench_function("perft_all_classical1", |b| {
        b.iter(|| perft_all(black_box(&pos1), 4))
    });
    c.bench_function("perft_all_classical2", |b| {
        b.iter(|| perft_all(black_box(&pos2), 3))
    });
}

criterion_group!(
    name = benches;
    config = fairy_criterion();
    targets = run_perft_all
);
criterion_main!(benches);
