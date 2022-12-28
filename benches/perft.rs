mod common;

use common::Perf;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fairy::board::Presets;
use fairy::perft::{fen, perft, perft_all, Position};
use fairy::player::Player::*;

fn run_perft_all(c: &mut Criterion<Perf>) {
    let pos1 = Position {
        board: Presets::classical(),
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
    config = Criterion::default().with_measurement(Perf);
    targets = run_perft_all
);
criterion_main!(benches);
