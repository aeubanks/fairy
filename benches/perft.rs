use criterion::{black_box, criterion_group, criterion_main, Criterion};

use fairy::board::Board;
use fairy::perft::{fen, perft_all, Position};
use fairy::player::Player::*;

fn run_perft_all(c: &mut Criterion) {
    let pos1 = Position {
        board: Board::classical(),
        player: White,
    };
    let pos2 = fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -");
    c.bench_function("classical1", |b| b.iter(|| perft_all(black_box(&pos1), 5)));
    c.bench_function("classical2", |b| b.iter(|| perft_all(black_box(&pos2), 4)));
}

criterion_group!(benches, run_perft_all);
criterion_main!(benches);
