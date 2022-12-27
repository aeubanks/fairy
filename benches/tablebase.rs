use criterion::{black_box, criterion_group, criterion_main, Criterion};

use fairy::piece::{Type::*, *};
use fairy::player::Player::*;
use fairy::tablebase::*;

fn three_piece_tablebase<const N: usize, const M: usize>(pieces: &[Piece]) -> Tablebase<N, M> {
    assert_eq!(pieces.len(), 3);
    let kk = [Piece::new(White, King), Piece::new(Black, King)];
    generate_tablebase::<N, M>(&[&kk, &pieces])
}

fn kqk(c: &mut Criterion) {
    let pieces = [
        Piece::new(White, King),
        Piece::new(White, Queen),
        Piece::new(Black, King),
    ];
    c.bench_function("kqk", |b| {
        b.iter(|| three_piece_tablebase::<5, 5>(black_box(&pieces)))
    });
}

criterion_group!(benches, kqk);
criterion_main!(benches);
