use criterion::{black_box, criterion_group, criterion_main, Criterion};

use fairy::piece::{Type::*, *};
use fairy::player::Player::*;
use fairy::tablebase::*;

fn three_piece_tablebase(pieces: &[Piece]) -> Tablebase {
    assert_eq!(pieces.len(), 3);
    let kk = [
        Piece {
            player: White,
            ty: King,
        },
        Piece {
            player: Black,
            ty: King,
        },
    ];
    generate_tablebase(6, 6, &[&kk, &pieces])
}

fn kqk(c: &mut Criterion) {
    let pieces = [
        Piece {
            player: White,
            ty: King,
        },
        Piece {
            player: White,
            ty: Queen,
        },
        Piece {
            player: Black,
            ty: King,
        },
    ];
    c.bench_function("kqk", |b| {
        b.iter(|| three_piece_tablebase(black_box(&pieces)))
    });
}

criterion_group!(benches, kqk);
criterion_main!(benches);
