use fairy::board::Presets;
use fairy::perft::{perft, Position};
use fairy::piece::{Type::*, *};
use fairy::player::Player::*;
use fairy::tablebase::*;

fn tablebase<const N: i8, const M: i8>(parallel: usize, only_three: bool) {
    let mut all_pieces = Vec::new();
    let mut tablebase = Tablebase::<N, M>::default();
    for ty in [Bishop, Knight, Rook, Queen, Cardinal, Empress, Amazon] {
        for pl in [White, Black] {
            all_pieces.push(Piece::new(pl, ty));
        }
    }
    let wk = Piece::new(White, King);
    let bk = Piece::new(Black, King);
    let mut sets = Vec::<PieceSet>::new();
    sets.push(PieceSet::new(&[wk, bk]));
    for p in &all_pieces {
        sets.push(PieceSet::new(&[wk, bk, *p]));
    }
    if !only_three {
        for p1 in &all_pieces {
            for p2 in &all_pieces {
                sets.push(PieceSet::new(&[wk, bk, *p1, *p2]));
            }
        }
    }
    if parallel != 0 {
        generate_tablebase_parallel(&mut tablebase, &sets, Some(parallel));
    } else {
        generate_tablebase(&mut tablebase, &sets);
    }
    tablebase.dump_stats();
}

fn main() {
    use env_logger::{Builder, Env};
    Builder::from_env(Env::default().default_filter_or("info")).init();

    println!(
        "perft(4): {}",
        perft(
            &Position {
                board: Presets::classical(),
                player: White
            },
            4
        )
    );

    use std::env::args;
    let parallel = args()
        .nth(1)
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(0);
    let only_three = args()
        .nth(2)
        .map(|s| s.parse::<bool>().unwrap())
        .unwrap_or(true);
    tablebase::<6, 6>(parallel, only_three);
}
