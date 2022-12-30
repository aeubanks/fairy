use fairy::board::Presets;
use fairy::perft::{perft, Position};
use fairy::piece::{Type::*, *};
use fairy::player::Player::*;
use fairy::tablebase::*;

fn tablebase<const N: i8, const M: i8>(parallel: bool, only_three: bool) {
    let mut all_pieces = Vec::new();
    let mut tablebase = Tablebase::<N, M>::default();
    generate_tablebase(
        &mut tablebase,
        &[Piece::new(White, King), Piece::new(Black, King)],
    );
    for ty in [Bishop, Knight, Rook, Queen, Archbishop, Chancellor, Amazon] {
        for pl in [White, Black] {
            all_pieces.push(Piece::new(pl, ty));
        }
    }
    let mut sets3 = Vec::new();
    let mut sets4 = Vec::new();
    for p in &all_pieces {
        sets3.push(vec![Piece::new(White, King), Piece::new(Black, King), *p]);
    }
    for p1 in &all_pieces {
        for p2 in &all_pieces {
            sets4.push(vec![
                Piece::new(White, King),
                Piece::new(Black, King),
                *p1,
                *p2,
            ]);
        }
    }
    if parallel {
        generate_tablebase_parallel(
            &mut tablebase,
            &sets3.iter().map(|p| p.as_slice()).collect::<Vec<_>>(),
            None,
        );
        if !only_three {
            generate_tablebase_parallel(
                &mut tablebase,
                &sets4.iter().map(|p| p.as_slice()).collect::<Vec<_>>(),
                None,
            );
        }
    } else {
        for set3 in &sets3 {
            generate_tablebase(&mut tablebase, set3);
        }
        if !only_three {
            for set4 in &sets4 {
                generate_tablebase(&mut tablebase, set4);
            }
        }
    }
    tablebase.dump_stats();
}

fn main() {
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
    tablebase::<6, 6>(true, true);
}
