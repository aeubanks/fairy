use fairy::piece::{Type::*, *};
use fairy::player::Player::*;
use log::info;
use std::env;
use std::fs;
use std::path::PathBuf;

fn run_perft() {
    use fairy::board::Presets;
    use fairy::perft::{perft, Position};
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
}

fn tablebase<const N: i8, const M: i8>(parallel: usize, only_three: bool) {
    use fairy::tablebase::*;

    let mut all_pieces = Vec::new();
    for ty in [Pawn, Knight, Bishop, Rook, Queen] {
        for pl in [White, Black] {
            all_pieces.push(Piece::new(pl, ty));
        }
    }
    let wk = Piece::new(White, King);
    let bk = Piece::new(Black, King);
    let mut sets = Vec::<PieceSet>::new();
    if only_three {
        for &p in &all_pieces {
            sets.push(PieceSet::new(&[wk, bk, p]));
        }
    } else {
        for &p1 in &all_pieces {
            for &p2 in &all_pieces {
                sets.push(PieceSet::new(&[wk, bk, p1, p2]));
            }
        }
    }
    let tablebase = if parallel != 0 {
        generate_tablebase_parallel::<N, M>(&sets, Some(parallel))
    } else {
        generate_tablebase(&sets)
    };
    tablebase.dump_stats();
    if let Ok(home) = env::var("HOME") {
        let mut path = PathBuf::new();
        path.push(home);
        path.push("tb");
        info!("writing tablebase to {:?}", path);
        fs::write(path, &tablebase.serialize()).unwrap();
    }
}

fn main() {
    use env_logger::{Builder, Env};
    Builder::from_env(Env::default().default_filter_or("info")).init();

    use std::env::args;
    if let Some(str) = args().nth(1) {
        if str == "tablebase" {
            let parallel = args()
                .nth(2)
                .map(|s| s.parse::<usize>().unwrap())
                .unwrap_or(0);
            let only_three = args()
                .nth(3)
                .map(|s| s.parse::<bool>().unwrap())
                .unwrap_or(true);
            tablebase::<6, 6>(parallel, only_three);
        } else if str == "perft" {
            run_perft();
        } else {
            println!("unexpected arg '{}'", str);
        }
    } else {
        println!("specify 'tablebase' or 'perft' as first arg");
    }
}
