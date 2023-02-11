use fairy::board::{Board, Move, Presets};
use fairy::coord::Coord;
use fairy::moves::all_moves;
use fairy::piece::{Type::*, *};
use fairy::player::Player::*;
use fairy::timer::Timer;
use log::info;
use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

fn play_game(cpu_as_black: bool) -> std::io::Result<()> {
    let mut board = Presets::los_alamos();
    let mut player = White;
    fn read_move(line: &str) -> Option<Move> {
        let split: Vec<_> = line.split_whitespace().collect();
        if split.len() != 4 {
            return None;
        }
        let a = split[0].parse::<i8>().ok()?;
        let b = split[1].parse::<i8>().ok()?;
        let c = split[2].parse::<i8>().ok()?;
        let d = split[3].parse::<i8>().ok()?;
        Some(Move {
            from: Coord::new(a, b),
            to: Coord::new(c, d),
        })
    }
    loop {
        println!("------------");
        println!();
        print!("  ");
        for x in 0..board.width() {
            print!("{}", x);
        }
        println!();
        println!();
        for y in (0..board.height()).rev() {
            print!("{} ", y);
            for x in 0..board.width() {
                print!(
                    "{}",
                    match board.get(Coord::new(x, y)) {
                        None => '.',
                        Some(p) => p.char(),
                    }
                );
            }
            println!(" {}", y);
        }
        println!();
        print!("  ");
        for x in 0..board.width() {
            print!("{}", x);
        }
        println!();
        println!();
        if board.maybe_king_coord(player).is_none() {
            break;
        }

        println!("{:?} turn", player);
        if cpu_as_black && player == Black {
            use rand::Rng;
            let all = all_moves(&board, player);
            let m = all[rand::thread_rng().gen_range(0..all.len())];
            println!("CPU made {:?}", m);
            board.make_move(m);
            player = player.next();
        } else {
            print!("> ");
            std::io::stdout().flush()?;

            let mut buf = Default::default();
            std::io::stdin().read_line(&mut buf)?;

            let line = buf.trim();
            if line == "exit" {
                break;
            } else if line == "help" {
                println!("exit");
                println!("help");
                println!("x_from y_from x_to y_to");
            } else if let Some(m) = read_move(line) {
                let all_moves = all_moves(&board, player);
                if all_moves.contains(&m) {
                    board.make_move(m);
                    player = player.next();
                } else {
                    println!("invalid move");
                }
            } else if !line.is_empty() {
                println!("invalid input");
            }
        }
    }
    Ok(())
}

fn run_perft() {
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
        let buf = tablebase.serialize();
        let mut path = PathBuf::new();
        path.push(home);
        path.push("tb");
        info!("writing tablebase to {:?}", path);
        let timer = Timer::new();
        fs::write(path, buf).unwrap();
        info!("writing took {:?}", timer.elapsed());
    }
}

fn main() {
    use env_logger::{Builder, Env};
    use std::process::exit;
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
        } else if str == "play" {
            if play_game(args().nth(2).map(|a| a == "cpu").unwrap_or(false)).is_err() {
                exit(1);
            }
        } else {
            println!("unexpected arg '{}'", str);
            exit(1);
        }
    } else {
        println!("specify 'tablebase', 'perft', or 'play' as first arg");
        exit(1);
    }
}
