use clap::{Parser, Subcommand};
use fairy::board::{Board, Move, Presets};
use fairy::coord::Coord;
use fairy::moves::all_moves;
use fairy::nn;
use fairy::piece::{Type::*, *};
use fairy::player::Player::*;
use fairy::tablebase::*;
use fairy::tablebase::{generate_tablebase, Tablebase};
use fairy::timer::Timer;
use log::info;
use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

fn play_game(cpu_as_black: bool) -> std::io::Result<()> {
    fn play_tablebase() -> Tablebase<6, 6> {
        let mut all_pieces = Vec::new();
        for ty in [Pawn, Knight, Rook, Queen] {
            for pl in [White, Black] {
                all_pieces.push(Piece::new(pl, ty));
            }
        }
        let wk = Piece::new(White, King);
        let bk = Piece::new(Black, King);
        let mut sets = vec![];
        for p in &all_pieces {
            sets.push(PieceSet::new(&[wk, bk, *p]));
        }
        generate_tablebase(&sets)
    }
    let tablebase = play_tablebase();
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
            let rand_move = || {
                use rand::Rng;
                println!("randomly moving");
                let all = all_moves(&board, player);
                all[rand::thread_rng().gen_range(0..all.len())]
            };
            let mut num_pieces = 0;
            board.foreach_piece(|_, _| num_pieces += 1);

            let m = if num_pieces > 3 {
                rand_move()
            } else {
                let (m, depth, move_type) = tablebase.result_for_real_board(player, &board);
                match move_type {
                    TBMoveType::Win => println!("{} moves until checkmating player", depth),
                    TBMoveType::Lose => {
                        println!("{} moves until checkmated by player (optimally)", depth)
                    }
                    TBMoveType::Draw => println!("moving to known safe position"),
                }
                m
            };
            println!("CPU made {:?}", m);
            board.make_move(m);
            player = player.next();
        } else {
            print!("> ");
            std::io::stdout().flush()?;

            let mut buf = Default::default();
            let bytes_read = std::io::stdin().read_line(&mut buf)?;

            let line = buf.trim();
            if bytes_read == 0 || line == "exit" {
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
                println!("invalid input (try \"help\")");
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

fn tablebase<const N: usize, const M: usize>(parallel: Option<usize>, num_pieces: usize) {
    assert!(num_pieces >= 1);
    assert!(num_pieces <= 5);

    let mut all_pieces = Vec::new();
    for ty in [Pawn, Knight, Bishop, Rook, Queen] {
        for pl in [White, Black] {
            all_pieces.push(Piece::new(pl, ty));
        }
    }
    let wk = Piece::new(White, King);
    let bk = Piece::new(Black, King);
    let mut sets = vec![vec![wk, bk]];
    for _ in 2..num_pieces {
        let copy = sets;
        sets = Vec::new();
        for c in copy {
            for p in &all_pieces {
                let mut clone = c.clone();
                clone.push(*p);
                sets.push(clone);
            }
        }
    }
    let piece_sets = sets
        .into_iter()
        .map(|v| PieceSet::new(&v))
        .collect::<Vec<_>>();
    let tablebase = if parallel.is_some() {
        generate_tablebase_parallel::<N, M>(&piece_sets, parallel)
    } else {
        generate_tablebase(&piece_sets)
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

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Tablebase {
        #[arg(long, short, default_value = "3")]
        num_pieces: usize,
        #[arg(long, short)]
        parallel: Option<usize>,
    },
    Perft,
    NnTablebasePolicy,
    NnTablebaseValue,
    Play {
        #[arg(long = "cpu", short)]
        cpu_as_black: bool,
    },
}

fn main() -> std::io::Result<()> {
    use env_logger::{Builder, Env};
    Builder::from_env(Env::default().default_filter_or("info")).init();

    use Command::*;
    match Cli::parse().command {
        Tablebase {
            num_pieces,
            parallel,
        } => tablebase::<6, 6>(parallel, num_pieces),
        Perft => run_perft(),
        NnTablebasePolicy => nn::train_nn_tablebase_policy::<6, 6>(500, 500, 500),
        NnTablebaseValue => nn::train_nn_tablebase_value::<6, 6>(500, 500, 500),
        Play { cpu_as_black } => play_game(cpu_as_black)?,
    }

    Ok(())
}
