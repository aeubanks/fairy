mod common;

use common::*;
use fairy::board::Board;
use fairy::coord::Coord;
use fairy::piece::{Piece, Type, Type::*};
use fairy::player::{Player, Player::*};
use rand::{thread_rng, Rng};

fn valid_piece_for_coord(board: &Board, piece: &Piece, coord: Coord) -> bool {
    match piece.ty {
        Pawn => coord.y != 0 && coord.y != board.height - 1,
        _ => true,
    }
}

fn add_piece_to_rand_coord<R: Rng + ?Sized>(rng: &mut R, board: &mut Board, piece: Piece) {
    loop {
        let coord = Coord::new(
            rng.gen_range(0..board.width),
            rng.gen_range(0..board.height),
        );
        if board[coord].is_some() {
            continue;
        }
        if !valid_piece_for_coord(board, &piece, coord) {
            continue;
        }
        board.add_piece(coord, piece);
        return;
    }
}

fn rand_non_king_type<R: Rng + ?Sized>(rng: &mut R) -> Type {
    loop {
        match rng.gen::<Type>() {
            King => {}
            t => return t,
        }
    }
}

fn rand_non_king_type_for_coord<R: Rng + ?Sized>(rng: &mut R, board: &Board, coord: Coord) -> Type {
    loop {
        match rand_non_king_type(rng) {
            Pawn => {
                if coord.y != 0 && coord.y != board.height - 1 {
                    return Pawn;
                }
            }
            t => return t,
        }
    }
}

fn rand_board<R: Rng + ?Sized>(rng: &mut R) -> Board {
    let mut board = Board::new(rng.gen_range(7..=9), rng.gen_range(7..=9));
    for player in [White, Black] {
        add_piece_to_rand_coord(rng, &mut board, Piece { player, ty: King });
    }

    if rng.gen() {
        for y in 0..board.height {
            for x in 0..board.width {
                let coord = Coord::new(x, y);
                if board[coord].is_some() {
                    continue;
                }
                if rng.gen_bool(1.0 / 8.0) {
                    let piece = Piece {
                        player: rng.gen::<Player>(),
                        ty: rand_non_king_type_for_coord(rng, &board, coord),
                    };
                    board.add_piece(coord, piece);
                }
            }
        }
    } else {
        for player in [White, Black] {
            for _ in 0..(rng.gen_range(5..10)) {
                let piece = Piece {
                    player,
                    ty: rand_non_king_type(rng),
                };
                add_piece_to_rand_coord(rng, &mut board, piece);
            }
        }
    }
    board
}

#[test]
fn fuzz_is_in_check() {
    let mut rng = thread_rng();
    for _ in 0..1000 {
        let mut board = rand_board(&mut rng);
        is_in_check(&board, Player::Black);
        board.advance_player();
        is_in_check(&board, Player::White);
    }
}
