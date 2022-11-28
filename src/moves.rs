use crate::board::{Board, ExistingPieceResult, Move};
use crate::coord::Coord;
use crate::piece::Piece;
use crate::piece::Type::*;

fn add_move_if_result(
    moves: &mut Vec<Coord>,
    board: &Board,
    coord: Coord,
    result: ExistingPieceResult,
) -> bool {
    assert!(board.in_bounds(coord));
    if board.existing_piece_result(coord) == result {
        moves.push(coord);
        return true;
    }
    false
}

fn add_move_if_in_bounds_and_result(
    moves: &mut Vec<Coord>,
    board: &Board,
    coord: Coord,
    result: ExistingPieceResult,
) {
    if board.in_bounds(coord) {
        add_move_if_result(moves, board, coord, result);
    }
}

fn add_moves_for_rider(moves: &mut Vec<Coord>, board: &Board, coord: Coord, offsets: &[Coord]) {
    for offset in offsets {
        let mut try_coord = coord + *offset;
        while board.in_bounds(try_coord) {
            match board.existing_piece_result(try_coord) {
                ExistingPieceResult::Empty => {
                    moves.push(try_coord);
                }
                ExistingPieceResult::Friend => {
                    break;
                }
                ExistingPieceResult::Opponent => {
                    moves.push(try_coord);
                    break;
                }
            }
            try_coord = try_coord + *offset;
        }
    }
}

fn add_moves_for_leaper(moves: &mut Vec<Coord>, board: &Board, coord: Coord, offsets: &[Coord]) {
    for offset in offsets {
        let try_coord = coord + *offset;
        if board.in_bounds(try_coord) {
            match board.existing_piece_result(try_coord) {
                ExistingPieceResult::Empty | ExistingPieceResult::Opponent => {
                    moves.push(try_coord);
                }
                ExistingPieceResult::Friend => {}
            }
        }
    }
}

fn offsets(offset: Coord) -> Vec<Coord> {
    assert!(offset.x >= 0);
    assert!(offset.y >= 0);
    assert!(offset.x > 0 || offset.y > 0);
    let mut ret = Vec::new();
    ret.push(Coord {
        x: offset.x,
        y: offset.y,
    });
    ret.push(Coord {
        x: -offset.y,
        y: offset.x,
    });
    ret.push(Coord {
        x: -offset.x,
        y: -offset.y,
    });
    ret.push(Coord {
        x: offset.y,
        y: -offset.x,
    });
    if offset.x != offset.y && offset.x != 0 && offset.y != 0 {
        ret.push(Coord {
            x: offset.y,
            y: offset.x,
        });
        ret.push(Coord {
            x: -offset.x,
            y: offset.y,
        });
        ret.push(Coord {
            x: -offset.y,
            y: -offset.x,
        });
        ret.push(Coord {
            x: offset.x,
            y: -offset.y,
        });
    }
    ret
}

fn add_rook_moves(moves: &mut Vec<Coord>, board: &Board, coord: Coord) {
    add_moves_for_rider(moves, board, coord, &offsets((1, 0).into()))
}

#[cfg(test)]
fn assert_moves_eq(expected: &[Coord], moves: Vec<Coord>) {
    use std::collections::HashSet;

    let mut set = HashSet::new();
    for e in expected {
        assert!(set.insert(*e), "duplicate expected");
    }
    let mut found = HashSet::new();
    for m in moves {
        assert!(found.insert(m), "duplicate move");
    }
    assert_eq!(set, found);
}

#[test]
fn test_rook() {
    let mut board = Board::new(4, 4);
    {
        let mut moves = Vec::new();
        add_rook_moves(&mut moves, &board, Coord::new(1, 2));
        assert_moves_eq(
            &[
                Coord::new(0, 2),
                Coord::new(1, 1),
                Coord::new(1, 0),
                Coord::new(2, 2),
                Coord::new(3, 2),
                Coord::new(1, 3),
            ],
            moves,
        );
    }
    board.add_piece(
        Coord::new(2, 2),
        Piece {
            player: 0,
            ty: Rook,
        },
    );
    board.add_piece(
        Coord::new(1, 1),
        Piece {
            player: 1,
            ty: Rook,
        },
    );
    {
        let mut moves = Vec::new();
        add_rook_moves(&mut moves, &board, Coord::new(1, 2));
        assert_moves_eq(
            &[Coord::new(0, 2), Coord::new(1, 1), Coord::new(1, 3)],
            moves,
        );
    }
}

fn add_bishop_moves(moves: &mut Vec<Coord>, board: &Board, coord: Coord) {
    add_moves_for_rider(moves, board, coord, &offsets((1, 1).into()))
}

#[test]
fn test_bishop() {
    let mut board = Board::new(6, 6);
    {
        let mut moves = Vec::new();
        add_bishop_moves(&mut moves, &board, Coord::new(1, 2));
        assert_moves_eq(
            &[
                Coord::new(0, 1),
                Coord::new(2, 3),
                Coord::new(3, 4),
                Coord::new(4, 5),
                Coord::new(0, 3),
                Coord::new(2, 1),
                Coord::new(3, 0),
            ],
            moves,
        );
    }
    board.add_piece(
        Coord::new(4, 5),
        Piece {
            player: 0,
            ty: Bishop,
        },
    );
    board.add_piece(
        Coord::new(2, 1),
        Piece {
            player: 1,
            ty: Bishop,
        },
    );
    {
        let mut moves = Vec::new();
        add_bishop_moves(&mut moves, &board, Coord::new(1, 2));
        assert_moves_eq(
            &[
                Coord::new(0, 1),
                Coord::new(2, 3),
                Coord::new(3, 4),
                Coord::new(0, 3),
                Coord::new(2, 1),
            ],
            moves,
        );
    }
}

fn add_queen_moves(moves: &mut Vec<Coord>, board: &Board, coord: Coord) {
    add_rook_moves(moves, board, coord);
    add_bishop_moves(moves, board, coord);
}

fn add_knight_moves(moves: &mut Vec<Coord>, board: &Board, coord: Coord) {
    add_moves_for_leaper(moves, board, coord, &offsets((2, 1).into()))
}

#[test]
fn test_knight() {
    let mut board = Board::new(6, 6);
    {
        let mut moves = Vec::new();
        add_knight_moves(&mut moves, &board, Coord::new(2, 2));
        assert_moves_eq(
            &[
                Coord::new(1, 0),
                Coord::new(1, 4),
                Coord::new(3, 0),
                Coord::new(3, 4),
                Coord::new(0, 1),
                Coord::new(0, 3),
                Coord::new(4, 1),
                Coord::new(4, 3),
            ],
            moves,
        );
    }
    {
        let mut moves = Vec::new();
        add_knight_moves(&mut moves, &board, Coord::new(0, 0));
        assert_moves_eq(&[Coord::new(1, 2), Coord::new(2, 1)], moves);
    }
    board.add_piece(
        Coord::new(1, 0),
        Piece {
            player: 0,
            ty: Knight,
        },
    );
    board.add_piece(
        Coord::new(1, 4),
        Piece {
            player: 1,
            ty: Knight,
        },
    );
    {
        let mut moves = Vec::new();
        add_knight_moves(&mut moves, &board, Coord::new(2, 2));
        assert_moves_eq(
            &[
                Coord::new(1, 4),
                Coord::new(3, 0),
                Coord::new(3, 4),
                Coord::new(0, 1),
                Coord::new(0, 3),
                Coord::new(4, 1),
                Coord::new(4, 3),
            ],
            moves,
        );
    }
}

fn add_king_moves(moves: &mut Vec<Coord>, board: &Board, coord: Coord) {
    add_moves_for_leaper(moves, board, coord, &offsets((1, 1).into()));
    add_moves_for_leaper(moves, board, coord, &offsets((1, 0).into()));

    // castling
    if !board.get_moved(coord) {
        fn is_friendly_unmoved_rook(board: &Board, coord: Coord) -> bool {
            if let Some(piece) = board[coord].as_ref() {
                piece.ty == Rook && !board.get_moved(coord) && piece.player == board.player_turn
            } else {
                false
            }
        }

        fn no_pieces_between_exc(board: &Board, c1: Coord, c2: Coord) -> bool {
            assert!(c1.y == c2.y);
            assert!(c1.x < c2.x);
            let mut x = c1.x + 1;
            while x != c2.x {
                if board.existing_piece_result(Coord::new(x, c1.y)) != ExistingPieceResult::Empty {
                    return false;
                }
                x += 1;
            }
            true
        }

        fn no_checks_between_inc(board: &Board, c1: Coord, c2: Coord) -> bool {
            assert!(c1.y == c2.y);
            assert!(c1.x < c2.x);
            for x in c1.x..=c2.x {
                // TODO: optimize
                if is_under_attack(board, Coord::new(x, c1.y)) {
                    return false;
                }
            }
            true
        }

        let left_rook = Coord::new(0, coord.y);
        let left_dest = Coord::new(coord.x - 2, coord.y);
        if is_friendly_unmoved_rook(board, left_rook)
            && no_pieces_between_exc(board, left_rook, coord)
            && no_checks_between_inc(board, left_dest, coord)
        {
            moves.push(left_dest);
        }
        let right_rook = Coord::new(board.width - 1, coord.y);
        let right_dest = Coord::new(coord.x + 2, coord.y);
        if is_friendly_unmoved_rook(board, right_rook)
            && no_pieces_between_exc(board, coord, right_rook)
            && no_checks_between_inc(board, coord, right_dest)
        {
            moves.push(right_dest);
        }
    }
}

#[test]
fn test_king() {
    {
        let board = Board::new(6, 5);
        let mut moves = Vec::new();
        add_king_moves(&mut moves, &board, Coord::new(2, 2));
        assert_moves_eq(
            &[
                Coord::new(1, 1),
                Coord::new(1, 2),
                Coord::new(1, 3),
                Coord::new(2, 1),
                Coord::new(2, 3),
                Coord::new(3, 1),
                Coord::new(3, 2),
                Coord::new(3, 3),
            ],
            moves,
        );
    }
    {
        let board = Board::new(6, 5);
        let mut moves = Vec::new();
        add_king_moves(&mut moves, &board, Coord::new(5, 4));
        assert_moves_eq(
            &[Coord::new(4, 3), Coord::new(5, 3), Coord::new(4, 4)],
            moves,
        );
    }
    {
        let mut board = Board::new(6, 5);
        board.add_piece(
            Coord::new(1, 2),
            Piece {
                player: 0,
                ty: Knight,
            },
        );
        board.add_piece(
            Coord::new(1, 3),
            Piece {
                player: 1,
                ty: Knight,
            },
        );
        let mut moves = Vec::new();
        add_king_moves(&mut moves, &board, Coord::new(2, 2));
        assert_moves_eq(
            &[
                Coord::new(1, 1),
                Coord::new(1, 3),
                Coord::new(2, 1),
                Coord::new(2, 3),
                Coord::new(3, 1),
                Coord::new(3, 2),
                Coord::new(3, 3),
            ],
            moves,
        );
    }
    {
        let mut board = Board::new(8, 8);
        board.add_piece(
            Coord::new(0, 0),
            Piece {
                player: 0,
                ty: Rook,
            },
        );
        board.add_piece(
            Coord::new(7, 0),
            Piece {
                player: 0,
                ty: Rook,
            },
        );
        {
            let mut moves = Vec::new();
            add_king_moves(&mut moves, &board, Coord::new(4, 0));
            assert!(moves.contains(&Coord::new(2, 0)));
            assert!(moves.contains(&Coord::new(6, 0)));
        }
        {
            let mut board2 = board.clone();
            board2.add_piece(
                Coord::new(4, 2),
                Piece {
                    player: 1,
                    ty: Rook,
                },
            );
            let mut moves = Vec::new();
            add_king_moves(&mut moves, &board2, Coord::new(4, 0));
            assert!(!moves.contains(&Coord::new(2, 0)));
            assert!(!moves.contains(&Coord::new(6, 0)));
        }
        {
            let mut board2 = board.clone();
            board2.add_piece(
                Coord::new(3, 2),
                Piece {
                    player: 1,
                    ty: Rook,
                },
            );
            let mut moves = Vec::new();
            add_king_moves(&mut moves, &board2, Coord::new(4, 0));
            assert!(!moves.contains(&Coord::new(2, 0)));
            assert!(moves.contains(&Coord::new(6, 0)));
        }
        {
            let mut board2 = board.clone();
            board2.add_piece(
                Coord::new(6, 2),
                Piece {
                    player: 1,
                    ty: Rook,
                },
            );
            let mut moves = Vec::new();
            add_king_moves(&mut moves, &board2, Coord::new(4, 0));
            assert!(moves.contains(&Coord::new(2, 0)));
            assert!(!moves.contains(&Coord::new(6, 0)));
        }
        {
            let mut board2 = board.clone();
            let mut moves = Vec::new();
            board2[(0, 0)] = Some(Piece {
                player: 0,
                ty: Knight,
            });
            add_king_moves(&mut moves, &board2, Coord::new(4, 0));
            assert!(!moves.contains(&Coord::new(2, 0)));
            assert!(moves.contains(&Coord::new(6, 0)));
        }
        {
            let mut board2 = board.clone();
            let mut moves = Vec::new();
            board2[(0, 0)] = Some(Piece {
                player: 1,
                ty: Knight,
            });
            add_king_moves(&mut moves, &board2, Coord::new(4, 0));
            assert!(!moves.contains(&Coord::new(2, 0)));
            assert!(moves.contains(&Coord::new(6, 0)));
        }
        {
            let mut board2 = board.clone();
            let mut moves = Vec::new();
            board2.add_piece(
                Coord::new(1, 0),
                Piece {
                    player: 0,
                    ty: Knight,
                },
            );
            board2.add_piece(
                Coord::new(6, 0),
                Piece {
                    player: 1,
                    ty: Knight,
                },
            );
            add_king_moves(&mut moves, &board2, Coord::new(4, 0));
            assert!(!moves.contains(&Coord::new(2, 0)));
            assert!(!moves.contains(&Coord::new(6, 0)));
        }
        {
            board.set_moved(Coord::new(0, 0));
            let mut moves = Vec::new();
            add_king_moves(&mut moves, &board, Coord::new(4, 0));
            assert!(!moves.contains(&Coord::new(2, 0)));
            assert!(moves.contains(&Coord::new(6, 0)));
        }
        {
            board.set_moved(Coord::new(4, 0));
            let mut moves = Vec::new();
            add_king_moves(&mut moves, &board, Coord::new(4, 0));
            assert!(!moves.contains(&Coord::new(2, 0)));
            assert!(!moves.contains(&Coord::new(6, 0)));
        }
    }
}

fn add_pawn_moves(moves: &mut Vec<Coord>, board: &Board, coord: Coord) {
    let dy = if board.player_turn == 0 { 1 } else { -1 };
    let front = coord + Coord { x: 0, y: dy };
    assert!(board.in_bounds(front));
    let front_empty = add_move_if_result(moves, board, front, ExistingPieceResult::Empty);
    let left = coord + Coord { x: -1, y: dy };
    add_move_if_in_bounds_and_result(moves, board, left, ExistingPieceResult::Opponent);
    let right = coord + Coord { x: 1, y: dy };
    add_move_if_in_bounds_and_result(moves, board, right, ExistingPieceResult::Opponent);

    if front_empty {
        let initial_y = if board.player_turn == 0 {
            1
        } else {
            board.height - 2
        };
        if coord.y == initial_y {
            let two_spaces = front + Coord { x: 0, y: dy };
            add_move_if_result(moves, board, two_spaces, ExistingPieceResult::Empty);
        }
    }

    if let Some(p) = board.last_pawn_double_move {
        if p.y == coord.y && (p.x - coord.x).abs() == 1 {
            moves.push(p + Coord { x: 0, y: dy });
        }
    }
}

#[test]
fn test_pawn() {
    {
        let board = Board::new(8, 8);
        let mut moves = Vec::new();
        add_pawn_moves(&mut moves, &board, Coord::new(2, 3));
        assert_moves_eq(&[Coord::new(2, 4)], moves);
    }
    {
        let mut board = Board::new(8, 8);
        board.add_piece(
            Coord::new(2, 4),
            Piece {
                player: 0,
                ty: Rook,
            },
        );
        let mut moves = Vec::new();
        add_pawn_moves(&mut moves, &board, Coord::new(2, 3));
        assert_moves_eq(&[], moves);
    }
    {
        let mut board = Board::new(8, 8);
        board.add_piece(
            Coord::new(2, 4),
            Piece {
                player: 1,
                ty: Rook,
            },
        );
        let mut moves = Vec::new();
        add_pawn_moves(&mut moves, &board, Coord::new(2, 3));
        assert_moves_eq(&[], moves);
    }
    {
        let board = Board::new(8, 8);
        let mut moves = Vec::new();
        add_pawn_moves(&mut moves, &board, Coord::new(2, 1));
        assert_moves_eq(&[Coord::new(2, 2), Coord::new(2, 3)], moves);
    }
    {
        let mut board = Board::new(8, 8);
        board.add_piece(
            Coord::new(2, 3),
            Piece {
                player: 0,
                ty: Rook,
            },
        );
        let mut moves = Vec::new();
        add_pawn_moves(&mut moves, &board, Coord::new(2, 1));
        assert_moves_eq(&[Coord::new(2, 2)], moves);
    }
    {
        let mut board = Board::new(8, 8);
        board.add_piece(
            Coord::new(2, 2),
            Piece {
                player: 1,
                ty: Rook,
            },
        );
        let mut moves = Vec::new();
        add_pawn_moves(&mut moves, &board, Coord::new(2, 1));
        assert_moves_eq(&[], moves);
    }
    {
        let mut board = Board::new(8, 8);
        board.add_piece(
            Coord::new(3, 3),
            Piece {
                player: 0,
                ty: Knight,
            },
        );
        board.add_piece(
            Coord::new(2, 3),
            Piece {
                player: 1,
                ty: Knight,
            },
        );
        board.add_piece(
            Coord::new(1, 3),
            Piece {
                player: 1,
                ty: Knight,
            },
        );
        let mut moves = Vec::new();
        add_pawn_moves(&mut moves, &board, Coord::new(2, 2));
        assert_moves_eq(&[Coord::new(1, 3)], moves);
    }
    {
        let mut board = Board::new(8, 8);
        board.player_turn = 1;
        let mut moves = Vec::new();
        add_pawn_moves(&mut moves, &board, Coord::new(2, 2));
        assert_moves_eq(&[Coord::new(2, 1)], moves);
    }
    {
        let mut board = Board::new(8, 8);
        board.player_turn = 1;
        let mut moves = Vec::new();
        add_pawn_moves(&mut moves, &board, Coord::new(2, 6));
        assert_moves_eq(&[Coord::new(2, 4), Coord::new(2, 5)], moves);
    }
}

fn add_moves_for_piece(moves: &mut Vec<Coord>, board: &Board, piece: &Piece, coord: Coord) {
    match piece.ty {
        Pawn => add_pawn_moves(moves, board, coord),
        Rook => add_rook_moves(moves, board, coord),
        Bishop => add_bishop_moves(moves, board, coord),
        Queen => add_queen_moves(moves, board, coord),
        Knight => add_knight_moves(moves, board, coord),
        King => add_king_moves(moves, board, coord),
    }
}

pub fn all_moves(board: &Board) -> Vec<Move> {
    let mut moves = Vec::new();

    for y in 0..board.height {
        for x in 0..board.width {
            if let Some(piece) = board[(x, y)].as_ref() {
                if piece.player == board.player_turn {
                    let coord = (x, y).into();
                    let mut piece_moves = Vec::new();
                    add_moves_for_piece(&mut piece_moves, board, piece, coord);
                    moves.append(
                        &mut piece_moves
                            .iter()
                            .map(|c| Move {
                                from: coord,
                                to: *c,
                            })
                            .collect(),
                    );
                }
            }
        }
    }

    moves
}

// TODO: optimize
pub fn is_under_attack(board: &Board, coord: Coord) -> bool {
    let mut copy = board.clone();
    copy.advance_player();
    all_moves(&copy).iter().any(|m| m.to == coord)
}

#[test]
fn test_under_attack() {
    let mut board = Board::new(8, 8);
    board.add_piece(
        Coord::new(0, 0),
        Piece {
            player: 0,
            ty: Rook,
        },
    );
    board.add_piece(
        Coord::new(7, 7),
        Piece {
            player: 1,
            ty: Rook,
        },
    );
    assert!(is_under_attack(&board, Coord::new(6, 7)));
    assert!(!is_under_attack(&board, Coord::new(1, 0)));
}
