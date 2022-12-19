use crate::board::{Board, ExistingPieceResult, Move};
use crate::coord::Coord;
use crate::piece::{Piece, Type, Type::*};
use crate::player::{Player, Player::*};

fn add_move_if_result(
    moves: &mut Vec<Coord>,
    board: &Board,
    coord: Coord,
    player: Player,
    result: ExistingPieceResult,
) -> bool {
    assert!(board.in_bounds(coord));
    if board.existing_piece_result(coord, player) == result {
        moves.push(coord);
        return true;
    }
    false
}

fn add_move_if_in_bounds_and_result(
    moves: &mut Vec<Coord>,
    board: &Board,
    coord: Coord,
    player: Player,
    result: ExistingPieceResult,
) {
    if board.in_bounds(coord) {
        add_move_if_result(moves, board, coord, player, result);
    }
}

fn add_moves_for_rider(
    moves: &mut Vec<Coord>,
    board: &Board,
    coord: Coord,
    player: Player,
    rider_offset: Coord,
) {
    for offset in offsets(rider_offset) {
        let mut try_coord = coord + offset;
        while board.in_bounds(try_coord) {
            match board.existing_piece_result(try_coord, player) {
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
            try_coord = try_coord + offset;
        }
    }
}

fn add_moves_for_leaper(
    moves: &mut Vec<Coord>,
    board: &Board,
    coord: Coord,
    player: Player,
    offsets: &[Coord],
) {
    for offset in offsets {
        let try_coord = coord + *offset;
        if board.in_bounds(try_coord) {
            match board.existing_piece_result(try_coord, player) {
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

fn add_rook_moves(moves: &mut Vec<Coord>, board: &Board, coord: Coord, player: Player) {
    add_moves_for_rider(moves, board, coord, player, Coord::new(1, 0))
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
        add_rook_moves(&mut moves, &board, Coord::new(1, 2), White);
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
            player: White,
            ty: Rook,
        },
    );
    board.add_piece(
        Coord::new(1, 1),
        Piece {
            player: Black,
            ty: Rook,
        },
    );
    {
        let mut moves = Vec::new();
        add_rook_moves(&mut moves, &board, Coord::new(1, 2), White);
        assert_moves_eq(
            &[Coord::new(0, 2), Coord::new(1, 1), Coord::new(1, 3)],
            moves,
        );
    }
}

fn add_bishop_moves(moves: &mut Vec<Coord>, board: &Board, coord: Coord, player: Player) {
    add_moves_for_rider(moves, board, coord, player, Coord::new(1, 1))
}

#[test]
fn test_bishop() {
    let mut board = Board::new(6, 6);
    {
        let mut moves = Vec::new();
        add_bishop_moves(&mut moves, &board, Coord::new(1, 2), White);
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
            player: White,
            ty: Bishop,
        },
    );
    board.add_piece(
        Coord::new(2, 1),
        Piece {
            player: Black,
            ty: Bishop,
        },
    );
    {
        let mut moves = Vec::new();
        add_bishop_moves(&mut moves, &board, Coord::new(1, 2), White);
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

fn add_queen_moves(moves: &mut Vec<Coord>, board: &Board, coord: Coord, player: Player) {
    add_rook_moves(moves, board, coord, player);
    add_bishop_moves(moves, board, coord, player);
}

fn add_chancellor_moves(moves: &mut Vec<Coord>, board: &Board, coord: Coord, player: Player) {
    add_rook_moves(moves, board, coord, player);
    add_knight_moves(moves, board, coord, player);
}

fn add_archbishop_moves(moves: &mut Vec<Coord>, board: &Board, coord: Coord, player: Player) {
    add_bishop_moves(moves, board, coord, player);
    add_knight_moves(moves, board, coord, player);
}

fn add_knight_moves(moves: &mut Vec<Coord>, board: &Board, coord: Coord, player: Player) {
    add_moves_for_leaper(moves, board, coord, player, &offsets((2, 1).into()))
}

#[test]
fn test_knight() {
    let mut board = Board::new(6, 6);
    {
        let mut moves = Vec::new();
        add_knight_moves(&mut moves, &board, Coord::new(2, 2), White);
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
        add_knight_moves(&mut moves, &board, Coord::new(0, 0), White);
        assert_moves_eq(&[Coord::new(1, 2), Coord::new(2, 1)], moves);
    }
    board.add_piece(
        Coord::new(1, 0),
        Piece {
            player: White,
            ty: Knight,
        },
    );
    board.add_piece(
        Coord::new(1, 4),
        Piece {
            player: Black,
            ty: Knight,
        },
    );
    {
        let mut moves = Vec::new();
        add_knight_moves(&mut moves, &board, Coord::new(2, 2), White);
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

fn add_king_moves(moves: &mut Vec<Coord>, board: &Board, coord: Coord, player: Player) {
    add_moves_for_leaper(moves, board, coord, player, &offsets((1, 1).into()));
    add_moves_for_leaper(moves, board, coord, player, &offsets((1, 0).into()));

    // castling

    fn no_pieces_between_inc(
        board: &Board,
        player: Player,
        y: i8,
        x1: i8,
        x2: i8,
        ignore_x_1: i8,
        ignore_x_2: i8,
    ) -> bool {
        let min_x = x1.min(x2);
        let max_x = x1.max(x2);
        for x in min_x..=max_x {
            if board.existing_piece_result(Coord::new(x, y), player) != ExistingPieceResult::Empty
                && ignore_x_1 != x
                && ignore_x_2 != x
            {
                return false;
            }
        }
        true
    }

    fn no_checks_between_inc(board: &Board, player: Player, y: i8, x1: i8, x2: i8) -> bool {
        let min_x = x1.min(x2);
        let max_x = x1.max(x2);
        for x in min_x..=max_x {
            // TODO: optimize
            if is_under_attack(board, Coord::new(x, y), player) {
                return false;
            }
        }
        true
    }

    let idx = match player {
        White => 0,
        Black => 2,
    };
    // make sure the first value is the left rook and the second value is the right rook
    if let Some(a) = board.castling_rights[idx] {
        if let Some(b) = board.castling_rights[idx + 1] {
            assert!(a.x < b.x);
        }
    }
    for (rook_coord, king_dest_x, rook_dest_x) in [
        (board.castling_rights[idx], 2, 3),
        (
            board.castling_rights[idx + 1],
            board.width - 2,
            board.width - 3,
        ),
    ] {
        if let Some(rook_coord) = rook_coord {
            {
                assert_eq!(coord.y, rook_coord.y);
                let piece = board[rook_coord].as_ref().unwrap();
                assert_eq!(piece.ty, Rook);
                assert_eq!(piece.player, player);
            }

            if no_pieces_between_inc(
                board,
                player,
                coord.y,
                coord.x,
                king_dest_x,
                coord.x,
                rook_coord.x,
            ) && no_pieces_between_inc(
                board,
                player,
                coord.y,
                rook_coord.x,
                rook_dest_x,
                coord.x,
                rook_coord.x,
            ) && no_checks_between_inc(board, player, coord.y, king_dest_x, coord.x)
            {
                moves.push(rook_coord);
            }
        }
    }
}

#[test]
fn test_king() {
    {
        let board = Board::new(6, 5);
        let mut moves = Vec::new();
        add_king_moves(&mut moves, &board, Coord::new(2, 2), White);
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
        add_king_moves(&mut moves, &board, Coord::new(5, 4), White);
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
                player: White,
                ty: Knight,
            },
        );
        board.add_piece(
            Coord::new(1, 3),
            Piece {
                player: Black,
                ty: Knight,
            },
        );
        let mut moves = Vec::new();
        add_king_moves(&mut moves, &board, Coord::new(2, 2), White);
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
                player: White,
                ty: Rook,
            },
        );
        board.add_piece(
            Coord::new(7, 0),
            Piece {
                player: White,
                ty: Rook,
            },
        );
        {
            let mut moves = Vec::new();
            add_king_moves(&mut moves, &board, Coord::new(4, 0), White);
            assert!(!moves.contains(&Coord::new(2, 0)));
            assert!(!moves.contains(&Coord::new(6, 0)));
        }
    }
    {
        let mut board = Board::new(8, 8);
        board.castling_rights = [
            Some(Coord::new(0, 0)),
            Some(Coord::new(7, 0)),
            Some(Coord::new(0, 7)),
            Some(Coord::new(7, 7)),
        ];
        board.add_piece(
            Coord::new(0, 0),
            Piece {
                player: White,
                ty: Rook,
            },
        );
        board.add_piece(
            Coord::new(7, 0),
            Piece {
                player: White,
                ty: Rook,
            },
        );
        {
            let mut moves = Vec::new();
            add_king_moves(&mut moves, &board, Coord::new(4, 0), White);
            assert!(moves.contains(&Coord::new(0, 0)));
            assert!(moves.contains(&Coord::new(7, 0)));
        }
        {
            let mut board2 = board.clone();
            board2.add_piece(
                Coord::new(4, 2),
                Piece {
                    player: Black,
                    ty: Rook,
                },
            );
            let mut moves = Vec::new();
            add_king_moves(&mut moves, &board2, Coord::new(4, 0), White);
            assert!(!moves.contains(&Coord::new(0, 0)));
            assert!(!moves.contains(&Coord::new(7, 0)));
        }
        {
            let mut board2 = board.clone();
            board2.add_piece(
                Coord::new(3, 2),
                Piece {
                    player: Black,
                    ty: Rook,
                },
            );
            let mut moves = Vec::new();
            add_king_moves(&mut moves, &board2, Coord::new(4, 0), White);
            assert!(!moves.contains(&Coord::new(0, 0)));
            assert!(moves.contains(&Coord::new(7, 0)));
        }
        {
            let mut board2 = board.clone();
            board2.add_piece(
                Coord::new(6, 2),
                Piece {
                    player: Black,
                    ty: Rook,
                },
            );
            let mut moves = Vec::new();
            add_king_moves(&mut moves, &board2, Coord::new(4, 0), White);
            assert!(moves.contains(&Coord::new(0, 0)));
            assert!(!moves.contains(&Coord::new(7, 0)));
        }
        {
            let mut board2 = board.clone();
            let mut moves = Vec::new();
            board2.add_piece(
                Coord::new(1, 0),
                Piece {
                    player: White,
                    ty: Knight,
                },
            );
            board2.add_piece(
                Coord::new(6, 0),
                Piece {
                    player: Black,
                    ty: Knight,
                },
            );
            add_king_moves(&mut moves, &board2, Coord::new(4, 0), White);
            assert!(!moves.contains(&Coord::new(0, 0)));
            assert!(!moves.contains(&Coord::new(7, 0)));
        }
        {
            board.castling_rights[0] = None;
            let mut moves = Vec::new();
            add_king_moves(&mut moves, &board, Coord::new(4, 0), White);
            assert!(!moves.contains(&Coord::new(0, 0)));
            assert!(moves.contains(&Coord::new(7, 0)));
        }
        {
            board.castling_rights[1] = None;
            let mut moves = Vec::new();
            add_king_moves(&mut moves, &board, Coord::new(4, 0), White);
            assert!(!moves.contains(&Coord::new(0, 0)));
            assert!(!moves.contains(&Coord::new(7, 0)));
        }
    }
    {
        let mut board = Board::with_pieces(
            8,
            8,
            &[
                (
                    Coord::new(0, 0),
                    Piece {
                        player: White,
                        ty: Rook,
                    },
                ),
                (
                    Coord::new(1, 0),
                    Piece {
                        player: White,
                        ty: King,
                    },
                ),
                (
                    Coord::new(7, 0),
                    Piece {
                        player: White,
                        ty: Rook,
                    },
                ),
            ],
        );
        board.castling_rights = [Some(Coord::new(0, 0)), Some(Coord::new(7, 0)), None, None];
        {
            let mut moves = Vec::new();
            add_king_moves(&mut moves, &board, Coord::new(1, 0), White);
            assert!(moves.contains(&Coord::new(0, 0)));
            assert!(moves.contains(&Coord::new(7, 0)));
        }
    }
    {
        let mut board = Board::with_pieces(
            8,
            8,
            &[
                (
                    Coord::new(0, 0),
                    Piece {
                        player: White,
                        ty: Rook,
                    },
                ),
                (
                    Coord::new(6, 0),
                    Piece {
                        player: White,
                        ty: King,
                    },
                ),
                (
                    Coord::new(7, 0),
                    Piece {
                        player: White,
                        ty: Rook,
                    },
                ),
            ],
        );
        board.castling_rights = [Some(Coord::new(0, 0)), Some(Coord::new(7, 0)), None, None];
        {
            let mut moves = Vec::new();
            add_king_moves(&mut moves, &board, Coord::new(6, 0), White);
            assert!(moves.contains(&Coord::new(0, 0)));
            assert!(moves.contains(&Coord::new(7, 0)));
        }
    }
    {
        let mut board = Board::with_pieces(
            8,
            8,
            &[
                (
                    Coord::new(1, 0),
                    Piece {
                        player: White,
                        ty: Rook,
                    },
                ),
                (
                    Coord::new(3, 0),
                    Piece {
                        player: White,
                        ty: King,
                    },
                ),
                (
                    Coord::new(5, 0),
                    Piece {
                        player: White,
                        ty: Rook,
                    },
                ),
            ],
        );
        board.castling_rights = [Some(Coord::new(1, 0)), Some(Coord::new(5, 0)), None, None];
        {
            let mut moves = Vec::new();
            add_king_moves(&mut moves, &board, Coord::new(3, 0), White);
            assert!(moves.contains(&Coord::new(1, 0)));
            assert!(moves.contains(&Coord::new(5, 0)));
        }
    }
    {
        let mut board = Board::with_pieces(
            8,
            8,
            &[
                (
                    Coord::new(1, 0),
                    Piece {
                        player: White,
                        ty: Rook,
                    },
                ),
                (
                    Coord::new(5, 0),
                    Piece {
                        player: White,
                        ty: King,
                    },
                ),
                (
                    Coord::new(6, 0),
                    Piece {
                        player: White,
                        ty: Rook,
                    },
                ),
            ],
        );
        board.castling_rights = [Some(Coord::new(1, 0)), Some(Coord::new(6, 0)), None, None];
        {
            let mut moves = Vec::new();
            add_king_moves(&mut moves, &board, Coord::new(5, 0), White);
            assert!(moves.contains(&Coord::new(1, 0)));
            assert!(moves.contains(&Coord::new(6, 0)));
        }
    }
    {
        let mut board = Board::with_pieces(
            8,
            8,
            &[
                (
                    Coord::new(0, 0),
                    Piece {
                        player: White,
                        ty: Rook,
                    },
                ),
                (
                    Coord::new(1, 0),
                    Piece {
                        player: White,
                        ty: King,
                    },
                ),
                (
                    Coord::new(2, 0),
                    Piece {
                        player: White,
                        ty: Rook,
                    },
                ),
            ],
        );
        board.castling_rights = [Some(Coord::new(0, 0)), Some(Coord::new(2, 0)), None, None];
        {
            let mut moves = Vec::new();
            add_king_moves(&mut moves, &board, Coord::new(1, 0), White);
            assert!(!moves.contains(&Coord::new(0, 0)));
            assert!(moves.contains(&Coord::new(2, 0)));
        }
    }
    {
        let mut board = Board::with_pieces(
            8,
            8,
            &[
                (
                    Coord::new(5, 0),
                    Piece {
                        player: White,
                        ty: Rook,
                    },
                ),
                (
                    Coord::new(6, 0),
                    Piece {
                        player: White,
                        ty: King,
                    },
                ),
                (
                    Coord::new(7, 0),
                    Piece {
                        player: White,
                        ty: Rook,
                    },
                ),
            ],
        );
        board.castling_rights = [Some(Coord::new(5, 0)), Some(Coord::new(7, 0)), None, None];
        {
            let mut moves = Vec::new();
            add_king_moves(&mut moves, &board, Coord::new(6, 0), White);
            assert!(moves.contains(&Coord::new(5, 0)));
            assert!(!moves.contains(&Coord::new(7, 0)));
        }
    }
    {
        let mut board = Board::with_pieces(
            8,
            8,
            &[
                (
                    Coord::new(3, 0),
                    Piece {
                        player: White,
                        ty: Rook,
                    },
                ),
                (
                    Coord::new(4, 0),
                    Piece {
                        player: White,
                        ty: King,
                    },
                ),
                (
                    Coord::new(5, 0),
                    Piece {
                        player: White,
                        ty: Rook,
                    },
                ),
            ],
        );
        board.castling_rights = [Some(Coord::new(3, 0)), Some(Coord::new(5, 0)), None, None];
        {
            let mut moves = Vec::new();
            add_king_moves(&mut moves, &board, Coord::new(4, 0), White);
            assert!(moves.contains(&Coord::new(3, 0)));
            assert!(moves.contains(&Coord::new(5, 0)));
        }
    }
    {
        let mut board = Board::with_pieces(
            8,
            8,
            &[
                (
                    Coord::new(0, 0),
                    Piece {
                        player: White,
                        ty: Rook,
                    },
                ),
                (
                    Coord::new(6, 0),
                    Piece {
                        player: White,
                        ty: King,
                    },
                ),
                (
                    Coord::new(7, 0),
                    Piece {
                        player: White,
                        ty: Rook,
                    },
                ),
            ],
        );
        board.castling_rights = [Some(Coord::new(0, 0)), Some(Coord::new(7, 0)), None, None];
        {
            let mut moves = Vec::new();
            add_king_moves(&mut moves, &board, Coord::new(6, 0), White);
            assert!(moves.contains(&Coord::new(0, 0)));
            assert!(moves.contains(&Coord::new(7, 0)));
        }
    }
    {
        let mut board = Board::with_pieces(
            8,
            8,
            &[
                (
                    Coord::new(0, 0),
                    Piece {
                        player: White,
                        ty: Rook,
                    },
                ),
                (
                    Coord::new(2, 0),
                    Piece {
                        player: White,
                        ty: King,
                    },
                ),
                (
                    Coord::new(7, 0),
                    Piece {
                        player: White,
                        ty: Rook,
                    },
                ),
            ],
        );
        board.castling_rights = [Some(Coord::new(0, 0)), Some(Coord::new(7, 0)), None, None];
        {
            let mut moves = Vec::new();
            add_king_moves(&mut moves, &board, Coord::new(2, 0), White);
            assert!(moves.contains(&Coord::new(0, 0)));
            assert!(moves.contains(&Coord::new(7, 0)));
        }
    }
}

fn add_pawn_moves(moves: &mut Vec<Coord>, board: &Board, coord: Coord, player: Player) {
    let dy = match player {
        White => 1,
        Black => -1,
    };

    let left = coord + Coord { x: -1, y: dy };
    add_move_if_in_bounds_and_result(moves, board, left, player, ExistingPieceResult::Opponent);
    let right = coord + Coord { x: 1, y: dy };
    add_move_if_in_bounds_and_result(moves, board, right, player, ExistingPieceResult::Opponent);

    let front = coord + Coord { x: 0, y: dy };
    assert!(board.in_bounds(front));
    let front_empty = add_move_if_result(moves, board, front, player, ExistingPieceResult::Empty);

    if front_empty {
        let initial_y = match player {
            White => 1,
            Black => board.height - 2,
        };
        if coord.y == initial_y {
            let max_forward = (board.height / 2 - 2).max(1);
            let mut front2 = front;
            for _ in 1..max_forward {
                front2 = front2 + Coord { x: 0, y: dy };
                let empty =
                    add_move_if_result(moves, board, front2, player, ExistingPieceResult::Empty);
                if !empty {
                    break;
                }
            }
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
        add_pawn_moves(&mut moves, &board, Coord::new(2, 3), White);
        assert_moves_eq(&[Coord::new(2, 4)], moves);
    }
    {
        let mut board = Board::new(8, 8);
        board.add_piece(
            Coord::new(2, 4),
            Piece {
                player: White,
                ty: Rook,
            },
        );
        let mut moves = Vec::new();
        add_pawn_moves(&mut moves, &board, Coord::new(2, 3), White);
        assert_moves_eq(&[], moves);
    }
    {
        let mut board = Board::new(8, 8);
        board.add_piece(
            Coord::new(2, 4),
            Piece {
                player: Black,
                ty: Rook,
            },
        );
        let mut moves = Vec::new();
        add_pawn_moves(&mut moves, &board, Coord::new(2, 3), White);
        assert_moves_eq(&[], moves);
    }
    {
        let board = Board::new(8, 8);
        let mut moves = Vec::new();
        add_pawn_moves(&mut moves, &board, Coord::new(2, 1), White);
        assert_moves_eq(&[Coord::new(2, 2), Coord::new(2, 3)], moves);
    }
    {
        let mut board = Board::new(8, 8);
        board.add_piece(
            Coord::new(2, 3),
            Piece {
                player: White,
                ty: Rook,
            },
        );
        let mut moves = Vec::new();
        add_pawn_moves(&mut moves, &board, Coord::new(2, 1), White);
        assert_moves_eq(&[Coord::new(2, 2)], moves);
    }
    {
        let mut board = Board::new(8, 8);
        board.add_piece(
            Coord::new(2, 2),
            Piece {
                player: Black,
                ty: Rook,
            },
        );
        let mut moves = Vec::new();
        add_pawn_moves(&mut moves, &board, Coord::new(2, 1), White);
        assert_moves_eq(&[], moves);
    }
    {
        let mut board = Board::new(8, 8);
        board.add_piece(
            Coord::new(3, 3),
            Piece {
                player: White,
                ty: Knight,
            },
        );
        board.add_piece(
            Coord::new(2, 3),
            Piece {
                player: Black,
                ty: Knight,
            },
        );
        board.add_piece(
            Coord::new(1, 3),
            Piece {
                player: Black,
                ty: Knight,
            },
        );
        let mut moves = Vec::new();
        add_pawn_moves(&mut moves, &board, Coord::new(2, 2), White);
        assert_moves_eq(&[Coord::new(1, 3)], moves);
    }
    {
        let board = Board::new(8, 8);
        let mut moves = Vec::new();
        add_pawn_moves(&mut moves, &board, Coord::new(2, 2), Black);
        assert_moves_eq(&[Coord::new(2, 1)], moves);
    }
    {
        let board = Board::new(8, 8);
        let mut moves = Vec::new();
        add_pawn_moves(&mut moves, &board, Coord::new(2, 6), Black);
        assert_moves_eq(&[Coord::new(2, 4), Coord::new(2, 5)], moves);
    }
    {
        let board = Board::new(8, 12);
        let mut moves = Vec::new();
        add_pawn_moves(&mut moves, &board, Coord::new(2, 1), White);
        assert_moves_eq(
            &[
                Coord::new(2, 2),
                Coord::new(2, 3),
                Coord::new(2, 4),
                Coord::new(2, 5),
            ],
            moves,
        );
    }
    {
        let board = Board::new(8, 13);
        let mut moves = Vec::new();
        add_pawn_moves(&mut moves, &board, Coord::new(2, 1), White);
        assert_moves_eq(
            &[
                Coord::new(2, 2),
                Coord::new(2, 3),
                Coord::new(2, 4),
                Coord::new(2, 5),
            ],
            moves,
        );
    }
    {
        let board = Board::new(8, 12);
        let mut moves = Vec::new();
        add_pawn_moves(&mut moves, &board, Coord::new(3, 10), Black);
        assert_moves_eq(
            &[
                Coord::new(3, 9),
                Coord::new(3, 8),
                Coord::new(3, 7),
                Coord::new(3, 6),
            ],
            moves,
        );
    }
    {
        let mut board = Board::new(8, 12);
        board.add_piece(
            Coord::new(3, 7),
            Piece {
                player: White,
                ty: Knight,
            },
        );
        let mut moves = Vec::new();
        add_pawn_moves(&mut moves, &board, Coord::new(3, 10), Black);
        assert_moves_eq(&[Coord::new(3, 9), Coord::new(3, 8)], moves);
    }
}

fn add_moves_for_piece(moves: &mut Vec<Coord>, board: &Board, piece: &Piece, coord: Coord) {
    match piece.ty {
        Pawn => add_pawn_moves(moves, board, coord, piece.player),
        Rook => add_rook_moves(moves, board, coord, piece.player),
        Bishop => add_bishop_moves(moves, board, coord, piece.player),
        Queen => add_queen_moves(moves, board, coord, piece.player),
        Knight => add_knight_moves(moves, board, coord, piece.player),
        King => add_king_moves(moves, board, coord, piece.player),
        Chancellor => add_chancellor_moves(moves, board, coord, piece.player),
        Archbishop => add_archbishop_moves(moves, board, coord, piece.player),
    }
}

pub fn all_moves(board: &Board, player: Player) -> Vec<Move> {
    let mut moves = Vec::new();

    for y in 0..board.height {
        for x in 0..board.width {
            if let Some(piece) = board[(x, y)].as_ref() {
                if piece.player == player {
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

fn enemy_piece_rider(board: &Board, coord: Coord, offset: Coord, player: Player) -> Option<Type> {
    let mut try_coord = coord + offset;
    while board.in_bounds(try_coord) {
        if let Some(p) = board[try_coord].as_ref() {
            if p.player != player {
                return Some(p.ty);
            } else {
                return None;
            }
        }
        try_coord = try_coord + offset;
    }
    None
}

fn enemy_piece_leaper(board: &Board, coord: Coord, offset: Coord, player: Player) -> Option<Type> {
    let try_coord = coord + offset;
    if board.in_bounds(try_coord) {
        if let Some(p) = board[try_coord].as_ref() {
            if p.player != player {
                return Some(p.ty);
            } else {
                return None;
            }
        }
    }
    None
}

pub fn is_under_attack(board: &Board, coord: Coord, player: Player) -> bool {
    if let Some(p) = board[coord].as_ref() {
        assert_eq!(p.player, player);
    }
    for o in offsets(Coord::new(1, 0)) {
        match enemy_piece_rider(board, coord, o, player) {
            Some(Queen) | Some(Rook) | Some(Chancellor) => return true,
            _ => {}
        }
    }
    for o in offsets(Coord::new(1, 1)) {
        match enemy_piece_rider(board, coord, o, player) {
            Some(Queen) | Some(Bishop) | Some(Archbishop) => return true,
            _ => {}
        }
    }
    for o in offsets(Coord::new(1, 2)) {
        match enemy_piece_leaper(board, coord, o, player) {
            Some(Knight) | Some(Archbishop) | Some(Chancellor) => return true,
            _ => {}
        }
    }
    for o in offsets(Coord::new(1, 1)) {
        match enemy_piece_leaper(board, coord, o, player) {
            Some(King) => return true,
            _ => {}
        }
    }
    for o in offsets(Coord::new(1, 0)) {
        match enemy_piece_leaper(board, coord, o, player) {
            Some(King) => return true,
            _ => {}
        }
    }
    fn has_enemy_pawn(board: &Board, coord: Coord, player: Player) -> bool {
        if board.in_bounds(coord) {
            if let Some(p) = board[coord].as_ref() {
                return p.player != player && p.ty == Pawn;
            }
        }
        false
    }
    if player == White {
        if has_enemy_pawn(board, coord + Coord::new(1, 1), player)
            || has_enemy_pawn(board, coord + Coord::new(-1, 1), player)
        {
            return true;
        }
    } else {
        if has_enemy_pawn(board, coord + Coord::new(1, -1), player)
            || has_enemy_pawn(board, coord + Coord::new(-1, -1), player)
        {
            return true;
        }
    }
    false
}

#[test]
fn test_under_attack() {
    {
        let board = Board::with_pieces(
            8,
            8,
            &[
                (
                    Coord::new(0, 0),
                    Piece {
                        player: White,
                        ty: Rook,
                    },
                ),
                (
                    Coord::new(3, 0),
                    Piece {
                        player: White,
                        ty: Pawn,
                    },
                ),
                (
                    Coord::new(0, 3),
                    Piece {
                        player: Black,
                        ty: Pawn,
                    },
                ),
                (
                    Coord::new(7, 7),
                    Piece {
                        player: Black,
                        ty: Rook,
                    },
                ),
            ],
        );
        assert!(is_under_attack(&board, Coord::new(6, 7), White));
        assert!(is_under_attack(&board, Coord::new(5, 7), White));
        assert!(!is_under_attack(&board, Coord::new(1, 0), White));
        assert!(!is_under_attack(&board, Coord::new(0, 0), White));

        assert!(is_under_attack(&board, Coord::new(1, 0), Black));
        assert!(is_under_attack(&board, Coord::new(2, 0), Black));
        assert!(!is_under_attack(&board, Coord::new(4, 0), Black));
        assert!(is_under_attack(&board, Coord::new(0, 1), Black));
        assert!(is_under_attack(&board, Coord::new(0, 2), Black));
        assert!(is_under_attack(&board, Coord::new(0, 3), Black));
        assert!(!is_under_attack(&board, Coord::new(0, 4), Black));
        assert!(!is_under_attack(&board, Coord::new(6, 7), Black));
        assert!(!is_under_attack(&board, Coord::new(7, 7), Black));
    }
    {
        let board = Board::with_pieces(
            8,
            8,
            &[(
                Coord::new(1, 1),
                Piece {
                    player: White,
                    ty: Bishop,
                },
            )],
        );
        assert!(is_under_attack(&board, Coord::new(0, 0), Black));
        assert!(is_under_attack(&board, Coord::new(2, 2), Black));
        assert!(is_under_attack(&board, Coord::new(3, 3), Black));
        assert!(is_under_attack(&board, Coord::new(0, 2), Black));
        assert!(is_under_attack(&board, Coord::new(2, 0), Black));
        assert!(!is_under_attack(&board, Coord::new(1, 0), Black));
        assert!(!is_under_attack(&board, Coord::new(0, 1), Black));
        assert!(!is_under_attack(&board, Coord::new(1, 2), Black));
        assert!(!is_under_attack(&board, Coord::new(2, 1), Black));
    }
    {
        let board = Board::with_pieces(
            8,
            8,
            &[(
                Coord::new(3, 3),
                Piece {
                    player: White,
                    ty: Archbishop,
                },
            )],
        );
        assert!(is_under_attack(&board, Coord::new(1, 2), Black));
        assert!(is_under_attack(&board, Coord::new(1, 4), Black));
        assert!(is_under_attack(&board, Coord::new(2, 1), Black));
        assert!(is_under_attack(&board, Coord::new(2, 5), Black));
        assert!(is_under_attack(&board, Coord::new(4, 1), Black));
        assert!(is_under_attack(&board, Coord::new(4, 5), Black));
        assert!(is_under_attack(&board, Coord::new(5, 2), Black));
        assert!(is_under_attack(&board, Coord::new(5, 4), Black));
        assert!(is_under_attack(&board, Coord::new(2, 2), Black));
        assert!(is_under_attack(&board, Coord::new(1, 1), Black));
        assert!(is_under_attack(&board, Coord::new(2, 4), Black));
        assert!(is_under_attack(&board, Coord::new(4, 2), Black));
        assert!(is_under_attack(&board, Coord::new(4, 4), Black));
        assert!(!is_under_attack(&board, Coord::new(2, 3), Black));
    }
    {
        let board = Board::with_pieces(
            8,
            8,
            &[(
                Coord::new(3, 3),
                Piece {
                    player: White,
                    ty: Chancellor,
                },
            )],
        );
        assert!(is_under_attack(&board, Coord::new(1, 2), Black));
        assert!(is_under_attack(&board, Coord::new(1, 4), Black));
        assert!(is_under_attack(&board, Coord::new(2, 1), Black));
        assert!(is_under_attack(&board, Coord::new(2, 5), Black));
        assert!(is_under_attack(&board, Coord::new(4, 1), Black));
        assert!(is_under_attack(&board, Coord::new(4, 5), Black));
        assert!(is_under_attack(&board, Coord::new(5, 2), Black));
        assert!(is_under_attack(&board, Coord::new(5, 4), Black));
        assert!(is_under_attack(&board, Coord::new(3, 4), Black));
        assert!(is_under_attack(&board, Coord::new(3, 5), Black));
        assert!(is_under_attack(&board, Coord::new(2, 3), Black));
        assert!(is_under_attack(&board, Coord::new(4, 3), Black));
        assert!(is_under_attack(&board, Coord::new(3, 2), Black));
        assert!(!is_under_attack(&board, Coord::new(2, 2), Black));
    }
    {
        let board = Board::with_pieces(
            8,
            8,
            &[(
                Coord::new(1, 1),
                Piece {
                    player: White,
                    ty: Queen,
                },
            )],
        );
        assert!(is_under_attack(&board, Coord::new(0, 0), Black));
        assert!(is_under_attack(&board, Coord::new(2, 2), Black));
        assert!(is_under_attack(&board, Coord::new(3, 3), Black));
        assert!(is_under_attack(&board, Coord::new(0, 2), Black));
        assert!(is_under_attack(&board, Coord::new(2, 0), Black));
        assert!(is_under_attack(&board, Coord::new(1, 0), Black));
        assert!(is_under_attack(&board, Coord::new(0, 1), Black));
        assert!(is_under_attack(&board, Coord::new(1, 2), Black));
        assert!(is_under_attack(&board, Coord::new(1, 3), Black));
        assert!(is_under_attack(&board, Coord::new(2, 1), Black));
        assert!(!is_under_attack(&board, Coord::new(2, 3), Black));
    }
    {
        let board = Board::with_pieces(
            8,
            8,
            &[(
                Coord::new(1, 1),
                Piece {
                    player: White,
                    ty: King,
                },
            )],
        );
        assert!(is_under_attack(&board, Coord::new(0, 0), Black));
        assert!(is_under_attack(&board, Coord::new(2, 2), Black));
        assert!(!is_under_attack(&board, Coord::new(3, 3), Black));
        assert!(is_under_attack(&board, Coord::new(0, 2), Black));
        assert!(is_under_attack(&board, Coord::new(2, 0), Black));
        assert!(is_under_attack(&board, Coord::new(1, 0), Black));
        assert!(is_under_attack(&board, Coord::new(0, 1), Black));
        assert!(is_under_attack(&board, Coord::new(1, 2), Black));
        assert!(!is_under_attack(&board, Coord::new(1, 3), Black));
        assert!(is_under_attack(&board, Coord::new(2, 1), Black));
        assert!(!is_under_attack(&board, Coord::new(2, 3), Black));
    }
    {
        let board = Board::with_pieces(
            8,
            8,
            &[(
                Coord::new(1, 1),
                Piece {
                    player: White,
                    ty: Knight,
                },
            )],
        );
        assert!(is_under_attack(&board, Coord::new(0, 3), Black));
        assert!(is_under_attack(&board, Coord::new(2, 3), Black));
        assert!(!is_under_attack(&board, Coord::new(3, 5), Black));
    }
    {
        let board = Board::with_pieces(
            8,
            8,
            &[
                (
                    Coord::new(1, 1),
                    Piece {
                        player: White,
                        ty: Pawn,
                    },
                ),
                (
                    Coord::new(6, 6),
                    Piece {
                        player: Black,
                        ty: Pawn,
                    },
                ),
            ],
        );
        assert!(is_under_attack(&board, Coord::new(5, 5), White));
        assert!(is_under_attack(&board, Coord::new(7, 5), White));
        assert!(!is_under_attack(&board, Coord::new(6, 5), White));
        assert!(!is_under_attack(&board, Coord::new(6, 4), White));
        assert!(!is_under_attack(&board, Coord::new(5, 7), White));
        assert!(!is_under_attack(&board, Coord::new(6, 7), White));
        assert!(!is_under_attack(&board, Coord::new(7, 7), White));

        assert!(is_under_attack(&board, Coord::new(0, 2), Black));
        assert!(is_under_attack(&board, Coord::new(2, 2), Black));
        assert!(!is_under_attack(&board, Coord::new(1, 2), Black));
        assert!(!is_under_attack(&board, Coord::new(1, 3), Black));
        assert!(!is_under_attack(&board, Coord::new(0, 0), Black));
        assert!(!is_under_attack(&board, Coord::new(1, 0), Black));
        assert!(!is_under_attack(&board, Coord::new(2, 0), Black));
    }
}
