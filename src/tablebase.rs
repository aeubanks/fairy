use crate::board::{king_coord, Board, Move};
use crate::coord::Coord;
use crate::moves::{all_moves, is_under_attack};
use crate::piece::Piece;
use crate::piece::Type::*;
use crate::player::Player::*;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

type MapTy = HashMap<u64, (Move, u16)>;
pub struct Tablebase {
    // table of best move to play on white's turn to force a win
    white_tablebase: MapTy,
    // table of best move to play on black's turn to prolong a loss
    black_tablebase: MapTy,
}

impl Tablebase {
    pub fn new() -> Self {
        Self {
            white_tablebase: MapTy::new(),
            black_tablebase: MapTy::new(),
        }
    }
    pub fn white_add(&mut self, board: &Board, m: Move, depth: u16) {
        self.white_tablebase.insert(hash(board), (m, depth));
    }
    pub fn white_contains(&self, board: &Board) -> bool {
        self.white_tablebase.contains_key(&hash(board))
    }
    pub fn white_move(&self, board: &Board) -> Option<Move> {
        self.white_tablebase.get(&hash(board)).map(|e| e.0)
    }
    pub fn white_depth(&self, board: &Board) -> Option<u16> {
        self.white_tablebase.get(&hash(board)).map(|e| e.1)
    }
    pub fn black_add(&mut self, board: &Board, m: Move, depth: u16) {
        self.black_tablebase.insert(hash(board), (m, depth));
    }
    pub fn black_contains(&self, board: &Board) -> bool {
        self.black_tablebase.contains_key(&hash(board))
    }
    pub fn black_move(&self, board: &Board) -> Option<Move> {
        self.black_tablebase.get(&hash(board)).map(|e| e.0)
    }
    pub fn black_depth(&self, board: &Board) -> Option<u16> {
        self.black_tablebase.get(&hash(board)).map(|e| e.1)
    }
}

fn hash(board: &Board) -> u64 {
    let mut hasher = DefaultHasher::new();
    for y in 0..board.height {
        for x in 0..board.width {
            board[Coord::new(x, y)].hash(&mut hasher);
        }
    }
    hasher.finish()
}

#[test]
fn test_hash() {
    let board1 = Board::with_pieces(
        8,
        8,
        &[
            (
                Coord::new(0, 0),
                Piece {
                    player: White,
                    ty: King,
                },
            ),
            (
                Coord::new(0, 1),
                Piece {
                    player: Black,
                    ty: King,
                },
            ),
        ],
    );
    let board2 = Board::with_pieces(
        8,
        8,
        &[
            (
                Coord::new(0, 0),
                Piece {
                    player: White,
                    ty: King,
                },
            ),
            (
                Coord::new(0, 2),
                Piece {
                    player: Black,
                    ty: King,
                },
            ),
        ],
    );
    assert_eq!(hash(&board1), hash(&board1));
    assert_eq!(hash(&board2), hash(&board2));
    assert_ne!(hash(&board1), hash(&board2));
}

fn generate_all_boards_impl(ret: &mut Vec<Board>, board: &Board, pieces: &[Piece]) {
    match pieces {
        [p, rest @ ..] => {
            for y in 0..board.height {
                for x in 0..board.width {
                    if p.ty == Pawn && (y == 0 || y == board.height - 1) {
                        continue;
                    }
                    let coord = Coord::new(x, y);
                    if board[coord].is_none() {
                        let mut clone = board.clone();
                        clone.add_piece(coord, p.clone());
                        generate_all_boards_impl(ret, &clone, rest);
                    }
                }
            }
        }
        [] => ret.push(board.clone()),
    }
}

pub fn generate_all_boards(width: i8, height: i8, pieces: &[Piece]) -> Vec<Board> {
    let board = Board::new(width, height);
    let mut ret = Vec::new();
    generate_all_boards_impl(&mut ret, &board, pieces);
    ret
}

#[test]
fn test_generate_all_boards() {
    {
        let boards = generate_all_boards(
            8,
            8,
            &[Piece {
                player: White,
                ty: King,
            }],
        );
        assert_eq!(boards.len(), 64);
    }
    {
        let boards = generate_all_boards(
            8,
            8,
            &[
                Piece {
                    player: White,
                    ty: King,
                },
                Piece {
                    player: White,
                    ty: Queen,
                },
            ],
        );
        assert_eq!(boards.len(), 64 * 63);
        assert_eq!(
            boards[0][(0, 0)],
            Some(Piece {
                player: White,
                ty: King
            })
        );
        assert_eq!(
            boards[0][(1, 0)],
            Some(Piece {
                player: White,
                ty: Queen
            })
        );
        assert_eq!(boards[0][(2, 0)], None);

        assert_eq!(
            boards[1][(0, 0)],
            Some(Piece {
                player: White,
                ty: King
            })
        );
        assert_eq!(boards[1][(1, 0)], None);
        assert_eq!(
            boards[1][(2, 0)],
            Some(Piece {
                player: White,
                ty: Queen
            })
        );
    }
}

fn populate_initial_wins(tablebase: &mut Tablebase, boards: &[Board]) {
    for b in boards {
        // white can capture black's king
        let opponent_king_coord = king_coord(b, Black);
        if is_under_attack(b, opponent_king_coord, Black) {
            let all_moves = all_moves(b, White);
            let m = all_moves
                .into_iter()
                .find(|m| m.to == opponent_king_coord)
                .unwrap();
            tablebase.white_add(b, m, 1);
        }
        // stalemate is a win
        if all_moves(b, Black).is_empty() {
            tablebase.black_add(
                b,
                // arbitrary move
                Move {
                    from: Coord::new(0, 0),
                    to: Coord::new(0, 0),
                },
                0,
            );
        }
    }
}

#[test]
fn test_populate_initial_tablebases() {
    let mut tablebase = Tablebase::new();
    let boards = [
        Board::with_pieces(
            8,
            8,
            &[
                (
                    Coord::new(0, 0),
                    Piece {
                        player: White,
                        ty: King,
                    },
                ),
                (
                    Coord::new(0, 1),
                    Piece {
                        player: Black,
                        ty: King,
                    },
                ),
            ],
        ),
        Board::with_pieces(
            8,
            8,
            &[
                (
                    Coord::new(0, 0),
                    Piece {
                        player: White,
                        ty: King,
                    },
                ),
                (
                    Coord::new(0, 2),
                    Piece {
                        player: Black,
                        ty: King,
                    },
                ),
            ],
        ),
    ];
    populate_initial_wins(&mut tablebase, &boards);
    assert_eq!(
        tablebase.white_move(&boards[0]),
        Some(Move {
            from: Coord::new(0, 0),
            to: Coord::new(0, 1)
        })
    );
    assert!(!tablebase.white_contains(&boards[1]));
    assert!(!tablebase.black_contains(&boards[0]));
    assert!(!tablebase.black_contains(&boards[1]));
}

#[test]
fn test_populate_initial_tablebases_stalemate() {
    let mut tablebase = Tablebase::new();
    populate_initial_wins(
        &mut tablebase,
        &generate_all_boards(
            1,
            8,
            &[
                Piece {
                    player: White,
                    ty: King,
                },
                Piece {
                    player: Black,
                    ty: Pawn,
                },
                Piece {
                    player: Black,
                    ty: King,
                },
            ],
        ),
    );
    assert_eq!(
        tablebase.black_depth(&Board::with_pieces(
            1,
            8,
            &[
                (
                    Coord::new(0, 7),
                    Piece {
                        player: White,
                        ty: King,
                    }
                ),
                (
                    Coord::new(0, 1),
                    Piece {
                        player: Black,
                        ty: Pawn,
                    }
                ),
                (
                    Coord::new(0, 0),
                    Piece {
                        player: Black,
                        ty: King,
                    }
                ),
            ]
        )),
        Some(0)
    );
    assert!(!tablebase.black_contains(&Board::with_pieces(
        1,
        8,
        &[
            (
                Coord::new(0, 7),
                Piece {
                    player: White,
                    ty: King,
                }
            ),
            (
                Coord::new(0, 2),
                Piece {
                    player: Black,
                    ty: Pawn,
                }
            ),
            (
                Coord::new(0, 0),
                Piece {
                    player: Black,
                    ty: King,
                }
            ),
        ]
    )));
}

fn iterate_black(tablebase: &mut Tablebase, boards: &[Board]) -> bool {
    let mut made_progress = false;
    for b in boards {
        if tablebase.black_contains(b) {
            continue;
        }
        // None means no forced loss
        // Some(depth) means forced loss in depth moves
        let mut max_depth = Some(0);
        let mut best_move = None;
        let black_moves = all_moves(b, Black);
        if black_moves.is_empty() {
            // loss?
            continue;
        }
        for black_move in black_moves {
            let mut clone = b.clone();
            clone.make_move(black_move, Black);

            if let Some(depth) = tablebase.white_depth(&clone) {
                max_depth = Some(match max_depth {
                    Some(md) => {
                        // use move that prolongs checkmate as long as possible
                        if depth > md {
                            best_move = Some(black_move);
                            depth
                        } else {
                            md
                        }
                    }
                    None => {
                        best_move = Some(black_move);
                        depth
                    }
                });
            } else {
                max_depth = None;
                break;
            }
        }
        if let Some(max_depth) = max_depth {
            tablebase.black_add(b, best_move.unwrap(), max_depth + 1);
            made_progress = true;
        }
    }
    made_progress
}

#[test]
fn test_iterate_black() {
    let wk = Piece {
        player: White,
        ty: King,
    };
    let bk = Piece {
        player: Black,
        ty: King,
    };
    let board = Board::with_pieces(4, 1, &[(Coord::new(0, 0), wk), (Coord::new(2, 0), bk)]);
    let mut tablebase = Tablebase::new();
    tablebase.white_add(
        &Board::with_pieces(4, 1, &[(Coord::new(0, 0), wk), (Coord::new(1, 0), bk)]),
        Move {
            from: Coord::new(0, 0),
            to: Coord::new(0, 0),
        },
        1,
    );
    assert!(!iterate_black(&mut tablebase, &[board.clone()]));
    assert!(tablebase.black_tablebase.is_empty());
    tablebase.white_add(
        &Board::with_pieces(4, 1, &[(Coord::new(0, 0), wk), (Coord::new(3, 0), bk)]),
        Move {
            from: Coord::new(0, 0),
            to: Coord::new(0, 0),
        },
        3,
    );
    assert!(iterate_black(&mut tablebase, &[board.clone()]));
    assert_eq!(tablebase.black_depth(&board), Some(4));
    assert_eq!(
        tablebase.black_move(&board),
        Some(Move {
            from: Coord::new(2, 0),
            to: Coord::new(3, 0)
        })
    );
}

fn iterate_white(tablebase: &mut Tablebase, boards: &[Board]) -> bool {
    let mut made_progress = false;
    for b in boards {
        if tablebase.white_contains(b) {
            continue;
        }
        // None means no forced win
        // Some(depth) means forced win in depth moves
        let mut min_depth: Option<u16> = None;
        let mut best_move = None;
        let white_moves = all_moves(b, White);
        if white_moves.is_empty() {
            // loss?
            continue;
        }
        for white_move in white_moves {
            let mut clone = b.clone();
            clone.make_move(white_move, White);

            if let Some(depth) = tablebase.black_depth(&clone) {
                min_depth = Some(match min_depth {
                    Some(md) => {
                        // use move that forces checkmate as quickly as possible
                        if depth < md {
                            best_move = Some(white_move);
                            depth
                        } else {
                            md
                        }
                    }
                    None => {
                        best_move = Some(white_move);
                        depth
                    }
                });
            }
        }
        if let Some(min_depth) = min_depth {
            tablebase.white_add(b, best_move.unwrap(), min_depth + 1);
            made_progress = true;
        }
    }
    made_progress
}

#[test]
fn test_iterate_white() {
    let wk = Piece {
        player: White,
        ty: King,
    };
    let bk = Piece {
        player: Black,
        ty: King,
    };
    let board = Board::with_pieces(5, 1, &[(Coord::new(1, 0), wk), (Coord::new(4, 0), bk)]);
    let mut tablebase = Tablebase::new();
    assert!(!iterate_white(&mut tablebase, &[board.clone()]));
    assert!(tablebase.white_tablebase.is_empty());

    tablebase.black_add(
        &Board::with_pieces(5, 1, &[(Coord::new(2, 0), wk), (Coord::new(4, 0), bk)]),
        Move {
            from: Coord::new(0, 0),
            to: Coord::new(0, 0),
        },
        2,
    );
    assert!(iterate_white(&mut tablebase, &[board.clone()]));
    assert_eq!(tablebase.white_depth(&board), Some(3));
    assert_eq!(
        tablebase.white_move(&board),
        Some(Move {
            from: Coord::new(1, 0),
            to: Coord::new(2, 0)
        })
    );
    tablebase.white_tablebase.clear();
    tablebase.black_add(
        &Board::with_pieces(5, 1, &[(Coord::new(0, 0), wk), (Coord::new(4, 0), bk)]),
        Move {
            from: Coord::new(0, 0),
            to: Coord::new(0, 0),
        },
        4,
    );
    assert!(iterate_white(&mut tablebase, &[board.clone()]));
    assert_eq!(tablebase.white_depth(&board), Some(3));
    assert_eq!(
        tablebase.white_move(&board),
        Some(Move {
            from: Coord::new(1, 0),
            to: Coord::new(2, 0)
        })
    );
}

fn verify_piece_set(pieces: &[Piece]) {
    let mut wk_count = 0;
    let mut bk_count = 0;
    for p in pieces {
        if p.ty == King {
            match p.player {
                White => wk_count += 1,
                Black => bk_count += 1,
            }
        }
    }
    assert_eq!(wk_count, 1);
    assert_eq!(bk_count, 1);
}

pub fn generate_tablebase(width: i8, height: i8, piece_sets: &[&[Piece]]) -> Tablebase {
    let mut tablebase = Tablebase::new();

    for set in piece_sets {
        verify_piece_set(set);
        let all = generate_all_boards(width, height, set);
        populate_initial_wins(&mut tablebase, &all);
        loop {
            if !iterate_black(&mut tablebase, &all) {
                break;
            }
            if !iterate_white(&mut tablebase, &all) {
                break;
            }
        }
    }
    tablebase
}

#[test]
fn test_generate_king_king_tablebase() {
    let pieces = [
        Piece {
            player: White,
            ty: King,
        },
        Piece {
            player: Black,
            ty: King,
        },
    ];
    let tablebase = generate_tablebase(8, 8, &[&pieces]);
    assert!(tablebase.black_tablebase.is_empty());
    let all = generate_all_boards(8, 8, &pieces);
    for b in all {
        if is_under_attack(&b, king_coord(&b, Black), Black) {
            assert_eq!(tablebase.white_depth(&b), Some(1));
        } else {
            assert_eq!(tablebase.white_depth(&b), None);
        }
    }
}
