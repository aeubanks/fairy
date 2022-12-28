use crate::board::{king_coord, Board, ExistingPieceResult, Move};
use crate::coord::Coord;
use crate::moves::{all_moves, all_moves_to_end_at_board_no_captures, under_attack_from_coord};
use crate::piece::Piece;
use crate::piece::Type::*;
use crate::player::{Player, Player::*};
use arrayvec::ArrayVec;
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

// TODO: factor out coord visiting
fn board_has_pawn<const W: usize, const H: usize>(board: &Board<W, H>) -> bool {
    for y in 0..H as i8 {
        for x in 0..W as i8 {
            let coord = Coord::new(x, y);
            if let Some(piece) = board[coord].as_ref() {
                if piece.ty() == Pawn {
                    return true;
                }
            }
        }
    }
    false
}

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
struct Symmetry {
    flip_x: bool,
    flip_y: bool,
    flip_diagonally: bool,
}

#[must_use]
fn flip_coord(mut c: Coord, sym: Symmetry, width: i8, height: i8) -> Coord {
    if sym.flip_x {
        c.x = width - 1 - c.x;
    }
    if sym.flip_y {
        c.y = height - 1 - c.y;
    }
    if sym.flip_diagonally {
        std::mem::swap(&mut c.x, &mut c.y);
    }
    c
}

#[test]
fn test_flip_coord() {
    assert_eq!(
        flip_coord(
            Coord::new(1, 2),
            Symmetry {
                flip_x: false,
                flip_y: false,
                flip_diagonally: false
            },
            4,
            4
        ),
        Coord::new(1, 2)
    );
    assert_eq!(
        flip_coord(
            Coord::new(1, 2),
            Symmetry {
                flip_x: true,
                flip_y: false,
                flip_diagonally: false
            },
            4,
            4
        ),
        Coord::new(2, 2)
    );
    assert_eq!(
        flip_coord(
            Coord::new(1, 2),
            Symmetry {
                flip_x: false,
                flip_y: true,
                flip_diagonally: false
            },
            4,
            4
        ),
        Coord::new(1, 1)
    );
    assert_eq!(
        flip_coord(
            Coord::new(1, 2),
            Symmetry {
                flip_x: false,
                flip_y: false,
                flip_diagonally: true
            },
            4,
            4
        ),
        Coord::new(2, 1)
    );
    assert_eq!(
        flip_coord(
            Coord::new(0, 2),
            Symmetry {
                flip_x: true,
                flip_y: true,
                flip_diagonally: true
            },
            4,
            4
        ),
        Coord::new(1, 3)
    );
}

#[must_use]
fn flip_move(mut m: Move, sym: Symmetry, width: i8, height: i8) -> Move {
    m.from = flip_coord(m.from, sym, width, height);
    m.to = flip_coord(m.to, sym, width, height);
    m
}

#[must_use]
fn unflip_coord(mut c: Coord, sym: Symmetry, width: i8, height: i8) -> Coord {
    if sym.flip_diagonally {
        std::mem::swap(&mut c.x, &mut c.y);
    }
    if sym.flip_y {
        c.y = height - 1 - c.y;
    }
    if sym.flip_x {
        c.x = width - 1 - c.x;
    }
    c
}

#[must_use]
fn unflip_move(mut m: Move, sym: Symmetry, width: i8, height: i8) -> Move {
    m.from = unflip_coord(m.from, sym, width, height);
    m.to = unflip_coord(m.to, sym, width, height);
    m
}

#[test]
fn test_flip_move() {
    assert_eq!(
        flip_move(
            Move {
                from: Coord::new(1, 0),
                to: Coord::new(3, 2)
            },
            Symmetry {
                flip_x: false,
                flip_y: true,
                flip_diagonally: false
            },
            4,
            4
        ),
        Move {
            from: Coord::new(1, 3),
            to: Coord::new(3, 1)
        },
    );
}

type MapTy = HashMap<u64, (Move, u16)>;

#[derive(Default)]
pub struct Tablebase<const W: usize, const H: usize> {
    // table of best move to play on white's turn to force a win
    white_tablebase: MapTy,
    // table of best move to play on black's turn to prolong a loss
    black_tablebase: MapTy,
}

impl<const W: usize, const H: usize> Tablebase<W, H> {
    pub fn white_result(&self, board: &Board<W, H>) -> Option<(Move, u16)> {
        let has_pawn = board_has_pawn(board);
        let (hash, sym) = hash(board, has_pawn);
        self.white_tablebase
            .get(&hash)
            .map(|e| (unflip_move(e.0, sym, W as i8, H as i8), e.1))
    }
    pub fn black_result(&self, board: &Board<W, H>) -> Option<(Move, u16)> {
        let has_pawn = board_has_pawn(board);
        let (hash, sym) = hash(board, has_pawn);
        self.black_tablebase
            .get(&hash)
            .map(|e| (unflip_move(e.0, sym, W as i8, H as i8), e.1))
    }
    fn white_add_impl(&mut self, board: &Board<W, H>, m: Move, depth: u16, has_pawn: bool) {
        let (hash, sym) = hash(board, has_pawn);
        debug_assert!(!self.white_tablebase.contains_key(&hash));
        self.white_tablebase
            .insert(hash, (flip_move(m, sym, W as i8, H as i8), depth));
    }
    fn white_contains_impl(&self, board: &Board<W, H>, has_pawn: bool) -> bool {
        self.white_tablebase.contains_key(&hash(board, has_pawn).0)
    }
    fn white_depth_impl(&self, board: &Board<W, H>, has_pawn: bool) -> Option<u16> {
        self.white_tablebase
            .get(&hash(board, has_pawn).0)
            .map(|e| e.1)
    }
    fn black_add_impl(&mut self, board: &Board<W, H>, m: Move, depth: u16, has_pawn: bool) {
        let (hash, sym) = hash(board, has_pawn);
        debug_assert!(!self.black_tablebase.contains_key(&hash));
        self.black_tablebase
            .insert(hash, (flip_move(m, sym, W as i8, H as i8), depth));
    }
    fn black_contains_impl(&self, board: &Board<W, H>, has_pawn: bool) -> bool {
        self.black_tablebase.contains_key(&hash(board, has_pawn).0)
    }
    fn black_depth_impl(&self, board: &Board<W, H>, has_pawn: bool) -> Option<u16> {
        self.black_tablebase
            .get(&hash(board, has_pawn).0)
            .map(|e| e.1)
    }
}

#[test]
fn test_generate_flip_unflip_move() {
    let board = Board::<4, 4>::with_pieces(&[
        (Coord::new(0, 0), Piece::new(White, King)),
        (Coord::new(1, 0), Piece::new(White, Queen)),
        (Coord::new(2, 0), Piece::new(Black, King)),
    ]);
    let m = Move {
        from: Coord::new(1, 0),
        to: Coord::new(2, 0),
    };
    let mut tablebase = Tablebase::default();
    tablebase.white_add_impl(&board, m, 1, false);
    assert_eq!(tablebase.white_result(&board), Some((m, 1)));
}

fn hash_one_board<const W: usize, const H: usize>(board: &Board<W, H>, sym: Symmetry) -> u64 {
    // need to visit in consistent order across symmetries
    let mut pieces = ArrayVec::<(Coord, Piece), 6>::new();
    for y in 0..H as i8 {
        for x in 0..W as i8 {
            if let Some(p) = board[(x, y)] {
                let c = flip_coord(Coord::new(x, y), sym, W as i8, H as i8);
                pieces.push((c, p));
            }
        }
    }
    pieces.sort_unstable_by(|(c1, _), (c2, _)| match c1.x.cmp(&c2.x) {
        Ordering::Equal => c1.y.cmp(&c2.y),
        o => o,
    });
    let mut hasher = DefaultHasher::new();
    for (c, p) in pieces {
        c.hash(&mut hasher);
        p.hash(&mut hasher);
    }
    hasher.finish()
}

fn hash<const W: usize, const H: usize>(board: &Board<W, H>, has_pawn: bool) -> (u64, Symmetry) {
    debug_assert_eq!(has_pawn, board_has_pawn(board));

    let mut symmetries_to_check = ArrayVec::<Symmetry, 8>::new();
    symmetries_to_check.push(Symmetry::default());

    let mut bk_coord = king_coord(board, Black);

    if W % 2 == 1 && bk_coord.x == W as i8 / 2 {
        let symmetries_copy = symmetries_to_check.clone();
        for mut s in symmetries_copy {
            s.flip_x = true;
            symmetries_to_check.push(s);
        }
    } else if bk_coord.x >= W as i8 / 2 {
        bk_coord = flip_coord(
            bk_coord,
            Symmetry {
                flip_x: true,
                flip_y: false,
                flip_diagonally: false,
            },
            W as i8,
            H as i8,
        );
        for s in symmetries_to_check.as_mut() {
            s.flip_x = true;
        }
    }
    // pawns are not symmetrical on the y axis or diagonally
    if !has_pawn {
        if H % 2 == 1 && bk_coord.y == H as i8 / 2 {
            let symmetries_copy = symmetries_to_check.clone();
            for mut s in symmetries_copy {
                s.flip_y = true;
                symmetries_to_check.push(s);
            }
        } else if bk_coord.y >= H as i8 / 2 {
            bk_coord = flip_coord(
                bk_coord,
                Symmetry {
                    flip_x: false,
                    flip_y: true,
                    flip_diagonally: false,
                },
                W as i8,
                H as i8,
            );
            for s in symmetries_to_check.as_mut() {
                s.flip_y = true;
            }
        }

        if W == H {
            if bk_coord.x == bk_coord.y {
                let symmetries_copy = symmetries_to_check.clone();
                for mut s in symmetries_copy {
                    s.flip_diagonally = true;
                    symmetries_to_check.push(s);
                }
            } else if bk_coord.x > bk_coord.y {
                for s in symmetries_to_check.as_mut() {
                    s.flip_diagonally = true;
                }
            }
        }
    }

    let mut min_hash = hash_one_board(board, symmetries_to_check[0]);
    let mut sym = symmetries_to_check[0];
    for s in symmetries_to_check.into_iter().skip(1) {
        let try_hash = hash_one_board(board, s);
        if try_hash < min_hash {
            min_hash = try_hash;
            sym = s;
        }
    }
    (min_hash, sym)
}

#[test]
fn test_hash() {
    let wk = Piece::new(White, King);
    let bk = Piece::new(Black, King);
    let wp = Piece::new(White, Pawn);
    let bp = Piece::new(Black, Pawn);
    let board1 = Board::<8, 8>::with_pieces(&[(Coord::new(0, 0), wk), (Coord::new(0, 1), bk)]);
    let board2 = Board::<8, 8>::with_pieces(&[(Coord::new(0, 0), wk), (Coord::new(0, 2), bk)]);

    fn assert_hash_eq<const W: usize, const H: usize>(b1: &Board<W, H>, b2: &Board<W, H>) {
        assert_eq!(
            hash(&b1, board_has_pawn(&b1)).0,
            hash(&b2, board_has_pawn(&b2)).0
        );
    }
    fn assert_hash_ne<const W: usize, const H: usize>(b1: &Board<W, H>, b2: &Board<W, H>) {
        assert_ne!(
            hash(&b1, board_has_pawn(&b1)).0,
            hash(&b2, board_has_pawn(&b2)).0
        );
    }
    assert_hash_eq(&board1, &board1);
    assert_hash_eq(&board2, &board2);
    assert_hash_ne(&board1, &board2);
    assert_hash_eq(
        &board1,
        &Board::<8, 8>::with_pieces(&[(Coord::new(0, 0), wk), (Coord::new(1, 0), bk)]),
    );
    assert_hash_eq(
        &board1,
        &Board::<8, 8>::with_pieces(&[(Coord::new(7, 7), wk), (Coord::new(7, 6), bk)]),
    );
    assert_hash_eq(
        &board1,
        &Board::<8, 8>::with_pieces(&[(Coord::new(7, 7), wk), (Coord::new(6, 7), bk)]),
    );
    assert_hash_eq(
        &Board::<5, 5>::with_pieces(&[(Coord::new(3, 2), wk), (Coord::new(2, 2), bk)]),
        &Board::<5, 5>::with_pieces(&[(Coord::new(1, 2), wk), (Coord::new(2, 2), bk)]),
    );
    assert_hash_eq(
        &Board::<5, 5>::with_pieces(&[(Coord::new(3, 2), wk), (Coord::new(2, 2), bk)]),
        &Board::<5, 5>::with_pieces(&[(Coord::new(2, 1), wk), (Coord::new(2, 2), bk)]),
    );
    assert_hash_eq(
        &Board::<5, 5>::with_pieces(&[(Coord::new(3, 2), wk), (Coord::new(2, 2), bk)]),
        &Board::<5, 5>::with_pieces(&[(Coord::new(2, 3), wk), (Coord::new(2, 2), bk)]),
    );
    assert_hash_eq(
        &Board::<5, 5>::with_pieces(&[(Coord::new(3, 3), wk), (Coord::new(2, 2), bk)]),
        &Board::<5, 5>::with_pieces(&[(Coord::new(3, 1), wk), (Coord::new(2, 2), bk)]),
    );
    assert_hash_eq(
        &Board::<5, 5>::with_pieces(&[(Coord::new(3, 3), wk), (Coord::new(2, 2), bk)]),
        &Board::<5, 5>::with_pieces(&[(Coord::new(1, 1), wk), (Coord::new(2, 2), bk)]),
    );
    assert_hash_eq(
        &Board::<5, 5>::with_pieces(&[(Coord::new(3, 3), wk), (Coord::new(2, 2), bk)]),
        &Board::<5, 5>::with_pieces(&[(Coord::new(1, 3), wk), (Coord::new(2, 2), bk)]),
    );
    assert_hash_ne(
        &Board::<5, 5>::with_pieces(&[(Coord::new(3, 3), wk), (Coord::new(2, 2), bk)]),
        &Board::<5, 5>::with_pieces(&[(Coord::new(1, 2), wk), (Coord::new(2, 2), bk)]),
    );
    assert_hash_eq(
        &Board::<8, 8>::with_pieces(&[
            (Coord::new(1, 0), wk),
            (Coord::new(1, 1), wp),
            (Coord::new(1, 7), bk),
            (Coord::new(1, 6), bp),
        ]),
        &Board::<8, 8>::with_pieces(&[
            (Coord::new(6, 0), wk),
            (Coord::new(6, 1), wp),
            (Coord::new(6, 7), bk),
            (Coord::new(6, 6), bp),
        ]),
    );
    assert_hash_ne(
        &Board::<8, 8>::with_pieces(&[
            (Coord::new(1, 0), wk),
            (Coord::new(1, 1), wp),
            (Coord::new(1, 7), bk),
            (Coord::new(1, 6), bp),
        ]),
        &Board::<8, 8>::with_pieces(&[
            (Coord::new(1, 7), wk),
            (Coord::new(1, 6), wp),
            (Coord::new(1, 0), bk),
            (Coord::new(1, 1), bp),
        ]),
    );
}

fn generate_all_boards_impl<const W: usize, const H: usize>(
    ret: &mut Vec<Board<W, H>>,
    board: &Board<W, H>,
    pieces: &[Piece],
) {
    match pieces {
        [p, rest @ ..] => {
            for y in 0..H as i8 {
                for x in 0..W as i8 {
                    if p.ty() == Pawn && (y == 0 || y == H as i8 - 1) {
                        continue;
                    }
                    let coord = Coord::new(x, y);
                    if board[coord].is_none() {
                        let mut clone = board.clone();
                        clone.add_piece(coord, *p);
                        generate_all_boards_impl(ret, &clone, rest);
                    }
                }
            }
        }
        [] => ret.push(board.clone()),
    }
}

pub fn generate_all_boards<const W: usize, const H: usize>(pieces: &[Piece]) -> Vec<Board<W, H>> {
    let board = Board::<W, H>::default();
    let mut ret = Vec::new();
    generate_all_boards_impl(&mut ret, &board, pieces);
    ret
}

#[test]
fn test_generate_all_boards() {
    {
        let boards = generate_all_boards::<8, 8>(&[Piece::new(White, King)]);
        assert_eq!(boards.len(), 64);
    }
    {
        let boards =
            generate_all_boards::<8, 8>(&[Piece::new(White, King), Piece::new(White, Queen)]);
        assert_eq!(boards.len(), 64 * 63);
        assert_eq!(boards[0][(0, 0)], Some(Piece::new(White, King)));
        assert_eq!(boards[0][(1, 0)], Some(Piece::new(White, Queen)));
        assert_eq!(boards[0][(2, 0)], None);

        assert_eq!(boards[1][(0, 0)], Some(Piece::new(White, King)));
        assert_eq!(boards[1][(1, 0)], None);
        assert_eq!(boards[1][(2, 0)], Some(Piece::new(White, Queen)));
    }
}

fn populate_initial_wins<const W: usize, const H: usize>(
    tablebase: &mut Tablebase<W, H>,
    boards: &[Board<W, H>],
    has_pawn: bool,
) -> Vec<Board<W, H>> {
    let mut ret = Vec::new();
    for b in boards {
        // white can capture black's king
        if !tablebase.white_contains_impl(b, has_pawn) {
            let opponent_king_coord = king_coord(b, Black);
            if let Some(c) = under_attack_from_coord(b, opponent_king_coord, Black) {
                tablebase.white_add_impl(
                    b,
                    Move {
                        from: c,
                        to: opponent_king_coord,
                    },
                    1,
                    has_pawn,
                );
                ret.push(b.clone());
            }
        }
        // don't support stalemate for now, it doesn't really happen on normal boards with few pieces, and takes up a noticeable chunk of tablebase generation time
        debug_assert!(!all_moves(b, Black).is_empty());
        // if !tablebase.black_contains_impl(b, has_pawn) {
        //     // stalemate is a win
        //     if all_moves(b, Black).is_empty() {
        //         tablebase.black_add_impl(
        //             b,
        //             // arbitrary move
        //             Move {
        //                 from: Coord::new(0, 0),
        //                 to: Coord::new(0, 0),
        //             },
        //             0,
        //             has_pawn,
        //         );
        //     }
        // }
    }
    ret
}

#[test]
fn test_populate_initial_tablebases() {
    let mut tablebase = Tablebase::default();
    let boards = [
        Board::<8, 8>::with_pieces(&[
            (Coord::new(0, 0), Piece::new(White, King)),
            (Coord::new(0, 1), Piece::new(Black, King)),
        ]),
        Board::<8, 8>::with_pieces(&[
            (Coord::new(0, 0), Piece::new(White, King)),
            (Coord::new(0, 2), Piece::new(Black, King)),
        ]),
    ];
    assert_eq!(
        populate_initial_wins(&mut tablebase, &boards, false).len(),
        1
    );
    assert_eq!(
        tablebase.white_result(&boards[0]),
        Some((
            Move {
                from: Coord::new(0, 0),
                to: Coord::new(0, 1)
            },
            1
        ))
    );
    assert!(!tablebase.white_contains_impl(&boards[1], false));
    assert!(!tablebase.black_contains_impl(&boards[0], false));
    assert!(!tablebase.black_contains_impl(&boards[1], false));
}

#[test]
#[ignore]
fn test_populate_initial_tablebases_stalemate() {
    let mut tablebase = Tablebase::default();
    populate_initial_wins(
        &mut tablebase,
        &generate_all_boards::<1, 8>(&[
            Piece::new(White, King),
            Piece::new(Black, Pawn),
            Piece::new(Black, King),
        ]),
        true,
    );
    assert_eq!(
        tablebase.black_depth_impl(
            &Board::<1, 8>::with_pieces(&[
                (Coord::new(0, 7), Piece::new(White, King,)),
                (Coord::new(0, 1), Piece::new(Black, Pawn,)),
                (Coord::new(0, 0), Piece::new(Black, King,)),
            ]),
            true
        ),
        Some(0)
    );
    assert!(!tablebase.black_contains_impl(
        &Board::<1, 8>::with_pieces(&[
            (Coord::new(0, 7), Piece::new(White, King,)),
            (Coord::new(0, 2), Piece::new(Black, Pawn,)),
            (Coord::new(0, 0), Piece::new(Black, King,)),
        ]),
        true
    ));
}

fn iterate_black<const W: usize, const H: usize>(
    tablebase: &mut Tablebase<W, H>,
    boards: &[Board<W, H>],
    has_pawn: bool,
) -> Vec<Board<W, H>> {
    let mut next_boards = Vec::new();
    for b in boards {
        if tablebase.black_contains_impl(b, has_pawn) {
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

            if let Some(depth) = tablebase.white_depth_impl(&clone, has_pawn) {
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
            tablebase.black_add_impl(b, best_move.unwrap(), max_depth + 1, has_pawn);
            next_boards.push(b.clone());
        }
    }
    next_boards
}

#[test]
fn test_iterate_black() {
    let wk = Piece::new(White, King);
    let bk = Piece::new(Black, King);
    let board = Board::<4, 1>::with_pieces(&[(Coord::new(0, 0), wk), (Coord::new(2, 0), bk)]);
    let mut tablebase = Tablebase::default();
    tablebase.white_add_impl(
        &Board::<4, 1>::with_pieces(&[(Coord::new(0, 0), wk), (Coord::new(1, 0), bk)]),
        Move {
            from: Coord::new(0, 0),
            to: Coord::new(0, 0),
        },
        1,
        false,
    );
    assert!(iterate_black(&mut tablebase, &[board.clone()], false).is_empty());
    assert!(tablebase.black_tablebase.is_empty());
    tablebase.white_add_impl(
        &Board::<4, 1>::with_pieces(&[(Coord::new(0, 0), wk), (Coord::new(3, 0), bk)]),
        Move {
            from: Coord::new(0, 0),
            to: Coord::new(0, 0),
        },
        3,
        false,
    );
    assert_eq!(
        iterate_black(&mut tablebase, &[board.clone()], false).len(),
        1
    );
    assert_eq!(
        tablebase.black_result(&board),
        Some((
            Move {
                from: Coord::new(2, 0),
                to: Coord::new(3, 0)
            },
            4
        ))
    );
}

fn iterate_white<const W: usize, const H: usize>(
    tablebase: &mut Tablebase<W, H>,
    boards: &[Board<W, H>],
    has_pawn: bool,
) -> Vec<Board<W, H>> {
    let mut next_boards = Vec::new();
    for b in boards {
        if tablebase.white_contains_impl(b, has_pawn) {
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

            if let Some(depth) = tablebase.black_depth_impl(&clone, has_pawn) {
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
            tablebase.white_add_impl(b, best_move.unwrap(), min_depth + 1, has_pawn);
            next_boards.push(b.clone());
        }
    }
    next_boards
}

#[test]
fn test_iterate_white() {
    let wk = Piece::new(White, King);
    let bk = Piece::new(Black, King);
    let board = Board::<5, 1>::with_pieces(&[(Coord::new(1, 0), wk), (Coord::new(4, 0), bk)]);
    let mut tablebase = Tablebase::default();
    assert!(iterate_white(&mut tablebase, &[board.clone()], false).is_empty());
    assert!(tablebase.white_tablebase.is_empty());

    tablebase.black_add_impl(
        &Board::<5, 1>::with_pieces(&[(Coord::new(2, 0), wk), (Coord::new(4, 0), bk)]),
        Move {
            from: Coord::new(0, 0),
            to: Coord::new(0, 0),
        },
        2,
        false,
    );
    assert_eq!(
        iterate_white(&mut tablebase, &[board.clone()], false).len(),
        1
    );
    assert_eq!(
        tablebase.white_result(&board),
        Some((
            Move {
                from: Coord::new(1, 0),
                to: Coord::new(2, 0)
            },
            3
        ))
    );
    tablebase.white_tablebase.clear();
    tablebase.black_add_impl(
        &Board::<5, 1>::with_pieces(&[(Coord::new(0, 0), wk), (Coord::new(4, 0), bk)]),
        Move {
            from: Coord::new(0, 0),
            to: Coord::new(0, 0),
        },
        4,
        false,
    );
    assert_eq!(
        iterate_white(&mut tablebase, &[board.clone()], false).len(),
        1
    );
    assert_eq!(
        tablebase.white_result(&board),
        Some((
            Move {
                from: Coord::new(1, 0),
                to: Coord::new(2, 0)
            },
            3
        ))
    );
}

fn verify_piece_set(pieces: &[Piece]) {
    let mut wk_count = 0;
    let mut bk_count = 0;
    for p in pieces {
        if p.ty() == King {
            match p.player() {
                White => wk_count += 1,
                Black => bk_count += 1,
            }
        }
    }
    assert_eq!(wk_count, 1);
    assert_eq!(bk_count, 1);
}

fn reachable_positions<const W: usize, const H: usize>(
    boards: &[Board<W, H>],
    player: Player,
) -> Vec<Board<W, H>> {
    let mut ret = Vec::new();
    for board in boards {
        for m in all_moves_to_end_at_board_no_captures(board, player) {
            assert_eq!(
                board.existing_piece_result(m.from, player),
                ExistingPieceResult::Empty
            );
            assert_eq!(
                board.existing_piece_result(m.to, player),
                ExistingPieceResult::Friend
            );
            let mut clone = board.clone();
            clone.swap(m.from, m.to);
            ret.push(clone);
        }
    }
    ret
}

pub fn generate_tablebase<const W: usize, const H: usize>(
    piece_sets: &[&[Piece]],
) -> Tablebase<W, H> {
    let mut tablebase = Tablebase::default();

    for set in piece_sets {
        print!("generating tablebase for ");
        for p in set.iter() {
            print!("{}", p.char());
        }
        println!();
        let has_pawn = set.iter().any(|p| p.ty() == Pawn);
        verify_piece_set(set);
        let mut boards_to_check = generate_all_boards::<W, H>(set);
        boards_to_check = populate_initial_wins(&mut tablebase, &boards_to_check, has_pawn);
        loop {
            boards_to_check = iterate_black(&mut tablebase, &boards_to_check, has_pawn);
            if boards_to_check.is_empty() {
                break;
            }
            boards_to_check = reachable_positions(&boards_to_check, White);

            boards_to_check = iterate_white(&mut tablebase, &boards_to_check, has_pawn);
            if boards_to_check.is_empty() {
                break;
            }
            boards_to_check = reachable_positions(&boards_to_check, Black);
        }
    }
    tablebase
}

#[test]
fn test_generate_king_king_tablebase() {
    fn test<const W: usize, const H: usize>() {
        let pieces = [Piece::new(White, King), Piece::new(Black, King)];
        let tablebase = generate_tablebase::<W, H>(&[&pieces]);
        // If white king couldn't capture on first move, no forced win.
        assert!(tablebase.black_tablebase.is_empty());
        let all = generate_all_boards::<W, H>(&pieces);
        for b in all {
            if crate::moves::is_under_attack(&b, king_coord(&b, Black), Black) {
                assert_eq!(tablebase.white_depth_impl(&b, false), Some(1));
            } else {
                assert_eq!(tablebase.white_depth_impl(&b, false), None);
            }
        }
    }
    test::<6, 6>();
    test::<5, 5>();
    test::<4, 5>();
    test::<4, 6>();
}

#[test]
fn test_tablebase_size() {
    let pieces1 = [Piece::new(White, King), Piece::new(Black, King)];
    let pieces2 = [
        Piece::new(White, King),
        Piece::new(White, Queen),
        Piece::new(Black, King),
    ];
    let tablebase = generate_tablebase::<4, 4>(&[&pieces1, &pieces2]);
    let all1 = generate_all_boards::<4, 4>(&pieces1);
    let all2 = generate_all_boards::<4, 4>(&pieces2);
    // With symmetry, we should expect a little over 1/8 of positions to be in the tablebase.
    assert!(tablebase.white_tablebase.len() < all1.len() + all2.len() / 6);
}
