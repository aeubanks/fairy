use crate::board::{king_coord, Board, Move};
use crate::coord::Coord;
use crate::moves::{all_moves, is_under_attack};
use crate::piece::Piece;
use crate::piece::Type::*;
use crate::player::Player::*;
use arrayvec::ArrayVec;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

// TODO: factor out coord visiting
fn board_has_pawn<const N: usize, const M: usize>(board: &Board<N, M>) -> bool {
    for y in 0..M as i8 {
        for x in 0..N as i8 {
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

fn flip_x<const N: usize, const M: usize>((board, sym): &mut (Board<N, M>, Symmetry)) {
    for y in 0..M as i8 {
        for x in 0..(N as i8 / 2) {
            let c = Coord::new(x, y);
            let c2 = Coord::new(N as i8 - 1 - x, y);
            board.swap(c, c2);
        }
    }
    debug_assert!(!sym.flip_x);
    sym.flip_x = true;
}

#[test]
fn test_flip_x() {
    let n = Piece::new(White, Knight);
    let b = Piece::new(White, Bishop);
    let mut bs = (
        Board::<3, 2>::with_pieces(&[(Coord::new(0, 0), n), (Coord::new(1, 0), b)]),
        Symmetry::default(),
    );

    flip_x(&mut bs);

    assert_eq!(
        bs.1,
        Symmetry {
            flip_x: true,
            flip_y: false,
            flip_diagonally: false
        }
    );
    assert_eq!(
        bs.0,
        Board::<3, 2>::with_pieces(&[(Coord::new(2, 0), n), (Coord::new(1, 0), b)])
    );
}

fn flip_y<const N: usize, const M: usize>((board, sym): &mut (Board<N, M>, Symmetry)) {
    for y in 0..(M as i8 / 2) {
        for x in 0..N as i8 {
            let c = Coord::new(x, y);
            let c2 = Coord::new(x, M as i8 - 1 - y);
            board.swap(c, c2);
        }
    }
    debug_assert!(!sym.flip_y);
    sym.flip_y = true;
}

#[test]
fn test_flip_y() {
    let n = Piece::new(White, Knight);
    let b = Piece::new(White, Bishop);
    let mut bs = (
        Board::<3, 2>::with_pieces(&[(Coord::new(0, 0), n), (Coord::new(1, 0), b)]),
        Symmetry::default(),
    );

    flip_y(&mut bs);

    assert_eq!(
        bs.1,
        Symmetry {
            flip_x: false,
            flip_y: true,
            flip_diagonally: false
        }
    );
    assert_eq!(
        bs.0,
        Board::<3, 2>::with_pieces(&[(Coord::new(0, 1), n), (Coord::new(1, 1), b)])
    );
}

fn flip_diagonal<const N: usize, const M: usize>((board, sym): &mut (Board<N, M>, Symmetry)) {
    assert_eq!(N, M);
    for y in 1..M as i8 {
        for x in 0..y {
            let c = Coord::new(x, y);
            let c2 = Coord::new(y, x);
            board.swap(c, c2);
        }
    }
    debug_assert!(!sym.flip_diagonally);
    sym.flip_diagonally = true;
}

#[test]
fn test_flip_diagonal() {
    let n = Piece::new(White, Knight);
    let b = Piece::new(White, Bishop);
    let mut bs = (
        Board::<3, 3>::with_pieces(&[(Coord::new(0, 0), n), (Coord::new(1, 0), b)]),
        Symmetry::default(),
    );

    flip_diagonal(&mut bs);

    assert_eq!(
        bs.1,
        Symmetry {
            flip_x: false,
            flip_y: false,
            flip_diagonally: true
        }
    );
    assert_eq!(
        bs.0,
        Board::<3, 3>::with_pieces(&[(Coord::new(0, 0), n), (Coord::new(0, 1), b)])
    );
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
pub struct Tablebase {
    // table of best move to play on white's turn to force a win
    white_tablebase: MapTy,
    // table of best move to play on black's turn to prolong a loss
    black_tablebase: MapTy,
}

impl Tablebase {
    pub fn white_result<const N: usize, const M: usize>(
        &self,
        board: &Board<N, M>,
    ) -> Option<(Move, u16)> {
        let has_pawn = board_has_pawn(board);
        let (hash, sym) = hash(board, has_pawn);
        self.white_tablebase
            .get(&hash)
            .map(|e| (unflip_move(e.0, sym, N as i8, M as i8), e.1))
    }
    pub fn black_result<const N: usize, const M: usize>(
        &self,
        board: &Board<N, M>,
    ) -> Option<(Move, u16)> {
        let has_pawn = board_has_pawn(board);
        let (hash, sym) = hash(board, has_pawn);
        self.black_tablebase
            .get(&hash)
            .map(|e| (unflip_move(e.0, sym, N as i8, M as i8), e.1))
    }
    fn white_add_impl<const N: usize, const M: usize>(
        &mut self,
        board: &Board<N, M>,
        m: Move,
        depth: u16,
        has_pawn: bool,
    ) {
        let (hash, sym) = hash(board, has_pawn);
        self.white_tablebase
            .insert(hash, (flip_move(m, sym, N as i8, M as i8), depth));
    }
    fn white_contains_impl<const N: usize, const M: usize>(
        &self,
        board: &Board<N, M>,
        has_pawn: bool,
    ) -> bool {
        self.white_tablebase.contains_key(&hash(board, has_pawn).0)
    }
    fn white_depth_impl<const N: usize, const M: usize>(
        &self,
        board: &Board<N, M>,
        has_pawn: bool,
    ) -> Option<u16> {
        self.white_tablebase
            .get(&hash(board, has_pawn).0)
            .map(|e| e.1)
    }
    fn black_add_impl<const N: usize, const M: usize>(
        &mut self,
        board: &Board<N, M>,
        m: Move,
        depth: u16,
        has_pawn: bool,
    ) {
        let (hash, sym) = hash(board, has_pawn);
        self.black_tablebase
            .insert(hash, (flip_move(m, sym, N as i8, M as i8), depth));
    }
    fn black_contains_impl<const N: usize, const M: usize>(
        &self,
        board: &Board<N, M>,
        has_pawn: bool,
    ) -> bool {
        self.black_tablebase.contains_key(&hash(board, has_pawn).0)
    }
    fn black_depth_impl<const N: usize, const M: usize>(
        &self,
        board: &Board<N, M>,
        has_pawn: bool,
    ) -> Option<u16> {
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

fn hash_one_board<const N: usize, const M: usize>(board: &Board<N, M>) -> u64 {
    let mut hasher = DefaultHasher::new();
    N.hash(&mut hasher);
    M.hash(&mut hasher);
    for y in 0..M as i8 {
        for x in 0..N as i8 {
            if let Some(p) = board[(x, y)] {
                x.hash(&mut hasher);
                y.hash(&mut hasher);
                p.hash(&mut hasher);
            }
        }
    }
    hasher.finish()
}

fn hash<const N: usize, const M: usize>(board: &Board<N, M>, has_pawn: bool) -> (u64, Symmetry) {
    // FIXME: this is faster right now...
    // return (hash_one_board(board), Symmetry::default());

    // TODO: add fast path when no symmetry to avoid clone?

    debug_assert_eq!(has_pawn, board_has_pawn(board));

    let mut boards_to_check = ArrayVec::<(Board<N, M>, Symmetry), 8>::new();
    boards_to_check.push((board.clone(), Symmetry::default()));

    let mut bk_coord = king_coord(board, Black);

    if N % 2 == 1 && bk_coord.x == N as i8 / 2 {
        let boards_copy = boards_to_check.clone();
        for mut c in boards_copy {
            flip_x(&mut c);
            boards_to_check.push(c);
        }
    } else if bk_coord.x >= N as i8 / 2 {
        bk_coord = flip_coord(
            bk_coord,
            Symmetry {
                flip_x: true,
                flip_y: false,
                flip_diagonally: false,
            },
            N as i8,
            M as i8,
        );
        for b in boards_to_check.as_mut() {
            flip_x(b);
        }
    }
    // pawns are not symmetrical on the y axis or diagonally
    if !has_pawn {
        if M % 2 == 1 && bk_coord.y == M as i8 / 2 {
            let boards_copy = boards_to_check.clone();
            for mut c in boards_copy {
                flip_y(&mut c);
                boards_to_check.push(c);
            }
        } else if bk_coord.y >= M as i8 / 2 {
            bk_coord = flip_coord(
                bk_coord,
                Symmetry {
                    flip_x: false,
                    flip_y: true,
                    flip_diagonally: false,
                },
                N as i8,
                M as i8,
            );
            for b in boards_to_check.as_mut() {
                flip_y(b);
            }
        }

        if N == M {
            if bk_coord.x == bk_coord.y {
                let boards_copy = boards_to_check.clone();
                for mut c in boards_copy {
                    flip_diagonal(&mut c);
                    boards_to_check.push(c);
                }
            } else if bk_coord.x > bk_coord.y {
                for b in boards_to_check.as_mut() {
                    flip_diagonal(b);
                }
            }
        }
    }

    let mut min_hash = hash_one_board(&boards_to_check[0].0);
    let mut sym = boards_to_check[0].1;
    for (b, s) in boards_to_check.into_iter().skip(1) {
        let try_hash = hash_one_board(&b);
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
    let board1 = Board::<8, 8>::with_pieces(&[(Coord::new(0, 0), wk), (Coord::new(0, 1), bk)]);
    let board2 = Board::<8, 8>::with_pieces(&[(Coord::new(0, 0), wk), (Coord::new(0, 2), bk)]);

    fn assert_hash_eq<const N: usize, const M: usize>(b1: &Board<N, M>, b2: &Board<N, M>) {
        assert_eq!(
            hash(&b1, board_has_pawn(&b1)).0,
            hash(&b2, board_has_pawn(&b2)).0
        );
    }
    fn assert_hash_ne<const N: usize, const M: usize>(b1: &Board<N, M>, b2: &Board<N, M>) {
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
}

fn generate_all_boards_impl<const N: usize, const M: usize>(
    ret: &mut Vec<Board<N, M>>,
    board: &Board<N, M>,
    pieces: &[Piece],
) {
    match pieces {
        [p, rest @ ..] => {
            for y in 0..M as i8 {
                for x in 0..N as i8 {
                    if p.ty() == Pawn && (y == 0 || y == M as i8 - 1) {
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

pub fn generate_all_boards<const N: usize, const M: usize>(pieces: &[Piece]) -> Vec<Board<N, M>> {
    let board = Board::<N, M>::default();
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

fn populate_initial_wins<const N: usize, const M: usize>(
    tablebase: &mut Tablebase,
    boards: &[Board<N, M>],
    has_pawn: bool,
) {
    for b in boards {
        // white can capture black's king
        let opponent_king_coord = king_coord(b, Black);
        if is_under_attack(b, opponent_king_coord, Black) {
            let all_moves = all_moves(b, White);
            let m = all_moves
                .into_iter()
                .find(|m| m.to == opponent_king_coord)
                .unwrap();
            tablebase.white_add_impl(b, m, 1, has_pawn);
        }
        // stalemate is a win
        if all_moves(b, Black).is_empty() {
            tablebase.black_add_impl(
                b,
                // arbitrary move
                Move {
                    from: Coord::new(0, 0),
                    to: Coord::new(0, 0),
                },
                0,
                has_pawn,
            );
        }
    }
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
    populate_initial_wins(&mut tablebase, &boards, false);
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

fn iterate_black<const N: usize, const M: usize>(
    tablebase: &mut Tablebase,
    boards: &[Board<N, M>],
    has_pawn: bool,
) -> bool {
    let mut made_progress = false;
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
            made_progress = true;
        }
    }
    made_progress
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
    assert!(!iterate_black(&mut tablebase, &[board.clone()], false));
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
    assert!(iterate_black(&mut tablebase, &[board.clone()], false));
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

fn iterate_white<const N: usize, const M: usize>(
    tablebase: &mut Tablebase,
    boards: &[Board<N, M>],
    has_pawn: bool,
) -> bool {
    let mut made_progress = false;
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
            made_progress = true;
        }
    }
    made_progress
}

#[test]
fn test_iterate_white() {
    let wk = Piece::new(White, King);
    let bk = Piece::new(Black, King);
    let board = Board::<5, 1>::with_pieces(&[(Coord::new(1, 0), wk), (Coord::new(4, 0), bk)]);
    let mut tablebase = Tablebase::default();
    assert!(!iterate_white(&mut tablebase, &[board.clone()], false));
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
    assert!(iterate_white(&mut tablebase, &[board.clone()], false));
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
    assert!(iterate_white(&mut tablebase, &[board.clone()], false));
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

pub fn generate_tablebase<const N: usize, const M: usize>(piece_sets: &[&[Piece]]) -> Tablebase {
    let mut tablebase = Tablebase::default();

    for set in piece_sets {
        let has_pawn = set.iter().any(|p| p.ty() == Pawn);
        verify_piece_set(set);
        let all = generate_all_boards::<N, M>(set);
        populate_initial_wins(&mut tablebase, &all, has_pawn);
        loop {
            if !iterate_black(&mut tablebase, &all, has_pawn) {
                break;
            }
            if !iterate_white(&mut tablebase, &all, has_pawn) {
                break;
            }
        }
    }
    tablebase
}

#[test]
fn test_generate_king_king_tablebase() {
    fn test<const N: usize, const M: usize>() {
        let pieces = [Piece::new(White, King), Piece::new(Black, King)];
        let tablebase = generate_tablebase::<N, M>(&[&pieces]);
        // If white king couldn't capture on first move, no forced win.
        assert!(tablebase.black_tablebase.is_empty());
        let all = generate_all_boards::<N, M>(&pieces);
        for b in all {
            if is_under_attack(&b, king_coord(&b, Black), Black) {
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
