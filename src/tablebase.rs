use crate::board::{Move, PieceWatcher};
use crate::coord::Coord;
use crate::moves::{all_moves, all_moves_to_end_at_board_no_captures, under_attack_from_coord};
use crate::piece::Piece;
use crate::piece::Type::*;
use crate::player::Player::*;
use arrayvec::ArrayVec;
use rustc_hash::FxHasher;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hash, Hasher};

#[derive(Clone, Default)]
pub struct TablebasePieceWatcher(ArrayVec<(Coord, Piece), 4>);

impl PieceWatcher for TablebasePieceWatcher {
    fn add_piece(&mut self, piece: Piece, coord: Coord) {
        self.0.push((coord, piece));
    }
    fn remove_piece(&mut self, piece: Piece, coord: Coord) {
        let idx = self
            .0
            .iter()
            .position(|(c, p)| *c == coord && *p == piece)
            .unwrap();
        self.0.swap_remove(idx);
    }
    fn foreach_piece<F>(&self, mut f: F) -> bool
    where
        F: FnMut(Piece, Coord),
    {
        for (c, p) in &self.0 {
            f(*p, *c);
        }
        true
    }
    fn piece_coord<F>(&self, mut f: F) -> Option<Option<Coord>>
    where
        F: FnMut(Piece) -> bool,
    {
        for (c, p) in &self.0 {
            if f(*p) {
                return Some(Some(*c));
            }
        }
        Some(None)
    }
}

pub type Board<const W: usize, const H: usize> = crate::board::Board<W, H, TablebasePieceWatcher>;

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

#[derive(Default)]
struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    fn write(&mut self, _bytes: &[u8]) {
        unimplemented!()
    }
    fn write_u64(&mut self, i: u64) {
        self.0 = i;
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

type MapTy = HashMap<u64, (Move, u16), BuildHasherDefault<IdentityHasher>>;

#[derive(Default, Clone)]
pub struct Tablebase<const W: usize, const H: usize> {
    // table of best move to play on white's turn to force a win
    white_tablebase: MapTy,
    // table of best move to play on black's turn to prolong a loss
    black_tablebase: MapTy,
}

impl<const W: usize, const H: usize> Tablebase<W, H> {
    pub fn white_result(&self, board: &Board<W, H>) -> Option<(Move, u16)> {
        let (hash, sym) = hash(board);
        self.white_tablebase
            .get(&hash)
            .map(|e| (unflip_move(e.0, sym, W as i8, H as i8), e.1))
    }
    pub fn black_result(&self, board: &Board<W, H>) -> Option<(Move, u16)> {
        let (hash, sym) = hash(board);
        self.black_tablebase
            .get(&hash)
            .map(|e| (unflip_move(e.0, sym, W as i8, H as i8), e.1))
    }
    fn white_add_impl(&mut self, board: &Board<W, H>, m: Move, depth: u16) {
        let (hash, sym) = hash(board);
        debug_assert!(!self.white_tablebase.contains_key(&hash));
        self.white_tablebase
            .insert(hash, (flip_move(m, sym, W as i8, H as i8), depth));
    }
    fn white_contains_impl(&self, board: &Board<W, H>) -> bool {
        self.white_tablebase.contains_key(&hash(board).0)
    }
    fn white_depth_impl(&self, board: &Board<W, H>) -> Option<u16> {
        self.white_tablebase.get(&hash(board).0).map(|e| e.1)
    }
    fn black_add_impl(&mut self, board: &Board<W, H>, m: Move, depth: u16) {
        let (hash, sym) = hash(board);
        debug_assert!(!self.black_tablebase.contains_key(&hash));
        self.black_tablebase
            .insert(hash, (flip_move(m, sym, W as i8, H as i8), depth));
    }
    fn black_contains_impl(&self, board: &Board<W, H>) -> bool {
        self.black_tablebase.contains_key(&hash(board).0)
    }
    fn black_depth_impl(&self, board: &Board<W, H>) -> Option<u16> {
        self.black_tablebase.get(&hash(board).0).map(|e| e.1)
    }
    fn merge(&mut self, other: &Self) {
        self.white_tablebase.extend(&other.white_tablebase);
        self.black_tablebase.extend(&other.black_tablebase);
    }
    pub fn dump_stats(&self) {
        println!("white positions: {}", self.white_tablebase.len());
        println!("black positions: {}", self.black_tablebase.len());
        let mut max_depth = 0;
        for v in self.white_tablebase.values() {
            if v.1 > max_depth {
                max_depth = v.1;
            }
        }
        println!("max depth: {}", max_depth);
    }
}

fn hash_one_board<const W: usize, const H: usize>(board: &Board<W, H>, sym: Symmetry) -> u64 {
    // need to visit in consistent order across symmetries
    let mut pieces = ArrayVec::<(Coord, Piece), 6>::new();
    board.foreach_piece(|piece, coord| {
        let c = flip_coord(coord, sym, W as i8, H as i8);
        pieces.push((c, piece));
    });

    pieces.sort_unstable_by(|(c1, _), (c2, _)| match c1.x.cmp(&c2.x) {
        Ordering::Equal => c1.y.cmp(&c2.y),
        o => o,
    });
    let mut hasher = FxHasher::default();
    for (c, p) in pieces {
        c.hash(&mut hasher);
        p.hash(&mut hasher);
    }
    hasher.finish()
}

fn hash<const W: usize, const H: usize>(board: &Board<W, H>) -> (u64, Symmetry) {
    let mut symmetries_to_check = ArrayVec::<Symmetry, 8>::new();
    symmetries_to_check.push(Symmetry::default());

    let mut bk_coord = board.king_coord(Black);

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
    let has_pawn = board.piece_coord(|piece| piece.ty() == Pawn).is_some();
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

pub struct GenerateAllBoards<const W: usize, const H: usize> {
    pieces: ArrayVec<Piece, 6>,
    stack: ArrayVec<Coord, 6>,
    board: Board<W, H>,
}

impl<const W: usize, const H: usize> GenerateAllBoards<W, H> {
    fn next_coord(mut c: Coord) -> Coord {
        c.x += 1;
        if c.x == W as i8 {
            c.x = 0;
            c.y += 1;
        }
        c
    }
    fn first_empty_coord_from(&self, mut c: Coord, piece: Piece) -> Option<Coord> {
        while c.y != H as i8 {
            if self.board[c].is_none() && (piece.ty() != Pawn || (c.y != 0 && c.y != H as i8 - 1)) {
                return Some(c);
            }
            c = Self::next_coord(c);
        }
        None
    }
    pub fn new(pieces: &[Piece]) -> Self {
        let mut ret = Self {
            pieces: pieces.iter().copied().collect(),
            stack: Default::default(),
            board: Default::default(),
        };
        for p in pieces {
            let c = ret.first_empty_coord_from(Coord::new(0, 0), *p).unwrap();
            ret.stack.push(c);
            ret.board.add_piece(c, *p);
        }
        ret
    }
}

impl<const W: usize, const H: usize> Iterator for GenerateAllBoards<W, H> {
    type Item = Board<W, H>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.stack.is_empty() {
            return None;
        }
        let ret = self.board.clone();
        let mut done = false;
        let mut add_from_idx = self.stack.len();
        for i in (0..self.stack.len()).rev() {
            assert_eq!(self.board[self.stack[i]], Some(self.pieces[i]));
            self.board.clear(self.stack[i]);
            if let Some(c) =
                self.first_empty_coord_from(Self::next_coord(self.stack[i]), self.pieces[i])
            {
                self.board.add_piece(c, self.pieces[i]);
                self.stack[i] = c;
                break;
            }
            add_from_idx = i;
            if i == 0 {
                done = true;
            }
        }
        if done {
            self.stack.clear();
        } else {
            for i in add_from_idx..self.stack.len() {
                self.stack[i] = self
                    .first_empty_coord_from(Coord::new(0, 0), self.pieces[i])
                    .unwrap();
                self.board.add_piece(self.stack[i], self.pieces[i]);
            }
        }
        Some(ret)
    }
}

fn populate_initial_wins<const W: usize, const H: usize>(
    tablebase: &mut Tablebase<W, H>,
    pieces: &[Piece],
) -> Vec<Board<W, H>> {
    let mut ret = Vec::new();
    for b in GenerateAllBoards::new(pieces) {
        // white can capture black's king
        if !tablebase.white_contains_impl(&b) {
            let opponent_king_coord = b.king_coord(Black);
            if let Some(c) = under_attack_from_coord(&b, opponent_king_coord, Black) {
                tablebase.white_add_impl(
                    &b,
                    Move {
                        from: c,
                        to: opponent_king_coord,
                    },
                    1,
                );
                ret.push(b.clone());
            }
        }
        // don't support stalemate for now, it doesn't really happen on normal boards with few pieces, and takes up a noticeable chunk of tablebase generation time
        debug_assert!(!all_moves(&b, Black).is_empty());
        // if !tablebase.black_contains_impl(b) {
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
        //         );
        //     }
        // }
    }
    ret
}

fn iterate_black<const W: usize, const H: usize>(
    tablebase: &mut Tablebase<W, H>,
    previous_boards: &[Board<W, H>],
) -> Vec<Board<W, H>> {
    let mut next_boards = Vec::new();
    for prev in previous_boards {
        for m in all_moves_to_end_at_board_no_captures(prev, Black) {
            let mut b = prev.clone();
            b.swap(m.from, m.to);
            if tablebase.black_contains_impl(&b) {
                continue;
            }
            // None means no forced loss
            // Some(depth) means forced loss in depth moves
            let mut max_depth = Some(0);
            let mut best_move = None;
            let black_moves = all_moves(&b, Black);
            if black_moves.is_empty() {
                // loss?
                continue;
            }
            for black_move in black_moves {
                let mut clone = b.clone();
                clone.make_move(black_move, Black);

                if let Some(depth) = tablebase.white_depth_impl(&clone) {
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
                tablebase.black_add_impl(&b, best_move.unwrap(), max_depth + 1);
                next_boards.push(b.clone());
            }
        }
    }
    next_boards
}

fn iterate_white<const W: usize, const H: usize>(
    tablebase: &mut Tablebase<W, H>,
    previous_boards: &[Board<W, H>],
) -> Vec<Board<W, H>> {
    let mut next_boards = Vec::new();
    for prev in previous_boards {
        for m in all_moves_to_end_at_board_no_captures(prev, White) {
            let mut b = prev.clone();
            b.swap(m.from, m.to);
            if tablebase.white_contains_impl(&b) {
                continue;
            }
            // None means no forced win
            // Some(depth) means forced win in depth moves
            let mut min_depth: Option<u16> = None;
            let mut best_move = None;
            let white_moves = all_moves(&b, White);
            if white_moves.is_empty() {
                // loss?
                continue;
            }
            for white_move in white_moves {
                let mut clone = b.clone();
                clone.make_move(white_move, White);

                if let Some(depth) = tablebase.black_depth_impl(&clone) {
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
                tablebase.white_add_impl(&b, best_move.unwrap(), min_depth + 1);
                next_boards.push(b.clone());
            }
        }
    }
    next_boards
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

pub fn generate_tablebase<const W: usize, const H: usize>(
    tablebase: &mut Tablebase<W, H>,
    pieces: &[Piece],
) {
    verify_piece_set(pieces);
    let mut boards_to_check = populate_initial_wins(tablebase, pieces);
    loop {
        // check that starting from all possible boards gives the same result as starting from all boards returned by previous step
        #[cfg(debug_assertions)]
        let all_black_count = iterate_black(
            &mut tablebase.clone(),
            &GenerateAllBoards::new(pieces).collect::<Vec<_>>(),
        )
        .len();

        boards_to_check = iterate_black(tablebase, &boards_to_check);

        #[cfg(debug_assertions)]
        debug_assert_eq!(all_black_count, boards_to_check.len());

        if boards_to_check.is_empty() {
            break;
        }

        #[cfg(debug_assertions)]
        let all_white_count = iterate_white(
            &mut tablebase.clone(),
            &GenerateAllBoards::new(pieces).collect::<Vec<_>>(),
        )
        .len();

        boards_to_check = iterate_white(tablebase, &boards_to_check);

        #[cfg(debug_assertions)]
        debug_assert_eq!(all_white_count, boards_to_check.len());

        if boards_to_check.is_empty() {
            break;
        }
    }
}

pub fn generate_tablebase_parallel<const W: usize, const H: usize>(
    tablebase: &mut Tablebase<W, H>,
    piece_sets: &[&[Piece]],
    parallelism: Option<usize>,
) {
    use std::sync::{Arc, Mutex};
    use std::thread;
    let sets = Arc::new(Mutex::new(
        piece_sets.iter().map(|s| s.to_vec()).collect::<Vec<_>>(),
    ));
    let mut handles = Vec::new();
    for _ in 0..parallelism.unwrap_or_else(|| thread::available_parallelism().unwrap().get()) {
        let sets = sets.clone();
        let mut clone = tablebase.clone();
        handles.push(thread::spawn(move || {
            loop {
                let set = { sets.lock().unwrap().pop() };
                if let Some(set) = set {
                    generate_tablebase(&mut clone, &set);
                } else {
                    break;
                }
            }
            clone
        }))
    }
    for handle in handles {
        let res = handle.join().unwrap();
        tablebase.merge(&res);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        tablebase.white_add_impl(&board, m, 1);
        assert_eq!(tablebase.white_result(&board), Some((m, 1)));
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
            assert_eq!(hash(b1).0, hash(b2).0);
        }
        fn assert_hash_ne<const W: usize, const H: usize>(b1: &Board<W, H>, b2: &Board<W, H>) {
            assert_ne!(hash(b1).0, hash(b2).0);
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
    #[test]
    fn test_generate_all_boards() {
        {
            let boards = GenerateAllBoards::<8, 8>::new(&[Piece::new(White, King)]);
            assert_eq!(boards.count(), 64);
        }
        {
            let boards = GenerateAllBoards::<8, 8>::new(&[
                Piece::new(White, King),
                Piece::new(White, Queen),
            ])
            .collect::<Vec<_>>();
            assert_eq!(boards.len(), 64 * 63);
            assert_eq!(boards[0][(0, 0)], Some(Piece::new(White, King)));
            assert_eq!(boards[0][(1, 0)], Some(Piece::new(White, Queen)));
            assert_eq!(boards[0][(2, 0)], None);

            assert_eq!(boards[1][(0, 0)], Some(Piece::new(White, King)));
            assert_eq!(boards[1][(1, 0)], None);
            assert_eq!(boards[1][(2, 0)], Some(Piece::new(White, Queen)));
        }
        {
            for b in GenerateAllBoards::<4, 4>::new(&[
                Piece::new(White, King),
                Piece::new(White, Pawn),
                Piece::new(White, Pawn),
            ]) {
                for y in 0..4_i8 {
                    for x in 0..4_i8 {
                        if let Some(p) = b[(x, y)] {
                            if p.ty() == Pawn {
                                assert_ne!(0, y);
                                assert_ne!(3, y);
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn test_populate_initial_tablebases() {
        let mut tablebase = Tablebase::default();
        let ret = populate_initial_wins::<4, 4>(
            &mut tablebase,
            &[Piece::new(White, King), Piece::new(Black, King)],
        );
        assert_ne!(ret.len(), 0);

        assert_eq!(
            tablebase.white_result(&Board::with_pieces(&[
                (Coord::new(0, 0), Piece::new(White, King)),
                (Coord::new(0, 1), Piece::new(Black, King))
            ])),
            Some((
                Move {
                    from: Coord::new(0, 0),
                    to: Coord::new(0, 1)
                },
                1
            ))
        );
        for board in ret {
            let wk_coord = board.king_coord(White);
            let bk_coord = board.king_coord(Black);
            assert!((wk_coord.x - bk_coord.x).abs() <= 1);
            assert!((wk_coord.y - bk_coord.y).abs() <= 1);
        }
    }

    #[test]
    #[ignore]
    fn test_populate_initial_tablebases_stalemate() {
        let mut tablebase = Tablebase::default();
        populate_initial_wins(
            &mut tablebase,
            &[
                Piece::new(White, King),
                Piece::new(Black, Pawn),
                Piece::new(Black, King),
            ],
        );
        assert_eq!(
            tablebase.black_depth_impl(&Board::<1, 8>::with_pieces(&[
                (Coord::new(0, 7), Piece::new(White, King)),
                (Coord::new(0, 1), Piece::new(Black, Pawn)),
                (Coord::new(0, 0), Piece::new(Black, King)),
            ])),
            Some(0)
        );
        assert!(
            !tablebase.black_contains_impl(&Board::<1, 8>::with_pieces(&[
                (Coord::new(0, 7), Piece::new(White, King)),
                (Coord::new(0, 2), Piece::new(Black, Pawn)),
                (Coord::new(0, 0), Piece::new(Black, King)),
            ]))
        );
    }
    #[test]
    fn test_generate_king_king_5_1_tablebase() {
        let wk = Piece::new(White, King);
        let bk = Piece::new(Black, King);
        let mut tablebase = Tablebase::<5, 1>::default();
        generate_tablebase(&mut tablebase, &[wk, bk]);

        assert_eq!(
            tablebase.white_result(&Board::<5, 1>::with_pieces(&[
                (Coord::new(0, 0), wk),
                (Coord::new(3, 0), bk)
            ])),
            Some((
                Move {
                    from: Coord::new(0, 0),
                    to: Coord::new(1, 0)
                },
                5
            ))
        );
        assert_eq!(
            tablebase.black_result(&Board::<5, 1>::with_pieces(&[
                (Coord::new(1, 0), wk),
                (Coord::new(3, 0), bk)
            ])),
            Some((
                Move {
                    from: Coord::new(3, 0),
                    to: Coord::new(4, 0)
                },
                4
            ))
        );
        assert_eq!(
            tablebase.white_result(&Board::<5, 1>::with_pieces(&[
                (Coord::new(1, 0), wk),
                (Coord::new(4, 0), bk)
            ])),
            Some((
                Move {
                    from: Coord::new(1, 0),
                    to: Coord::new(2, 0)
                },
                3
            ))
        );
        assert_eq!(
            tablebase.black_result(&Board::<5, 1>::with_pieces(&[
                (Coord::new(2, 0), wk),
                (Coord::new(4, 0), bk)
            ])),
            Some((
                Move {
                    from: Coord::new(4, 0),
                    to: Coord::new(3, 0)
                },
                2
            ))
        );
        assert_eq!(
            tablebase.white_result(&Board::<5, 1>::with_pieces(&[
                (Coord::new(2, 0), wk),
                (Coord::new(3, 0), bk)
            ])),
            Some((
                Move {
                    from: Coord::new(2, 0),
                    to: Coord::new(3, 0)
                },
                1
            ))
        );
    }

    #[test]
    fn test_generate_king_king_tablebase() {
        fn test<const W: usize, const H: usize>() {
            let pieces = [Piece::new(White, King), Piece::new(Black, King)];
            let mut tablebase = Tablebase::<W, H>::default();
            generate_tablebase(&mut tablebase, &pieces);
            // If white king couldn't capture on first move, no forced win.
            assert!(tablebase.black_tablebase.is_empty());
            for b in GenerateAllBoards::new(&pieces) {
                if crate::moves::is_under_attack(&b, b.king_coord(Black), Black) {
                    assert_eq!(tablebase.white_depth_impl(&b), Some(1));
                } else {
                    assert_eq!(tablebase.white_depth_impl(&b), None);
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
        let mut tablebase = Tablebase::<4, 4>::default();
        generate_tablebase(&mut tablebase, &pieces1);
        generate_tablebase(&mut tablebase, &pieces2);
        let all1_count = GenerateAllBoards::<4, 4>::new(&pieces1).count();
        let all2_count = GenerateAllBoards::<4, 4>::new(&pieces2).count();
        // With symmetry, we should expect a little over 1/8 of positions to be in the tablebase.
        assert!(tablebase.white_tablebase.len() < all1_count + all2_count / 6);
    }

    #[test]
    fn test_generate_tablebase_parallel() {
        let mut tablebase1 = Tablebase::<4, 4>::default();
        let mut tablebase2 = Tablebase::<4, 4>::default();
        let kk = [Piece::new(White, King), Piece::new(Black, King)];
        let kzk = [
            Piece::new(White, King),
            Piece::new(White, Amazon),
            Piece::new(Black, King),
        ];
        let kkz = [
            Piece::new(White, King),
            Piece::new(Black, King),
            Piece::new(Black, Amazon),
        ];
        generate_tablebase(&mut tablebase1, &kk);
        generate_tablebase(&mut tablebase1, &kzk);
        generate_tablebase(&mut tablebase1, &kkz);

        generate_tablebase(&mut tablebase2, &kk);
        generate_tablebase_parallel(&mut tablebase2, &[&kzk, &kkz], Some(2));

        assert_eq!(
            tablebase1.white_tablebase.len(),
            tablebase2.white_tablebase.len()
        );
        assert_eq!(
            tablebase1.black_tablebase.len(),
            tablebase2.black_tablebase.len()
        );
    }

    #[test]
    fn test_hash_collision() {
        let mut hashes = std::collections::HashSet::new();
        let kk = [Piece::new(White, King), Piece::new(Black, King)];
        let kqk = [
            Piece::new(White, King),
            Piece::new(White, Queen),
            Piece::new(Black, King),
        ];
        let _krkr = [
            Piece::new(White, King),
            Piece::new(White, Rook),
            Piece::new(Black, King),
            Piece::new(Black, Rook),
        ];
        #[cfg(debug_assertions)]
        let sets = [kk.as_slice(), kqk.as_slice()];
        #[cfg(not(debug_assertions))]
        let sets = [kk.as_slice(), kqk.as_slice(), _krkr.as_slice()];
        for pieces in &sets {
            for b in GenerateAllBoards::<6, 6>::new(pieces) {
                assert!(hashes.insert(hash_one_board(&b, Symmetry::default())));
            }
        }
    }
}
