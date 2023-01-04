use crate::board::{Board, Move};
use crate::coord::Coord;
use crate::moves::{
    add_moves_for_piece_to_end_at_board_no_captures, all_moves,
    all_moves_to_end_at_board_no_captures, under_attack_from_coord,
};
use crate::piece::Piece;
use crate::piece::Type::*;
use crate::player::Player::*;
use arrayvec::ArrayVec;
use log::info;
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

const MAX_PIECES: usize = 4;

#[derive(Default, PartialEq, Eq, Hash, Clone, Debug)]
pub struct PieceSet(ArrayVec<Piece, MAX_PIECES>);

impl PieceSet {
    pub fn new(pieces: &[Piece]) -> Self {
        let mut arr = ArrayVec::<Piece, MAX_PIECES>::default();
        for p in pieces {
            arr.push(*p);
        }
        arr.sort_unstable_by_key(|a| a.val());
        Self(arr)
    }

    fn remove(&mut self, piece: Piece) {
        let idx = self.0.iter().position(|&p| p == piece).unwrap();
        self.0.remove(idx);
    }
}

impl Deref for PieceSet {
    type Target = [Piece];

    fn deref(&self) -> &Self::Target {
        self.0.as_slice()
    }
}

impl DerefMut for PieceSet {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut_slice()
    }
}

impl IntoIterator for PieceSet {
    type Item = <ArrayVec<Piece, MAX_PIECES> as IntoIterator>::Item;
    type IntoIter = <ArrayVec<Piece, MAX_PIECES> as IntoIterator>::IntoIter;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a PieceSet {
    type Item = <&'a ArrayVec<Piece, MAX_PIECES> as IntoIterator>::Item;
    type IntoIter = <&'a ArrayVec<Piece, MAX_PIECES> as IntoIterator>::IntoIter;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

pub type TBBoard<const W: i8, const H: i8> = crate::board::BoardPiece<W, H, MAX_PIECES>;

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

type KeyTy = ArrayVec<(i8, Piece), MAX_PIECES>;
type MapTy = FxHashMap<KeyTy, (Move, u16)>;

#[derive(Default, Clone)]
pub struct Tablebase<const W: i8, const H: i8> {
    // table of best move to play on white's turn to force a win
    white_tablebase: MapTy,
    // table of best move to play on black's turn to prolong a loss
    black_tablebase: MapTy,
}

impl<const W: i8, const H: i8> Tablebase<W, H> {
    pub fn white_result(&self, board: &TBBoard<W, H>) -> Option<(Move, u16)> {
        let (hash, sym) = canonical_board(board);
        self.white_tablebase
            .get(&hash)
            .map(|e| (unflip_move(e.0, sym, W, H), e.1))
    }
    pub fn black_result(&self, board: &TBBoard<W, H>) -> Option<(Move, u16)> {
        let (hash, sym) = canonical_board(board);
        self.black_tablebase
            .get(&hash)
            .map(|e| (unflip_move(e.0, sym, W, H), e.1))
    }
    fn white_add_impl(&mut self, board: &TBBoard<W, H>, m: Move, depth: u16) {
        let (hash, sym) = canonical_board(board);
        debug_assert!(!self.white_tablebase.contains_key(&hash));
        self.white_tablebase
            .insert(hash, (flip_move(m, sym, W, H), depth));
    }
    fn white_contains_impl(&self, board: &TBBoard<W, H>) -> bool {
        self.white_tablebase.contains_key(&canonical_board(board).0)
    }
    fn white_depth_impl(&self, board: &TBBoard<W, H>) -> Option<u16> {
        self.white_tablebase
            .get(&canonical_board(board).0)
            .map(|e| e.1)
    }
    fn black_add_impl(&mut self, board: &TBBoard<W, H>, m: Move, depth: u16) {
        let (hash, sym) = canonical_board(board);
        debug_assert!(!self.black_tablebase.contains_key(&hash));
        self.black_tablebase
            .insert(hash, (flip_move(m, sym, W, H), depth));
    }
    fn black_contains_impl(&self, board: &TBBoard<W, H>) -> bool {
        self.black_tablebase.contains_key(&canonical_board(board).0)
    }
    fn black_depth_impl(&self, board: &TBBoard<W, H>) -> Option<u16> {
        self.black_tablebase
            .get(&canonical_board(board).0)
            .map(|e| e.1)
    }
    fn merge(&mut self, other: Self) {
        self.white_tablebase.reserve(other.white_tablebase.len());
        self.white_tablebase
            .extend(other.white_tablebase.iter().map(|(k, v)| (k.clone(), *v)));
        self.black_tablebase.reserve(other.black_tablebase.len());
        self.black_tablebase
            .extend(other.black_tablebase.iter().map(|(k, v)| (k.clone(), *v)));
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

fn board_key<const W: i8, const H: i8>(board: &TBBoard<W, H>, sym: Symmetry) -> KeyTy {
    let mut ret = KeyTy::new();
    board.foreach_piece(|piece, coord| {
        let c = flip_coord(coord, sym, W, H);
        ret.push((c.x + c.y * W, piece));
    });

    ret.sort_unstable_by(|(c1, _), (c2, _)| c1.cmp(c2));
    ret
}

fn canonical_board<const W: i8, const H: i8>(board: &TBBoard<W, H>) -> (KeyTy, Symmetry) {
    let mut symmetries_to_check = ArrayVec::<Symmetry, 8>::new();
    symmetries_to_check.push(Symmetry::default());

    let mut bk_coord = board.king_coord(Black);

    if W % 2 == 1 && bk_coord.x == W / 2 {
        let symmetries_copy = symmetries_to_check.clone();
        for mut s in symmetries_copy {
            s.flip_x = true;
            symmetries_to_check.push(s);
        }
    } else if bk_coord.x >= W / 2 {
        bk_coord = flip_coord(
            bk_coord,
            Symmetry {
                flip_x: true,
                flip_y: false,
                flip_diagonally: false,
            },
            W,
            H,
        );
        for s in symmetries_to_check.as_mut() {
            s.flip_x = true;
        }
    }
    // pawns are not symmetrical on the y axis or diagonally
    let has_pawn = board.piece_coord(|piece| piece.ty() == Pawn).is_some();
    if !has_pawn {
        if H % 2 == 1 && bk_coord.y == H / 2 {
            let symmetries_copy = symmetries_to_check.clone();
            for mut s in symmetries_copy {
                s.flip_y = true;
                symmetries_to_check.push(s);
            }
        } else if bk_coord.y >= H / 2 {
            bk_coord = flip_coord(
                bk_coord,
                Symmetry {
                    flip_x: false,
                    flip_y: true,
                    flip_diagonally: false,
                },
                W,
                H,
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

    let mut min_key = board_key(board, symmetries_to_check[0]);
    let mut sym = symmetries_to_check[0];
    for s in symmetries_to_check.into_iter().skip(1) {
        let try_key = board_key(board, s);
        let mut use_this_one = false;
        for (a, b) in min_key.iter().zip(try_key.iter()) {
            match a.0.cmp(&b.0) {
                Ordering::Equal => match a.1.val().cmp(&b.1.val()) {
                    Ordering::Equal => {}
                    Ordering::Greater => {
                        use_this_one = false;
                        break;
                    }
                    Ordering::Less => {
                        use_this_one = true;
                        break;
                    }
                },
                Ordering::Greater => {
                    use_this_one = false;
                    break;
                }
                Ordering::Less => {
                    use_this_one = true;
                    break;
                }
            }
        }

        if use_this_one {
            min_key = try_key;
            sym = s;
        }
    }
    (min_key, sym)
}

#[cfg(test)]
fn generate_literally_all_boards<const W: i8, const H: i8>(
    piece_sets: &[PieceSet],
) -> Vec<TBBoard<W, H>> {
    fn generate_literally_all_boards_impl<const W: i8, const H: i8>(
        ret: &mut Vec<TBBoard<W, H>>,
        board: TBBoard<W, H>,
        pieces: &[Piece],
    ) {
        let valid_piece_coord = |c: Coord, piece: Piece| -> bool {
            // Pawns cannot be on bottom/top row
            piece.ty() != Pawn || (c.y != 0 && c.y != H - 1)
        };
        match pieces {
            [piece, rest @ ..] => {
                for x in 0..W {
                    for y in 0..H {
                        let c = Coord::new(x, y);
                        if !valid_piece_coord(c, *piece) || board.get(c).is_some() {
                            continue;
                        }
                        let mut clone = board.clone();
                        clone.add_piece(c, *piece);
                        generate_literally_all_boards_impl(ret, clone, rest);
                    }
                }
            }
            [] => ret.push(board),
        }
    }

    let mut ret = Vec::new();
    for pieces in piece_sets {
        generate_literally_all_boards_impl(&mut ret, TBBoard::default(), pieces);
    }
    ret
}

fn generate_all_boards<const W: i8, const H: i8>(pieces: &PieceSet) -> Vec<TBBoard<W, H>> {
    fn generate_all_boards_impl<const W: i8, const H: i8>(
        ret: &mut Vec<TBBoard<W, H>>,
        board: TBBoard<W, H>,
        pieces: &[Piece],
        has_pawn: bool,
    ) {
        let valid_piece_coord = |c: Coord, piece: Piece| -> bool {
            // Can do normal symmetry optimizations here.
            if piece == Piece::new(Black, King) {
                if c.x > (W - 1) / 2 {
                    return false;
                }
                if !has_pawn {
                    if c.y > (H - 1) / 2 {
                        return false;
                    }
                    if W == H && c.x > c.y {
                        return false;
                    }
                }
                return true;
            }
            // Pawns cannot be on bottom/top row
            piece.ty() != Pawn || (c.y != 0 && c.y != H - 1)
        };
        match pieces {
            [piece, rest @ ..] => {
                for x in 0..W {
                    for y in 0..H {
                        let c = Coord::new(x, y);
                        if !valid_piece_coord(c, *piece) || board.get(c).is_some() {
                            continue;
                        }
                        let mut clone = board.clone();
                        clone.add_piece(c, *piece);
                        generate_all_boards_impl(ret, clone, rest, has_pawn);
                    }
                }
            }
            [] => ret.push(board),
        }
    }

    let mut ret = Vec::new();
    // for pieces in piece_sets {
    let has_pawn = pieces.iter().any(|&p| p.ty() == Pawn);
    let bk = Piece::new(Black, King);
    // Only support at most one black king if symmetry optimizations are on.
    assert!(pieces.iter().filter(|&&p| p == bk).count() <= 1);
    let mut clone = pieces.clone();
    // Since we deduplicate symmetric positions via the black king, make sure it's placed first.
    if let Some(idx) = clone.iter().position(|&p| p == bk) {
        clone.swap(idx, 0);
    }
    generate_all_boards_impl(&mut ret, TBBoard::default(), &clone, has_pawn);
    ret
}

const MIN_SPLIT_COUNT: usize = 10000;

#[derive(Default, Clone)]
struct PieceSets {
    maybe_reverse_capture: Vec<PieceSet>,
    no_reverse_capture: Vec<PieceSet>,
    // FIXME: this should be per-PieceSet
    black_pieces_to_add: Vec<Piece>,
}

struct PieceSetsSplitIter<'a> {
    remaining: usize,
    no_slice: &'a [PieceSet],
    no_per_slice_count: usize,
    maybe_slice: &'a [PieceSet],
    maybe_per_slice_count: usize,
}

impl<'a> Iterator for PieceSetsSplitIter<'a> {
    type Item = PieceSets;
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 || (self.no_slice.is_empty() && self.maybe_slice.is_empty()) {
            return None;
        }
        self.remaining -= 1;
        if self.remaining == 0 {
            return Some(PieceSets {
                maybe_reverse_capture: self.maybe_slice.to_vec(),
                no_reverse_capture: self.no_slice.to_vec(),
                ..Default::default()
            });
        }
        let this_maybe_slice;
        (this_maybe_slice, self.maybe_slice) = self
            .maybe_slice
            .split_at(self.maybe_per_slice_count.min(self.maybe_slice.len()));
        let this_no_slice;
        (this_no_slice, self.no_slice) = self
            .no_slice
            .split_at(self.no_per_slice_count.min(self.no_slice.len()));
        Some(PieceSets {
            maybe_reverse_capture: this_maybe_slice.to_vec(),
            no_reverse_capture: this_no_slice.to_vec(),
            ..Default::default()
        })
    }
}

impl PieceSets {
    fn split(&self, count: usize) -> PieceSetsSplitIter {
        assert_ne!(count, 0);
        PieceSetsSplitIter {
            remaining: count,
            no_slice: self.no_reverse_capture.as_slice(),
            no_per_slice_count: MIN_SPLIT_COUNT.max(self.no_reverse_capture.len() / count),
            maybe_slice: self.maybe_reverse_capture.as_slice(),
            maybe_per_slice_count: MIN_SPLIT_COUNT.max(self.maybe_reverse_capture.len() / count),
        }
    }
}

#[derive(Default)]
struct BoardsToVisit<const W: i8, const H: i8> {
    maybe_reverse_capture: Vec<TBBoard<W, H>>,
    no_reverse_capture: Vec<TBBoard<W, H>>,
}

struct BoardsToVisitSplitIter<'a, const W: i8, const H: i8> {
    remaining: usize,
    no_slice: &'a [TBBoard<W, H>],
    no_per_slice_count: usize,
    maybe_slice: &'a [TBBoard<W, H>],
    maybe_per_slice_count: usize,
}

impl<'a, const W: i8, const H: i8> Iterator for BoardsToVisitSplitIter<'a, W, H> {
    type Item = BoardsToVisit<W, H>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 || (self.no_slice.is_empty() && self.maybe_slice.is_empty()) {
            return None;
        }
        self.remaining -= 1;
        if self.remaining == 0 {
            return Some(BoardsToVisit {
                maybe_reverse_capture: self.maybe_slice.to_vec(),
                no_reverse_capture: self.no_slice.to_vec(),
            });
        }
        let this_maybe_slice;
        (this_maybe_slice, self.maybe_slice) = self
            .maybe_slice
            .split_at(self.maybe_per_slice_count.min(self.maybe_slice.len()));
        let this_no_slice;
        (this_no_slice, self.no_slice) = self
            .no_slice
            .split_at(self.no_per_slice_count.min(self.no_slice.len()));
        Some(BoardsToVisit {
            maybe_reverse_capture: this_maybe_slice.to_vec(),
            no_reverse_capture: this_no_slice.to_vec(),
        })
    }
}

impl<const W: i8, const H: i8> BoardsToVisit<W, H> {
    fn split(&self, count: usize) -> BoardsToVisitSplitIter<W, H> {
        assert_ne!(count, 0);
        BoardsToVisitSplitIter {
            remaining: count,
            no_slice: self.no_reverse_capture.as_slice(),
            no_per_slice_count: MIN_SPLIT_COUNT.max(self.no_reverse_capture.len() / count),
            maybe_slice: self.maybe_reverse_capture.as_slice(),
            maybe_per_slice_count: MIN_SPLIT_COUNT.max(self.maybe_reverse_capture.len() / count),
        }
    }

    fn is_empty(&self) -> bool {
        self.maybe_reverse_capture.is_empty() && self.no_reverse_capture.is_empty()
    }

    fn merge(&mut self, other: Self) {
        self.maybe_reverse_capture
            .reserve(other.maybe_reverse_capture.len());
        self.maybe_reverse_capture
            .extend(other.maybe_reverse_capture);
        self.no_reverse_capture
            .reserve(other.no_reverse_capture.len());
        self.no_reverse_capture.extend(other.no_reverse_capture);
    }
}

fn populate_initial_wins_one<const W: i8, const H: i8>(
    tablebase: &mut Tablebase<W, H>,
    b: &TBBoard<W, H>,
) -> bool {
    // white can capture black's king
    if !tablebase.white_contains_impl(b) {
        let opponent_king_coord = b.king_coord(Black);
        if let Some(c) = under_attack_from_coord(b, opponent_king_coord, Black) {
            tablebase.white_add_impl(
                b,
                Move {
                    from: c,
                    to: opponent_king_coord,
                },
                1,
            );
            return true;
        }
    }
    // don't support stalemate for now, it doesn't really happen on normal boards with few pieces, and takes up a noticeable chunk of tablebase generation time
    #[cfg(not(tablebase_stalemate_win))]
    debug_assert!(!all_moves(b, Black).is_empty());
    #[cfg(tablebase_stalemate_win)]
    if !tablebase.black_contains_impl(b) {
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
            );
            return true;
        }
    }
    false
}

fn populate_initial_wins<const W: i8, const H: i8>(
    tablebase: &mut Tablebase<W, H>,
    piece_sets: &PieceSets,
) -> BoardsToVisit<W, H> {
    let mut ret = BoardsToVisit::default();
    for set in &piece_sets.maybe_reverse_capture {
        for b in generate_all_boards(set) {
            if populate_initial_wins_one(tablebase, &b) {
                ret.maybe_reverse_capture.push(b);
            }
        }
    }
    for set in &piece_sets.no_reverse_capture {
        for b in generate_all_boards(set) {
            if populate_initial_wins_one(tablebase, &b) {
                ret.no_reverse_capture.push(b);
            }
        }
    }
    ret
}

fn iterate_black_once<const W: i8, const H: i8>(
    tablebase: &Tablebase<W, H>,
    out_tablebase: &mut Tablebase<W, H>,
    board: &TBBoard<W, H>,
) -> bool {
    if tablebase.black_contains_impl(board) || out_tablebase.black_contains_impl(board) {
        return false;
    }
    // None means no forced loss
    // Some(depth) means forced loss in depth moves
    let mut max_depth = Some(0);
    let mut best_move = None;
    let black_moves = all_moves(board, Black);
    if black_moves.is_empty() {
        // loss?
        return false;
    }
    for black_move in black_moves {
        let mut clone = board.clone();
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
        out_tablebase.black_add_impl(board, best_move.unwrap(), max_depth + 1);
        return true;
    }
    false
}

fn iterate_black<const W: i8, const H: i8>(
    tablebase: &Tablebase<W, H>,
    previous_boards: BoardsToVisit<W, H>,
) -> (Tablebase<W, H>, BoardsToVisit<W, H>) {
    let mut next_boards = BoardsToVisit::default();
    let mut out_tablebase = Tablebase::default();
    for prev in previous_boards.no_reverse_capture {
        for m in all_moves_to_end_at_board_no_captures(&prev, Black) {
            let mut b = prev.clone();
            b.swap(m.from, m.to);
            if iterate_black_once(tablebase, &mut out_tablebase, &b) {
                next_boards.no_reverse_capture.push(b);
            }
        }
    }
    for prev in previous_boards.maybe_reverse_capture {
        for m in all_moves_to_end_at_board_no_captures(&prev, Black) {
            let mut b = prev.clone();
            b.swap(m.from, m.to);
            if iterate_black_once(tablebase, &mut out_tablebase, &b) {
                next_boards.maybe_reverse_capture.push(b);
            }
        }
    }
    (out_tablebase, next_boards)
}

fn iterate_white_once<const W: i8, const H: i8>(
    tablebase: &Tablebase<W, H>,
    out_tablebase: &mut Tablebase<W, H>,
    board: &TBBoard<W, H>,
) -> bool {
    if tablebase.white_contains_impl(board) || out_tablebase.white_contains_impl(board) {
        return false;
    }
    // None means no forced win
    // Some(depth) means forced win in depth moves
    let mut min_depth = None;
    let mut best_move = None;
    let white_moves = all_moves(board, White);
    if white_moves.is_empty() {
        // loss?
        return false;
    }
    for white_move in white_moves {
        let mut clone = board.clone();
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
        out_tablebase.white_add_impl(board, best_move.unwrap(), min_depth + 1);
        return true;
    }
    false
}

fn iterate_white<const W: i8, const H: i8>(
    tablebase: &Tablebase<W, H>,
    previous_boards: BoardsToVisit<W, H>,
    piece_sets: &PieceSets,
) -> (Tablebase<W, H>, BoardsToVisit<W, H>) {
    let mut next_boards = BoardsToVisit::default();
    let mut out_tablebase = Tablebase::default();
    for prev in previous_boards.no_reverse_capture {
        for m in all_moves_to_end_at_board_no_captures(&prev, White) {
            let mut b = prev.clone();
            b.swap(m.from, m.to);
            if iterate_white_once(tablebase, &mut out_tablebase, &b) {
                next_boards.no_reverse_capture.push(b);
            }
        }
    }
    fn board_pieces<const W: i8, const H: i8>(b: &TBBoard<W, H>) -> PieceSet {
        let mut set = ArrayVec::<Piece, MAX_PIECES>::default();
        b.foreach_piece(|p, _| set.push(p));
        PieceSet::new(&set)
    }
    for prev in previous_boards.maybe_reverse_capture {
        for m in all_moves_to_end_at_board_no_captures(&prev, White) {
            let mut b = prev.clone();
            b.swap(m.from, m.to);
            if iterate_white_once(tablebase, &mut out_tablebase, &b) {
                next_boards.maybe_reverse_capture.push(b);
            }
        }

        for black_piece in &piece_sets.black_pieces_to_add {
            prev.foreach_piece(|p, c| {
                if p.player() == White {
                    // TODO: make more ergonomic
                    let mut moves = Vec::new();
                    add_moves_for_piece_to_end_at_board_no_captures(&mut moves, &prev, p, c);
                    for m in moves {
                        let mut clone = prev.clone();
                        assert_eq!(clone.get(m), None);
                        clone.swap(m, c);
                        clone.add_piece(c, *black_piece);
                        if iterate_white_once(tablebase, &mut out_tablebase, &clone) {
                            // TODO: optimize this
                            if piece_sets
                                .maybe_reverse_capture
                                .contains(&board_pieces(&clone))
                            {
                                next_boards.maybe_reverse_capture.push(clone);
                            } else {
                                next_boards.no_reverse_capture.push(clone);
                            }
                        }
                    }
                }
            });
        }
    }

    (out_tablebase, next_boards)
}

fn verify_piece_sets(piece_sets: &[PieceSet]) {
    for pieces in piece_sets {
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
}

#[cfg(test)]
fn generate_tablebase_no_opt<const W: i8, const H: i8>(piece_sets: &[PieceSet]) -> Tablebase<W, H> {
    let mut tablebase = Tablebase::default();
    verify_piece_sets(piece_sets);
    let all = generate_literally_all_boards(piece_sets);
    for b in &all {
        populate_initial_wins_one(&mut tablebase, b);
    }
    loop {
        let mut changed = false;
        let mut black_out = Tablebase::default();
        for b in &all {
            changed |= iterate_black_once(&tablebase, &mut black_out, b);
        }
        if !changed {
            break;
        }
        tablebase.merge(black_out);

        changed = false;
        let mut white_out = Tablebase::default();
        for b in &all {
            changed |= iterate_white_once(&tablebase, &mut white_out, b);
        }
        if !changed {
            break;
        }
        tablebase.merge(white_out);
    }
    tablebase
}

fn calculate_piece_sets(piece_sets: &[PieceSet]) -> PieceSets {
    // clone and sort/canonicalize piece sets
    let piece_sets = piece_sets.to_vec();
    let mut set = HashSet::new();
    for pieces in &piece_sets {
        for &p in pieces.iter() {
            if p.player() == Black && p.ty() != King {
                let mut subset = pieces.clone();
                subset.remove(p);
                set.insert(subset);
            }
        }
    }
    let mut ret = PieceSets::default();
    let mut black_non_king_pieces = HashSet::new();
    for pieces in piece_sets {
        for &p in &pieces {
            if p.player() == Black && p.ty() != King {
                black_non_king_pieces.insert(p);
            }
        }
        if set.contains(&pieces) {
            ret.maybe_reverse_capture.push(pieces);
        } else {
            ret.no_reverse_capture.push(pieces);
        }
    }
    ret.black_pieces_to_add = black_non_king_pieces.into_iter().collect();
    ret
}

fn info_tablebase<const W: i8, const H: i8>(tablebase: &Tablebase<W, H>) {
    info!(
        "tablebase white size {}, black size {}",
        tablebase.white_tablebase.len(),
        tablebase.black_tablebase.len()
    );
}

pub fn generate_tablebase<const W: i8, const H: i8>(piece_sets: &[PieceSet]) -> Tablebase<W, H> {
    info!("generating tablebases for {:?}", piece_sets);
    verify_piece_sets(piece_sets);
    let piece_sets = calculate_piece_sets(piece_sets);
    info!("populating initial wins");
    let mut tablebase = Tablebase::default();
    let mut boards_to_check = populate_initial_wins(&mut tablebase, &piece_sets);
    info_tablebase(&tablebase);
    let mut i = 0;
    loop {
        info!("iteration {}", i);
        info!("iterate_black");
        let black_out;
        (black_out, boards_to_check) = iterate_black(&tablebase, boards_to_check);
        info_tablebase(&tablebase);
        if boards_to_check.is_empty() {
            break;
        }
        tablebase.merge(black_out);

        info!("iterate_white");
        let white_out;
        (white_out, boards_to_check) = iterate_white(&tablebase, boards_to_check, &piece_sets);
        tablebase.merge(white_out);
        info_tablebase(&tablebase);
        if boards_to_check.is_empty() {
            break;
        }
        i += 1;
    }
    info!("done");
    tablebase
}

pub fn generate_tablebase_parallel<const W: i8, const H: i8>(
    piece_sets: &[PieceSet],
    parallelism: Option<usize>,
) -> Tablebase<W, H> {
    info!("generating tablebases (in parallel) for {:?}", piece_sets);

    use std::sync::mpsc::channel;

    let pool = {
        let mut builder = threadpool::Builder::new();
        if let Some(p) = parallelism {
            builder = builder.num_threads(p);
        }
        builder.build()
    };
    let pool_count = pool.max_count();

    verify_piece_sets(piece_sets);
    let piece_sets = calculate_piece_sets(piece_sets);

    info!("populating initial wins");
    let mut tablebase = Tablebase::default();
    let mut boards_to_check = {
        let (tx, rx) = channel();
        for set_clone in piece_sets.split(pool_count) {
            let mut tablebase_clone = Tablebase::default();
            let tx = tx.clone();
            pool.execute(move || {
                let boards = populate_initial_wins(&mut tablebase_clone, &set_clone);
                tx.send((boards, tablebase_clone)).unwrap();
            });
        }
        drop(tx);
        let mut boards_to_check = BoardsToVisit::<W, H>::default();
        for (boards, tablebase_clone) in rx {
            boards_to_check.merge(boards);
            tablebase.merge(tablebase_clone);
        }
        boards_to_check
    };
    info_tablebase(&tablebase);

    let mut i = 0;
    loop {
        info!("iteration {}", i);
        info!("iterate_black");
        boards_to_check = {
            let tablebase_arc = Arc::new(tablebase);
            let (tx, rx) = channel();
            for boards_clone in boards_to_check.split(pool_count) {
                let tablebase_clone = tablebase_arc.clone();
                let tx = tx.clone();
                pool.execute(move || {
                    let ret = iterate_black(&tablebase_clone, boards_clone);
                    tx.send(ret).unwrap();
                });
            }
            drop(tx);
            let mut boards_to_check = BoardsToVisit::<W, H>::default();
            let mut tablebases_to_merge = Vec::new();
            for (tablebase_to_merge, boards) in rx {
                boards_to_check.merge(boards);
                tablebases_to_merge.push(tablebase_to_merge);
            }
            tablebase = match Arc::try_unwrap(tablebase_arc) {
                Ok(t) => t,
                Err(_) => panic!("threads didn't finish using tablebase?"),
            };
            for to_merge in tablebases_to_merge {
                tablebase.merge(to_merge);
            }
            boards_to_check
        };
        info_tablebase(&tablebase);
        if boards_to_check.is_empty() {
            break;
        }

        info!("iterate_white");
        boards_to_check = {
            let tablebase_arc = Arc::new(tablebase);
            let (tx, rx) = channel();
            for boards_clone in boards_to_check.split(pool_count) {
                let tablebase_clone = tablebase_arc.clone();
                let piece_sets_clone = piece_sets.clone();
                let tx = tx.clone();
                pool.execute(move || {
                    let ret = iterate_white(&tablebase_clone, boards_clone, &piece_sets_clone);
                    tx.send(ret).unwrap();
                });
            }
            drop(tx);
            let mut boards_to_check = BoardsToVisit::<W, H>::default();
            let mut tablebases_to_merge = Vec::new();
            for (tablebase_to_merge, boards) in rx {
                boards_to_check.merge(boards);
                tablebases_to_merge.push(tablebase_to_merge);
            }
            tablebase = match Arc::try_unwrap(tablebase_arc) {
                Ok(t) => t,
                Err(_) => panic!("threads didn't finish using tablebase?"),
            };
            for to_merge in tablebases_to_merge {
                tablebase.merge(to_merge);
            }
            boards_to_check
        };
        info_tablebase(&tablebase);
        if boards_to_check.is_empty() {
            break;
        }
        i += 1;
    }
    info!("done");
    tablebase
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moves::is_under_attack;
    use crate::piece::Type;
    use crate::player::next_player;

    #[test]
    fn test_piece_sets_split_iter() {
        let mut set = PieceSets::default();
        {
            let mut iter = set.split(1);
            assert!(iter.next().is_none());
        }
        {
            let mut iter = set.split(2);
            assert!(iter.next().is_none());
        }

        set.maybe_reverse_capture
            .resize(MIN_SPLIT_COUNT, PieceSet::default());
        {
            let mut iter = set.split(1);
            {
                let v = iter.next().unwrap();
                assert_eq!(v.maybe_reverse_capture.len(), MIN_SPLIT_COUNT);
                assert!(v.no_reverse_capture.is_empty());
            }
            assert!(iter.next().is_none());
        }
        {
            let mut iter = set.split(2);
            {
                let v = iter.next().unwrap();
                assert_eq!(v.maybe_reverse_capture.len(), MIN_SPLIT_COUNT);
                assert!(v.no_reverse_capture.is_empty());
            }
            assert!(iter.next().is_none());
        }

        set.no_reverse_capture
            .resize(MIN_SPLIT_COUNT + 2, PieceSet::default());
        {
            let mut iter = set.split(1);
            {
                let v = iter.next().unwrap();
                assert_eq!(v.maybe_reverse_capture.len(), MIN_SPLIT_COUNT);
                assert_eq!(v.no_reverse_capture.len(), MIN_SPLIT_COUNT + 2);
            }
            assert!(iter.next().is_none());
        }
        {
            let mut iter = set.split(2);
            {
                let v = iter.next().unwrap();
                assert_eq!(v.maybe_reverse_capture.len(), MIN_SPLIT_COUNT);
                assert_eq!(v.no_reverse_capture.len(), MIN_SPLIT_COUNT);
            }
            {
                let v = iter.next().unwrap();
                assert_eq!(v.maybe_reverse_capture.len(), 0);
                assert_eq!(v.no_reverse_capture.len(), 2);
            }
            assert!(iter.next().is_none());
        }
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
        let board = TBBoard::<4, 4>::with_pieces(&[
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
    fn test_canonical_board() {
        let wk = Piece::new(White, King);
        let bk = Piece::new(Black, King);
        let wp = Piece::new(White, Pawn);
        let bp = Piece::new(Black, Pawn);
        let board1 =
            TBBoard::<8, 8>::with_pieces(&[(Coord::new(0, 0), wk), (Coord::new(0, 1), bk)]);
        let board2 =
            TBBoard::<8, 8>::with_pieces(&[(Coord::new(0, 0), wk), (Coord::new(0, 2), bk)]);
        let board3 =
            TBBoard::<8, 8>::with_pieces(&[(Coord::new(0, 2), bk), (Coord::new(0, 0), wk)]);

        fn assert_canon_eq<const W: i8, const H: i8>(b1: &TBBoard<W, H>, b2: &TBBoard<W, H>) {
            assert_eq!(canonical_board(b1).0, canonical_board(b2).0);
        }
        fn assert_canon_ne<const W: i8, const H: i8>(b1: &TBBoard<W, H>, b2: &TBBoard<W, H>) {
            assert_ne!(canonical_board(b1).0, canonical_board(b2).0);
        }
        assert_canon_eq(&board1, &board1);
        assert_canon_eq(&board2, &board2);
        assert_canon_ne(&board1, &board2);
        assert_canon_eq(&board2, &board3);
        assert_canon_eq(
            &board1,
            &TBBoard::<8, 8>::with_pieces(&[(Coord::new(0, 0), wk), (Coord::new(1, 0), bk)]),
        );
        assert_canon_eq(
            &board1,
            &TBBoard::<8, 8>::with_pieces(&[(Coord::new(7, 7), wk), (Coord::new(7, 6), bk)]),
        );
        assert_canon_eq(
            &board1,
            &TBBoard::<8, 8>::with_pieces(&[(Coord::new(7, 7), wk), (Coord::new(6, 7), bk)]),
        );
        assert_canon_eq(
            &TBBoard::<5, 5>::with_pieces(&[(Coord::new(3, 2), wk), (Coord::new(2, 2), bk)]),
            &TBBoard::<5, 5>::with_pieces(&[(Coord::new(1, 2), wk), (Coord::new(2, 2), bk)]),
        );
        assert_canon_eq(
            &TBBoard::<5, 5>::with_pieces(&[(Coord::new(3, 2), wk), (Coord::new(2, 2), bk)]),
            &TBBoard::<5, 5>::with_pieces(&[(Coord::new(2, 1), wk), (Coord::new(2, 2), bk)]),
        );
        assert_canon_eq(
            &TBBoard::<5, 5>::with_pieces(&[(Coord::new(3, 2), wk), (Coord::new(2, 2), bk)]),
            &TBBoard::<5, 5>::with_pieces(&[(Coord::new(2, 3), wk), (Coord::new(2, 2), bk)]),
        );
        assert_canon_eq(
            &TBBoard::<5, 5>::with_pieces(&[(Coord::new(3, 3), wk), (Coord::new(2, 2), bk)]),
            &TBBoard::<5, 5>::with_pieces(&[(Coord::new(3, 1), wk), (Coord::new(2, 2), bk)]),
        );
        assert_canon_eq(
            &TBBoard::<5, 5>::with_pieces(&[(Coord::new(3, 3), wk), (Coord::new(2, 2), bk)]),
            &TBBoard::<5, 5>::with_pieces(&[(Coord::new(1, 1), wk), (Coord::new(2, 2), bk)]),
        );
        assert_canon_eq(
            &TBBoard::<5, 5>::with_pieces(&[(Coord::new(3, 3), wk), (Coord::new(2, 2), bk)]),
            &TBBoard::<5, 5>::with_pieces(&[(Coord::new(1, 3), wk), (Coord::new(2, 2), bk)]),
        );
        assert_canon_ne(
            &TBBoard::<5, 5>::with_pieces(&[(Coord::new(3, 3), wk), (Coord::new(2, 2), bk)]),
            &TBBoard::<5, 5>::with_pieces(&[(Coord::new(1, 2), wk), (Coord::new(2, 2), bk)]),
        );
        assert_canon_eq(
            &TBBoard::<8, 8>::with_pieces(&[
                (Coord::new(1, 0), wk),
                (Coord::new(1, 1), wp),
                (Coord::new(1, 7), bk),
                (Coord::new(1, 6), bp),
            ]),
            &TBBoard::<8, 8>::with_pieces(&[
                (Coord::new(6, 0), wk),
                (Coord::new(6, 1), wp),
                (Coord::new(6, 7), bk),
                (Coord::new(6, 6), bp),
            ]),
        );
        assert_canon_ne(
            &TBBoard::<8, 8>::with_pieces(&[
                (Coord::new(1, 0), wk),
                (Coord::new(1, 1), wp),
                (Coord::new(1, 7), bk),
                (Coord::new(1, 6), bp),
            ]),
            &TBBoard::<8, 8>::with_pieces(&[
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
            let boards = generate_all_boards::<8, 8>(&PieceSet::new(&[Piece::new(White, King)]));
            assert_eq!(boards.len(), 64);
        }
        {
            let boards = generate_all_boards::<8, 8>(&PieceSet::new(&[
                Piece::new(White, King),
                Piece::new(White, Queen),
            ]));
            assert_eq!(boards.len(), 64 * 63);
            assert_eq!(
                boards[0].get(Coord::new(0, 0)),
                Some(Piece::new(White, Queen))
            );
            assert_eq!(
                boards[0].get(Coord::new(0, 1)),
                Some(Piece::new(White, King))
            );
            assert_eq!(boards[0].get(Coord::new(0, 2)), None);

            assert_eq!(
                boards[1].get(Coord::new(0, 0)),
                Some(Piece::new(White, Queen))
            );
            assert_eq!(boards[1].get(Coord::new(0, 1)), None);
            assert_eq!(
                boards[1].get(Coord::new(0, 2)),
                Some(Piece::new(White, King))
            );
        }
        {
            for b in generate_all_boards::<4, 4>(&PieceSet::new(&[
                Piece::new(White, King),
                Piece::new(White, Pawn),
                Piece::new(White, Pawn),
            ])) {
                for y in 0..4_i8 {
                    for x in 0..4_i8 {
                        if let Some(p) = b.get(Coord::new(x, y)) {
                            if p.ty() == Pawn {
                                assert_ne!(0, y);
                                assert_ne!(3, y);
                            }
                        }
                    }
                }
            }
        }

        for b in generate_all_boards::<4, 4>(&PieceSet::new(&[Piece::new(Black, King)])) {
            let c = b.king_coord(Black);
            assert!(c.x <= 1);
            assert!(c.y <= 1);
            assert!(c.x <= c.y);
        }
        for b in generate_all_boards::<5, 5>(&PieceSet::new(&[Piece::new(Black, King)])) {
            let c = b.king_coord(Black);
            assert!(c.x <= 2);
            assert!(c.y <= 2);
            assert!(c.x <= c.y);
        }
        assert_eq!(
            generate_all_boards::<4, 4>(&PieceSet::new(&[Piece::new(Black, King)])).len(),
            3
        );
        assert_eq!(
            generate_all_boards::<5, 5>(&PieceSet::new(&[Piece::new(Black, King)])).len(),
            6
        );
        assert_eq!(
            generate_all_boards::<5, 4>(&PieceSet::new(&[Piece::new(Black, King)])).len(),
            6
        );
        generate_all_boards::<4, 4>(&PieceSet::new(&[
            Piece::new(Black, Rook),
            Piece::new(Black, Amazon),
            Piece::new(Black, Queen),
            Piece::new(Black, King),
        ]));
    }
    #[test]
    fn test_generate_literally_all_boards() {
        let kk = PieceSet::new(&[Piece::new(Black, King), Piece::new(White, King)]);
        let krk = PieceSet::new(&[
            Piece::new(White, Rook),
            Piece::new(Black, King),
            Piece::new(White, King),
        ]);
        let kkr = PieceSet::new(&[
            Piece::new(Black, Rook),
            Piece::new(Black, King),
            Piece::new(White, King),
        ]);
        let krkr = PieceSet::new(&[
            Piece::new(Black, Rook),
            Piece::new(White, Rook),
            Piece::new(Black, King),
            Piece::new(White, King),
        ]);
        let kpk = PieceSet::new(&[
            Piece::new(White, Pawn),
            Piece::new(Black, King),
            Piece::new(White, King),
        ]);
        assert_eq!(
            generate_literally_all_boards::<3, 3>(&[kk.clone()]).len(),
            9 * 8
        );
        assert_eq!(
            generate_literally_all_boards::<3, 3>(&[kk.clone(), krk.clone()]).len(),
            9 * 8 * 7 + 9 * 8
        );
        assert_eq!(
            generate_literally_all_boards::<3, 3>(&[kk.clone(), kpk.clone()]).len(),
            3 * 8 * 7 + 9 * 8
        );
        let bs = generate_literally_all_boards::<3, 3>(&[kk, krk, kkr, krkr]);
        assert_eq!(bs.len(), 9 * 8 + 9 * 8 * 7 * 2 + 9 * 8 * 7 * 6);
        let count = |b: &TBBoard<3, 3>| -> usize {
            let mut count = 0;
            b.foreach_piece(|_, _| count += 1);
            count
        };
        assert!(bs.iter().any(|b| count(b) == 2));
        assert!(bs.iter().any(|b| count(b) == 3));
        assert!(bs.iter().any(|b| count(b) == 4));
    }
    #[test]
    fn test_populate_initial_tablebases() {
        let mut tablebase = Tablebase::default();
        let piece_sets = calculate_piece_sets(&[PieceSet::new(&[
            Piece::new(White, King),
            Piece::new(Black, King),
        ])]);

        let ret = populate_initial_wins::<4, 4>(&mut tablebase, &piece_sets);
        assert!(ret.maybe_reverse_capture.is_empty());
        assert!(!ret.no_reverse_capture.is_empty());

        assert_eq!(
            tablebase.white_result(&TBBoard::with_pieces(&[
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
        for board in ret.no_reverse_capture {
            let wk_coord = board.king_coord(White);
            let bk_coord = board.king_coord(Black);
            assert!((wk_coord.x - bk_coord.x).abs() <= 1);
            assert!((wk_coord.y - bk_coord.y).abs() <= 1);
        }
    }

    #[cfg(tablebase_stalemate_win)]
    mod stalemate {
        #[test]
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
                tablebase.black_depth_impl(&TBBoard::<1, 8>::with_pieces(&[
                    (Coord::new(0, 7), Piece::new(White, King)),
                    (Coord::new(0, 1), Piece::new(Black, Pawn)),
                    (Coord::new(0, 0), Piece::new(Black, King)),
                ])),
                Some(0)
            );
            assert!(
                !tablebase.black_contains_impl(&TBBoard::<1, 8>::with_pieces(&[
                    (Coord::new(0, 7), Piece::new(White, King)),
                    (Coord::new(0, 2), Piece::new(Black, Pawn)),
                    (Coord::new(0, 0), Piece::new(Black, King)),
                ]))
            );
        }
    }

    fn test_tablebase<const W: i8, const H: i8>(sets: &[PieceSet]) -> Tablebase<W, H> {
        let tablebase1 = generate_tablebase(sets);
        let tablebase2 = generate_tablebase_no_opt(sets);
        verify_tablebases_equal(&tablebase1, &tablebase2, sets);
        tablebase1
    }

    fn test_tablebase_parallel<const W: i8, const H: i8>(sets: &[PieceSet]) -> Tablebase<W, H> {
        let tablebase1 = generate_tablebase(sets);
        let tablebase2 = generate_tablebase_parallel(sets, Some(2));
        verify_tablebases_equal(&tablebase1, &tablebase2, sets);
        tablebase1
    }

    #[test]
    fn test_generate_tablebase_parallel() {
        let kk = PieceSet::new(&[Piece::new(White, King), Piece::new(Black, King)]);
        let kak = PieceSet::new(&[
            Piece::new(White, King),
            Piece::new(White, Amazon),
            Piece::new(Black, King),
        ]);
        let kka = PieceSet::new(&[
            Piece::new(White, King),
            Piece::new(Black, King),
            Piece::new(Black, Amazon),
        ]);
        let sets = [kk, kak, kka];
        test_tablebase_parallel::<4, 4>(&sets);
    }

    fn verify_board_tablebase<const W: i8, const H: i8>(
        board: &TBBoard<W, H>,
        tablebase: &Tablebase<W, H>,
    ) {
        fn black_king_exists<const W: i8, const H: i8>(board: &TBBoard<W, H>) -> bool {
            board
                .piece_coord(|piece| piece.player() == Black && piece.ty() == King)
                .is_some()
        }

        let mut board = board.clone();
        if let Some((_, mut expected_depth)) = tablebase.white_result(&board) {
            let mut player = White;

            while expected_depth > 0 {
                assert!(black_king_exists(&board));
                let (m, depth) = match player {
                    White => tablebase.white_result(&board),
                    Black => tablebase.black_result(&board),
                }
                .unwrap();
                assert_eq!(depth, expected_depth);
                board.make_move(m, player);
                expected_depth -= 1;
                player = next_player(player);
            }
            assert!(!black_king_exists(&board));
        }
    }

    fn verify_tablebases_equal<const W: i8, const H: i8>(
        tb1: &Tablebase<W, H>,
        tb2: &Tablebase<W, H>,
        piece_sets: &[PieceSet],
    ) {
        for set in piece_sets {
            for b in generate_all_boards::<W, H>(set) {
                let w1 = tb1.white_result(&b).map(|(_, d)| d);
                let w2 = tb2.white_result(&b).map(|(_, d)| d);
                if w1 != w2 {
                    println!("{:?}", &b);
                    println!("w1: {:?}", w1);
                    println!("w2: {:?}", w2);
                    panic!("white_result mismatch");
                }
                let b1 = tb1.black_result(&b).map(|(_, d)| d);
                let b2 = tb2.black_result(&b).map(|(_, d)| d);
                if b1 != b2 {
                    println!("{:?}", &b);
                    println!("b1: {:?}", b1);
                    println!("b2: {:?}", b2);
                    panic!("black_result mismatch");
                }
                verify_board_tablebase(&b, tb1);
            }
        }
    }

    fn verify_all_three_piece_positions_forced_win(ty: Type) {
        let set = PieceSet::new(&[
            Piece::new(White, King),
            Piece::new(Black, King),
            Piece::new(White, ty),
        ]);
        let kk = PieceSet::new(&[Piece::new(White, King), Piece::new(Black, King)]);
        let sets = [kk, set.clone()];

        let tablebase = test_tablebase(&sets);

        for b in generate_all_boards::<4, 4>(&set) {
            let wd = tablebase.white_result(&b);
            let bd = tablebase.black_result(&b);
            assert!(wd.unwrap().1 % 2 == 1);
            assert!(bd.is_none() || bd.unwrap().1 % 2 == 0);
        }
    }

    #[test]
    fn test_kqk() {
        verify_all_three_piece_positions_forced_win(Queen);
    }

    #[test]
    fn test_krk() {
        verify_all_three_piece_positions_forced_win(Rook);
    }

    #[test]
    fn test_kek() {
        verify_all_three_piece_positions_forced_win(Empress);
    }

    #[test]
    fn test_kck() {
        verify_all_three_piece_positions_forced_win(Cardinal);
    }

    #[test]
    fn test_kak() {
        verify_all_three_piece_positions_forced_win(Amazon);
    }

    #[test]
    fn test_kk_5_1() {
        let wk = Piece::new(White, King);
        let bk = Piece::new(Black, King);
        let kk = PieceSet::new(&[wk, bk]);
        let tablebase = test_tablebase(&[kk]);

        assert_eq!(
            tablebase.white_result(&TBBoard::<5, 1>::with_pieces(&[
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
            tablebase.black_result(&TBBoard::<5, 1>::with_pieces(&[
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
            tablebase.white_result(&TBBoard::<5, 1>::with_pieces(&[
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
            tablebase.black_result(&TBBoard::<5, 1>::with_pieces(&[
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
            tablebase.white_result(&TBBoard::<5, 1>::with_pieces(&[
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
    fn test_kk() {
        fn test<const W: i8, const H: i8>() {
            let pieces = PieceSet::new(&[Piece::new(White, King), Piece::new(Black, King)]);
            let tablebase = test_tablebase::<W, H>(&[pieces.clone()]);
            // If white king couldn't capture on first move, no forced win.
            for b in generate_all_boards(&pieces) {
                if is_under_attack(&b, b.king_coord(Black), Black) {
                    assert_eq!(tablebase.white_result(&b).unwrap().1, 1);
                } else {
                    assert_eq!(tablebase.white_result(&b), None);
                }
                assert_eq!(tablebase.black_result(&b), None);
            }
        }
        test::<6, 6>();
        test::<5, 5>();
        test::<4, 5>();
        test::<4, 6>();
    }

    #[test]
    fn test_kqkq() {
        let wk = Piece::new(White, King);
        let wq = Piece::new(White, Queen);
        let bk = Piece::new(Black, King);
        let bq = Piece::new(Black, Queen);
        let kk = PieceSet::new(&[wk, bk]);
        let kqk = PieceSet::new(&[wk, bk, wq]);
        let kkq = PieceSet::new(&[wk, bk, bq]);
        let kqkq = PieceSet::new(&[wk, bk, wq, bq]);
        let sets = [kk, kqk, kkq, kqkq];
        test_tablebase::<4, 4>(&sets);
    }

    #[test]
    fn test_kqkq_piece_order() {
        let wk = Piece::new(White, King);
        let wq = Piece::new(White, Queen);
        let bk = Piece::new(Black, King);
        let bq = Piece::new(Black, Queen);
        let kk = PieceSet::new(&[wk, bk]);
        let kqk = PieceSet::new(&[wk, bk, wq]);
        let kkq = PieceSet::new(&[wk, bk, bq]);
        let kqkq = PieceSet::new(&[wk, bk, wq, bq]);
        let sets = [kk.clone(), kqk.clone(), kkq.clone(), kqkq.clone()];
        let sets2 = [kqkq, kkq, kqk, kk];
        let tb1 = generate_tablebase::<4, 4>(&sets);
        let tb2 = generate_tablebase(&sets2);
        verify_tablebases_equal(&tb1, &tb2, &sets2);
    }

    #[test]
    fn test_kqkr() {
        let wk = Piece::new(White, King);
        let wq = Piece::new(White, Queen);
        let bk = Piece::new(Black, King);
        let br = Piece::new(Black, Rook);
        let kk = PieceSet::new(&[wk, bk]);
        let kqk = PieceSet::new(&[wk, bk, wq]);
        let kkr = PieceSet::new(&[wk, bk, br]);
        let kqkr = PieceSet::new(&[wk, bk, wq, br]);
        let sets = [kk, kqk, kkr, kqkr];

        let tablebase = test_tablebase::<4, 4>(&sets);

        // ..k.
        // ....
        // .K..
        // r..Q
        // Don't capture the rook, it's slower to checkmate overall.
        let res = tablebase.white_result(&TBBoard::with_pieces(&[
            (Coord::new(0, 0), br),
            (Coord::new(3, 0), wq),
            (Coord::new(1, 1), wk),
            (Coord::new(2, 3), bk),
        ]));
        assert_ne!(res.unwrap().0.to, Coord::new(0, 0));
        assert_eq!(res.unwrap().1, 5);
    }

    #[test]
    fn test_kqkr_parallel() {
        let wk = Piece::new(White, King);
        let wq = Piece::new(White, Queen);
        let bk = Piece::new(Black, King);
        let br = Piece::new(Black, Rook);
        let kk = PieceSet::new(&[wk, bk]);
        let kqk = PieceSet::new(&[wk, bk, wq]);
        let kkr = PieceSet::new(&[wk, bk, br]);
        let kqkr = PieceSet::new(&[wk, bk, wq, br]);
        let sets = [kk, kqk, kkr, kqkr];
        test_tablebase_parallel::<4, 4>(&sets);
    }
}
