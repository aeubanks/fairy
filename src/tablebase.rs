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
use rustc_hash::FxHashMap;
use std::cmp::Ordering;

pub type TBBoard<const W: i8, const H: i8> = crate::board::BoardPiece<W, H, 4>;

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

type KeyTy = ArrayVec<(i8, Piece), 4>;
type MapTy = FxHashMap<KeyTy, (Move, u16)>;

#[derive(Default, Clone)]
pub struct Tablebase<const W: i8, const H: i8> {
    // table of best move to play on white's turn to force a win
    white_tablebase: MapTy,
    // table of best move to play on black's turn to prolong a loss
    black_tablebase: MapTy,
    // winning white positions per piece set
    white_positions_per_piece_set: FxHashMap<ArrayVec<Piece, 4>, Vec<TBBoard<W, H>>>,
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
    fn merge(&mut self, other: &Self) {
        self.white_tablebase
            .extend(other.white_tablebase.iter().map(|(k, v)| (k.clone(), *v)));
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
fn generate_literally_all_boards<const W: i8, const H: i8>(pieces: &[Piece]) -> Vec<TBBoard<W, H>> {
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
    // Only support exactly one white and black king.
    let wk = Piece::new(White, King);
    let bk = Piece::new(Black, King);
    let mut pieces = pieces.iter().copied().collect::<ArrayVec<_, 4>>();
    assert!(pieces.iter().filter(|&&p| p == wk).count() == 1);
    assert!(pieces.iter().filter(|&&p| p == bk).count() == 1);
    {
        let wk_pos = pieces.iter().position(|&p| p == wk).unwrap();
        pieces.swap_remove(wk_pos);
        let bk_pos = pieces.iter().position(|&p| p == bk).unwrap();
        pieces.swap_remove(bk_pos);
    }
    // all combinations of yes/no to adding piece on board (except kings)
    for enabled in 0..(1 << pieces.len()) {
        let mut iter_pieces = ArrayVec::<_, 4>::new();
        for (i, piece) in pieces.iter().copied().enumerate() {
            if enabled & (1 << i) != 0 {
                iter_pieces.push(piece);
            }
        }
        iter_pieces.push(wk);
        iter_pieces.push(bk);
        generate_literally_all_boards_impl(&mut ret, TBBoard::default(), &iter_pieces);
    }
    ret
}

pub struct GenerateAllBoards<const W: i8, const H: i8> {
    pieces: ArrayVec<Piece, 6>,
    stack: ArrayVec<Coord, 6>,
    board: TBBoard<W, H>,
    has_pawn: bool,
}

impl<const W: i8, const H: i8> GenerateAllBoards<W, H> {
    fn next_coord(mut c: Coord) -> Coord {
        c.x += 1;
        if c.x == W {
            c.x = 0;
            c.y += 1;
        }
        c
    }
    fn valid_piece_coord(&self, c: Coord, piece: Piece) -> bool {
        // Can do normal symmetry optimizations here.
        if piece == Piece::new(Black, King) {
            if c.x > (W - 1) / 2 {
                return false;
            }
            if !self.has_pawn {
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
    }
    fn first_empty_coord_from(&self, mut c: Coord, piece: Piece) -> Option<Coord> {
        while c.y != H {
            if self.board.get(c).is_none() && self.valid_piece_coord(c, piece) {
                return Some(c);
            }
            c = Self::next_coord(c);
        }
        None
    }
    pub fn new(pieces: &[Piece]) -> Self {
        let bk = Piece::new(Black, King);
        let has_pawn = pieces.iter().any(|p| p.ty() == Pawn);
        // Only support at most one black king if symmetry optimizations are on.
        assert!(pieces.iter().filter(|&&p| p == bk).count() <= 1);
        let mut pieces = pieces.iter().copied().collect::<ArrayVec<_, 6>>();
        // Since we deduplicate symmetric positions via the black king, make sure it's placed first.
        if let Some(idx) = pieces.iter().position(|&p| p == bk) {
            pieces.swap(idx, 0);
        }
        let mut ret = Self {
            pieces,
            stack: Default::default(),
            board: Default::default(),
            has_pawn,
        };
        for p in &ret.pieces {
            let c = ret.first_empty_coord_from(Coord::new(0, 0), *p).unwrap();
            ret.stack.push(c);
            ret.board.add_piece(c, *p);
        }
        ret
    }
}

impl<const W: i8, const H: i8> Iterator for GenerateAllBoards<W, H> {
    type Item = TBBoard<W, H>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.stack.is_empty() {
            return None;
        }
        let ret = self.board.clone();
        let mut done = false;
        let mut add_from_idx = self.stack.len();
        for i in (0..self.stack.len()).rev() {
            assert_eq!(self.board.get(self.stack[i]), Some(self.pieces[i]));
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
    pieces: &[Piece],
) -> Vec<TBBoard<W, H>> {
    let mut ret = Vec::new();
    for b in GenerateAllBoards::new(pieces) {
        if populate_initial_wins_one(tablebase, &b) {
            ret.push(b);
        }
    }
    ret
}

fn iterate_black_once<const W: i8, const H: i8>(
    tablebase: &mut Tablebase<W, H>,
    board: &TBBoard<W, H>,
) -> bool {
    if tablebase.black_contains_impl(board) {
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
        tablebase.black_add_impl(board, best_move.unwrap(), max_depth + 1);
        return true;
    }
    false
}

fn iterate_black<const W: i8, const H: i8>(
    tablebase: &mut Tablebase<W, H>,
    previous_boards: &[TBBoard<W, H>],
) -> Vec<TBBoard<W, H>> {
    let mut next_boards = Vec::new();
    for prev in previous_boards {
        for m in all_moves_to_end_at_board_no_captures(prev, Black) {
            let mut b = prev.clone();
            b.swap(m.from, m.to);
            if iterate_black_once(tablebase, &b) {
                next_boards.push(b);
            }
        }
    }
    next_boards
}

fn iterate_white_once<const W: i8, const H: i8>(
    tablebase: &mut Tablebase<W, H>,
    board: &TBBoard<W, H>,
) -> bool {
    if tablebase.white_contains_impl(board) {
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
        tablebase.white_add_impl(board, best_move.unwrap(), min_depth + 1);
        return true;
    }
    false
}

fn iterate_white<const W: i8, const H: i8>(
    pieces: &[Piece],
    tablebase: &mut Tablebase<W, H>,
    previous_boards: &[TBBoard<W, H>],
) -> Vec<TBBoard<W, H>> {
    let mut next_boards = Vec::new();
    for prev in previous_boards {
        for m in all_moves_to_end_at_board_no_captures(prev, White) {
            let mut b = prev.clone();
            b.swap(m.from, m.to);
            if iterate_white_once(tablebase, &b) {
                next_boards.push(b);
            }
        }
    }
    for e in extras(tablebase, pieces) {
        if iterate_white_once(tablebase, &e) {
            next_boards.push(e);
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

fn canonical_piece_set(pieces: &[Piece]) -> ArrayVec<Piece, 4> {
    let mut ret = pieces.iter().copied().collect::<ArrayVec<Piece, 4>>();
    ret.sort_unstable_by_key(|a| a.val());
    ret
}

fn extras<const W: i8, const H: i8>(
    tablebase: &mut Tablebase<W, H>,
    pieces: &[Piece],
) -> Vec<TBBoard<W, H>> {
    let mut ret = Vec::new();
    for (i, &piece_to_remove) in pieces.iter().enumerate() {
        if piece_to_remove.player() == Black && piece_to_remove.ty() != King {
            let mut pieces_minus_one = pieces.iter().copied().collect::<ArrayVec<_, 4>>();
            pieces_minus_one.remove(i);
            let boards_minus_one = tablebase
                .white_positions_per_piece_set
                .get(&pieces_minus_one)
                .expect("didn't populate dependent piece set tablebase");
            for b in boards_minus_one {
                b.foreach_piece(|p, c| {
                    if p.player() == White {
                        // TODO: make more ergonomic
                        let mut moves = Vec::new();
                        add_moves_for_piece_to_end_at_board_no_captures(&mut moves, b, p, c);
                        for m in moves {
                            let mut clone = b.clone();
                            assert_eq!(clone.get(m), None);
                            clone.swap(m, c);
                            clone.add_piece(c, piece_to_remove);
                            ret.push(clone);
                        }
                    }
                });
            }
        }
    }
    ret
}

// must generate all piece combinations at once
#[cfg(test)]
fn generate_tablebase_no_opt<const W: i8, const H: i8>(
    tablebase: &mut Tablebase<W, H>,
    pieces: &[Piece],
) {
    assert!(tablebase.white_tablebase.is_empty());
    verify_piece_set(pieces);
    let all = generate_literally_all_boards(pieces);
    for b in &all {
        populate_initial_wins_one(tablebase, b);
    }
    loop {
        let mut changed = false;
        for b in &all {
            changed |= iterate_black_once(tablebase, b);
        }
        if !changed {
            break;
        }

        changed = false;
        for b in &all {
            changed |= iterate_white_once(tablebase, b);
        }
        if !changed {
            break;
        }
    }
}

pub fn generate_tablebase<const W: i8, const H: i8>(
    tablebase: &mut Tablebase<W, H>,
    pieces: &[Piece],
) {
    verify_piece_set(pieces);
    let pieces = canonical_piece_set(pieces);
    let mut boards_to_check = populate_initial_wins(tablebase, &pieces);
    let mut all_added_white_boards = boards_to_check.clone();
    loop {
        boards_to_check = iterate_black(tablebase, &boards_to_check);
        if boards_to_check.is_empty() {
            break;
        }

        boards_to_check = iterate_white(&pieces, tablebase, &boards_to_check);
        all_added_white_boards.reserve(boards_to_check.len());
        for b in &boards_to_check {
            all_added_white_boards.push(b.clone());
        }
        if boards_to_check.is_empty() {
            break;
        }
    }
    tablebase
        .white_positions_per_piece_set
        .insert(pieces, all_added_white_boards);
}

pub fn generate_tablebase_parallel<const W: i8, const H: i8>(
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
    use crate::moves::is_under_attack;
    use crate::player::next_player;

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
            assert_eq!(
                boards[0].get(Coord::new(0, 0)),
                Some(Piece::new(White, King))
            );
            assert_eq!(
                boards[0].get(Coord::new(1, 0)),
                Some(Piece::new(White, Queen))
            );
            assert_eq!(boards[0].get(Coord::new(2, 0)), None);

            assert_eq!(
                boards[1].get(Coord::new(0, 0)),
                Some(Piece::new(White, King))
            );
            assert_eq!(boards[1].get(Coord::new(1, 0)), None);
            assert_eq!(
                boards[1].get(Coord::new(2, 0)),
                Some(Piece::new(White, Queen))
            );
        }
        {
            for b in GenerateAllBoards::<4, 4>::new(&[
                Piece::new(White, King),
                Piece::new(White, Pawn),
                Piece::new(White, Pawn),
            ]) {
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

        for b in GenerateAllBoards::<4, 4>::new(&[Piece::new(Black, King)]) {
            let c = b.king_coord(Black);
            assert!(c.x <= 1);
            assert!(c.y <= 1);
            assert!(c.x <= c.y);
        }
        for b in GenerateAllBoards::<5, 5>::new(&[Piece::new(Black, King)]) {
            let c = b.king_coord(Black);
            assert!(c.x <= 2);
            assert!(c.y <= 2);
            assert!(c.x <= c.y);
        }
        assert_eq!(
            GenerateAllBoards::<4, 4>::new(&[Piece::new(Black, King)]).count(),
            3
        );
        assert_eq!(
            GenerateAllBoards::<5, 5>::new(&[Piece::new(Black, King)]).count(),
            6
        );
        assert_eq!(
            GenerateAllBoards::<5, 4>::new(&[Piece::new(Black, King)]).count(),
            6
        );
        GenerateAllBoards::<4, 4>::new(&[
            Piece::new(Black, Rook),
            Piece::new(Black, Amazon),
            Piece::new(Black, Queen),
            Piece::new(Black, King),
        ]);
    }
    #[test]
    fn test_generate_literally_all_boards() {
        assert_eq!(
            generate_literally_all_boards::<3, 3>(&[
                Piece::new(Black, King),
                Piece::new(White, King),
            ])
            .len(),
            9 * 8
        );
        assert_eq!(
            generate_literally_all_boards::<3, 3>(&[
                Piece::new(Black, King),
                Piece::new(White, King),
                Piece::new(White, Rook),
            ])
            .len(),
            9 * 8 * 7 + 9 * 8
        );
        assert_eq!(
            generate_literally_all_boards::<3, 3>(&[
                Piece::new(Black, King),
                Piece::new(White, King),
                Piece::new(White, Pawn),
            ])
            .len(),
            3 * 8 * 7 + 9 * 8
        );
        let bs = generate_literally_all_boards::<3, 3>(&[
            Piece::new(Black, King),
            Piece::new(White, King),
            Piece::new(White, Rook),
            Piece::new(Black, Rook),
        ]);
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
        let ret = populate_initial_wins::<4, 4>(
            &mut tablebase,
            &[Piece::new(White, King), Piece::new(Black, King)],
        );
        assert_ne!(ret.len(), 0);

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
        for board in ret {
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
        let (_, mut expected_depth) = tablebase.white_result(&board).unwrap();
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

    fn verify_tablebases_equal<const W: i8, const H: i8>(
        tb1: &Tablebase<W, H>,
        tb2: &Tablebase<W, H>,
        pieces: &[Piece],
    ) {
        for b in GenerateAllBoards::<W, H>::new(pieces) {
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
        }
    }

    fn verify_all_three_piece_positions_forced_win(pieces: &[Piece]) {
        assert_eq!(pieces.len(), 3);
        let mut tablebase = Tablebase::<4, 4>::default();
        let mut tablebase_no_optimize = Tablebase::<4, 4>::default();
        let kk = [Piece::new(White, King), Piece::new(Black, King)];
        generate_tablebase(&mut tablebase, &kk);
        generate_tablebase(&mut tablebase, pieces);

        generate_tablebase_no_opt(&mut tablebase_no_optimize, pieces);

        verify_tablebases_equal(&tablebase, &tablebase_no_optimize, pieces);

        for b in GenerateAllBoards::<4, 4>::new(pieces) {
            let wd = tablebase.white_result(&b);
            let bd = tablebase.black_result(&b);
            assert!(wd.unwrap().1 % 2 == 1);
            assert!(bd.is_none() || bd.unwrap().1 % 2 == 0);
            verify_board_tablebase(&b, &tablebase);
        }
    }

    #[test]
    fn test_kqk() {
        let pieces = [
            Piece::new(White, King),
            Piece::new(White, Queen),
            Piece::new(Black, King),
        ];
        verify_all_three_piece_positions_forced_win(&pieces);
    }

    #[test]
    fn test_krk() {
        let pieces = [
            Piece::new(White, King),
            Piece::new(White, Rook),
            Piece::new(Black, King),
        ];
        verify_all_three_piece_positions_forced_win(&pieces);
    }

    #[test]
    fn test_kek() {
        let pieces = [
            Piece::new(White, King),
            Piece::new(White, Empress),
            Piece::new(Black, King),
        ];
        verify_all_three_piece_positions_forced_win(&pieces);
    }

    #[test]
    fn test_kck() {
        let pieces = [
            Piece::new(White, King),
            Piece::new(White, Cardinal),
            Piece::new(Black, King),
        ];
        verify_all_three_piece_positions_forced_win(&pieces);
    }

    #[test]
    fn test_kak() {
        let pieces = [
            Piece::new(White, King),
            Piece::new(White, Cardinal),
            Piece::new(Black, King),
        ];
        verify_all_three_piece_positions_forced_win(&pieces);
    }

    #[test]
    fn test_kk_5_1() {
        let wk = Piece::new(White, King);
        let bk = Piece::new(Black, King);
        let mut tablebase = Tablebase::<5, 1>::default();
        generate_tablebase(&mut tablebase, &[wk, bk]);

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
            let pieces = [Piece::new(White, King), Piece::new(Black, King)];
            let mut tablebase = Tablebase::<W, H>::default();
            generate_tablebase(&mut tablebase, &pieces);
            // If white king couldn't capture on first move, no forced win.
            for b in GenerateAllBoards::new(&pieces) {
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
        let mut tablebase = Tablebase::<4, 4>::default();
        for set in [
            [wk, bk].as_slice(),
            &[wk, bk, wq],
            &[wk, bk, bq],
            &[wk, wq, bk, bq],
        ] {
            generate_tablebase(&mut tablebase, set);
        }
        let mut tablebase_no_optimize = Tablebase::<4, 4>::default();
        generate_tablebase_no_opt(&mut tablebase_no_optimize, &[wk, wq, bk, bq]);
        verify_tablebases_equal(&tablebase, &tablebase_no_optimize, &[wk, wq, bk, bq]);
    }

    #[test]
    // FIXME
    #[should_panic]
    fn test_kqkr() {
        let wk = Piece::new(White, King);
        let wq = Piece::new(White, Queen);
        let bk = Piece::new(Black, King);
        let br = Piece::new(Black, Rook);
        let mut tablebase = Tablebase::<4, 4>::default();
        for set in [
            [wk, bk].as_slice(),
            &[wk, bk, wq],
            &[wk, bk, br],
            &[wk, wq, bk, br],
        ] {
            generate_tablebase(&mut tablebase, set);
        }
        let mut tablebase_no_optimize = Tablebase::<4, 4>::default();
        generate_tablebase_no_opt(&mut tablebase_no_optimize, &[wk, wq, bk, br]);
        verify_tablebases_equal(&tablebase, &tablebase_no_optimize, &[wk, wq, bk, br]);
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
}
