// Tablebase: given a position, which move should be played to win the quickest (white)/lose the slowest (black) and how many moves until the black king is taken
//   White is arbitrarily used as player to win, and black to lose
//   If a position is not in the tablebase, the position is a draw as long as you do not reach another position in the tablebase
// Creating tablebases:
//   Canonicalize identical positions by symmetry
//     If no pawns on board, can mirror along x and y axis, plus flip over diagonal (if board is square)
//     If pawns on board, can only mirror along x axis
//   The set of piece sets to include in the tablebase is specified up front
//     (e.g. all combinations of 3 pieces, [WK, WQ, BK, BR])
//   Start with all piece sets with black king removed
//   From all possible positions with piece sets above, find all backwards captures and put a black king there
//   Now that we have the initial positions, alternate between black moves and white moves
//     For each position we added in the previous step, add positions from all possible reverse moves, captures, and promotions that can result in the previous position (if they haven't been seen before with the current player moving)
//   Repeat until no new positions are found
//   We can parallelize by partitioning the previous positions
//     There may be overlap in the new positions between partitions so make sure to deduplicate

use crate::board::{board_square_to_piece, Board, BoardSquare, Move};
use crate::coord::Coord;
use crate::moves::{
    all_legal_moves, all_legal_moves_to_end_at_board_captures,
    all_legal_moves_to_end_at_board_no_captures, all_moves_for_piece, under_attack_from_coord,
};
use crate::piece::Piece;
use crate::piece::Type::*;
use crate::player::Player;
use crate::player::Player::*;
use crate::timer::Timer;
use arrayvec::ArrayVec;
use log::info;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Ordering;
use std::ops::Deref;
use std::sync::Arc;

const MAX_PIECES: usize = 4;

const BK: Piece = Piece::new(Black, King);

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

pub type TBBoard<const W: usize, const H: usize> = crate::board::BoardPiece<W, H, MAX_PIECES>;

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
struct Symmetry {
    flip_x: bool,
    flip_y: bool,
    flip_diagonally: bool,
}

#[must_use]
fn flip_coord<const W: usize, const H: usize>(mut c: Coord, sym: Symmetry) -> Coord {
    if sym.flip_x {
        c.x = W as i8 - 1 - c.x;
    }
    if sym.flip_y {
        c.y = H as i8 - 1 - c.y;
    }
    if sym.flip_diagonally {
        std::mem::swap(&mut c.x, &mut c.y);
    }
    c
}

#[must_use]
fn flip_board<const W: usize, const H: usize>(b: &TBBoard<W, H>, sym: Symmetry) -> TBBoard<W, H> {
    let mut ret = TBBoard::default();
    b.foreach_piece(|p, c| ret.add_piece(flip_coord::<W, H>(c, sym), p));
    ret
}

#[must_use]
fn flip_move<const W: usize, const H: usize>(mut m: Move, sym: Symmetry) -> Move {
    m.from = flip_coord::<W, H>(m.from, sym);
    m.to = flip_coord::<W, H>(m.to, sym);
    m
}

#[must_use]
fn flip_x<const W: usize, const H: usize>(b: &TBBoard<W, H>, c: Coord) -> (TBBoard<W, H>, Coord) {
    let sym = Symmetry {
        flip_x: true,
        flip_y: false,
        flip_diagonally: false,
    };
    (flip_board::<W, H>(b, sym), flip_coord::<W, H>(c, sym))
}

#[must_use]
fn flip_y<const W: usize, const H: usize>(b: &TBBoard<W, H>, c: Coord) -> (TBBoard<W, H>, Coord) {
    let sym = Symmetry {
        flip_x: false,
        flip_y: true,
        flip_diagonally: false,
    };
    (flip_board::<W, H>(b, sym), flip_coord::<W, H>(c, sym))
}

#[must_use]
fn flip_diagonally<const W: usize, const H: usize>(
    b: &TBBoard<W, H>,
    c: Coord,
) -> (TBBoard<W, H>, Coord) {
    let sym = Symmetry {
        flip_x: false,
        flip_y: false,
        flip_diagonally: true,
    };
    (flip_board::<W, H>(b, sym), flip_coord::<W, H>(c, sym))
}

#[must_use]
fn unflip_coord(mut c: Coord, sym: Symmetry, width: usize, height: usize) -> Coord {
    if sym.flip_diagonally {
        std::mem::swap(&mut c.x, &mut c.y);
    }
    if sym.flip_y {
        c.y = height as i8 - 1 - c.y;
    }
    if sym.flip_x {
        c.x = width as i8 - 1 - c.x;
    }
    c
}

#[must_use]
fn unflip_move(mut m: Move, sym: Symmetry, width: usize, height: usize) -> Move {
    m.from = unflip_coord(m.from, sym, width, height);
    m.to = unflip_coord(m.to, sym, width, height);
    m
}

type KeyTy = ArrayVec<(i8, Piece), MAX_PIECES>;
type MapTy = FxHashMap<KeyTy, (Move, u16)>;
type VisitedTy = FxHashSet<KeyTy>;

#[derive(Default, Clone)]
pub struct Tablebase<const W: usize, const H: usize> {
    // table of best move to play on white's turn to force a win
    white_tablebase: MapTy,
    // table of best move to play on black's turn to prolong a loss
    black_tablebase: MapTy,
}

#[derive(PartialEq, Eq, Debug)]
pub enum TBMoveType {
    Win,
    Lose,
    Draw,
}

impl<const W: usize, const H: usize> Tablebase<W, H> {
    pub fn result(&self, player: Player, board: &TBBoard<W, H>) -> Option<(Move, u16)> {
        let (key, sym) = canonical_board(board);
        self.tablebase_for_player(player)
            .get(&key)
            .map(|e| (unflip_move(e.0, sym, W, H), e.1))
    }
    // given a board/player, return the optimal move
    // assumes that this tablebase can handle the input position
    pub fn result_for_real_board(
        &self,
        player: Player,
        board: &BoardSquare<W, H>,
    ) -> (Move, u16, TBMoveType) {
        let flip_sym = Symmetry {
            flip_x: false,
            flip_y: true,
            flip_diagonally: false,
        };
        let mut tb_board = board_square_to_piece(board);
        if player == Black {
            tb_board = tb_board.make_player_white(Black);
        }
        if let Some((mut m, num)) = self.result(White, &tb_board) {
            if player == Black {
                m = flip_move::<W, H>(m, flip_sym);
            }
            return (m, num, TBMoveType::Win);
        }

        tb_board = tb_board.make_player_white(Black);
        if let Some((mut m, num)) = self.result(Black, &tb_board) {
            if player == White {
                m = flip_move::<W, H>(m, flip_sym);
            }
            return (m, num, TBMoveType::Lose);
        }

        // a move may turn a draw into a loss; keep looking through all possible moves until we find one that doesn't end in a win for the opponent
        let mut all = all_legal_moves(board, player);
        loop {
            let try_move = all.pop().unwrap();
            let mut clone = board.clone();
            clone.make_move(try_move);
            tb_board = board_square_to_piece(&clone);
            if player == White {
                tb_board = tb_board.make_player_white(Black);
            }
            if self.result(White, &tb_board).is_none() {
                return (try_move, 0, TBMoveType::Draw);
            }
        }
    }
    fn tablebase_for_player(&self, player: Player) -> &MapTy {
        match player {
            White => &self.white_tablebase,
            Black => &self.black_tablebase,
        }
    }
    fn tablebase_for_player_mut(&mut self, player: Player) -> &mut MapTy {
        match player {
            White => &mut self.white_tablebase,
            Black => &mut self.black_tablebase,
        }
    }
    fn add_impl(&mut self, player: Player, board: &TBBoard<W, H>, m: Move, depth: u16) {
        let map = self.tablebase_for_player_mut(player);
        let (key, sym) = canonical_board(board);
        debug_assert!(!map.contains_key(&key));
        map.insert(key, (flip_move::<W, H>(m, sym), depth));
    }
    fn contains_impl(&self, player: Player, board_key: &KeyTy) -> bool {
        self.tablebase_for_player(player).contains_key(board_key)
    }
    fn depth_impl(&self, player: Player, board_key: &KeyTy) -> Option<u16> {
        self.tablebase_for_player(player)
            .get(board_key)
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
    pub fn serialize(&self) -> Vec<u8> {
        let timer = Timer::new();
        let mut buf = Vec::with_capacity(
            10 + (1 + 2 * MAX_PIECES + 4)
                * (self.white_tablebase.len() + self.black_tablebase.len()),
        );
        let cap = buf.capacity();
        buf.push(W as u8);
        buf.push(H as u8);
        buf.extend((self.white_tablebase.len() as u64).to_be_bytes());
        for (k, v) in self
            .white_tablebase
            .iter()
            .chain(self.black_tablebase.iter())
        {
            buf.push(k.len() as u8);
            for (a, b) in k {
                buf.push(*a as u8);
                buf.push(b.val());
            }
            buf.push((v.0.from.x + v.0.from.y * W as i8) as u8);
            buf.push((v.0.to.x + v.0.to.y * W as i8) as u8);
            buf.extend(v.1.to_be_bytes());
        }
        assert_eq!(buf.capacity(), cap);
        let ret = miniz_oxide::deflate::compress_to_vec(&buf, 1);
        info!(
            "compression ratio: {} / {} = {:.2}%",
            ret.len(),
            buf.len(),
            ret.len() as f64 / buf.len() as f64 * 100.0
        );
        info!("compression took {:?}", timer.elapsed());
        ret
    }
    pub fn deserialize(buf: &[u8]) -> Option<Self> {
        let w = W as i8;
        let h = H as i8;
        let timer = Timer::new();
        let decompressed = match miniz_oxide::inflate::decompress_to_vec(buf) {
            Ok(b) => b,
            Err(_) => return None,
        };
        let mut buf = decompressed.as_slice();
        let mut tb = Self::default();
        if buf.len() < 10 {
            return None;
        }
        if buf[0] as i8 != w || buf[1] as i8 != h {
            return None;
        }
        let white_len = u64::from_be_bytes(buf[2..10].try_into().unwrap()) as usize;
        buf = &buf[10..];
        while !buf.is_empty() {
            let k_len = buf[0] as usize;
            buf = &buf[1..];
            if buf.len() < k_len * 2 + 4 {
                return None;
            }
            let mut k = KeyTy::new();
            for _ in 0..k_len {
                k.push((buf[0] as i8, Piece::from_val(buf[1])));
                buf = &buf[2..];
            }
            let from = Coord::new(buf[0] as i8 % w, buf[0] as i8 / w);
            let to = Coord::new(buf[1] as i8 % w, buf[1] as i8 / w);
            let depth = u16::from_be_bytes(buf[2..4].try_into().unwrap());
            buf = &buf[4..];
            let tb_to_add = if tb.white_tablebase.len() < white_len {
                &mut tb.white_tablebase
            } else {
                &mut tb.black_tablebase
            };
            tb_to_add.insert(k, (Move { from, to }, depth));
        }
        if tb.white_tablebase.len() != white_len {
            return None;
        }
        info!("decompression took {:?}", timer.elapsed());
        Some(tb)
    }
    pub fn dump_stats(&self) {
        info!("white positions: {}", self.white_tablebase.len());
        info!("black positions: {}", self.black_tablebase.len());
        let mut max_depth = 0;
        for v in self.white_tablebase.values() {
            if v.1 > max_depth {
                max_depth = v.1;
            }
        }
        info!("max depth: {}", max_depth);
    }
}

fn insertion_sort<T: Copy>(slice: &mut [T], less_than: fn(T, T) -> bool) {
    for i in 1..slice.len() {
        let val = slice[i];
        for j in (0..i).rev() {
            if less_than(val, slice[j]) {
                slice.swap(j, j + 1);
            } else {
                break;
            }
        }
    }
}

fn board_key<const W: usize, const H: usize>(board: &TBBoard<W, H>, sym: Symmetry) -> KeyTy {
    let mut ret = KeyTy::new();
    board.foreach_piece(|piece, coord| {
        let c = flip_coord::<W, H>(coord, sym);
        ret.push((c.x + c.y * W as i8, piece));
    });

    insertion_sort(&mut ret, |(c1, _), (c2, _)| c1 < c2);
    ret
}

fn canonical_board<const W: usize, const H: usize>(board: &TBBoard<W, H>) -> (KeyTy, Symmetry) {
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
        bk_coord = flip_coord::<W, H>(
            bk_coord,
            Symmetry {
                flip_x: true,
                flip_y: false,
                flip_diagonally: false,
            },
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
            bk_coord = flip_coord::<W, H>(
                bk_coord,
                Symmetry {
                    flip_x: false,
                    flip_y: true,
                    flip_diagonally: false,
                },
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

fn generate_all_boards<const W: usize, const H: usize>(
    pieces: &[Piece],
) -> GenerateAllBoards<W, H, true> {
    GenerateAllBoards::new(pieces)
}

struct GenerateAllBoards<const W: usize, const H: usize, const OPT: bool> {
    pieces: ArrayVec<Piece, MAX_PIECES>,
    stack: ArrayVec<Coord, MAX_PIECES>,
    board: TBBoard<W, H>,
    has_pawn: bool,
}

impl<const W: usize, const H: usize, const OPT: bool> GenerateAllBoards<W, H, OPT> {
    fn next_coord(mut c: Coord) -> Coord {
        c.x += 1;
        if c.x == W as i8 {
            c.x = 0;
            c.y += 1;
        }
        c
    }
    fn valid_piece_coord(&self, c: Coord, piece: Piece, symmetry_opt: bool) -> bool {
        // Can do normal symmetry optimizations here.
        if OPT && symmetry_opt {
            if c.x > (W as i8 - 1) / 2 {
                return false;
            }
            if !self.has_pawn {
                if c.y > (H as i8 - 1) / 2 {
                    return false;
                }
                if W == H && c.x > c.y {
                    return false;
                }
            }
        }
        valid_piece_coord::<W, H>(piece, c)
    }
    fn first_empty_coord_from(
        &self,
        mut c: Coord,
        piece: Piece,
        symmetry_opt: bool,
    ) -> Option<Coord> {
        while c.y != H as i8 {
            if self.board.get(c).is_none() && self.valid_piece_coord(c, piece, symmetry_opt) {
                return Some(c);
            }
            c = Self::next_coord(c);
        }
        None
    }
    fn new(pieces: &[Piece]) -> Self {
        let has_pawn = pieces.iter().any(|p| p.ty() == Pawn);
        let mut pieces = pieces.iter().copied().collect::<ArrayVec<_, MAX_PIECES>>();
        // Since we deduplicate symmetric positions via non-pawn pieces at the beginning, make sure the first piece is a non-pawn if one exisets.
        if let Some(idx) = pieces.iter().position(|&p| p.ty() != Pawn) {
            pieces.swap(idx, 0);
        }
        let mut ret = Self {
            pieces,
            stack: Default::default(),
            board: Default::default(),
            has_pawn,
        };
        for (i, p) in ret.pieces.iter().enumerate() {
            let c = ret
                .first_empty_coord_from(Coord::new(0, 0), *p, i == 0)
                .unwrap();
            ret.stack.push(c);
            ret.board.add_piece(c, *p);
        }
        ret
    }
}

impl<const W: usize, const H: usize, const OPT: bool> Iterator for GenerateAllBoards<W, H, OPT> {
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
                self.first_empty_coord_from(Self::next_coord(self.stack[i]), self.pieces[i], i == 0)
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
                    .first_empty_coord_from(Coord::new(0, 0), self.pieces[i], i == 0)
                    .unwrap();
                self.board.add_piece(self.stack[i], self.pieces[i]);
            }
        }
        Some(ret)
    }
}

#[derive(Default, Clone)]
struct PieceSets {
    piece_sets: Vec<PieceSet>,
    pieces_to_add: FxHashMap<(Player, PieceSet), Vec<Piece>>,
    can_reverse_promote: FxHashSet<(Player, PieceSet)>,
}

#[derive(Default)]
struct BoardsToVisit<const W: usize, const H: usize> {
    boards: FxHashSet<KeyTy>,
}

impl<const W: usize, const H: usize> BoardsToVisit<W, H> {
    fn add(&mut self, board: TBBoard<W, H>) {
        let key = board_key(&board, Symmetry::default());
        self.boards.insert(key);
    }

    fn consume_as_vec(self) -> Vec<TBBoard<W, H>> {
        self.boards
            .into_iter()
            .map(|k| {
                let mut b = TBBoard::default();
                for (c, p) in k {
                    b.add_piece(Coord::new(c % W as i8, c / W as i8), p);
                }
                b
            })
            .collect()
    }
}

impl<const W: usize, const H: usize> BoardsToVisit<W, H> {
    fn is_empty(&self) -> bool {
        self.boards.is_empty()
    }

    fn merge(&mut self, other: Self) {
        self.boards.reserve(other.boards.len());
        self.boards.extend(other.boards);
    }
}

fn populate_initial_wins_one<const W: usize, const H: usize>(
    tablebase: &mut Tablebase<W, H>,
    b: &TBBoard<W, H>,
) -> bool {
    // white can capture black's king
    if !tablebase.contains_impl(White, &canonical_board(b).0) {
        let opponent_king_coord = b.king_coord(Black);
        if let Some(c) = under_attack_from_coord(b, opponent_king_coord, Black) {
            tablebase.add_impl(
                White,
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
    false
}

fn populate_initial_wins<const W: usize, const H: usize>(
    tablebase: &mut Tablebase<W, H>,
    piece_sets: &PieceSets,
) -> BoardsToVisit<W, H> {
    let mut ret = BoardsToVisit::default();
    for set in &piece_sets.piece_sets {
        let mut set_without_bk = set.clone();
        set_without_bk.remove(BK);
        let can_reverse_promote = piece_sets
            .can_reverse_promote
            .contains(&(White, set.clone()));

        for board in generate_all_boards(&set_without_bk) {
            board.foreach_piece(|p, c| {
                if p.player() != White {
                    return;
                }
                let moves = all_legal_moves_to_end_at_board_captures(&board, p, c);
                for m in moves {
                    let mut clone = board.clone();
                    debug_assert_eq!(clone.get(m), None);
                    let take = clone.take(c);
                    clone.add_piece(m, take.unwrap());
                    clone.add_piece(c, BK);
                    if populate_initial_wins_one(tablebase, &clone) {
                        ret.add(clone);
                    }
                }
                if can_reverse_promote && p == Piece::new(White, Queen) {
                    visit_reverse_promotion(&board, White, |b, c| {
                        let moves =
                            all_legal_moves_to_end_at_board_captures(b, Piece::new(White, Pawn), c);
                        for m in moves {
                            let mut clone = b.clone();
                            debug_assert_eq!(clone.get(m), None);
                            clone.clear(c);
                            clone.add_piece(m, Piece::new(White, Pawn));
                            clone.add_piece(c, BK);
                            if populate_initial_wins_one(tablebase, &clone) {
                                ret.add(clone);
                            }
                        }
                    });
                }
            });
        }
    }
    ret
}

// We can only introduce at most one pawn at most per move
fn visit_board_pawn_symmetry<const W: usize, const H: usize>(
    tablebase: &Tablebase<W, H>,
    out_tablebase: &mut Tablebase<W, H>,
    next_boards: &mut BoardsToVisit<W, H>,
    rejected_boards: &mut VisitedTy,
    player: Player,
    board: &TBBoard<W, H>,
    mut maybe_pawn_coord: Coord,
    maybe_pawn_piece: Piece,
    cur_max_depth: u16,
) {
    let mut board = board.clone();
    if valid_piece_coord::<W, H>(maybe_pawn_piece, maybe_pawn_coord) {
        visit_board(
            tablebase,
            out_tablebase,
            next_boards,
            rejected_boards,
            player,
            &board,
            None,
            cur_max_depth,
        );
    }
    (board, maybe_pawn_coord) = flip_y(&board, maybe_pawn_coord);
    if valid_piece_coord::<W, H>(maybe_pawn_piece, maybe_pawn_coord) {
        visit_board(
            tablebase,
            out_tablebase,
            next_boards,
            rejected_boards,
            player,
            &board,
            None,
            cur_max_depth,
        );
    }
    if W == H {
        (board, maybe_pawn_coord) = flip_diagonally(&board, maybe_pawn_coord);
        if valid_piece_coord::<W, H>(maybe_pawn_piece, maybe_pawn_coord) {
            visit_board(
                tablebase,
                out_tablebase,
                next_boards,
                rejected_boards,
                player,
                &board,
                None,
                cur_max_depth,
            );
        }
        (board, maybe_pawn_coord) = flip_y(&board, maybe_pawn_coord);
        if valid_piece_coord::<W, H>(maybe_pawn_piece, maybe_pawn_coord) {
            visit_board(
                tablebase,
                out_tablebase,
                next_boards,
                rejected_boards,
                player,
                &board,
                None,
                cur_max_depth,
            );
        }
    }
}

fn visit_board<const W: usize, const H: usize>(
    tablebase: &Tablebase<W, H>,
    out_tablebase: &mut Tablebase<W, H>,
    next_boards: &mut BoardsToVisit<W, H>,
    rejected_boards: &mut VisitedTy,
    player: Player,
    board: &TBBoard<W, H>,
    m: Option<Move>,
    cur_max_depth: u16,
) {
    let canon_board = canonical_board(board).0;
    if rejected_boards.contains(&canon_board)
        || tablebase.contains_impl(player, &canon_board)
        || out_tablebase.contains_impl(player, &canon_board)
    {
        return;
    }
    #[cfg(debug_assertions)]
    board.foreach_piece(|p, c| {
        if p.ty() == Pawn {
            assert_ne!(c.y, 0);
            assert_ne!(c.y, H as i8 - 1);
        }
    });
    let mut add = player == Black;
    let mut any_move = None;
    // if we were passed in the move that caused us to get to this position, we can directly use that for white
    // for black, all possible moves need to lead to mate, so knowing what position lead us here isn't useful
    if player == White {
        if let Some(m) = m {
            #[cfg(debug_assertions)]
            {
                let mut clone = board.clone();
                clone.make_move(m);
                debug_assert_eq!(
                    tablebase
                        .depth_impl(player.next(), &canonical_board(&clone).0)
                        .unwrap(),
                    cur_max_depth
                );
            }
            add = true;
            any_move = Some(m);
        }
    }
    if any_move.is_none() {
        board.foreach_piece(|piece, coord| {
            // bail out early if we've made our final decision to add/not add this position to the tablebase
            if add != (player == Black) {
                return;
            }
            if piece.player() != player {
                return;
            }
            for to in all_moves_for_piece(board, piece, coord) {
                let m = Move { from: coord, to };
                let mut clone = board.clone();
                #[cfg(debug_assertions)]
                if clone.get(to) == Some(BK) {
                    dbg!(to);
                    dbg!(&clone);
                    panic!();
                }
                clone.make_move(m);

                let maybe_depth = tablebase.depth_impl(player.next(), &canonical_board(&clone).0);
                match player {
                    // if we find any move that leads to mate we're done
                    White => {
                        if let Some(_d) = maybe_depth {
                            debug_assert_eq!(_d, cur_max_depth);
                            any_move = Some(m);
                            add = true;
                            break;
                        }
                    }
                    // if we find any move that doesn't lead to mate we're done
                    Black => {
                        if let Some(depth) = maybe_depth {
                            if depth == cur_max_depth {
                                any_move = Some(m);
                            }
                        } else {
                            add = false;
                            break;
                        }
                    }
                }
            }
        });
    }
    if add {
        if let Some(m) = any_move {
            #[cfg(debug_assertions)]
            if board.get(any_move.unwrap().to) == Some(Piece::new(White, King)) {
                dbg!(board);
                panic!();
            }
            next_boards.add(board.clone());
            out_tablebase.add_impl(player, board, m, cur_max_depth + 1);
        }
    } else {
        rejected_boards.insert(canon_board);
    }
}

fn visit_reverse_moves<const W: usize, const H: usize>(
    tablebase: &Tablebase<W, H>,
    out_tablebase: &mut Tablebase<W, H>,
    board: &TBBoard<W, H>,
    player: Player,
    next_boards: &mut BoardsToVisit<W, H>,
    rejected_boards: &mut VisitedTy,
    cur_max_depth: u16,
) {
    board.foreach_piece(|p, c| {
        if p.player() == player {
            let moves = all_legal_moves_to_end_at_board_no_captures(board, p, c);
            for m in moves {
                let mut clone = board.clone();
                assert_eq!(clone.get(m), None);
                clone.swap(m, c);
                visit_board(
                    tablebase,
                    out_tablebase,
                    next_boards,
                    rejected_boards,
                    player,
                    &clone,
                    Some(Move { from: m, to: c }),
                    cur_max_depth,
                );
            }
        }
    });
}

fn board_pieces<const W: usize, const H: usize>(b: &TBBoard<W, H>) -> PieceSet {
    let mut set = ArrayVec::<Piece, MAX_PIECES>::default();
    b.foreach_piece(|p, _| set.push(p));
    PieceSet::new(&set)
}

fn board_has_pawn<const W: usize, const H: usize>(board: &TBBoard<W, H>) -> bool {
    let mut ret = false;
    board.foreach_piece(|p, _| {
        if p.ty() == Pawn {
            ret = true;
        }
    });
    ret
}

fn visit_reverse_promotion<const W: usize, const H: usize, F>(
    board: &TBBoard<W, H>,
    player: Player,
    mut callback: F,
) where
    F: FnMut(&TBBoard<W, H>, Coord),
{
    let has_pawn = board_has_pawn(board);
    board.foreach_piece(|p, mut c| {
        if p == Piece::new(player, Queen) {
            if has_pawn {
                if c.y
                    == match player {
                        White => H as i8 - 1,
                        Black => 0,
                    }
                {
                    callback(board, c);
                }
            } else if c.y == H as i8 - 1 || c.y == 0 {
                let mut clone = board.clone();
                if (c.y == 0) == (player == White) {
                    (clone, c) = flip_y(&clone, c);
                }
                callback(&clone, c);
                if W == H {
                    if c.x == 0 || c.x == W as i8 - 1 {
                        if (c.x == 0) == (player == White) {
                            (clone, c) = flip_x(&clone, c);
                        }
                        (clone, c) = flip_diagonally(&clone, c);
                        callback(&clone, c);
                    }
                }
            } else if c.x == W as i8 - 1 || c.x == 0 {
                if W == H {
                    let mut clone = board.clone();
                    if (c.x == 0) == (player == White) {
                        (clone, c) = flip_x(&clone, c);
                    }
                    (clone, c) = flip_diagonally(&clone, c);
                    callback(&clone, c);
                }
            }
        }
    });
}

fn valid_piece_coord<const W: usize, const H: usize>(piece: Piece, c: Coord) -> bool {
    if piece.ty() == Pawn {
        return c.y != 0 && c.y != H as i8 - 1;
    }
    true
}

fn visit_reverse_capture<const W: usize, const H: usize>(
    tablebase: &Tablebase<W, H>,
    out_tablebase: &mut Tablebase<W, H>,
    pieces_to_add: &[Piece],
    board: &TBBoard<W, H>,
    player: Player,
    next_boards: &mut BoardsToVisit<W, H>,
    rejected_boards: &mut VisitedTy,
    cur_max_depth: u16,
) {
    let board_has_pawn = board_has_pawn(board);
    for &piece_to_add in pieces_to_add {
        board.foreach_piece(|p, c| {
            if p.player() != player {
                return;
            }
            if board_has_pawn && !valid_piece_coord::<W, H>(piece_to_add, c) {
                return;
            }
            let moves = all_legal_moves_to_end_at_board_captures(board, p, c);
            for m in moves {
                let mut clone = board.clone();
                debug_assert_eq!(clone.get(m), None);
                let take = clone.take(c);
                clone.add_piece(m, take.unwrap());
                clone.add_piece(c, piece_to_add);

                if board_has_pawn || piece_to_add.ty() != Pawn {
                    visit_board(
                        tablebase,
                        out_tablebase,
                        next_boards,
                        rejected_boards,
                        player,
                        &clone,
                        Some(Move { from: m, to: c }),
                        cur_max_depth,
                    );
                } else {
                    // if we introduced a pawn, need to visit more symmetries
                    visit_board_pawn_symmetry(
                        tablebase,
                        out_tablebase,
                        next_boards,
                        rejected_boards,
                        player,
                        &clone,
                        c,
                        piece_to_add,
                        cur_max_depth,
                    );
                }
            }
        });
    }
}

fn iterate<const W: usize, const H: usize>(
    tablebase: &Tablebase<W, H>,
    previous_boards: &[TBBoard<W, H>],
    piece_sets: &PieceSets,
    player: Player,
    cur_max_depth: u16,
) -> (Tablebase<W, H>, BoardsToVisit<W, H>) {
    let mut next_boards = BoardsToVisit::default();
    let mut out_tablebase = Tablebase::default();
    let mut rejected_boards = VisitedTy::default();
    for prev in previous_boards {
        visit_reverse_moves(
            tablebase,
            &mut out_tablebase,
            prev,
            player,
            &mut next_boards,
            &mut rejected_boards,
            cur_max_depth,
        );

        let key = (player, board_pieces(prev));
        let promote = piece_sets.can_reverse_promote.contains(&key);
        if promote {
            visit_reverse_promotion(prev, player, |b: &TBBoard<W, H>, c: Coord| {
                debug_assert_eq!(
                    c.y,
                    match player {
                        White => b.height() - 1,
                        Black => 0,
                    }
                );
                debug_assert_eq!(b.get(c), Some(Piece::new(player, Queen)));
                for m in all_legal_moves_to_end_at_board_no_captures(b, Piece::new(player, Pawn), c)
                {
                    let mut clone = b.clone();
                    let pawn = Piece::new(player, Pawn);
                    clone.clear(c);
                    clone.add_piece(m, pawn);

                    if board_has_pawn(b) {
                        visit_board(
                            tablebase,
                            &mut out_tablebase,
                            &mut next_boards,
                            &mut rejected_boards,
                            player,
                            &clone,
                            Some(Move { from: m, to: c }),
                            cur_max_depth,
                        );
                    } else {
                        // if we introduced a pawn, need to visit more symmetries
                        visit_board_pawn_symmetry(
                            tablebase,
                            &mut out_tablebase,
                            &mut next_boards,
                            &mut rejected_boards,
                            player,
                            &clone,
                            m,
                            pawn,
                            cur_max_depth,
                        );
                    }
                }
            });
        }
        if let Some(pieces_to_add) = piece_sets.pieces_to_add.get(&key) {
            visit_reverse_capture(
                tablebase,
                &mut out_tablebase,
                pieces_to_add,
                prev,
                player,
                &mut next_boards,
                &mut rejected_boards,
                cur_max_depth,
            );

            if promote {
                visit_reverse_promotion(prev, player, |b: &TBBoard<W, H>, c: Coord| {
                    debug_assert_eq!(
                        c.y,
                        match player {
                            White => b.height() - 1,
                            Black => 0,
                        }
                    );
                    debug_assert_eq!(b.get(c), Some(Piece::new(player, Queen)));
                    let pawn = Piece::new(player, Pawn);
                    for m in all_legal_moves_to_end_at_board_captures(b, pawn, c) {
                        for &piece_to_add in pieces_to_add {
                            if !valid_piece_coord::<W, H>(piece_to_add, c) {
                                continue;
                            }
                            let mut clone = b.clone();
                            clone.clear(c);
                            clone.add_piece(m, pawn);
                            clone.add_piece(c, piece_to_add);

                            if board_has_pawn(b) {
                                visit_board(
                                    tablebase,
                                    &mut out_tablebase,
                                    &mut next_boards,
                                    &mut rejected_boards,
                                    player,
                                    &clone,
                                    Some(Move { from: m, to: c }),
                                    cur_max_depth,
                                );
                            } else {
                                // if we introduced a pawn, need to visit more symmetries
                                visit_board_pawn_symmetry(
                                    tablebase,
                                    &mut out_tablebase,
                                    &mut next_boards,
                                    &mut rejected_boards,
                                    player,
                                    &clone,
                                    m,
                                    pawn,
                                    cur_max_depth,
                                );
                            }
                        }
                    }
                });
            }
        }
    }

    (out_tablebase, next_boards)
}

fn verify_piece_sets(piece_sets: &[PieceSet]) {
    for pieces in piece_sets {
        let mut bk_count = 0;
        for &p in pieces {
            if p == BK {
                bk_count += 1;
            }
        }
        assert_eq!(bk_count, 1, "must be exactly one black king in PieceSet");
    }
}

fn calculate_piece_sets(piece_sets: &[PieceSet]) -> PieceSets {
    verify_piece_sets(piece_sets);

    let mut visited = FxHashSet::<PieceSet>::default();
    let mut stack = piece_sets.to_vec();
    let mut pieces_to_add = FxHashMap::<(Player, PieceSet), Vec<Piece>>::default();
    let mut can_reverse_promote = FxHashSet::default();
    while let Some(s) = stack.pop() {
        if !s.iter().any(|&p| p.player() == White) {
            continue;
        }
        if !visited.insert(s.clone()) {
            continue;
        }
        let mut last_p = None;
        for (i, &p) in s.iter().enumerate() {
            if p.ty() == King || Some(p) == last_p {
                continue;
            }
            last_p = Some(p);
            let mut captured = s.clone();
            captured.remove(p);
            let set_to_add_to = pieces_to_add
                .entry((p.player().next(), captured.clone()))
                .or_default();
            set_to_add_to.push(p);
            stack.push(captured);
            if p.ty() == Pawn {
                let mut pawn_to_queen = s.iter().as_slice().to_vec();
                let q = Piece::new(p.player(), Queen);
                pawn_to_queen[i] = q;
                let pawn_to_queen_set = PieceSet::new(&pawn_to_queen);
                can_reverse_promote.insert((p.player(), pawn_to_queen_set.clone()));
                stack.push(pawn_to_queen_set);
            }
        }
    }

    let mut maybe_reverse_capture_set = FxHashSet::default();
    for pieces in &visited {
        for &p in pieces.iter() {
            if p.ty() != King {
                let mut subset = pieces.clone();
                subset.remove(p);
                maybe_reverse_capture_set.insert(subset);
            }
        }
    }

    pieces_to_add.shrink_to_fit();
    can_reverse_promote.shrink_to_fit();

    let ret = PieceSets {
        piece_sets: visited.into_iter().collect(),
        pieces_to_add,
        can_reverse_promote,
    };

    #[cfg(debug_assertions)]
    {
        verify_piece_sets(&ret.piece_sets);
        for (k, v) in &ret.pieces_to_add {
            for p in v {
                assert_ne!(k.0, p.player());
            }
            let mut slice = v.as_slice();
            while let Some((p, subslice)) = slice.split_last() {
                assert!(!subslice.contains(p));
                slice = subslice;
            }
        }
    }

    ret
}

fn info_tablebase<const W: usize, const H: usize>(tablebase: &Tablebase<W, H>) {
    info!(
        "tablebase white size {}, black size {}",
        tablebase.white_tablebase.len(),
        tablebase.black_tablebase.len()
    );
}

pub fn generate_tablebase<const W: usize, const H: usize>(
    piece_sets: &[PieceSet],
) -> Tablebase<W, H> {
    let total_timer = Timer::new();

    info!("generating tablebases for {:?}", piece_sets);
    let piece_sets = calculate_piece_sets(piece_sets);

    info!("populating initial wins");
    let mut tablebase = Tablebase::default();
    let mut timer = Timer::new();
    let mut boards_to_check = populate_initial_wins(&mut tablebase, &piece_sets);
    info!("took {:?}", timer.elapsed());
    info_tablebase(&tablebase);
    info!("");

    let mut i = 1;
    let mut player = Black;
    loop {
        info!("iteration {} ({:?})", i, player);
        let out;
        timer = Timer::new();
        (out, boards_to_check) = iterate(
            &tablebase,
            &boards_to_check.consume_as_vec(),
            &piece_sets,
            player,
            i,
        );
        info!("took {:?}", timer.elapsed());
        info_tablebase(&tablebase);
        info!("");
        if boards_to_check.is_empty() {
            break;
        }
        tablebase.merge(out);
        player = player.next();

        i += 1;
    }
    info!("done in {:?}", total_timer.elapsed());
    tablebase
}

pub fn generate_tablebase_parallel<const W: usize, const H: usize>(
    piece_sets: &[PieceSet],
    parallelism: Option<usize>,
) -> Tablebase<W, H> {
    use std::sync::mpsc::channel;

    let total_timer = Timer::new();

    let pool = {
        let mut builder = threadpool::Builder::new();
        if let Some(p) = parallelism {
            builder = builder.num_threads(p);
        }
        builder.build()
    };
    let pool_count = pool.max_count();

    info!("generating tablebases (in parallel) for {:?}", piece_sets);
    let piece_sets = calculate_piece_sets(piece_sets);

    info!("populating initial wins");
    let mut tablebase = Tablebase::default();
    let mut timer = Timer::new();
    let mut boards_to_check = {
        let (tx, rx) = channel();
        let mut sets = piece_sets.piece_sets.clone();
        let per_count = sets.len() / pool_count + 1;
        while !sets.is_empty() {
            let split_off = sets.len() - per_count.min(sets.len());
            let this_sets = sets.split_off(split_off);
            let this_piece_sets = PieceSets {
                piece_sets: this_sets,
                pieces_to_add: piece_sets.pieces_to_add.clone(),
                can_reverse_promote: piece_sets.can_reverse_promote.clone(),
            };
            let tx = tx.clone();
            pool.execute(move || {
                let mut tablebase_clone = Tablebase::default();
                let boards = populate_initial_wins(&mut tablebase_clone, &this_piece_sets);
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
    if pool.panic_count() != 0 {
        panic!();
    }
    info!("took {:?}", timer.elapsed());
    info_tablebase(&tablebase);
    info!("");

    let mut i = 1;
    let mut player = Black;
    loop {
        info!("iteration {} {:?}", i, player);
        timer = Timer::new();
        boards_to_check = {
            let tablebase_arc = Arc::new(tablebase);
            let (tx, rx) = channel();
            let mut to_check = boards_to_check.consume_as_vec();
            const MIN_SPLIT_COUNT: usize = 10000;
            let per_slice_count = MIN_SPLIT_COUNT.max(to_check.len() / pool_count + 1);
            while !to_check.is_empty() {
                let split_off = to_check.len() - per_slice_count.min(to_check.len());
                let batch = to_check.split_off(split_off);
                let tablebase_clone = tablebase_arc.clone();
                let piece_sets_clone = piece_sets.clone();
                let tx = tx.clone();
                pool.execute(move || {
                    let ret = iterate(&tablebase_clone, &batch, &piece_sets_clone, player, i);
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
            if pool.panic_count() != 0 {
                panic!();
            }
            boards_to_check
        };
        info!("took {:?}", timer.elapsed());
        info_tablebase(&tablebase);
        info!("");
        if boards_to_check.is_empty() {
            break;
        }

        i += 1;
        player = player.next();
    }
    info!("done in {:?}", total_timer.elapsed());
    tablebase
}

#[allow(clippy::redundant_clone)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::moves::is_under_attack;
    use crate::piece::Type;

    const WK: Piece = Piece::new(White, King);
    const WB: Piece = Piece::new(White, Bishop);
    const WQ: Piece = Piece::new(White, Queen);
    const WP: Piece = Piece::new(White, Pawn);
    const WA: Piece = Piece::new(White, Amazon);
    const WR: Piece = Piece::new(White, Rook);
    const BQ: Piece = Piece::new(Black, Queen);
    const BR: Piece = Piece::new(Black, Rook);
    const BP: Piece = Piece::new(Black, Pawn);
    const BA: Piece = Piece::new(Black, Amazon);
    const BB: Piece = Piece::new(Black, Bishop);

    fn generate_literally_all_boards<const W: usize, const H: usize>(
        pieces: &[Piece],
    ) -> GenerateAllBoards<W, H, false> {
        GenerateAllBoards::new(pieces)
    }

    fn generate_tablebase_no_opt<const W: usize, const H: usize>(
        piece_sets: &[PieceSet],
    ) -> Tablebase<W, H> {
        let mut tablebase = Tablebase::default();
        let mut all = Vec::new();
        let piece_sets = calculate_piece_sets(piece_sets);
        for s in &piece_sets.piece_sets {
            all.extend(generate_literally_all_boards(s));
        }
        for b in &all {
            populate_initial_wins_one(&mut tablebase, b);
        }
        let mut player = Black;
        let mut i = 1;
        loop {
            let mut changed = false;
            let mut black_out = Tablebase::default();
            let mut ignore = BoardsToVisit::default();
            let mut ignore2 = VisitedTy::default();
            for b in &all {
                visit_board(
                    &tablebase,
                    &mut black_out,
                    &mut ignore,
                    &mut ignore2,
                    player,
                    b,
                    None,
                    i,
                );
                changed |= !ignore.boards.is_empty();
                ignore.boards.clear();
            }
            if !changed {
                break;
            }
            tablebase.merge(black_out);
            player = player.next();
            i += 1;
        }
        tablebase
    }

    #[test]
    fn test_flip_coord() {
        assert_eq!(
            flip_coord::<4, 4>(
                Coord::new(1, 2),
                Symmetry {
                    flip_x: false,
                    flip_y: false,
                    flip_diagonally: false
                },
            ),
            Coord::new(1, 2)
        );
        assert_eq!(
            flip_coord::<4, 4>(
                Coord::new(1, 2),
                Symmetry {
                    flip_x: true,
                    flip_y: false,
                    flip_diagonally: false
                },
            ),
            Coord::new(2, 2)
        );
        assert_eq!(
            flip_coord::<4, 4>(
                Coord::new(1, 2),
                Symmetry {
                    flip_x: false,
                    flip_y: true,
                    flip_diagonally: false
                },
            ),
            Coord::new(1, 1)
        );
        assert_eq!(
            flip_coord::<4, 4>(
                Coord::new(1, 2),
                Symmetry {
                    flip_x: false,
                    flip_y: false,
                    flip_diagonally: true
                },
            ),
            Coord::new(2, 1)
        );
        assert_eq!(
            flip_coord::<4, 4>(
                Coord::new(0, 2),
                Symmetry {
                    flip_x: true,
                    flip_y: true,
                    flip_diagonally: true
                },
            ),
            Coord::new(1, 3)
        );
    }
    #[test]
    fn test_flip_move() {
        assert_eq!(
            flip_move::<4, 4>(
                Move {
                    from: Coord::new(1, 0),
                    to: Coord::new(3, 2)
                },
                Symmetry {
                    flip_x: false,
                    flip_y: true,
                    flip_diagonally: false
                },
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
            (Coord::new(0, 0), WK),
            (Coord::new(1, 0), WQ),
            (Coord::new(2, 0), BK),
        ]);
        let m = Move {
            from: Coord::new(1, 0),
            to: Coord::new(2, 0),
        };
        let mut tablebase = Tablebase::default();
        tablebase.add_impl(White, &board, m, 1);
        assert_eq!(tablebase.result(White, &board), Some((m, 1)));
    }
    #[test]
    fn test_canonical_board() {
        let board1 =
            TBBoard::<8, 8>::with_pieces(&[(Coord::new(0, 0), WK), (Coord::new(0, 1), BK)]);
        let board2 =
            TBBoard::<8, 8>::with_pieces(&[(Coord::new(0, 0), WK), (Coord::new(0, 2), BK)]);
        let board3 =
            TBBoard::<8, 8>::with_pieces(&[(Coord::new(0, 2), BK), (Coord::new(0, 0), WK)]);

        fn assert_canon_eq<const W: usize, const H: usize>(b1: &TBBoard<W, H>, b2: &TBBoard<W, H>) {
            assert_eq!(canonical_board(b1).0, canonical_board(b2).0);
        }
        fn assert_canon_ne<const W: usize, const H: usize>(b1: &TBBoard<W, H>, b2: &TBBoard<W, H>) {
            assert_ne!(canonical_board(b1).0, canonical_board(b2).0);
        }
        assert_canon_eq(&board1, &board1);
        assert_canon_eq(&board2, &board2);
        assert_canon_ne(&board1, &board2);
        assert_canon_eq(&board2, &board3);
        assert_canon_eq(
            &board1,
            &TBBoard::<8, 8>::with_pieces(&[(Coord::new(0, 0), WK), (Coord::new(1, 0), BK)]),
        );
        assert_canon_eq(
            &board1,
            &TBBoard::<8, 8>::with_pieces(&[(Coord::new(7, 7), WK), (Coord::new(7, 6), BK)]),
        );
        assert_canon_eq(
            &board1,
            &TBBoard::<8, 8>::with_pieces(&[(Coord::new(7, 7), WK), (Coord::new(6, 7), BK)]),
        );
        assert_canon_eq(
            &TBBoard::<5, 5>::with_pieces(&[(Coord::new(3, 2), WK), (Coord::new(2, 2), BK)]),
            &TBBoard::<5, 5>::with_pieces(&[(Coord::new(1, 2), WK), (Coord::new(2, 2), BK)]),
        );
        assert_canon_eq(
            &TBBoard::<5, 5>::with_pieces(&[(Coord::new(3, 2), WK), (Coord::new(2, 2), BK)]),
            &TBBoard::<5, 5>::with_pieces(&[(Coord::new(2, 1), WK), (Coord::new(2, 2), BK)]),
        );
        assert_canon_eq(
            &TBBoard::<5, 5>::with_pieces(&[(Coord::new(3, 2), WK), (Coord::new(2, 2), BK)]),
            &TBBoard::<5, 5>::with_pieces(&[(Coord::new(2, 3), WK), (Coord::new(2, 2), BK)]),
        );
        assert_canon_eq(
            &TBBoard::<5, 5>::with_pieces(&[(Coord::new(3, 3), WK), (Coord::new(2, 2), BK)]),
            &TBBoard::<5, 5>::with_pieces(&[(Coord::new(3, 1), WK), (Coord::new(2, 2), BK)]),
        );
        assert_canon_eq(
            &TBBoard::<5, 5>::with_pieces(&[(Coord::new(3, 3), WK), (Coord::new(2, 2), BK)]),
            &TBBoard::<5, 5>::with_pieces(&[(Coord::new(1, 1), WK), (Coord::new(2, 2), BK)]),
        );
        assert_canon_eq(
            &TBBoard::<5, 5>::with_pieces(&[(Coord::new(3, 3), WK), (Coord::new(2, 2), BK)]),
            &TBBoard::<5, 5>::with_pieces(&[(Coord::new(1, 3), WK), (Coord::new(2, 2), BK)]),
        );
        assert_canon_ne(
            &TBBoard::<5, 5>::with_pieces(&[(Coord::new(3, 3), WK), (Coord::new(2, 2), BK)]),
            &TBBoard::<5, 5>::with_pieces(&[(Coord::new(1, 2), WK), (Coord::new(2, 2), BK)]),
        );
        assert_canon_eq(
            &TBBoard::<8, 8>::with_pieces(&[
                (Coord::new(1, 0), WK),
                (Coord::new(1, 1), WP),
                (Coord::new(1, 7), BK),
                (Coord::new(1, 6), BP),
            ]),
            &TBBoard::<8, 8>::with_pieces(&[
                (Coord::new(6, 0), WK),
                (Coord::new(6, 1), WP),
                (Coord::new(6, 7), BK),
                (Coord::new(6, 6), BP),
            ]),
        );
        assert_canon_ne(
            &TBBoard::<8, 8>::with_pieces(&[
                (Coord::new(1, 0), WK),
                (Coord::new(1, 1), WP),
                (Coord::new(1, 7), BK),
                (Coord::new(1, 6), BP),
            ]),
            &TBBoard::<8, 8>::with_pieces(&[
                (Coord::new(1, 7), WK),
                (Coord::new(1, 6), WP),
                (Coord::new(1, 0), BK),
                (Coord::new(1, 1), BP),
            ]),
        );
    }
    #[test]
    fn test_calculate_piece_sets() {
        let kk = PieceSet::new(&[WK, BK]);
        let kqk = PieceSet::new(&[WK, WQ, BK]);
        let kkq = PieceSet::new(&[WK, BQ, BK]);
        let kpk = PieceSet::new(&[WK, WP, BK]);
        let kqkr = PieceSet::new(&[WK, WQ, BK, BR]);
        let kpkr = PieceSet::new(&[WK, WP, BK, BR]);
        let kkr = PieceSet::new(&[WK, BK, BR]);
        let bk = PieceSet::new(&[WB, BK]);
        let bbk = PieceSet::new(&[WB, WB, BK]);
        fn assert_sets_equal(s1: PieceSets, s2: &[PieceSet]) {
            assert_eq!(
                s1.piece_sets.into_iter().collect::<FxHashSet<_>>(),
                s2.iter().cloned().collect()
            );
        }
        {
            let s = calculate_piece_sets(&[kk.clone()]);
            assert_sets_equal(s, &[kk.clone()]);
        }
        {
            let s = calculate_piece_sets(&[kqk.clone()]);
            assert_sets_equal(s, &[kk.clone(), kqk.clone()]);
        }
        {
            let s = calculate_piece_sets(&[kkq.clone()]);
            assert_sets_equal(s, &[kk.clone(), kkq.clone()]);
        }
        {
            let s = calculate_piece_sets(&[kkq.clone(), kqk.clone()]);
            assert_sets_equal(s, &[kk.clone(), kkq.clone(), kqk.clone()]);
        }
        {
            let s = calculate_piece_sets(&[kpk.clone()]);
            assert_sets_equal(s, &[kk.clone(), kpk.clone(), kqk.clone()]);
        }
        {
            let s = calculate_piece_sets(&[bk.clone()]);
            assert_sets_equal(s, &[bk.clone()]);
        }
        {
            let s = calculate_piece_sets(&[bbk.clone()]);
            assert_sets_equal(s, &[bk.clone(), bbk.clone()]);
        }
        {
            let s = calculate_piece_sets(&[kpkr.clone()]);
            assert_sets_equal(
                s,
                &[
                    kk.clone(),
                    kpk.clone(),
                    kqk.clone(),
                    kk.clone(),
                    kkr.clone(),
                    kpkr.clone(),
                    kqkr.clone(),
                ],
            );
        }
    }
    #[test]
    fn test_generate_all_boards() {
        {
            let boards = generate_all_boards::<8, 8>(&PieceSet::new(&[WK]));
            assert_eq!(boards.count(), 10);
        }
        {
            let boards = generate_all_boards::<8, 8>(&PieceSet::new(&[WP]));
            assert_eq!(boards.count(), 24);
        }
        {
            let mut boards = generate_all_boards::<8, 8>(&PieceSet::new(&[WK, WQ]));
            let b0 = boards.next().unwrap();
            assert_eq!(b0.get(Coord::new(0, 0)), Some(WK));
            assert_eq!(b0.get(Coord::new(1, 0)), Some(WQ));
            assert_eq!(b0.get(Coord::new(2, 0)), None);

            let b1 = boards.next().unwrap();
            assert_eq!(b1.get(Coord::new(0, 0)), Some(WK));
            assert_eq!(b1.get(Coord::new(1, 0)), None);
            assert_eq!(b1.get(Coord::new(2, 0)), Some(WQ));
        }
        {
            let boards = generate_all_boards::<8, 8>(&PieceSet::new(&[WK, WQ]));
            assert_eq!(boards.count(), 10 * 63);
        }
        {
            let boards = generate_all_boards::<8, 8>(&PieceSet::new(&[WP, WQ]));
            assert_eq!(boards.count(), 24 * 63);
        }
        {
            for b in generate_all_boards::<4, 4>(&PieceSet::new(&[WK, WP, WP])) {
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

        for b in generate_all_boards::<4, 4>(&PieceSet::new(&[BK])) {
            let c = b.king_coord(Black);
            assert!(c.x <= 1);
            assert!(c.y <= 1);
            assert!(c.x <= c.y);
        }
        for b in generate_all_boards::<5, 5>(&PieceSet::new(&[BK])) {
            let c = b.king_coord(Black);
            assert!(c.x <= 2);
            assert!(c.y <= 2);
            assert!(c.x <= c.y);
        }
        assert_eq!(
            generate_all_boards::<4, 4>(&PieceSet::new(&[BK])).count(),
            3
        );
        assert_eq!(
            generate_all_boards::<5, 5>(&PieceSet::new(&[BK])).count(),
            6
        );
        assert_eq!(
            generate_all_boards::<5, 4>(&PieceSet::new(&[BK])).count(),
            6
        );
        generate_all_boards::<4, 4>(&PieceSet::new(&[BR, BA, BQ, BK]));
    }
    #[test]
    fn test_generate_literally_all_boards() {
        let kk = PieceSet::new(&[BK, WK]);
        let krk = PieceSet::new(&[WR, BK, WK]);
        let kpk = PieceSet::new(&[WP, BK, WK]);
        assert_eq!(generate_literally_all_boards::<3, 3>(&kk).count(), 9 * 8);
        assert_eq!(
            generate_literally_all_boards::<3, 3>(&krk).count(),
            9 * 8 * 7
        );
        assert_eq!(
            generate_literally_all_boards::<3, 3>(&kpk).count(),
            3 * 8 * 7
        );
    }

    #[test]
    #[should_panic(expected = "must be exactly one black king in PieceSet")]
    fn test_generate_tablebase_panic_no_bk() {
        let set = PieceSet::new(&[WK, WA, BQ]);
        generate_tablebase::<6, 6>(&[set]);
    }

    #[test]
    #[should_panic]
    fn test_piece_set_too_large() {
        PieceSet::new(&[WK, WA, BQ, BK, BP, WP]);
    }

    fn test_tablebase<const W: usize, const H: usize>(sets: &[PieceSet]) -> Tablebase<W, H> {
        let tablebase1 = generate_tablebase(sets);
        let tablebase2 = generate_tablebase_no_opt(sets);
        verify_tablebases_equal(&tablebase1, &tablebase2, sets);
        tablebase1
    }

    fn test_tablebase_parallel<const W: usize, const H: usize>(
        sets: &[PieceSet],
    ) -> Tablebase<W, H> {
        let tablebase1 = generate_tablebase(sets);
        let tablebase2 = generate_tablebase_parallel(sets, Some(2));
        verify_tablebases_equal(&tablebase1, &tablebase2, sets);
        tablebase1
    }

    #[test]
    fn test_generate_tablebase_parallel() {
        let kak = PieceSet::new(&[WK, WA, BK]);
        let kka = PieceSet::new(&[WK, BK, BA]);
        let sets = [kak, kka];
        test_tablebase_parallel::<4, 4>(&sets);
    }

    fn verify_board_tablebase<const W: usize, const H: usize>(
        board: &TBBoard<W, H>,
        tablebase: &Tablebase<W, H>,
    ) {
        fn black_king_exists<const W: usize, const H: usize>(board: &TBBoard<W, H>) -> bool {
            board
                .piece_coord(|piece| piece.player() == Black && piece.ty() == King)
                .is_some()
        }

        let mut board = board.clone();
        if let Some((_, mut expected_depth)) = tablebase.result(White, &board) {
            let mut player = White;

            while expected_depth > 0 {
                assert!(black_king_exists(&board));
                let (m, depth) = tablebase.result(player, &board).unwrap();
                assert_eq!(depth, expected_depth);
                board.make_move(m);
                expected_depth -= 1;
                player = player.next();
            }
            assert!(!black_king_exists(&board));
        }
    }

    fn verify_tablebases_equal<const W: usize, const H: usize>(
        tb1: &Tablebase<W, H>,
        tb2: &Tablebase<W, H>,
        piece_sets: &[PieceSet],
    ) {
        for set in piece_sets {
            for b in generate_all_boards::<W, H>(set) {
                for player in [White, Black] {
                    let r1 = tb1.result(player, &b).map(|(_, d)| d);
                    let r2 = tb2.result(player, &b).map(|(_, d)| d);
                    if r1 != r2 {
                        println!("{:?}", &b);
                        println!("tb1: {:?}", tb1.result(player, &b));
                        println!("tb2: {:?}", tb2.result(player, &b));
                        panic!("result({:?}) mismatch", player);
                    }
                }
                verify_board_tablebase(&b, tb1);
            }
        }
    }

    fn verify_all_three_piece_positions_forced_win(ty: Type) -> Tablebase<4, 4> {
        let set = PieceSet::new(&[WK, BK, Piece::new(White, ty)]);

        let tablebase = test_tablebase(&[set.clone()]);

        for b in generate_all_boards::<4, 4>(&set) {
            let wd = tablebase.result(White, &b);
            let bd = tablebase.result(Black, &b);
            assert!(wd.unwrap().1 % 2 == 1);
            assert!(bd.is_none() || bd.unwrap().1 % 2 == 0);
        }

        tablebase
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
        let kk = PieceSet::new(&[WK, BK]);
        let tablebase = test_tablebase(&[kk]);

        assert_eq!(
            tablebase.result(
                White,
                &TBBoard::<5, 1>::with_pieces(&[(Coord::new(0, 0), WK), (Coord::new(3, 0), BK)])
            ),
            Some((
                Move {
                    from: Coord::new(0, 0),
                    to: Coord::new(1, 0)
                },
                5
            ))
        );
        assert_eq!(
            tablebase.result(
                Black,
                &TBBoard::<5, 1>::with_pieces(&[(Coord::new(1, 0), WK), (Coord::new(3, 0), BK)])
            ),
            Some((
                Move {
                    from: Coord::new(3, 0),
                    to: Coord::new(4, 0)
                },
                4
            ))
        );
        assert_eq!(
            tablebase.result(
                White,
                &TBBoard::<5, 1>::with_pieces(&[(Coord::new(1, 0), WK), (Coord::new(4, 0), BK)])
            ),
            Some((
                Move {
                    from: Coord::new(1, 0),
                    to: Coord::new(2, 0)
                },
                3
            ))
        );
        assert_eq!(
            tablebase.result(
                Black,
                &TBBoard::<5, 1>::with_pieces(&[(Coord::new(2, 0), WK), (Coord::new(4, 0), BK)])
            ),
            Some((
                Move {
                    from: Coord::new(4, 0),
                    to: Coord::new(3, 0)
                },
                2
            ))
        );
        assert_eq!(
            tablebase.result(
                White,
                &TBBoard::<5, 1>::with_pieces(&[(Coord::new(2, 0), WK), (Coord::new(3, 0), BK)])
            ),
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
        fn test<const W: usize, const H: usize>() {
            let pieces = PieceSet::new(&[WK, BK]);
            let tablebase = test_tablebase::<W, H>(&[pieces.clone()]);
            // If white king couldn't capture on first move, no forced win.
            for b in generate_all_boards(&pieces) {
                if is_under_attack(&b, b.king_coord(Black), Black) {
                    assert_eq!(tablebase.result(White, &b).unwrap().1, 1);
                } else {
                    assert_eq!(tablebase.result(White, &b), None);
                }
                assert_eq!(tablebase.result(Black, &b), None);
            }
        }
        test::<6, 6>();
        test::<5, 5>();
        test::<4, 5>();
        test::<4, 6>();
    }

    #[test]
    fn test_qk() {
        let set = PieceSet::new(&[WQ, BK]);
        test_tablebase::<4, 4>(&[set]);
    }

    #[test]
    fn test_qbk() {
        let set = PieceSet::new(&[WQ, WB, BK]);
        test_tablebase::<4, 4>(&[set]);
    }

    #[test]
    fn test_pbk() {
        let set = PieceSet::new(&[WP, WB, BK]);
        test_tablebase::<4, 4>(&[set]);
    }

    #[test]
    fn test_pawn_double_move() {
        let set = PieceSet::new(&[BK, WP, BP]);
        test_tablebase::<3, 7>(&[set]);
    }

    #[test]
    fn test_pawn_multiple_move() {
        let set = PieceSet::new(&[BK, WP]);
        test_tablebase::<3, 9>(&[set]);
    }

    #[test]
    fn test_kpk() {
        let set = PieceSet::new(&[WK, BK, WP]);
        test_tablebase::<4, 4>(&[set]);
    }

    #[test]
    fn test_kpk_rect() {
        let set = PieceSet::new(&[WK, BK, WP]);
        test_tablebase::<5, 4>(&[set.clone()]);
        test_tablebase::<4, 5>(&[set]);
    }

    #[test]
    fn test_kqkp() {
        let set = PieceSet::new(&[WK, BK, WQ, BP]);
        test_tablebase::<4, 4>(&[set]);
    }

    #[test]
    fn test_kpkp() {
        let set = PieceSet::new(&[WK, BK, WP, BP]);
        test_tablebase::<4, 4>(&[set]);
    }

    #[test]
    fn test_pkb() {
        let set = PieceSet::new(&[BK, WP, BB]);
        test_tablebase::<4, 4>(&[set]);
    }

    #[test]
    fn test_kqkq() {
        let set = PieceSet::new(&[WK, BK, WQ, BQ]);
        test_tablebase::<4, 4>(&[set]);
    }

    #[test]
    fn test_bqkb() {
        let set = PieceSet::new(&[WB, WQ, BK, BB]);
        test_tablebase::<4, 4>(&[set]);
    }

    #[test]
    fn test_kqkr() {
        let set = PieceSet::new(&[WK, BK, WQ, BR]);

        let tablebase = test_tablebase::<4, 4>(&[set]);

        // ..k.
        // ....
        // .K..
        // r..Q
        // Don't capture the rook, it's slower to checkmate overall.
        let res = tablebase.result(
            White,
            &TBBoard::with_pieces(&[
                (Coord::new(0, 0), BR),
                (Coord::new(3, 0), WQ),
                (Coord::new(1, 1), WK),
                (Coord::new(2, 3), BK),
            ]),
        );
        assert_ne!(res.unwrap().0.to, Coord::new(0, 0));
        assert_eq!(res.unwrap().1, 5);
    }

    #[test]
    fn test_kqkr_parallel() {
        let set = PieceSet::new(&[WK, BK, WQ, BR]);
        test_tablebase_parallel::<4, 4>(&[set]);
    }

    #[test]
    fn test_kpk_parallel() {
        let set = PieceSet::new(&[WK, BK, WP]);
        test_tablebase_parallel::<4, 4>(&[set]);
    }

    #[test]
    fn test_kpk_parallel_1() {
        generate_tablebase::<4, 4>(&[PieceSet::new(&[WK, BK, WP])]);
        generate_tablebase_parallel::<4, 4>(&[PieceSet::new(&[WK, BK, WP])], Some(1));
    }

    #[test]
    #[ignore]
    fn test_all_sets() {
        use derive_enum::EnumFrom;
        let mut all_pieces = Vec::new();
        for ty in Type::all() {
            for pl in Player::all() {
                let p = Piece::new(pl, ty);
                if p != Piece::new(Black, King) {
                    all_pieces.push(p);
                }
            }
        }
        for &p1 in &all_pieces {
            for &p2 in &all_pieces {
                let set = PieceSet::new(dbg!(&[p1, p2, BK]));
                if !set.iter().any(|&p| p.player() == White) {
                    continue;
                }
                test_tablebase::<4, 4>(&[set]);
            }
        }
    }
    #[test]
    fn test_result_for_real_board_win() {
        let tablebase = generate_tablebase::<4, 4>(&[PieceSet::new(&[WK, WQ, BK, BQ])]);
        let board = BoardSquare::<4, 4>::with_pieces(&[
            (Coord::new(0, 0), WK),
            (Coord::new(1, 2), WQ),
            (Coord::new(3, 3), BK),
            (Coord::new(2, 1), BQ),
        ]);
        let r = tablebase.result_for_real_board(White, &board);
        assert_eq!(
            r.0,
            Move {
                from: Coord::new(1, 2),
                to: Coord::new(2, 1)
            }
        );
        assert_eq!(r.2, TBMoveType::Win);

        let r = tablebase.result_for_real_board(Black, &board);
        assert_eq!(
            r.0,
            Move {
                from: Coord::new(2, 1),
                to: Coord::new(1, 2)
            }
        );
        assert_eq!(r.2, TBMoveType::Win);
    }
    #[test]
    fn test_result_for_real_board_lose() {
        let tablebase = generate_tablebase::<4, 4>(&[PieceSet::new(&[WK, WQ, BK, BQ])]);
        {
            let board = BoardSquare::<4, 4>::with_pieces(&[
                (Coord::new(0, 0), WK),
                (Coord::new(3, 3), BK),
                (Coord::new(2, 1), BQ),
            ]);
            let r = tablebase.result_for_real_board(White, &board);
            assert_eq!(r.0.from, Coord::new(0, 0));
            assert_eq!(r.2, TBMoveType::Lose);
        }
        {
            let board = BoardSquare::<4, 4>::with_pieces(&[
                (Coord::new(0, 0), WK),
                (Coord::new(1, 2), WQ),
                (Coord::new(3, 3), BK),
            ]);
            let r = tablebase.result_for_real_board(Black, &board);
            assert_eq!(r.0.from, Coord::new(3, 3));
            assert_eq!(r.2, TBMoveType::Lose);
        }
    }
    #[test]
    fn test_result_for_real_board_draw() {
        let tablebase = generate_tablebase::<5, 5>(&[PieceSet::new(&[WK, BK])]);
        {
            let board =
                BoardSquare::<5, 5>::with_pieces(&[(Coord::new(0, 2), WK), (Coord::new(2, 2), BK)]);
            let r = tablebase.result_for_real_board(White, &board);
            assert_eq!(r.0.from, Coord::new(0, 2));
            assert_eq!(r.0.to.x, 0);
            assert_eq!(r.2, TBMoveType::Draw);
        }
        {
            let board =
                BoardSquare::<5, 5>::with_pieces(&[(Coord::new(2, 2), WK), (Coord::new(4, 2), BK)]);
            let r = tablebase.result_for_real_board(Black, &board);
            assert_eq!(r.0.from, Coord::new(4, 2));
            assert_eq!(r.0.to.x, 4);
            assert_eq!(r.2, TBMoveType::Draw);
        }
        {
            let board =
                BoardSquare::<5, 5>::with_pieces(&[(Coord::new(2, 4), WK), (Coord::new(2, 2), BK)]);
            let r = tablebase.result_for_real_board(White, &board);
            assert_eq!(r.0.from, Coord::new(2, 4));
            assert_eq!(r.0.to.y, 4);
            assert_eq!(r.2, TBMoveType::Draw);
        }
        {
            let board =
                BoardSquare::<5, 5>::with_pieces(&[(Coord::new(2, 2), WK), (Coord::new(2, 0), BK)]);
            let r = tablebase.result_for_real_board(Black, &board);
            assert_eq!(r.0.from, Coord::new(2, 0));
            assert_eq!(r.0.to.y, 0);
            assert_eq!(r.2, TBMoveType::Draw);
        }
    }
    #[test]
    fn test_serde() {
        let tb = generate_tablebase::<4, 4>(&[PieceSet::new(&[WK, BK])]);
        let buf = tb.serialize();
        let tb2 = Tablebase::<4, 4>::deserialize(&buf).unwrap();
        assert_eq!(tb.white_tablebase, tb2.white_tablebase);
        assert_eq!(tb.black_tablebase, tb2.black_tablebase);
        assert!(Tablebase::<4, 8>::deserialize(&buf).is_none());
        assert!(Tablebase::<4, 4>::deserialize(&[]).is_none());
    }
    #[test]
    fn test_insertion_sort() {
        let mut arr = [1, 3, 5, 3, 2];
        insertion_sort(&mut arr, |a, b| a < b);
        assert_eq!(arr, [1, 2, 3, 3, 5]);

        let mut arr = [5, 4, 3, 2, 1];
        insertion_sort(&mut arr, |a, b| a < b);
        assert_eq!(arr, [1, 2, 3, 4, 5]);

        let mut arr = [1, 2, 3, 4, 5];
        insertion_sort(&mut arr, |a, b| a < b);
        assert_eq!(arr, [1, 2, 3, 4, 5]);
    }
}
