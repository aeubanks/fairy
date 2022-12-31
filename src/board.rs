use crate::coord::Coord;
use crate::piece::{Piece, Type, Type::*};
use crate::player::{Player, Player::*};
use arrayvec::ArrayVec;
use derive_enum::EnumCount;
use rand::Rng;
use static_assertions::const_assert_eq;
use std::fmt::Debug;

#[derive(PartialEq, Eq, Debug)]
pub enum ExistingPieceResult {
    Empty,
    Friend,
    Opponent,
}

pub trait Board: Default + Debug + Clone {
    fn width(&self) -> i8;
    fn height(&self) -> i8;
    fn in_bounds(&self, coord: Coord) -> bool {
        coord.x < self.width() as i8
            && coord.x >= 0
            && coord.y < self.height() as i8
            && coord.y >= 0
    }
    fn get(&self, coord: Coord) -> Option<Piece>;
    fn clear(&mut self, coord: Coord);
    fn add_piece(&mut self, coord: Coord, piece: Piece);
    fn take(&mut self, coord: Coord) -> Option<Piece>;

    fn existing_piece_result(&self, coord: Coord, player: Player) -> ExistingPieceResult {
        use ExistingPieceResult::*;
        match &self.get(coord) {
            None => Empty,
            Some(other_piece) => {
                if other_piece.player() == player {
                    Friend
                } else {
                    Opponent
                }
            }
        }
    }

    fn swap(&mut self, c1: Coord, c2: Coord) {
        assert_ne!(c1, c2);
        let p1 = self.take(c1);
        let p2 = self.take(c2);
        if let Some(p1) = p1 {
            self.add_piece(c2, p1);
        }
        if let Some(p2) = p2 {
            self.add_piece(c1, p2);
        }
    }

    fn set_last_pawn_double_move(&mut self, c: Option<Coord>);
    fn get_last_pawn_double_move(&self) -> Option<Coord>;
    fn set_castling_rights(&mut self, m: Move, piece: Piece);
    // TODO: make side into enum
    fn get_castling_rights(&self, player: Player, side: bool) -> Option<Coord>;

    fn make_move(&mut self, m: Move, player: Player) {
        assert_ne!(m.from, m.to);
        assert!(self.existing_piece_result(m.from, player) == ExistingPieceResult::Friend);
        let to_res = self.existing_piece_result(m.to, player);
        let mut piece = self.take(m.from).unwrap();
        // pawn double moves
        if piece.ty() == Pawn && (m.from.y - m.to.y).abs() == 2 {
            self.set_last_pawn_double_move(Some(m.to));
        } else {
            self.set_last_pawn_double_move(None);
        }
        // en passant
        if piece.ty() == Pawn && m.from.x != m.to.x && self.get(m.to).is_none() {
            let opponent_pawn_coord = Coord::new(m.to.x, m.from.y);
            assert!(
                self.existing_piece_result(opponent_pawn_coord, player)
                    == ExistingPieceResult::Opponent
            );
            assert!(self.get(opponent_pawn_coord).unwrap().ty() == Pawn);
            self.clear(opponent_pawn_coord);
        }
        // promotion
        if piece.ty() == Pawn && (m.to.y == 0 || m.to.y == self.height() - 1) {
            // TODO: support more than promoting to queen
            piece = Piece::new(piece.player(), Queen);
        }
        // keep track of castling rights
        self.set_castling_rights(m, piece);
        // castling
        if piece.ty() == King && to_res == ExistingPieceResult::Friend {
            let rook = self.take(m.to).unwrap();
            // king moves to rook to castle with
            // king should always be between two rooks to castle with
            let (dest, rook_dest) = if m.from.x > m.to.x {
                (Coord::new(2, m.from.y), Coord::new(3, m.from.y))
            } else {
                (
                    Coord::new(self.width() - 2, m.from.y),
                    Coord::new(self.width() - 3, m.from.y),
                )
            };
            self.add_piece(rook_dest, rook);
            self.add_piece(dest, piece);
        } else {
            assert!(to_res != ExistingPieceResult::Friend);
            self.clear(m.to);
            self.add_piece(m.to, piece);
        }
    }

    fn foreach_piece<F>(&self, f: F)
    where
        F: FnMut(Piece, Coord);

    fn piece_coord<F>(&self, f: F) -> Option<Coord>
    where
        F: FnMut(Piece) -> bool;

    // TODO: return bit vector?
    fn piece_types_of_player(&self, _player: Player) -> Option<[bool; Type::COUNT]> {
        None
    }

    fn king_coord(&self, player: Player) -> Coord {
        let ret = self
            .piece_coord(|piece| piece.player() == player && piece.ty() == King)
            .unwrap();
        debug_assert_eq!(self.get(ret).unwrap().ty(), King);
        ret
    }
    fn with_pieces(pieces: &[(Coord, Piece)]) -> Self {
        let mut board = Self::default();
        for (c, p) in pieces {
            board.add_piece(*c, *p);
        }
        board
    }
    fn to_str(&self) -> String {
        let mut ret = String::new();
        for y in (0..self.height()).rev() {
            for x in 0..self.width() {
                let c = match self.get(Coord::new(x, y)) {
                    None => '.',
                    Some(p) => {
                        let c = p.ty().char();
                        if p.player() == White {
                            c
                        } else {
                            c.to_lowercase().next().unwrap()
                        }
                    }
                };
                ret.push(c);
            }
            ret.push('\n');
        }
        ret
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct BoardSquare<const W: usize, const H: usize> {
    pieces: [[Option<Piece>; H]; W],
    pub castling_rights: [Option<Coord>; 4],
    pub last_pawn_double_move: Option<Coord>,
}

const_assert_eq!(79, std::mem::size_of::<BoardSquare<8, 8>>());

// This really should be derivable...
impl<const W: usize, const H: usize> Default for BoardSquare<W, H> {
    fn default() -> Self {
        assert!(W > 0);
        assert!(W < i8::MAX as usize);
        assert!(H > 0);
        assert!(H < i8::MAX as usize);
        Self {
            pieces: [[None; H]; W],
            castling_rights: [None; 4],
            last_pawn_double_move: None,
        }
    }
}

impl<const W: usize, const H: usize> BoardSquare<W, H> {
    fn get_mut(&mut self, coord: Coord) -> &mut Option<Piece> {
        &mut self.pieces[coord.x as usize][coord.y as usize]
    }
}

impl<const W: usize, const H: usize> Board for BoardSquare<W, H> {
    fn width(&self) -> i8 {
        W as i8
    }
    fn height(&self) -> i8 {
        H as i8
    }

    fn get(&self, coord: Coord) -> Option<Piece> {
        self.pieces[coord.x as usize][coord.y as usize]
    }

    fn clear(&mut self, coord: Coord) {
        *self.get_mut(coord) = None;
    }

    fn add_piece(&mut self, coord: Coord, piece: Piece) {
        assert!(self.get(coord).is_none());

        *self.get_mut(coord) = Some(piece);
    }

    fn take(&mut self, coord: Coord) -> Option<Piece> {
        self.get_mut(coord).take()
    }

    fn existing_piece_result(&self, coord: Coord, player: Player) -> ExistingPieceResult {
        use ExistingPieceResult::*;
        match &self.get(coord) {
            None => Empty,
            Some(other_piece) => {
                if other_piece.player() == player {
                    Friend
                } else {
                    Opponent
                }
            }
        }
    }

    fn swap(&mut self, c1: Coord, c2: Coord) {
        assert_ne!(c1, c2);
        let p1 = self.take(c1);
        let p2 = self.take(c2);
        if let Some(p1) = p1 {
            self.add_piece(c2, p1);
        }
        if let Some(p2) = p2 {
            self.add_piece(c1, p2);
        }
    }

    fn foreach_piece<F>(&self, mut f: F)
    where
        F: FnMut(Piece, Coord),
    {
        for (x, ps) in self.pieces.iter().enumerate() {
            for (y, p) in ps.iter().enumerate() {
                if let Some(p) = p {
                    f(*p, Coord::new(x as i8, y as i8));
                }
            }
        }
    }

    fn piece_coord<F>(&self, mut f: F) -> Option<Coord>
    where
        F: FnMut(Piece) -> bool,
    {
        for (x, ps) in self.pieces.iter().enumerate() {
            for (y, p) in ps.iter().enumerate() {
                if let Some(p) = p {
                    if f(*p) {
                        return Some(Coord::new(x as i8, y as i8));
                    }
                }
            }
        }
        None
    }
    fn set_last_pawn_double_move(&mut self, c: Option<Coord>) {
        self.last_pawn_double_move = c;
    }
    fn get_last_pawn_double_move(&self) -> Option<Coord> {
        self.last_pawn_double_move
    }
    fn set_castling_rights(&mut self, m: Move, piece: Piece) {
        // keep track of castling rights
        for cr in self.castling_rights.as_mut() {
            if let Some(c) = cr {
                if *c == m.to || *c == m.from {
                    *cr = None;
                }
            }
        }
        if piece.ty() == King {
            match piece.player() {
                White => {
                    self.castling_rights[0] = None;
                    self.castling_rights[1] = None;
                }
                Black => {
                    self.castling_rights[2] = None;
                    self.castling_rights[3] = None;
                }
            }
        }
    }
    fn get_castling_rights(&self, player: Player, side: bool) -> Option<Coord> {
        self.castling_rights[match (player, side) {
            (White, false) => 0,
            (White, true) => 1,
            (Black, false) => 2,
            (Black, true) => 3,
        }]
    }
}

impl<const W: usize, const H: usize> Debug for BoardSquare<W, H> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_str())
    }
}

#[derive(Clone, Default)]
pub struct BoardPiece<const W: i8, const H: i8, const N: usize>(ArrayVec<(Coord, Piece), N>);

const_assert_eq!(16, std::mem::size_of::<BoardPiece<12, 13, 4>>());

impl<const W: i8, const H: i8, const N: usize> Board for BoardPiece<W, H, N> {
    fn width(&self) -> i8 {
        W
    }
    fn height(&self) -> i8 {
        H
    }
    fn get(&self, coord: Coord) -> Option<Piece> {
        assert!(self.in_bounds(coord));
        for (c, p) in &self.0 {
            if *c == coord {
                return Some(*p);
            }
        }
        None
    }
    fn clear(&mut self, coord: Coord) {
        assert!(self.in_bounds(coord));
        if let Some(idx) = self.0.iter().position(|(c, _)| *c == coord) {
            self.0.swap_remove(idx);
        }
    }
    fn take(&mut self, coord: Coord) -> Option<Piece> {
        assert!(self.in_bounds(coord));
        if let Some(idx) = self.0.iter().position(|(c, _)| *c == coord) {
            Some(self.0.swap_remove(idx).1)
        } else {
            None
        }
    }
    fn set_last_pawn_double_move(&mut self, _: Option<Coord>) {
        // not implemented
    }
    fn get_last_pawn_double_move(&self) -> Option<Coord> {
        // not implemented
        None
    }
    fn set_castling_rights(&mut self, _: Move, _: Piece) {
        // not implemented
    }
    fn get_castling_rights(&self, _: Player, _: bool) -> Option<Coord> {
        // not implemented
        None
    }
    fn add_piece(&mut self, coord: Coord, piece: Piece) {
        assert!(self.in_bounds(coord));
        debug_assert!(!self.0.iter().any(|(c, _)| *c == coord));
        self.0.push((coord, piece));
    }
    fn foreach_piece<F>(&self, mut f: F)
    where
        F: FnMut(Piece, Coord),
    {
        for (c, p) in &self.0 {
            f(*p, *c);
        }
    }
    fn piece_coord<F>(&self, mut f: F) -> Option<Coord>
    where
        F: FnMut(Piece) -> bool,
    {
        for (c, p) in &self.0 {
            if f(*p) {
                return Some(*c);
            }
        }
        None
    }
    fn piece_types_of_player(&self, player: Player) -> Option<[bool; Type::COUNT]> {
        let mut ret = [false; Type::COUNT];
        for (_, p) in &self.0 {
            if p.player() == player {
                ret[p.ty() as usize] = true;
            }
        }
        Some(ret)
    }
}

impl<const W: i8, const H: i8, const N: usize> Debug for BoardPiece<W, H, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_str())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Move {
    pub from: Coord,
    pub to: Coord,
}

#[allow(dead_code)]
pub struct Presets;

#[allow(dead_code)]
impl Presets {
    pub fn los_alamos() -> BoardSquare<6, 6> {
        BoardSquare::<6, 6>::setup_with_pawns(false, &[Rook, Knight, Queen, King, Knight, Rook])
    }

    pub fn classical() -> BoardSquare<8, 8> {
        BoardSquare::<8, 8>::setup_with_pawns(
            true,
            &[Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook],
        )
    }

    pub fn embassy() -> BoardSquare<10, 8> {
        BoardSquare::<10, 8>::setup_with_pawns(
            true,
            &[
                Rook, Knight, Bishop, Queen, King, Empress, Cardinal, Bishop, Knight, Rook,
            ],
        )
    }

    fn set_nth_empty(mut n: usize, pieces: &mut [Option<Type>], ty: Type) {
        let mut i = 0;
        loop {
            while pieces[i].is_some() {
                i += 1;
            }
            if n == 0 {
                break;
            } else {
                n -= 1;
            }
        }
        pieces[i] = Some(ty);
    }

    pub fn chess960<R: Rng + ?Sized>(rng: &mut R) -> BoardSquare<8, 8> {
        let mut pieces: [Option<Type>; 8] = [None; 8];
        Self::set_nth_empty(rng.gen_range(0..4) * 2, &mut pieces, Bishop);
        Self::set_nth_empty(rng.gen_range(0..4) * 2 + 1, &mut pieces, Bishop);
        Self::set_nth_empty(rng.gen_range(0..6), &mut pieces, Queen);
        Self::set_nth_empty(rng.gen_range(0..5), &mut pieces, Knight);
        Self::set_nth_empty(rng.gen_range(0..4), &mut pieces, Knight);
        Self::set_nth_empty(0, &mut pieces, Rook);
        Self::set_nth_empty(0, &mut pieces, King);
        Self::set_nth_empty(0, &mut pieces, Rook);

        BoardSquare::<8, 8>::setup_with_pawns(true, &pieces.map(|p| p.unwrap()))
    }

    pub fn capablanca_random<R: Rng + ?Sized>(rng: &mut R) -> BoardSquare<10, 8> {
        let mut evens: [Option<Type>; 5] = [None; 5];
        let mut odds: [Option<Type>; 5] = [None; 5];
        evens[rng.gen_range(0..5)] = Some(Bishop);
        odds[rng.gen_range(0..5)] = Some(Bishop);
        let (qa1, qa2) = if rng.gen() {
            (Queen, Cardinal)
        } else {
            (Cardinal, Queen)
        };
        Self::set_nth_empty(rng.gen_range(0..4), &mut evens, qa1);
        Self::set_nth_empty(rng.gen_range(0..4), &mut odds, qa2);

        let mut pieces: [Option<Type>; 10] = [None; 10];
        for (i, t) in evens.into_iter().enumerate() {
            pieces[i * 2] = t;
        }
        for (i, t) in odds.into_iter().enumerate() {
            pieces[i * 2 + 1] = t;
        }

        Self::set_nth_empty(rng.gen_range(0..6), &mut pieces, Empress);
        Self::set_nth_empty(rng.gen_range(0..5), &mut pieces, Knight);
        Self::set_nth_empty(rng.gen_range(0..4), &mut pieces, Knight);
        Self::set_nth_empty(0, &mut pieces, Rook);
        Self::set_nth_empty(0, &mut pieces, King);
        Self::set_nth_empty(0, &mut pieces, Rook);

        BoardSquare::<10, 8>::setup_with_pawns(true, &pieces.map(|p| p.unwrap()))
    }
}

impl<const W: usize, const H: usize> BoardSquare<W, H> {
    fn setup_with_pawns(castling: bool, pieces: &[Type]) -> Self {
        let mut board = Self::default();
        for i in 0..W as i8 {
            board.add_piece(Coord::new(i, 1), Piece::new(White, Pawn));
            board.add_piece(Coord::new(i, H as i8 - 2), Piece::new(Black, Pawn));
        }
        assert!(pieces.len() == W);
        for (i, ty) in pieces.iter().enumerate() {
            let white_coord = Coord::new(i as i8, 0);
            board.add_piece(white_coord, Piece::new(White, *ty));
            let black_coord = Coord::new(i as i8, H as i8 - 1);
            board.add_piece(black_coord, Piece::new(Black, *ty));
            if castling {
                if board.castling_rights[0].is_none() {
                    board.castling_rights[0] = Some(white_coord);
                    board.castling_rights[2] = Some(black_coord);
                } else {
                    board.castling_rights[1] = Some(white_coord);
                    board.castling_rights[3] = Some(black_coord);
                }
            }
        }
        if castling {
            assert!(board.castling_rights.iter().all(|cr| cr.is_some()));
        }
        board
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_board_swap() {
        fn test<T: Board>() {
            let n = Piece::new(White, Knight);
            let b = Piece::new(White, Bishop);
            let mut board = T::with_pieces(&[(Coord::new(0, 0), n), (Coord::new(1, 0), b)]);

            assert_eq!(board.get(Coord::new(0, 0)), Some(n));
            assert_eq!(board.get(Coord::new(1, 0)), Some(b));
            assert_eq!(board.get(Coord::new(2, 0)), None);

            board.swap(Coord::new(1, 0), Coord::new(0, 0));
            assert_eq!(board.get(Coord::new(0, 0)), Some(b));
            assert_eq!(board.get(Coord::new(1, 0)), Some(n));
            assert_eq!(board.get(Coord::new(2, 0)), None);

            board.swap(Coord::new(2, 0), Coord::new(0, 0));
            assert_eq!(board.get(Coord::new(0, 0)), None);
            assert_eq!(board.get(Coord::new(1, 0)), Some(n));
            assert_eq!(board.get(Coord::new(2, 0)), Some(b));
        }
        test::<BoardSquare<3, 1>>();
        test::<BoardPiece<3, 1, 4>>();
    }
    #[test]
    fn test_board() {
        fn test<T: Board>() {
            let mut b = T::default();
            let p1 = Piece::new(White, Bishop);
            let p2 = Piece::new(Black, Knight);
            b.add_piece(Coord::new(0, 0), p1);
            b.add_piece(Coord::new(3, 3), p2);
            assert_eq!(b.get(Coord::new(0, 0)), Some(p1));
            assert_eq!(b.get(Coord::new(3, 3)), Some(p2));
            assert_eq!(b.get(Coord::new(0, 3)), None);
            b.clear(Coord::new(0, 0));
            assert_eq!(b.get(Coord::new(0, 0)), None);
        }
        test::<BoardSquare<4, 4>>();
        test::<BoardPiece<4, 4, 4>>();
    }

    #[test]
    #[should_panic]
    fn test_board_square_panic_x_1() {
        let b = BoardSquare::<2, 3>::default();
        let _ = b.get(Coord::new(2, 1));
    }

    #[test]
    #[should_panic]
    fn test_board_square_panic_x_2() {
        let b = BoardSquare::<3, 1>::default();
        let _ = b.get(Coord::new(-1, 1));
    }

    #[test]
    #[should_panic]
    fn test_board_square_panic_y_1() {
        let b = BoardSquare::<2, 3>::default();
        let _ = b.get(Coord::new(1, 3));
    }

    #[test]
    #[should_panic]
    fn test_board_square_panic_y_2() {
        let b = BoardSquare::<2, 3>::default();
        let _ = b.get(Coord::new(1, -1));
    }

    #[test]
    #[should_panic]
    fn test_mut_board_square_panic_x_1() {
        let mut b = BoardSquare::<2, 3>::default();
        b.clear(Coord::new(2, 1));
    }

    #[test]
    #[should_panic]
    fn test_mut_board_square_panic_x_2() {
        let mut b = BoardSquare::<2, 3>::default();
        b.clear(Coord::new(-1, 1));
    }

    #[test]
    #[should_panic]
    fn test_mut_board_square_panic_y_1() {
        let mut b = BoardSquare::<2, 3>::default();
        b.clear(Coord::new(1, 3));
    }

    #[test]
    #[should_panic]
    fn test_mut_board_square_panic_y_2() {
        let mut b = BoardSquare::<2, 3>::default();
        b.clear(Coord::new(1, -1));
    }

    #[test]
    #[should_panic]
    fn test_board_square_set_existing_piece_panic() {
        let mut b = BoardSquare::<2, 3>::default();
        b.add_piece(Coord::new(1, 1), Piece::new(White, King));
        b.add_piece(Coord::new(1, 1), Piece::new(White, King));
    }

    #[test]
    #[should_panic]
    fn test_board_piece_panic_x_1() {
        let b = BoardPiece::<2, 3, 4>::default();
        let _ = b.get(Coord::new(2, 1));
    }

    #[test]
    #[should_panic]
    fn test_board_piece_panic_x_2() {
        let b = BoardPiece::<3, 1, 4>::default();
        let _ = b.get(Coord::new(-1, 1));
    }

    #[test]
    #[should_panic]
    fn test_board_piece_panic_y_1() {
        let b = BoardPiece::<2, 3, 4>::default();
        let _ = b.get(Coord::new(1, 3));
    }

    #[test]
    #[should_panic]
    fn test_board_piece_panic_y_2() {
        let b = BoardPiece::<2, 3, 4>::default();
        let _ = b.get(Coord::new(1, -1));
    }

    #[test]
    #[should_panic]
    fn test_mut_board_piece_panic_x_1() {
        let mut b = BoardPiece::<2, 3, 4>::default();
        b.clear(Coord::new(2, 1));
    }

    #[test]
    #[should_panic]
    fn test_mut_board_piece_panic_x_2() {
        let mut b = BoardPiece::<2, 3, 4>::default();
        b.clear(Coord::new(-1, 1));
    }

    #[test]
    #[should_panic]
    fn test_mut_board_piece_panic_y_1() {
        let mut b = BoardPiece::<2, 3, 4>::default();
        b.clear(Coord::new(1, 3));
    }

    #[test]
    #[should_panic]
    fn test_mut_board_piece_panic_y_2() {
        let mut b = BoardPiece::<2, 3, 4>::default();
        b.clear(Coord::new(1, -1));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn test_board_piece_set_existing_piece_panic() {
        let mut b = BoardPiece::<2, 3, 4>::default();
        b.add_piece(Coord::new(1, 1), Piece::new(White, King));
        b.add_piece(Coord::new(1, 1), Piece::new(White, King));
    }
    #[test]
    fn test_dump() {
        fn test<T: Board>() {
            let board = T::with_pieces(&[
                (Coord::new(0, 0), Piece::new(White, King)),
                (Coord::new(2, 0), Piece::new(Black, King)),
                (Coord::new(2, 2), Piece::new(White, Empress)),
                (Coord::new(3, 3), Piece::new(Black, Bishop)),
            ]);
            assert_eq!(format!("{:?}", board), "...b\n..E.\n....\nK.k.\n");
        }
        test::<BoardSquare<4, 4>>();
        test::<BoardPiece<4, 4, 4>>();
    }

    #[test]
    fn test_move() {
        fn test<T: Board>() {
            let mut board = T::default();
            let p = Piece::new(White, Rook);
            board.add_piece(Coord::new(1, 0), p);
            assert_eq!(board.get(Coord::new(1, 0)), Some(p));
            assert_eq!(board.get(Coord::new(1, 1)), None);
            board.make_move(
                Move {
                    from: Coord::new(1, 0),
                    to: Coord::new(1, 1),
                },
                White,
            );
            assert_eq!(board.get(Coord::new(1, 1)), Some(p));
            assert_eq!(board.get(Coord::new(1, 0)), None);
        }
        test::<BoardSquare<8, 8>>();
        test::<BoardPiece<8, 8, 4>>();
    }

    #[test]
    fn test_pawn_double_move() {
        let mut board = BoardSquare::<8, 8>::default();
        board.add_piece(Coord::new(2, 1), Piece::new(White, Pawn));
        board.add_piece(Coord::new(3, 6), Piece::new(Black, Pawn));
        assert_eq!(board.get_last_pawn_double_move(), None);
        assert!(board.get(Coord::new(2, 1)).is_some());

        board.make_move(
            Move {
                from: Coord::new(2, 1),
                to: Coord::new(2, 3),
            },
            White,
        );
        assert!(board.get(Coord::new(2, 1)).is_none());
        assert!(board.get(Coord::new(2, 3)).is_some());
        assert_eq!(board.get_last_pawn_double_move(), Some(Coord::new(2, 3)));
        board.make_move(
            Move {
                from: Coord::new(3, 6),
                to: Coord::new(3, 4),
            },
            Black,
        );
        assert_eq!(board.get_last_pawn_double_move(), Some(Coord::new(3, 4)));

        board.make_move(
            Move {
                from: Coord::new(2, 3),
                to: Coord::new(2, 4),
            },
            White,
        );
        assert_eq!(board.get_last_pawn_double_move(), None);
    }

    #[test]
    fn test_en_passant() {
        fn test<T: Board>() {
            let mut board = T::default();
            board.add_piece(Coord::new(2, 4), Piece::new(White, Pawn));
            board.add_piece(Coord::new(3, 4), Piece::new(Black, Pawn));
            assert!(board.get(Coord::new(3, 4)).is_some());

            board.make_move(
                Move {
                    from: Coord::new(2, 4),
                    to: Coord::new(3, 5),
                },
                White,
            );
            assert!(board.get(Coord::new(3, 4)).is_none());
        }
        test::<BoardSquare<8, 8>>();
        test::<BoardPiece<8, 8, 4>>();
    }

    #[test]
    fn test_promotion() {
        fn test<T: Board>() {
            let mut board = T::default();
            board.add_piece(Coord::new(2, 5), Piece::new(White, Pawn));
            board.add_piece(Coord::new(3, 2), Piece::new(Black, Pawn));

            board.make_move(
                Move {
                    from: Coord::new(2, 5),
                    to: Coord::new(2, 6),
                },
                White,
            );
            assert!(board.get(Coord::new(2, 6)).unwrap().ty() == Pawn);

            board.make_move(
                Move {
                    from: Coord::new(3, 2),
                    to: Coord::new(3, 1),
                },
                Black,
            );
            assert!(board.get(Coord::new(3, 1)).unwrap().ty() == Pawn);

            board.make_move(
                Move {
                    from: Coord::new(2, 6),
                    to: Coord::new(2, 7),
                },
                White,
            );
            assert!(board.get(Coord::new(2, 7)).unwrap().ty() == Queen);

            board.make_move(
                Move {
                    from: Coord::new(3, 1),
                    to: Coord::new(3, 0),
                },
                Black,
            );
            assert!(board.get(Coord::new(3, 0)).unwrap().ty() == Queen);
        }
        test::<BoardSquare<8, 8>>();
        test::<BoardPiece<8, 8, 4>>();
    }

    #[test]
    fn test_castling_rights() {
        let mut board = BoardSquare::<8, 8>::with_pieces(&[
            (Coord::new(0, 0), Piece::new(White, Rook)),
            (Coord::new(7, 0), Piece::new(White, Rook)),
            (Coord::new(4, 0), Piece::new(White, King)),
            (Coord::new(0, 7), Piece::new(Black, Rook)),
            (Coord::new(7, 7), Piece::new(Black, Rook)),
            (Coord::new(4, 7), Piece::new(Black, King)),
        ]);
        board.castling_rights = [
            Some(Coord::new(0, 0)),
            Some(Coord::new(7, 0)),
            Some(Coord::new(0, 7)),
            Some(Coord::new(7, 7)),
        ];
        {
            let mut board2 = board.clone();
            board2.make_move(
                Move {
                    from: Coord::new(4, 0),
                    to: Coord::new(4, 1),
                },
                White,
            );
            assert_eq!(
                board2.castling_rights,
                [None, None, Some(Coord::new(0, 7)), Some(Coord::new(7, 7))]
            );
            board2.make_move(
                Move {
                    from: Coord::new(4, 7),
                    to: Coord::new(4, 6),
                },
                Black,
            );
            assert_eq!(board2.castling_rights, [None, None, None, None]);
        }
        {
            let mut board2 = board.clone();
            board2.make_move(
                Move {
                    from: Coord::new(0, 0),
                    to: Coord::new(1, 0),
                },
                White,
            );
            assert_eq!(
                board2.castling_rights,
                [
                    None,
                    Some(Coord::new(7, 0)),
                    Some(Coord::new(0, 7)),
                    Some(Coord::new(7, 7)),
                ]
            );
            board2.make_move(
                Move {
                    from: Coord::new(0, 7),
                    to: Coord::new(0, 6),
                },
                Black,
            );
            assert_eq!(
                board2.castling_rights,
                [None, Some(Coord::new(7, 0)), None, Some(Coord::new(7, 7))]
            );
            board2.make_move(
                Move {
                    from: Coord::new(7, 0),
                    to: Coord::new(7, 7),
                },
                White,
            );
            assert_eq!(board2.castling_rights, [None, None, None, None]);
        }
    }

    #[test]
    fn test_castle() {
        fn test<T: Board>() {
            let board = BoardSquare::<8, 8>::with_pieces(&[
                (Coord::new(0, 0), Piece::new(White, Rook)),
                (Coord::new(7, 0), Piece::new(White, Rook)),
                (Coord::new(4, 0), Piece::new(White, King)),
                (Coord::new(0, 7), Piece::new(Black, Rook)),
                (Coord::new(7, 7), Piece::new(Black, Rook)),
                (Coord::new(4, 7), Piece::new(Black, King)),
            ]);
            {
                let mut board2 = board.clone();
                board2.make_move(
                    Move {
                        from: Coord::new(4, 0),
                        to: Coord::new(0, 0),
                    },
                    White,
                );
                assert_eq!(board2.get(Coord::new(2, 0)).unwrap().ty(), King);
                assert_eq!(board2.get(Coord::new(3, 0)).unwrap().ty(), Rook);
                assert_eq!(board2.get(Coord::new(7, 0)).unwrap().ty(), Rook);
                assert!(board2.get(Coord::new(0, 0)).is_none());
                assert!(board2.get(Coord::new(4, 0)).is_none());
            }
            {
                let mut board2 = board.clone();
                board2.make_move(
                    Move {
                        from: Coord::new(4, 0),
                        to: Coord::new(7, 0),
                    },
                    White,
                );
                assert_eq!(board2.get(Coord::new(0, 0)).unwrap().ty(), Rook);
                assert_eq!(board2.get(Coord::new(5, 0)).unwrap().ty(), Rook);
                assert_eq!(board2.get(Coord::new(6, 0)).unwrap().ty(), King);
                assert!(board2.get(Coord::new(4, 0)).is_none());
                assert!(board2.get(Coord::new(7, 0)).is_none());
            }
            {
                let mut board2 = board.clone();
                board2.make_move(
                    Move {
                        from: Coord::new(4, 7),
                        to: Coord::new(0, 7),
                    },
                    Black,
                );
                assert_eq!(board2.get(Coord::new(2, 7)).unwrap().ty(), King);
                assert_eq!(board2.get(Coord::new(3, 7)).unwrap().ty(), Rook);
                assert_eq!(board2.get(Coord::new(7, 7)).unwrap().ty(), Rook);
                assert!(board2.get(Coord::new(0, 7)).is_none());
                assert!(board2.get(Coord::new(4, 7)).is_none());
            }
            {
                let mut board2 = board.clone();
                board2.make_move(
                    Move {
                        from: Coord::new(4, 7),
                        to: Coord::new(7, 7),
                    },
                    Black,
                );
                assert_eq!(board2.get(Coord::new(0, 7)).unwrap().ty(), Rook);
                assert_eq!(board2.get(Coord::new(5, 7)).unwrap().ty(), Rook);
                assert_eq!(board2.get(Coord::new(6, 7)).unwrap().ty(), King);
                assert!(board2.get(Coord::new(4, 7)).is_none());
                assert!(board2.get(Coord::new(7, 7)).is_none());
            }
            {
                let mut board = BoardSquare::<8, 8>::with_pieces(&[
                    (Coord::new(0, 0), Piece::new(White, Rook)),
                    (Coord::new(7, 0), Piece::new(White, Rook)),
                    (Coord::new(1, 0), Piece::new(White, King)),
                ]);
                board.make_move(
                    Move {
                        from: Coord::new(1, 0),
                        to: Coord::new(0, 0),
                    },
                    White,
                );
                assert_eq!(board.get(Coord::new(2, 0)).unwrap().ty(), King);
                assert_eq!(board.get(Coord::new(3, 0)).unwrap().ty(), Rook);
                assert_eq!(board.get(Coord::new(7, 0)).unwrap().ty(), Rook);
                assert!(board.get(Coord::new(0, 0)).is_none());
                assert!(board.get(Coord::new(1, 0)).is_none());
            }
        }
        test::<BoardSquare<8, 8>>();
        test::<BoardPiece<8, 8, 4>>();
    }
    #[test]
    fn test_premade_boards() {
        Presets::classical();
        Presets::los_alamos();
        Presets::embassy();

        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            Presets::chess960(&mut rng);
            Presets::capablanca_random(&mut rng);
        }
    }
}
