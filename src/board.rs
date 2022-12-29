use crate::coord::Coord;
use crate::piece::{Piece, Type, Type::*};
use crate::player::{Player, Player::*};
use rand::Rng;
use static_assertions::const_assert_eq;
use std::ops::Index;

#[derive(Clone, PartialEq, Eq)]
pub struct Board<const W: usize, const H: usize> {
    pieces: [[Option<Piece>; H]; W],
    pub castling_rights: [Option<Coord>; 4],
    pub last_pawn_double_move: Option<Coord>,
}

#[derive(PartialEq, Eq, Debug)]
pub enum ExistingPieceResult {
    Empty,
    Friend,
    Opponent,
}

// This really should be derivable...
impl<const W: usize, const H: usize> Default for Board<W, H> {
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

const_assert_eq!(79, std::mem::size_of::<Board<8, 8>>());

impl<const W: usize, const H: usize> Board<W, H> {
    #[cfg(test)]
    pub fn with_pieces(pieces: &[(Coord, Piece)]) -> Self {
        let mut board = Self::default();
        for (c, p) in pieces {
            board.add_piece(*c, p.clone());
        }
        board
    }

    pub fn in_bounds(&self, coord: Coord) -> bool {
        coord.x < W as i8 && coord.x >= 0 && coord.y < H as i8 && coord.y >= 0
    }

    pub fn clear(&mut self, coord: Coord) {
        self.pieces[coord.x as usize][coord.y as usize] = None;
    }

    pub fn add_piece(&mut self, coord: Coord, piece: Piece) {
        self.set(coord, Some(piece));
    }

    pub fn set(&mut self, coord: Coord, piece: Option<Piece>) {
        assert!(self.in_bounds(coord));
        assert!(self[coord].is_none());

        self.pieces[coord.x as usize][coord.y as usize] = piece;
    }

    pub fn take(&mut self, coord: Coord) -> Option<Piece> {
        assert!(self.in_bounds(coord));

        self.pieces[coord.x as usize][coord.y as usize].take()
    }

    pub fn existing_piece_result(&self, coord: Coord, player: Player) -> ExistingPieceResult {
        use ExistingPieceResult::*;
        match &self[coord] {
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

    pub fn swap(&mut self, c1: Coord, c2: Coord) {
        assert_ne!(c1, c2);
        let p1 = self.take(c1);
        let p2 = self.take(c2);
        self.set(c1, p2);
        self.set(c2, p1);
    }

    pub fn pieces_fn<F>(&self, mut f: F)
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

    pub fn pieces_fn_first<F>(&self, mut f: F) -> Option<Coord>
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
}

#[test]
fn test_board_swap() {
    let n = Piece::new(White, Knight);
    let b = Piece::new(White, Bishop);
    let mut board = Board::<3, 1>::with_pieces(&[(Coord::new(0, 0), n), (Coord::new(1, 0), b)]);

    assert_eq!(board[(0, 0)], Some(n));
    assert_eq!(board[(1, 0)], Some(b));
    assert_eq!(board[(2, 0)], None);

    board.swap(Coord::new(1, 0), Coord::new(0, 0));
    assert_eq!(board[(0, 0)], Some(b));
    assert_eq!(board[(1, 0)], Some(n));
    assert_eq!(board[(2, 0)], None);

    board.swap(Coord::new(2, 0), Coord::new(0, 0));
    assert_eq!(board[(0, 0)], None);
    assert_eq!(board[(1, 0)], Some(n));
    assert_eq!(board[(2, 0)], Some(b));
}

impl<const W: usize, const H: usize> Index<Coord> for Board<W, H> {
    type Output = Option<Piece>;

    fn index(&self, coord: Coord) -> &Self::Output {
        assert!(self.in_bounds(coord));
        &self.pieces[coord.x as usize][coord.y as usize]
    }
}

impl<const W: usize, const H: usize> Index<(i8, i8)> for Board<W, H> {
    type Output = Option<Piece>;

    fn index(&self, (x, y): (i8, i8)) -> &Self::Output {
        self.index(Coord { x, y })
    }
}

#[test]
fn test_board() {
    use crate::piece::*;
    let mut b = Board::<4, 4>::default();
    let p1 = Piece::new(White, Bishop);
    let p2 = Piece::new(Black, Knight);
    b.add_piece(Coord::new(0, 0), p1);
    b.set(Coord::new(3, 3), Some(p2));
    assert_eq!(b[(0, 0)], Some(p1));
    assert_eq!(b[(3, 3)], Some(p2));
    assert_eq!(b[(0, 3)], None);
    b.clear(Coord::new(0, 0));
    assert_eq!(b[(0, 0)], None);
}

#[test]
#[should_panic]
fn test_board_panic_x_1() {
    let b = Board::<2, 3>::default();
    let _ = b[(2, 1)];
}

#[test]
#[should_panic]
fn test_board_panic_x_2() {
    let b = Board::<2, 3>::default();
    let _ = b[(-1, 1)];
}

#[test]
#[should_panic]
fn test_board_panic_y_1() {
    let b = Board::<2, 3>::default();
    let _ = b[(1, 3)];
}

#[test]
#[should_panic]
fn test_board_panic_y_2() {
    let b = Board::<2, 3>::default();
    let _ = b[(1, -1)];
}

#[test]
#[should_panic]
fn test_mut_board_panic_x_1() {
    let mut b = Board::<2, 3>::default();
    b.set(Coord::new(2, 1), None);
}

#[test]
#[should_panic]
fn test_mut_board_panic_x_2() {
    let mut b = Board::<2, 3>::default();
    b.set(Coord::new(-1, 1), None);
}

#[test]
#[should_panic]
fn test_mut_board_panic_y_1() {
    let mut b = Board::<2, 3>::default();
    b.set(Coord::new(1, 3), None);
}

#[test]
#[should_panic]
fn test_mut_board_panic_y_2() {
    let mut b = Board::<2, 3>::default();
    b.set(Coord::new(1, -1), None);
}

#[test]
#[should_panic]
fn test_board_set_existing_piece_panic_1() {
    let mut b = Board::<2, 3>::default();
    b.add_piece(Coord::new(1, 1), Piece::new(White, King));
    b.add_piece(Coord::new(1, 1), Piece::new(White, King));
}

#[test]
#[should_panic]
fn test_board_set_existing_piece_panic_2() {
    let mut b = Board::<2, 3>::default();
    b.add_piece(Coord::new(1, 1), Piece::new(White, King));
    b.set(Coord::new(1, 1), Some(Piece::new(White, King)));
}

#[test]
#[should_panic]
fn test_board_set_existing_piece_panic_3() {
    let mut b = Board::<2, 3>::default();
    b.add_piece(Coord::new(1, 1), Piece::new(White, King));
    b.set(Coord::new(1, 1), None);
}

impl<const W: usize, const H: usize> std::fmt::Debug for Board<W, H> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for y in (0..H as i8).rev() {
            for x in 0..W as i8 {
                let c = match self[(x, y)] {
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
                write!(f, "{}", c)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[test]
fn test_dump() {
    let board = Board::<4, 4>::with_pieces(&[
        (Coord::new(0, 0), Piece::new(White, King)),
        (Coord::new(2, 0), Piece::new(Black, King)),
        (Coord::new(2, 2), Piece::new(White, Chancellor)),
        (Coord::new(3, 3), Piece::new(Black, Bishop)),
    ]);
    assert_eq!(format!("{:?}", board), "...b\n..C.\n....\nK.k.\n");
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Move {
    pub from: Coord,
    pub to: Coord,
}

impl<const W: usize, const H: usize> Board<W, H> {
    pub fn make_move(&mut self, m: Move, player: Player) {
        assert_ne!(m.from, m.to);
        assert!(self.existing_piece_result(m.from, player) == ExistingPieceResult::Friend);
        let to_res = self.existing_piece_result(m.to, player);
        let mut piece = self.take(m.from).unwrap();
        // pawn double moves
        if piece.ty() == Pawn && (m.from.y - m.to.y).abs() == 2 {
            self.last_pawn_double_move = Some(m.to);
        } else {
            self.last_pawn_double_move = None;
        }
        // en passant
        if piece.ty() == Pawn && m.from.x != m.to.x && self[m.to].is_none() {
            let opponent_pawn_coord = Coord::new(m.to.x, m.from.y);
            assert!(
                self.existing_piece_result(opponent_pawn_coord, player)
                    == ExistingPieceResult::Opponent
            );
            assert!(self[opponent_pawn_coord].unwrap().ty() == Pawn);
            self.clear(opponent_pawn_coord);
        }
        // promotion
        if piece.ty() == Pawn && (m.to.y == 0 || m.to.y == H as i8 - 1) {
            // TODO: support more than promoting to queen
            piece = Piece::new(piece.player(), Queen);
        }
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
        // castling
        if piece.ty() == King && to_res == ExistingPieceResult::Friend {
            let rook = self.take(m.to);
            assert_eq!(rook.unwrap().ty(), Rook);
            // king moves to rook to castle with
            // king should always be between two rooks to castle with
            let (dest, rook_dest) = if m.from.x > m.to.x {
                (Coord::new(2, m.from.y), Coord::new(3, m.from.y))
            } else {
                (
                    Coord::new(W as i8 - 2, m.from.y),
                    Coord::new(W as i8 - 3, m.from.y),
                )
            };
            assert!(self[rook_dest].is_none());
            self.set(rook_dest, rook);
            self.add_piece(dest, piece);
        } else {
            assert!(to_res != ExistingPieceResult::Friend);
            self.clear(m.to);
            self.add_piece(m.to, piece);
        }
    }
}

#[test]
fn test_make_move() {
    let mut board = Board::<8, 8>::default();
    board.add_piece(Coord::new(2, 1), Piece::new(White, Pawn));
    board.add_piece(Coord::new(3, 6), Piece::new(Black, Pawn));
    assert_eq!(board.last_pawn_double_move, None);
    assert!(board[(2, 1)].is_some());

    board.make_move(
        Move {
            from: Coord::new(2, 1),
            to: Coord::new(2, 3),
        },
        White,
    );
    assert!(board[(2, 1)].is_none());
    assert!(board[(2, 3)].is_some());
    assert_eq!(board.last_pawn_double_move, Some(Coord::new(2, 3)));
    board.make_move(
        Move {
            from: Coord::new(3, 6),
            to: Coord::new(3, 4),
        },
        Black,
    );
    assert_eq!(board.last_pawn_double_move, Some(Coord::new(3, 4)));

    board.make_move(
        Move {
            from: Coord::new(2, 3),
            to: Coord::new(2, 4),
        },
        White,
    );
    assert_eq!(board.last_pawn_double_move, None);
}

#[test]
fn test_en_passant() {
    let mut board = Board::<8, 8>::default();
    board.add_piece(Coord::new(2, 4), Piece::new(White, Pawn));
    board.add_piece(Coord::new(3, 4), Piece::new(Black, Pawn));
    board.last_pawn_double_move = Some(Coord::new(3, 4));
    assert!(board[(3, 4)].is_some());

    board.make_move(
        Move {
            from: Coord::new(2, 4),
            to: Coord::new(3, 5),
        },
        White,
    );
    assert!(board[(3, 4)].is_none());
}

#[test]
fn test_en_promotion() {
    let mut board = Board::<8, 8>::default();
    board.add_piece(Coord::new(2, 5), Piece::new(White, Pawn));
    board.add_piece(Coord::new(3, 2), Piece::new(Black, Pawn));

    board.make_move(
        Move {
            from: Coord::new(2, 5),
            to: Coord::new(2, 6),
        },
        White,
    );
    assert!(board[(2, 6)].unwrap().ty() == Pawn);

    board.make_move(
        Move {
            from: Coord::new(3, 2),
            to: Coord::new(3, 1),
        },
        Black,
    );
    assert!(board[(3, 1)].unwrap().ty() == Pawn);

    board.make_move(
        Move {
            from: Coord::new(2, 6),
            to: Coord::new(2, 7),
        },
        White,
    );
    assert!(board[(2, 7)].unwrap().ty() == Queen);

    board.make_move(
        Move {
            from: Coord::new(3, 1),
            to: Coord::new(3, 0),
        },
        Black,
    );
    assert!(board[(3, 0)].unwrap().ty() == Queen);
}

#[test]
fn test_castling_rights() {
    let mut board = Board::<8, 8>::with_pieces(&[
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
    let board = Board::<8, 8>::with_pieces(&[
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
        assert_eq!(board2[(2, 0)].unwrap().ty(), King);
        assert_eq!(board2[(3, 0)].unwrap().ty(), Rook);
        assert_eq!(board2[(7, 0)].unwrap().ty(), Rook);
        assert!(board2[(0, 0)].is_none());
        assert!(board2[(4, 0)].is_none());
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
        assert_eq!(board2[(0, 0)].unwrap().ty(), Rook);
        assert_eq!(board2[(5, 0)].unwrap().ty(), Rook);
        assert_eq!(board2[(6, 0)].unwrap().ty(), King);
        assert!(board2[(4, 0)].is_none());
        assert!(board2[(7, 0)].is_none());
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
        assert_eq!(board2[(2, 7)].unwrap().ty(), King);
        assert_eq!(board2[(3, 7)].unwrap().ty(), Rook);
        assert_eq!(board2[(7, 7)].unwrap().ty(), Rook);
        assert!(board2[(0, 7)].is_none());
        assert!(board2[(4, 7)].is_none());
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
        assert_eq!(board2[(0, 7)].unwrap().ty(), Rook);
        assert_eq!(board2[(5, 7)].unwrap().ty(), Rook);
        assert_eq!(board2[(6, 7)].unwrap().ty(), King);
        assert!(board2[(4, 7)].is_none());
        assert!(board2[(7, 7)].is_none());
    }
    {
        let mut board = Board::<8, 8>::with_pieces(&[
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
        assert_eq!(board[(2, 0)].unwrap().ty(), King);
        assert_eq!(board[(3, 0)].unwrap().ty(), Rook);
        assert_eq!(board[(7, 0)].unwrap().ty(), Rook);
        assert!(board[(0, 0)].is_none());
        assert!(board[(1, 0)].is_none());
    }
}

#[allow(dead_code)]
pub struct Presets;

#[allow(dead_code)]
impl Presets {
    pub fn los_alamos() -> Board<6, 6> {
        Board::<6, 6>::setup_with_pawns(false, &[Rook, Knight, Queen, King, Knight, Rook])
    }

    pub fn classical() -> Board<8, 8> {
        Board::<8, 8>::setup_with_pawns(
            true,
            &[Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook],
        )
    }

    pub fn embassy() -> Board<10, 8> {
        Board::<10, 8>::setup_with_pawns(
            true,
            &[
                Rook, Knight, Bishop, Queen, King, Chancellor, Archbishop, Bishop, Knight, Rook,
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

    pub fn chess960<R: Rng + ?Sized>(rng: &mut R) -> Board<8, 8> {
        let mut pieces: [Option<Type>; 8] = [None; 8];
        Self::set_nth_empty(rng.gen_range(0..4) * 2, &mut pieces, Bishop);
        Self::set_nth_empty(rng.gen_range(0..4) * 2 + 1, &mut pieces, Bishop);
        Self::set_nth_empty(rng.gen_range(0..6), &mut pieces, Queen);
        Self::set_nth_empty(rng.gen_range(0..5), &mut pieces, Knight);
        Self::set_nth_empty(rng.gen_range(0..4), &mut pieces, Knight);
        Self::set_nth_empty(0, &mut pieces, Rook);
        Self::set_nth_empty(0, &mut pieces, King);
        Self::set_nth_empty(0, &mut pieces, Rook);

        Board::<8, 8>::setup_with_pawns(true, &pieces.map(|p| p.unwrap()))
    }

    pub fn capablanca_random<R: Rng + ?Sized>(rng: &mut R) -> Board<10, 8> {
        let mut evens: [Option<Type>; 5] = [None; 5];
        let mut odds: [Option<Type>; 5] = [None; 5];
        evens[rng.gen_range(0..5)] = Some(Bishop);
        odds[rng.gen_range(0..5)] = Some(Bishop);
        let (qa1, qa2) = if rng.gen() {
            (Queen, Archbishop)
        } else {
            (Archbishop, Queen)
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

        Self::set_nth_empty(rng.gen_range(0..6), &mut pieces, Chancellor);
        Self::set_nth_empty(rng.gen_range(0..5), &mut pieces, Knight);
        Self::set_nth_empty(rng.gen_range(0..4), &mut pieces, Knight);
        Self::set_nth_empty(0, &mut pieces, Rook);
        Self::set_nth_empty(0, &mut pieces, King);
        Self::set_nth_empty(0, &mut pieces, Rook);

        Board::<10, 8>::setup_with_pawns(true, &pieces.map(|p| p.unwrap()))
    }
}

impl<const W: usize, const H: usize> Board<W, H> {
    fn setup_with_pawns(castling: bool, pieces: &[Type]) -> Self {
        let mut board = Self::default();
        for i in 0..W as i8 {
            board.add_piece(Coord::new(i, 1), Piece::new(White, Pawn));
            board.add_piece(Coord::new(i, H as i8 - 2), Piece::new(Black, Pawn));
        }
        assert!(pieces.len() == W);
        for (i, ty) in pieces.into_iter().enumerate() {
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

pub fn king_coord<const W: usize, const H: usize>(board: &Board<W, H>, player: Player) -> Coord {
    board
        .pieces_fn_first(|piece| piece.player() == player && piece.ty() == King)
        .unwrap()
}
