use crate::coord::Coord;
use crate::piece::{Piece, Type, Type::*};
use crate::player::{Player, Player::*};
use rand::Rng;
use std::ops::{Index, IndexMut};

#[derive(Clone, PartialEq, Eq)]
pub struct Board<const N: usize, const M: usize> {
    pieces: [[Option<Piece>; M]; N],
    pub castling_rights: [Option<Coord>; 4],
    pub last_pawn_double_move: Option<Coord>,
}

#[derive(PartialEq, Eq)]
pub enum ExistingPieceResult {
    Empty,
    Friend,
    Opponent,
}

// This really should be derivable...
impl<const N: usize, const M: usize> Default for Board<N, M> {
    fn default() -> Self {
        assert!(N > 0);
        assert!(N < i8::MAX as usize);
        assert!(M > 0);
        assert!(M < i8::MAX as usize);
        Self {
            pieces: [[None; M]; N],
            castling_rights: [None; 4],
            last_pawn_double_move: None,
        }
    }
}

impl<const N: usize, const M: usize> Board<N, M> {
    #[cfg(test)]
    pub fn with_pieces(pieces: &[(Coord, Piece)]) -> Self {
        let mut board = Self::default();
        for (c, p) in pieces {
            board.add_piece(*c, p.clone());
        }
        board
    }

    pub fn in_bounds(&self, coord: Coord) -> bool {
        coord.x < N as i8 && coord.x >= 0 && coord.y < M as i8 && coord.y >= 0
    }

    pub fn add_piece(&mut self, coord: Coord, piece: Piece) {
        assert!(self[coord].is_none());
        self[coord] = Some(piece);
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
        let mut a = self[c1].take();
        std::mem::swap(&mut a, &mut self[c2]);
        std::mem::swap(&mut a, &mut self[c1]);
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

impl<const N: usize, const M: usize> Index<Coord> for Board<N, M> {
    type Output = Option<Piece>;

    fn index(&self, coord: Coord) -> &Self::Output {
        assert!(self.in_bounds(coord));
        &self.pieces[coord.x as usize][coord.y as usize]
    }
}

impl<const N: usize, const M: usize> Index<(i8, i8)> for Board<N, M> {
    type Output = Option<Piece>;

    fn index(&self, (x, y): (i8, i8)) -> &Self::Output {
        self.index(Coord { x, y })
    }
}

impl<const N: usize, const M: usize> IndexMut<Coord> for Board<N, M> {
    fn index_mut(&mut self, coord: Coord) -> &mut Self::Output {
        assert!(self.in_bounds(coord));

        &mut self.pieces[coord.x as usize][coord.y as usize]
    }
}

impl<const N: usize, const M: usize> IndexMut<(i8, i8)> for Board<N, M> {
    fn index_mut(&mut self, (x, y): (i8, i8)) -> &mut Self::Output {
        self.index_mut(Coord { x, y })
    }
}

#[test]
fn test_board() {
    use crate::piece::*;
    let mut b = Board::<4, 4>::default();
    let p1 = Some(Piece::new(White, Bishop));
    let p2 = Some(Piece::new(Black, Knight));
    b[(0, 0)] = p1.clone();
    b[(3, 3)] = p2.clone();
    assert_eq!(b[(0, 0)], p1);
    assert_eq!(b[(3, 3)], p2);
    assert_eq!(b[(0, 3)], None);
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
    b[(2, 1)] = None;
}

#[test]
#[should_panic]
fn test_mut_board_panic_x_2() {
    let mut b = Board::<2, 3>::default();
    b[(-1, 1)] = None;
}

#[test]
#[should_panic]
fn test_mut_board_panic_y_1() {
    let mut b = Board::<2, 3>::default();
    b[(1, 3)] = None;
}

#[test]
#[should_panic]
fn test_mut_board_panic_y_2() {
    let mut b = Board::<2, 3>::default();
    b[(1, -1)] = None;
}

impl<const N: usize, const M: usize> std::fmt::Debug for Board<N, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for y in (0..M as i8).rev() {
            for x in 0..N as i8 {
                let c = match self[(x, y)].as_ref() {
                    None => '.',
                    Some(p) => {
                        let c = match p.ty() {
                            Pawn => 'P',
                            Knight => 'N',
                            Bishop => 'B',
                            Rook => 'R',
                            Queen => 'Q',
                            King => 'K',
                            Chancellor => 'C',
                            Archbishop => 'A',
                            Amazon => 'Z',
                        };
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

impl<const N: usize, const M: usize> Board<N, M> {
    pub fn make_move(&mut self, m: Move, player: Player) {
        assert_ne!(m.from, m.to);
        assert!(self.existing_piece_result(m.from, player) == ExistingPieceResult::Friend);
        let to_res = self.existing_piece_result(m.to, player);
        let mut piece = self[m.from].take().unwrap();
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
            assert!(self[opponent_pawn_coord].as_ref().unwrap().ty() == Pawn);
            self[opponent_pawn_coord] = None;
        }
        // promotion
        if piece.ty() == Pawn && (m.to.y == 0 || m.to.y == M as i8 - 1) {
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
            let rook = self[m.to].take();
            assert_eq!(rook.as_ref().unwrap().ty(), Rook);
            // king moves to rook to castle with
            // king should always be between two rooks to castle with
            let (dest, rook_dest) = if m.from.x > m.to.x {
                (Coord::new(2, m.from.y), Coord::new(3, m.from.y))
            } else {
                (
                    Coord::new(N as i8 - 2, m.from.y),
                    Coord::new(N as i8 - 3, m.from.y),
                )
            };
            assert!(self[rook_dest].as_ref().is_none());
            self[rook_dest] = rook;
            self[dest] = Some(piece);
        } else {
            assert!(to_res != ExistingPieceResult::Friend);
            self[m.to] = Some(piece);
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
    assert!(board[(2, 6)].as_ref().unwrap().ty() == Pawn);

    board.make_move(
        Move {
            from: Coord::new(3, 2),
            to: Coord::new(3, 1),
        },
        Black,
    );
    assert!(board[(3, 1)].as_ref().unwrap().ty() == Pawn);

    board.make_move(
        Move {
            from: Coord::new(2, 6),
            to: Coord::new(2, 7),
        },
        White,
    );
    assert!(board[(2, 7)].as_ref().unwrap().ty() == Queen);

    board.make_move(
        Move {
            from: Coord::new(3, 1),
            to: Coord::new(3, 0),
        },
        Black,
    );
    assert!(board[(3, 0)].as_ref().unwrap().ty() == Queen);
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
        assert_eq!(board2[(2, 0)].as_ref().unwrap().ty(), King);
        assert_eq!(board2[(3, 0)].as_ref().unwrap().ty(), Rook);
        assert_eq!(board2[(7, 0)].as_ref().unwrap().ty(), Rook);
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
        assert_eq!(board2[(0, 0)].as_ref().unwrap().ty(), Rook);
        assert_eq!(board2[(5, 0)].as_ref().unwrap().ty(), Rook);
        assert_eq!(board2[(6, 0)].as_ref().unwrap().ty(), King);
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
        assert_eq!(board2[(2, 7)].as_ref().unwrap().ty(), King);
        assert_eq!(board2[(3, 7)].as_ref().unwrap().ty(), Rook);
        assert_eq!(board2[(7, 7)].as_ref().unwrap().ty(), Rook);
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
        assert_eq!(board2[(0, 7)].as_ref().unwrap().ty(), Rook);
        assert_eq!(board2[(5, 7)].as_ref().unwrap().ty(), Rook);
        assert_eq!(board2[(6, 7)].as_ref().unwrap().ty(), King);
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
        assert_eq!(board[(2, 0)].as_ref().unwrap().ty(), King);
        assert_eq!(board[(3, 0)].as_ref().unwrap().ty(), Rook);
        assert_eq!(board[(7, 0)].as_ref().unwrap().ty(), Rook);
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

impl<const N: usize, const M: usize> Board<N, M> {
    fn setup_with_pawns(castling: bool, pieces: &[Type]) -> Self {
        let mut board = Self::default();
        for i in 0..N as i8 {
            board.add_piece(Coord::new(i, 1), Piece::new(White, Pawn));
            board.add_piece(Coord::new(i, M as i8 - 2), Piece::new(Black, Pawn));
        }
        assert!(pieces.len() == N);
        for (i, ty) in pieces.into_iter().enumerate() {
            let white_coord = Coord::new(i as i8, 0);
            board.add_piece(white_coord, Piece::new(White, *ty));
            let black_coord = Coord::new(i as i8, M as i8 - 1);
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

pub fn king_coord<const N: usize, const M: usize>(board: &Board<N, M>, player: Player) -> Coord {
    for y in 0..M as i8 {
        for x in 0..N as i8 {
            let coord = Coord::new(x, y);
            if let Some(piece) = board[coord].as_ref() {
                if piece.player() == player && piece.ty() == King {
                    return coord;
                }
            }
        }
    }
    panic!()
}

// TODO: factor out coord visiting
pub fn has_pawn<const N: usize, const M: usize>(board: &Board<N, M>) -> bool {
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
