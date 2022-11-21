use crate::coord::Coord;
use crate::piece::{Piece, Type::*};
use std::ops::{Index, IndexMut};

pub struct Board {
    pieces: Vec<Option<Piece>>,
    pub width: i8,
    pub height: i8,
    pub player_turn: u8,
    pub last_pawn_double_move: Option<Coord>,
}

#[derive(PartialEq, Eq)]
pub enum ExistingPieceResult {
    Empty,
    Friend,
    Opponent,
}

impl Board {
    pub fn new(width: i8, height: i8) -> Self {
        Self {
            pieces: vec![None; (width * height) as usize],
            width,
            height,
            player_turn: 0,
            last_pawn_double_move: None,
        }
    }

    pub fn classical() -> Self {
        let mut board = Self::new(8, 8);
        for i in 0..board.width {
            board.add_piece(
                Coord::new(i, 1),
                Piece {
                    player: 0,
                    ty: Pawn,
                },
            );
            board.add_piece(
                Coord::new(i, 6),
                Piece {
                    player: 1,
                    ty: Pawn,
                },
            );
        }
        for (i, ty) in [Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook]
            .into_iter()
            .enumerate()
        {
            board.add_piece(Coord::new(i as i8, 0), Piece { player: 0, ty });
            board.add_piece(Coord::new(i as i8, 7), Piece { player: 1, ty });
        }
        board
    }

    pub fn in_bounds(&self, coord: Coord) -> bool {
        coord.x < self.width && coord.x >= 0 && coord.y < self.height && coord.y >= 0
    }

    pub fn add_piece(&mut self, coord: Coord, piece: Piece) {
        assert!(self[coord].is_none());
        self[coord] = Some(piece);
    }

    pub fn existing_piece_result(&self, coord: Coord) -> ExistingPieceResult {
        use ExistingPieceResult::*;
        match &self[coord] {
            None => Empty,
            Some(other_piece) => {
                if other_piece.player == self.player_turn {
                    Friend
                } else {
                    Opponent
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
pub struct Move {
    pub from: Coord,
    pub to: Coord,
}

impl Board {
    pub fn make_move(&mut self, m: Move) {
        debug_assert!(self.existing_piece_result(m.from) == ExistingPieceResult::Friend);
        debug_assert!(self.existing_piece_result(m.to) != ExistingPieceResult::Friend);
        let piece = self[m.from].take().unwrap();
        if piece.ty == Pawn && (m.from.y - m.to.y).abs() == 2 {
            self.last_pawn_double_move = Some(m.to);
        } else {
            self.last_pawn_double_move = None;
        }
        if piece.ty == Pawn && m.from.x != m.to.x && self[m.to].is_none() {
            // en passant
            let opponent_pawn_coord = Coord::new(m.to.x, m.from.y);
            debug_assert!(
                self.existing_piece_result(opponent_pawn_coord) == ExistingPieceResult::Opponent
            );
            debug_assert!(self[opponent_pawn_coord].as_ref().unwrap().ty == Pawn);
            self[opponent_pawn_coord] = None;
        }
        self[m.to] = Some(piece);
        self.player_turn = (self.player_turn + 1) % 2;
    }
}

#[test]
fn test_make_move() {
    let mut board = Board::new(8, 8);
    board.add_piece(
        Coord::new(2, 1),
        Piece {
            player: 0,
            ty: Pawn,
        },
    );
    board.add_piece(
        Coord::new(3, 6),
        Piece {
            player: 1,
            ty: Pawn,
        },
    );
    assert_eq!(board.last_pawn_double_move, None);
    assert!(board[(2, 1)].is_some());
    assert_eq!(board.player_turn, 0);

    board.make_move(Move {
        from: Coord::new(2, 1),
        to: Coord::new(2, 3),
    });
    assert!(board[(2, 1)].is_none());
    assert!(board[(2, 3)].is_some());
    assert_eq!(board.player_turn, 1);
    assert_eq!(board.last_pawn_double_move, Some(Coord::new(2, 3)));

    board.make_move(Move {
        from: Coord::new(3, 6),
        to: Coord::new(3, 4),
    });
    assert_eq!(board.player_turn, 0);
    assert_eq!(board.last_pawn_double_move, Some(Coord::new(3, 4)));

    board.make_move(Move {
        from: Coord::new(2, 3),
        to: Coord::new(2, 4),
    });
    assert_eq!(board.player_turn, 1);
    assert_eq!(board.last_pawn_double_move, None);
}

#[test]
fn test_en_passant() {
    let mut board = Board::new(8, 8);
    board.add_piece(
        Coord::new(2, 4),
        Piece {
            player: 0,
            ty: Pawn,
        },
    );
    board.add_piece(
        Coord::new(3, 4),
        Piece {
            player: 1,
            ty: Pawn,
        },
    );
    board.last_pawn_double_move = Some(Coord::new(3, 4));
    assert!(board[(3, 4)].is_some());

    board.make_move(Move {
        from: Coord::new(2, 4),
        to: Coord::new(3, 5),
    });
    assert!(board[(3, 4)].is_none());
}

impl Index<Coord> for Board {
    type Output = Option<Piece>;

    fn index(&self, coord: Coord) -> &Self::Output {
        debug_assert!(self.in_bounds(coord));
        &self.pieces[(coord.y * self.width + coord.x) as usize]
    }
}

impl Index<(i8, i8)> for Board {
    type Output = Option<Piece>;

    fn index(&self, (x, y): (i8, i8)) -> &Self::Output {
        self.index(Coord { x, y })
    }
}

impl IndexMut<Coord> for Board {
    fn index_mut(&mut self, coord: Coord) -> &mut Self::Output {
        debug_assert!(self.in_bounds(coord));
        &mut self.pieces[(coord.y * self.width + coord.x) as usize]
    }
}

impl IndexMut<(i8, i8)> for Board {
    fn index_mut(&mut self, (x, y): (i8, i8)) -> &mut Self::Output {
        self.index_mut(Coord { x, y })
    }
}

#[test]
fn test_board() {
    use crate::piece::*;
    let mut b = Board::new(4, 4);
    let p1 = Some(Piece {
        player: 0,
        ty: Type::Bishop,
    });
    let p2 = Some(Piece {
        player: 1,
        ty: Type::Knight,
    });
    b[(0, 0)] = p1.clone();
    b[(3, 3)] = p2.clone();
    assert_eq!(b[(0, 0)], p1);
    assert_eq!(b[(3, 3)], p2);
    assert_eq!(b[(0, 3)], None);
}

#[test]
#[should_panic]
#[cfg(debug_assertions)]
fn test_board_panic_x_1() {
    let b = Board::new(2, 3);
    let _ = b[(2, 1)];
}

#[test]
#[should_panic]
#[cfg(debug_assertions)]
fn test_board_panic_x_2() {
    let b = Board::new(2, 3);
    let _ = b[(-1, 1)];
}

#[test]
#[should_panic]
#[cfg(debug_assertions)]
fn test_board_panic_y_1() {
    let b = Board::new(2, 3);
    let _ = b[(1, 3)];
}

#[test]
#[should_panic]
#[cfg(debug_assertions)]
fn test_board_panic_y_2() {
    let b = Board::new(2, 3);
    let _ = b[(1, -1)];
}

#[test]
#[should_panic]
#[cfg(debug_assertions)]
fn test_mut_board_panic_x_1() {
    let mut b = Board::new(2, 3);
    b[(2, 1)] = None;
}

#[test]
#[should_panic]
#[cfg(debug_assertions)]
fn test_mut_board_panic_x_2() {
    let mut b = Board::new(2, 3);
    b[(-1, 1)] = None;
}

#[test]
#[should_panic]
#[cfg(debug_assertions)]
fn test_mut_board_panic_y_1() {
    let mut b = Board::new(2, 3);
    b[(1, 3)] = None;
}

#[test]
#[should_panic]
#[cfg(debug_assertions)]
fn test_mut_board_panic_y_2() {
    let mut b = Board::new(2, 3);
    b[(1, -1)] = None;
}
