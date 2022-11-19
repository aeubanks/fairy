use crate::coord::Coord;
use crate::piece::Piece;
use std::ops::{Index, IndexMut};

pub struct Board {
    pieces: Vec<Option<Piece>>,
    width: i8,
    height: i8,
}

impl Board {
    pub fn new(width: i8, height: i8) -> Self {
        Self {
            pieces: vec![None; (width * height) as usize],
            width,
            height,
        }
    }

    pub fn in_bounds(&self, coord: Coord) -> bool {
        coord.x < self.width && coord.x >= 0 && coord.y < self.height && coord.y >= 0
    }
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
