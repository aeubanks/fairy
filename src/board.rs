use crate::piece::Piece;
use std::ops::{Index, IndexMut};

pub struct Board {
    pieces: Vec<Option<Piece>>,
    width: usize,
    height: usize,
}

impl Board {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            pieces: vec![None; width * height],
            width,
            height,
        }
    }
}

impl Index<(usize, usize)> for Board {
    type Output = Option<Piece>;

    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        assert!(x < self.width);
        assert!(y < self.height);
        &self.pieces[y * self.width + x]
    }
}

impl IndexMut<(usize, usize)> for Board {
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut Self::Output {
        assert!(x < self.width);
        assert!(y < self.height);
        &mut self.pieces[y * self.width + x]
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
fn test_board_panic_x_1() {
    let b = Board::new(2, 3);
    let _ = b[(2, 1)];
}

#[test]
#[should_panic]
fn test_board_panic_x_2() {
    let b = Board::new(2, 3);
    let _ = b[(3, 1)];
}

#[test]
#[should_panic]
fn test_board_panic_y_1() {
    let b = Board::new(2, 3);
    let _ = b[(1, 3)];
}

#[test]
#[should_panic]
fn test_board_panic_y_2() {
    let b = Board::new(2, 3);
    let _ = b[(1, 4)];
}

#[test]
#[should_panic]
fn test_mut_board_panic_x_1() {
    let mut b = Board::new(2, 3);
    b[(2, 1)] = None;
}

#[test]
#[should_panic]
fn test_mut_board_panic_x_2() {
    let mut b = Board::new(2, 3);
    b[(3, 1)] = None;
}

#[test]
#[should_panic]
fn test_mut_board_panic_y_1() {
    let mut b = Board::new(2, 3);
    b[(1, 3)] = None;
}

#[test]
#[should_panic]
fn test_mut_board_panic_y_2() {
    let mut b = Board::new(2, 3);
    b[(1, 4)] = None;
}