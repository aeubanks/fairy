mod board;
mod piece;

use board::Board;
use piece::{Piece, Type::*};

fn create_board() -> Board {
    let mut board = Board::new(8, 8);
    board[(0, 0)] = Some(Piece {
        player: 0,
        ty: Rook,
    });
    board[(7, 7)] = Some(Piece {
        player: 1,
        ty: Rook,
    });
    board
}

fn main() {
    let _ = create_board();
}
