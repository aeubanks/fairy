mod board;
mod coord;
mod moves;
mod piece;
mod player;

use board::Board;
use coord::Coord;

fn main() {
    let mut board = Board::classical();
    println!("1: {} moves", moves::all_moves(&board).len());
    board.make_move(board::Move {
        from: Coord::new(4, 1),
        to: Coord::new(4, 3),
    });
    println!("2: {} moves", moves::all_moves(&board).len());
    board.make_move(board::Move {
        from: Coord::new(4, 6),
        to: Coord::new(4, 4),
    });
    println!("3: {} moves", moves::all_moves(&board).len());
}
