use fairy::board::{Board, Move};
use fairy::coord::Coord;
use fairy::moves::all_moves;

fn main() {
    let mut board = Board::classical();
    println!("1: {} moves", all_moves(&board).len());
    board.make_move(Move {
        from: Coord::new(4, 1),
        to: Coord::new(4, 3),
    });
    println!("2: {} moves", all_moves(&board).len());
    board.make_move(Move {
        from: Coord::new(4, 6),
        to: Coord::new(4, 4),
    });
    println!("3: {} moves", all_moves(&board).len());
}
