use fairy::board::Board;
use fairy::moves::all_moves;
use fairy::player::Player;

fn main() {
    let board = Board::classical();
    println!("1: {} moves", all_moves(&board, Player::White).len());
}
