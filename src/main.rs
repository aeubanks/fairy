use fairy::board::Presets;
use fairy::moves::all_moves;
use fairy::player::Player;

fn main() {
    let board = Presets::classical();
    println!("1: {} moves", all_moves(&board, Player::White).len());
}
