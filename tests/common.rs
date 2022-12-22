use fairy::board::{king_coord, Board};
use fairy::moves::is_under_attack;
use fairy::player::Player;

pub fn is_in_check(board: &Board, player: Player) -> bool {
    let king_coord = king_coord(board, player);
    is_under_attack(board, king_coord, player)
}
