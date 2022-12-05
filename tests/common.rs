use fairy::board::Board;
use fairy::coord::Coord;
use fairy::moves::*;
use fairy::piece::Type::*;
use fairy::player::Player;

pub fn king_coord(board: &Board, player: Player) -> Coord {
    for y in 0..board.height {
        for x in 0..board.width {
            let coord = Coord::new(x, y);
            if let Some(piece) = board[coord].as_ref() {
                if piece.player == player && piece.ty == King {
                    return coord;
                }
            }
        }
    }
    panic!()
}

pub fn is_in_check(board: &Board, player: Player) -> bool {
    let king_coord = king_coord(board, player);
    is_under_attack(board, king_coord, player)
}
