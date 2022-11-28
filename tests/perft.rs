use fairy::board::Board;
use fairy::coord::Coord;
use fairy::moves::*;
use fairy::piece::Type;

fn king_coord(board: &Board) -> Coord {
    for y in 0..board.height {
        for x in 0..board.width {
            let coord = Coord::new(x, y);
            if let Some(piece) = board[coord].as_ref() {
                if piece.player == board.player_turn && piece.ty == Type::King {
                    return coord;
                }
            }
        }
    }
    panic!()
}

fn perft(board: &Board, depth: u64) -> u64 {
    assert_ne!(depth, 0);
    let moves = all_moves(board);
    let king_coord = king_coord(board);
    let mut sum = 0;
    for m in moves {
        let mut copy = board.clone();
        copy.make_move(m);
        let opponent_moves = all_moves(&copy);
        if opponent_moves.into_iter().any(|om| om.to == king_coord) {
            continue;
        }
        if depth == 1 {
            sum += 1
        } else {
            sum += perft(&copy, depth - 1);
        }
    }
    sum
}

#[test]
fn classical() {
    let board = Board::classical();
    assert_eq!(perft(&board, 1), 20);
    assert_eq!(perft(&board, 2), 400);
    assert_eq!(perft(&board, 3), 8902);
    // assert_eq!(perft(&board, 4), 197281);
    // assert_eq!(perft(&board, 5), 4865609);
    // assert_eq!(perft(&board, 6), 119060324);
}
