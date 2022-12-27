use fairy::board::Board;
use fairy::coord::Coord;
use fairy::piece::{Piece, Type::*};
use fairy::player::{next_player, Player::*};
use fairy::tablebase::{generate_all_boards, generate_tablebase, Tablebase};

fn black_king_exists<const N: usize, const M: usize>(board: &Board<N, M>) -> bool {
    for y in 0..M as i8 {
        for x in 0..N as i8 {
            let coord = Coord::new(x, y);
            if let Some(piece) = board[coord].as_ref() {
                if piece.player() == Black && piece.ty() == King {
                    return true;
                }
            }
        }
    }
    false
}

fn verify_board_tablebase<const N: usize, const M: usize>(
    board: &Board<N, M>,
    tablebase: &Tablebase<N, M>,
) {
    let mut board = board.clone();
    let (_, mut expected_depth) = tablebase.white_result(&board).unwrap();
    let mut player = White;

    while expected_depth > 0 {
        assert!(black_king_exists(&board));
        let (m, depth) = match player {
            White => tablebase.white_result(&board),
            Black => tablebase.black_result(&board),
        }
        .unwrap();
        assert_eq!(depth, expected_depth);
        board.make_move(m, player);
        expected_depth -= 1;
        player = next_player(player);
    }
    assert!(!black_king_exists(&board));
}

fn verify_all_three_piece_positions_forced_win(pieces: &[Piece]) {
    assert_eq!(pieces.len(), 3);
    let kk = [Piece::new(White, King), Piece::new(Black, King)];
    let tablebase = generate_tablebase::<4, 4>(&[&kk, &pieces]);
    let all = generate_all_boards::<4, 4>(pieces);

    for b in all {
        let wd = tablebase.white_result(&b);
        let bd = tablebase.black_result(&b);
        assert!(wd.unwrap().1 % 2 == 1);
        assert!(bd.is_none() || bd.unwrap().1 % 2 == 0);
        verify_board_tablebase(&b, &tablebase);
    }
}

#[test]
fn test_kqk_tablebase() {
    let pieces = [
        Piece::new(White, King),
        Piece::new(White, Queen),
        Piece::new(Black, King),
    ];
    verify_all_three_piece_positions_forced_win(&pieces);
}

#[test]
fn test_krk_tablebase() {
    let pieces = [
        Piece::new(White, King),
        Piece::new(White, Rook),
        Piece::new(Black, King),
    ];
    verify_all_three_piece_positions_forced_win(&pieces);
}

#[test]
fn test_kck_tablebase() {
    let pieces = [
        Piece::new(White, King),
        Piece::new(White, Chancellor),
        Piece::new(Black, King),
    ];
    verify_all_three_piece_positions_forced_win(&pieces);
}

#[test]
fn test_kak_tablebase() {
    let pieces = [
        Piece::new(White, King),
        Piece::new(White, Archbishop),
        Piece::new(Black, King),
    ];
    verify_all_three_piece_positions_forced_win(&pieces);
}

#[test]
fn test_kzk_tablebase() {
    let pieces = [
        Piece::new(White, King),
        Piece::new(White, Archbishop),
        Piece::new(Black, King),
    ];
    verify_all_three_piece_positions_forced_win(&pieces);
}
