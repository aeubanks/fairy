use fairy::board::Board;
use fairy::coord::Coord;
use fairy::piece::{Piece, Type::*};
use fairy::player::{next_player, Player::*};
use fairy::tablebase::{generate_all_boards, generate_tablebase, Tablebase};

fn black_king_exists(board: &Board) -> bool {
    for y in 0..board.height {
        for x in 0..board.width {
            let coord = Coord::new(x, y);
            if let Some(piece) = board[coord].as_ref() {
                if piece.player == Black && piece.ty == King {
                    return true;
                }
            }
        }
    }
    false
}

fn verify_board_tablebase(board: &Board, tablebase: &Tablebase) {
    let mut board = board.clone();
    let mut expected_depth = tablebase.white_depth(&board).unwrap();
    let mut player = White;

    while expected_depth > 0 {
        assert!(black_king_exists(&board));
        let (depth, m) = match player {
            White => (tablebase.white_depth(&board), tablebase.white_move(&board)),
            Black => (tablebase.black_depth(&board), tablebase.black_move(&board)),
        };
        assert_eq!(depth, Some(expected_depth));
        board.make_move(m.unwrap(), player);
        expected_depth -= 1;
        player = next_player(player);
    }
    assert!(!black_king_exists(&board));
}

fn verify_all_three_piece_positions_forced_win(pieces: &[Piece]) {
    assert_eq!(pieces.len(), 3);
    let kk = [
        Piece {
            player: White,
            ty: King,
        },
        Piece {
            player: Black,
            ty: King,
        },
    ];
    let tablebase = generate_tablebase(4, 4, &[&kk, &pieces]);
    let all = generate_all_boards(4, 4, pieces);

    for b in all {
        let wd = tablebase.white_depth(&b);
        let bd = tablebase.black_depth(&b);
        assert!(wd.unwrap() % 2 == 1);
        assert!(bd.is_none() || bd.unwrap() % 2 == 0);
        verify_board_tablebase(&b, &tablebase);
    }
}

#[test]
fn test_kqk_tablebase() {
    let pieces = [
        Piece {
            player: White,
            ty: King,
        },
        Piece {
            player: White,
            ty: Queen,
        },
        Piece {
            player: Black,
            ty: King,
        },
    ];
    verify_all_three_piece_positions_forced_win(&pieces);
}

#[test]
fn test_krk_tablebase() {
    let pieces = [
        Piece {
            player: White,
            ty: King,
        },
        Piece {
            player: White,
            ty: Rook,
        },
        Piece {
            player: Black,
            ty: King,
        },
    ];
    verify_all_three_piece_positions_forced_win(&pieces);
}

#[test]
fn test_kck_tablebase() {
    let pieces = [
        Piece {
            player: White,
            ty: King,
        },
        Piece {
            player: White,
            ty: Chancellor,
        },
        Piece {
            player: Black,
            ty: King,
        },
    ];
    verify_all_three_piece_positions_forced_win(&pieces);
}

#[test]
fn test_kak_tablebase() {
    let pieces = [
        Piece {
            player: White,
            ty: King,
        },
        Piece {
            player: White,
            ty: Archbishop,
        },
        Piece {
            player: Black,
            ty: King,
        },
    ];
    verify_all_three_piece_positions_forced_win(&pieces);
}
