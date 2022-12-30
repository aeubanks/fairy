use fairy::board::Board;
use fairy::piece::{Piece, Type::*};
use fairy::player::{next_player, Player::*};
use fairy::tablebase::{generate_tablebase, GenerateAllBoards, TBBoard, Tablebase};

fn black_king_exists<const W: i8, const H: i8>(board: &TBBoard<W, H>) -> bool {
    board
        .piece_coord(|piece| piece.player() == Black && piece.ty() == King)
        .is_some()
}

fn verify_board_tablebase<const W: i8, const H: i8>(
    board: &TBBoard<W, H>,
    tablebase: &Tablebase<W, H>,
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
    let mut tablebase = Tablebase::<4, 4>::default();
    let kk = [Piece::new(White, King), Piece::new(Black, King)];
    generate_tablebase(&mut tablebase, &kk);
    generate_tablebase(&mut tablebase, pieces);

    for b in GenerateAllBoards::<4, 4>::new(pieces) {
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
