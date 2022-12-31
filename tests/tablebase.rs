use fairy::board::{Board, Move};
use fairy::coord::Coord;
use fairy::moves::is_under_attack;
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
fn test_kqk() {
    let pieces = [
        Piece::new(White, King),
        Piece::new(White, Queen),
        Piece::new(Black, King),
    ];
    verify_all_three_piece_positions_forced_win(&pieces);
}

#[test]
fn test_krk() {
    let pieces = [
        Piece::new(White, King),
        Piece::new(White, Rook),
        Piece::new(Black, King),
    ];
    verify_all_three_piece_positions_forced_win(&pieces);
}

#[test]
fn test_kek() {
    let pieces = [
        Piece::new(White, King),
        Piece::new(White, Empress),
        Piece::new(Black, King),
    ];
    verify_all_three_piece_positions_forced_win(&pieces);
}

#[test]
fn test_kck() {
    let pieces = [
        Piece::new(White, King),
        Piece::new(White, Cardinal),
        Piece::new(Black, King),
    ];
    verify_all_three_piece_positions_forced_win(&pieces);
}

#[test]
fn test_kak() {
    let pieces = [
        Piece::new(White, King),
        Piece::new(White, Cardinal),
        Piece::new(Black, King),
    ];
    verify_all_three_piece_positions_forced_win(&pieces);
}

#[test]
fn test_kk_5_1() {
    let wk = Piece::new(White, King);
    let bk = Piece::new(Black, King);
    let mut tablebase = Tablebase::<5, 1>::default();
    generate_tablebase(&mut tablebase, &[wk, bk]);

    assert_eq!(
        tablebase.white_result(&TBBoard::<5, 1>::with_pieces(&[
            (Coord::new(0, 0), wk),
            (Coord::new(3, 0), bk)
        ])),
        Some((
            Move {
                from: Coord::new(0, 0),
                to: Coord::new(1, 0)
            },
            5
        ))
    );
    assert_eq!(
        tablebase.black_result(&TBBoard::<5, 1>::with_pieces(&[
            (Coord::new(1, 0), wk),
            (Coord::new(3, 0), bk)
        ])),
        Some((
            Move {
                from: Coord::new(3, 0),
                to: Coord::new(4, 0)
            },
            4
        ))
    );
    assert_eq!(
        tablebase.white_result(&TBBoard::<5, 1>::with_pieces(&[
            (Coord::new(1, 0), wk),
            (Coord::new(4, 0), bk)
        ])),
        Some((
            Move {
                from: Coord::new(1, 0),
                to: Coord::new(2, 0)
            },
            3
        ))
    );
    assert_eq!(
        tablebase.black_result(&TBBoard::<5, 1>::with_pieces(&[
            (Coord::new(2, 0), wk),
            (Coord::new(4, 0), bk)
        ])),
        Some((
            Move {
                from: Coord::new(4, 0),
                to: Coord::new(3, 0)
            },
            2
        ))
    );
    assert_eq!(
        tablebase.white_result(&TBBoard::<5, 1>::with_pieces(&[
            (Coord::new(2, 0), wk),
            (Coord::new(3, 0), bk)
        ])),
        Some((
            Move {
                from: Coord::new(2, 0),
                to: Coord::new(3, 0)
            },
            1
        ))
    );
}

#[test]
fn test_kk() {
    fn test<const W: i8, const H: i8>() {
        let pieces = [Piece::new(White, King), Piece::new(Black, King)];
        let mut tablebase = Tablebase::<W, H>::default();
        generate_tablebase(&mut tablebase, &pieces);
        // If white king couldn't capture on first move, no forced win.
        for b in GenerateAllBoards::new(&pieces) {
            if is_under_attack(&b, b.king_coord(Black), Black) {
                assert_eq!(tablebase.white_result(&b).unwrap().1, 1);
            } else {
                assert_eq!(tablebase.white_result(&b), None);
            }
            assert_eq!(tablebase.black_result(&b), None);
        }
    }
    test::<6, 6>();
    test::<5, 5>();
    test::<4, 5>();
    test::<4, 6>();
}
