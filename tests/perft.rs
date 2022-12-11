mod common;

use common::*;
use fairy::board::Board;
use fairy::coord::Coord;
use fairy::moves::*;
use fairy::piece::{Piece, Type::*};
use fairy::player::Player::*;

fn perft(board: &Board, depth: u64) -> u64 {
    assert_ne!(depth, 0);
    let moves = all_moves(board);
    let player = board.player_turn;
    let mut sum = 0;
    for m in moves {
        let mut copy = board.clone();
        copy.make_move(m);
        if is_in_check(&copy, player) {
            continue;
        }
        if board[m.from].as_ref().unwrap().ty == Pawn && (m.to.y == 0 || m.to.y == board.height - 1)
        {
            for ty in [Knight, Bishop, Rook] {
                let mut promotion_copy = copy.clone();
                promotion_copy[m.to].as_mut().unwrap().ty = ty;
                if depth == 1 {
                    sum += 1
                } else {
                    sum += perft(&promotion_copy, depth - 1);
                }
            }
        }
        if depth == 1 {
            sum += 1
        } else {
            sum += perft(&copy, depth - 1);
        }
    }
    sum
}

fn fen(fen: &str) -> Board {
    let mut board = Board::new(8, 8);
    let space_split: Vec<&str> = fen.split(' ').collect();
    assert!(space_split.len() == 6 || space_split.len() == 4);

    {
        let slash_split: Vec<&str> = space_split[0].split('/').collect();
        assert_eq!(slash_split.len(), 8);
        let mut y = 7;
        for s in slash_split {
            let mut x = 0;
            for c in s.chars() {
                if let Some(n) = c.to_digit(10) {
                    x += n as i8;
                } else {
                    let (player, ty) = match c {
                        'P' => (White, Pawn),
                        'p' => (Black, Pawn),
                        'N' => (White, Knight),
                        'n' => (Black, Knight),
                        'B' => (White, Bishop),
                        'b' => (Black, Bishop),
                        'R' => (White, Rook),
                        'r' => (Black, Rook),
                        'Q' => (White, Queen),
                        'q' => (Black, Queen),
                        'K' => (White, King),
                        'k' => (Black, King),
                        _ => panic!(),
                    };
                    board.add_piece(Coord::new(x, y), Piece { player, ty });
                    x += 1;
                }
            }
            y -= 1;
        }
    }

    {
        let turn = space_split[1];
        assert!(turn == "w" || turn == "b");
        board.player_turn = if turn == "w" { White } else { Black };
    }

    {
        let castling = space_split[2];
        if castling != "-" {
            assert!(castling.len() > 0);
            assert!(castling.len() <= 4);
            for c in castling.chars() {
                let x = match c.to_lowercase().next().unwrap() {
                    'a' | 'q' => 0,
                    'b' => 1,
                    'c' => 2,
                    'd' => 3,
                    'e' => 4,
                    'f' => 5,
                    'g' => 6,
                    'h' | 'k' => 7,
                    _ => panic!(),
                };
                let player = if c.is_uppercase() { White } else { Black };
                let king_coord = king_coord(&board, player);
                let y = if player == White { 0 } else { 7 };
                let color_idx = if player == White { 0 } else { 2 };
                let idx = color_idx + if x < king_coord.x { 0 } else { 1 };
                board.castling_rights[idx] = Some(Coord::new(x, y));
            }
        }
    }

    {
        let en_passant = space_split[3];
        if en_passant != "-" {
            assert_eq!(en_passant.len(), 2);
            let x = en_passant.chars().nth(0).unwrap() as i32 - 'a' as i32;
            assert!(x >= 0 && x < 8);
            let rank = en_passant.chars().nth(1).unwrap().to_digit(10).unwrap();
            assert!(rank == 3 || rank == 6);
            let y = if rank == 3 { 3 } else { 4 };
            board.last_pawn_double_move = Some(Coord::new(x as i8, y));
        }
    }

    if space_split.len() == 6 {
        assert!(space_split[4].parse::<i32>().is_ok());
        assert!(space_split[5].parse::<i32>().is_ok());
    }

    board
}

#[test]
fn test_fen() {
    {
        let board = fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let classical = Board::classical();
        for y in 0..board.height {
            for x in 0..board.width {
                let coord = Coord::new(x, y);
                assert_eq!(board[coord], classical[coord]);

                assert_eq!(
                    board.castling_rights,
                    [
                        Some(Coord::new(0, 0)),
                        Some(Coord::new(7, 0)),
                        Some(Coord::new(0, 7)),
                        Some(Coord::new(7, 7))
                    ]
                );
            }
        }
        assert_eq!(board.player_turn, White);
        assert!(board.last_pawn_double_move.is_none());
    }
    {
        let board = fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 b kq - 0 1");
        assert_eq!(
            board[(0, 2)],
            Some(Piece {
                player: Black,
                ty: Queen
            })
        );
        assert_eq!(board[(1, 2)], None);
        assert_eq!(
            board.castling_rights,
            [None, None, Some(Coord::new(0, 7)), Some(Coord::new(7, 7))]
        );
        assert_eq!(board.player_turn, Black);
        assert!(board.last_pawn_double_move.is_none());
    }
    {
        let board = fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 b qk - 0 1");
        assert_eq!(
            board.castling_rights,
            [None, None, Some(Coord::new(0, 7)), Some(Coord::new(7, 7))]
        );
    }
    {
        let board = fen("1r2k1r1/8/8/8/8/8/8/1R2K1R1 b Bg - 0 1");
        assert_eq!(
            board.castling_rights,
            [Some(Coord::new(1, 0)), None, None, Some(Coord::new(6, 7))]
        );
    }
    {
        let board = fen("1r2k1r1/8/8/8/8/8/8/1R2K1R1 b bG - 0 1");
        assert_eq!(
            board.castling_rights,
            [None, Some(Coord::new(6, 0)), Some(Coord::new(1, 7)), None]
        );
    }
    {
        let board = fen("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2");
        assert_eq!(board.last_pawn_double_move, Some(Coord::new(2, 4)));
    }
}

#[test]
fn classical_1() {
    let board = Board::classical();
    assert_eq!(perft(&board, 1), 20);
    assert_eq!(perft(&board, 2), 400);
    assert_eq!(perft(&board, 3), 8902);
    assert_eq!(perft(&board, 4), 197281);
}

#[test]
fn classical_2() {
    let board = fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -");
    assert_eq!(perft(&board, 1), 48);
    assert_eq!(perft(&board, 2), 2039);
    assert_eq!(perft(&board, 3), 97862);
}

#[test]
fn classical_3() {
    let board = fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -");
    assert_eq!(perft(&board, 1), 14);
    assert_eq!(perft(&board, 2), 191);
    assert_eq!(perft(&board, 3), 2812);
    assert_eq!(perft(&board, 4), 43238);
}

#[test]
fn classical_4() {
    let board = fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
    assert_eq!(perft(&board, 1), 44);
    assert_eq!(perft(&board, 2), 1486);
    assert_eq!(perft(&board, 3), 62379);
}

#[test]
fn chess960_1() {
    let board = fen("bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9");
    assert_eq!(perft(&board, 1), 21);
    assert_eq!(perft(&board, 2), 528);
    assert_eq!(perft(&board, 3), 12189);
}

#[test]
fn chess960_2() {
    let board = fen("2nnrbkr/p1qppppp/8/1ppb4/6PP/3PP3/PPP2P2/BQNNRBKR w HEhe - 1 9");
    assert_eq!(perft(&board, 1), 21);
    assert_eq!(perft(&board, 2), 807);
    assert_eq!(perft(&board, 3), 18002);
}

#[test]
fn chess960_3() {
    let board = fen("b1q1rrkb/pppppppp/3nn3/8/P7/1PPP4/4PPPP/BQNNRKRB w GE - 1 9");
    assert_eq!(perft(&board, 1), 20);
    assert_eq!(perft(&board, 2), 479);
    assert_eq!(perft(&board, 3), 10471);
}

#[test]
fn chess960_4() {
    let board = fen("qbbnnrkr/2pp2pp/p7/1p2pp2/8/P3PP2/1PPP1KPP/QBBNNR1R w hf - 0 9");
    assert_eq!(perft(&board, 1), 22);
    assert_eq!(perft(&board, 2), 593);
    assert_eq!(perft(&board, 3), 13440);
}
