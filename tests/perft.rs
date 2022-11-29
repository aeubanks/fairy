use fairy::board::Board;
use fairy::coord::Coord;
use fairy::moves::*;
use fairy::piece::{Piece, Type, Type::*};

fn king_coord(board: &Board, player: u8) -> Coord {
    for y in 0..board.height {
        for x in 0..board.width {
            let coord = Coord::new(x, y);
            if let Some(piece) = board[coord].as_ref() {
                if piece.player == player && piece.ty == Type::King {
                    return coord;
                }
            }
        }
    }
    panic!()
}

fn is_in_check(board: &Board, player: u8) -> bool {
    let king_coord = king_coord(board, player);
    let is_check_1 = is_under_attack(board, king_coord, player);
    let is_check_2 = all_moves(board).into_iter().any(|om| om.to == king_coord);
    // TODO: do more fuzz testing of checks?
    assert_eq!(is_check_1, is_check_2);
    is_check_1
}

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
        if depth == 1 {
            sum += 1
        } else {
            sum += perft(&copy, depth - 1);
        }
    }
    sum
}

fn fen(fen: &str) -> Board {
    let mut board = Board::new_with_castling(8, 8);
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
                        'P' => (0, Pawn),
                        'p' => (1, Pawn),
                        'N' => (0, Knight),
                        'n' => (1, Knight),
                        'B' => (0, Bishop),
                        'b' => (1, Bishop),
                        'R' => (0, Rook),
                        'r' => (1, Rook),
                        'Q' => (0, Queen),
                        'q' => (1, Queen),
                        'K' => (0, King),
                        'k' => (1, King),
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
        board.player_turn = if turn == "w" { 0 } else { 1 };
    }

    {
        let castling = space_split[2];
        let white_king = king_coord(&board, 0);
        let black_king = king_coord(&board, 1);
        if castling == "-" {
            board.set_moved(white_king);
            board.set_moved(black_king);
        } else {
            assert!(castling.len() > 0);
            assert!(castling.len() <= 4);
            assert!(castling
                .chars()
                .into_iter()
                .all(|c| c == 'K' || c == 'Q' || c == 'k' || c == 'q'));
            for (c, rook_coord) in [
                ('K', Coord::new(7, white_king.y)),
                ('Q', Coord::new(0, white_king.y)),
                ('k', Coord::new(7, black_king.y)),
                ('q', Coord::new(0, black_king.y)),
            ] {
                if !castling.chars().into_iter().any(|cc| cc == c) {
                    board.set_moved(rook_coord);
                }
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

                assert!(!board.get_moved(coord));
            }
        }
        assert_eq!(board.player_turn, 0);
        assert!(board.last_pawn_double_move.is_none());
    }
    {
        let board = fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 b kq - 0 1");
        assert_eq!(
            board[(0, 2)],
            Some(Piece {
                player: 1,
                ty: Queen
            })
        );
        assert_eq!(board[(1, 2)], None);
        assert!(board.get_moved(Coord::new(0, 0)));
        assert!(board.get_moved(Coord::new(7, 0)));
        assert!(!board.get_moved(Coord::new(0, 7)));
        assert!(!board.get_moved(Coord::new(7, 7)));
        assert_eq!(board.player_turn, 1);
        assert!(board.last_pawn_double_move.is_none());
    }
    {
        let board = fen("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2");
        assert_eq!(board.last_pawn_double_move, Some(Coord::new(2, 4)));
    }
}

#[test]
fn classical() {
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
