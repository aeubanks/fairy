use crate::board::Board;
use crate::coord::Coord;
use crate::moves::*;
use crate::piece::{Piece, Type::*};
use crate::player::{next_player, Player, Player::*};

pub fn is_in_check<const W: usize, const H: usize>(board: &Board<W, H>, player: Player) -> bool {
    is_under_attack(board, board.king_coord(player), player)
}

fn perft_impl<const W: usize, const H: usize>(
    board: &Board<W, H>,
    player: Player,
    depth: u64,
) -> u64 {
    assert_ne!(depth, 0);
    let moves = all_moves(board, player);
    let mut sum = 0;
    for m in moves {
        let mut copy = board.clone();
        copy.make_move(m, player);
        if is_in_check(&copy, player) {
            continue;
        }
        let next_player = next_player(player);
        if board[m.from].unwrap().ty() == Pawn && (m.to.y == 0 || m.to.y == H as i8 - 1) {
            for ty in [Knight, Bishop, Rook] {
                let mut promotion_copy = copy.clone();
                promotion_copy.clear(m.to);
                promotion_copy.add_piece(m.to, Piece::new(player, ty));
                if depth == 1 {
                    sum += 1
                } else {
                    sum += perft_impl(&promotion_copy, next_player, depth - 1);
                }
            }
        }
        if depth == 1 {
            sum += 1
        } else {
            sum += perft_impl(&copy, next_player, depth - 1);
        }
    }
    sum
}

pub struct Position<const W: usize, const H: usize> {
    pub board: Board<W, H>,
    pub player: Player,
}

pub fn perft<const W: usize, const H: usize>(position: &Position<W, H>, depth: u64) -> u64 {
    perft_impl(&position.board, position.player, depth)
}

fn perft_all_impl<const W: usize, const H: usize>(
    board: &Board<W, H>,
    player: Player,
    depth: u64,
) -> u64 {
    assert_ne!(depth, 0);
    let moves = all_moves(board, player);
    let mut sum = 0;
    for m in moves {
        if let Some(p) = board[m.to] {
            if p.ty() == King {
                continue;
            }
        }
        let mut copy = board.clone();
        copy.make_move(m, player);
        if depth == 1 {
            sum += 1
        } else {
            sum += perft_all_impl(&copy, next_player(player), depth - 1);
        }
    }
    sum
}

pub fn perft_all<const W: usize, const H: usize>(position: &Position<W, H>, depth: u64) -> u64 {
    perft_all_impl(&position.board, position.player, depth)
}

pub fn fen(fen: &str) -> Position<8, 8> {
    let mut board = Board::<8, 8>::default();
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
                    board.add_piece(Coord::new(x, y), Piece::new(player, ty));
                    x += 1;
                }
            }
            y -= 1;
        }
    }

    let player = {
        let turn = space_split[1];
        assert!(turn == "w" || turn == "b");
        if turn == "w" {
            White
        } else {
            Black
        }
    };

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
                let king_coord = board.king_coord(player);
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
            let x = en_passant.chars().next().unwrap() as i32 - 'a' as i32;
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

    Position { board, player }
}

#[test]
fn test_fen() {
    {
        let Position { board, player } =
            fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let classical = crate::board::Presets::classical();
        for y in 0..8 {
            for x in 0..8 {
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
        assert_eq!(player, White);
        assert!(board.last_pawn_double_move.is_none());
    }
    {
        let Position { board, player } =
            fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 b kq - 0 1");
        assert_eq!(board[(0, 2)], Some(Piece::new(Black, Queen)));
        assert_eq!(board[(1, 2)], None);
        assert_eq!(
            board.castling_rights,
            [None, None, Some(Coord::new(0, 7)), Some(Coord::new(7, 7))]
        );
        assert_eq!(player, Black);
        assert!(board.last_pawn_double_move.is_none());
    }
    {
        let Position { board, player: _ } =
            fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 b qk - 0 1");
        assert_eq!(
            board.castling_rights,
            [None, None, Some(Coord::new(0, 7)), Some(Coord::new(7, 7))]
        );
    }
    {
        let Position { board, player: _ } = fen("1r2k1r1/8/8/8/8/8/8/1R2K1R1 b Bg - 0 1");
        assert_eq!(
            board.castling_rights,
            [Some(Coord::new(1, 0)), None, None, Some(Coord::new(6, 7))]
        );
    }
    {
        let Position { board, player: _ } = fen("1r2k1r1/8/8/8/8/8/8/1R2K1R1 b bG - 0 1");
        assert_eq!(
            board.castling_rights,
            [None, Some(Coord::new(6, 0)), Some(Coord::new(1, 7)), None]
        );
    }
    {
        let Position { board, player: _ } =
            fen("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2");
        assert_eq!(board.last_pawn_double_move, Some(Coord::new(2, 4)));
    }
}
