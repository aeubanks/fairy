use fairy::board::{Board, BoardPiece, BoardSquare};
use fairy::coord::Coord;
use fairy::moves::all_moves;
use fairy::perft::is_in_check;
use fairy::piece::{Piece, Type, Type::*};
use fairy::player::{next_player, Player, Player::*};
use rand::{thread_rng, Rng};

fn valid_piece_for_coord(piece: Piece, coord: Coord, height: i8) -> bool {
    match piece.ty() {
        Pawn => coord.y != 0 && coord.y != height - 1,
        _ => true,
    }
}

fn add_piece_to_rand_coord<R: Rng + ?Sized, T: Board>(rng: &mut R, board: &mut T, piece: Piece) {
    loop {
        let coord = Coord::new(
            rng.gen_range(0..board.width() as i8),
            rng.gen_range(0..board.height() as i8),
        );
        if board.get(coord).is_some() {
            continue;
        }
        if !valid_piece_for_coord(piece, coord, board.height() as i8) {
            continue;
        }
        board.add_piece(coord, piece);
        return;
    }
}

fn rand_non_king_type<R: Rng + ?Sized>(rng: &mut R) -> Type {
    loop {
        match rng.gen::<Type>() {
            King => {}
            t => return t,
        }
    }
}

fn rand_non_king_type_for_coord<R: Rng + ?Sized>(rng: &mut R, coord: Coord, height: i8) -> Type {
    loop {
        match rand_non_king_type(rng) {
            Pawn => {
                if coord.y != 0 && coord.y != height - 1 {
                    return Pawn;
                }
            }
            t => return t,
        }
    }
}

fn rand_board_piece<const W: i8, const H: i8, R: Rng + ?Sized>(rng: &mut R) -> BoardPiece<W, H, 4> {
    let mut board = BoardPiece::<W, H, 4>::default();
    for player in [White, Black] {
        add_piece_to_rand_coord(rng, &mut board, Piece::new(player, King));
    }

    for _ in 0..2 {
        let piece = Piece::new(rng.gen::<Player>(), rand_non_king_type(rng));
        add_piece_to_rand_coord(rng, &mut board, piece);
    }

    board
}

fn rand_board_square<const W: usize, const H: usize, R: Rng + ?Sized>(
    rng: &mut R,
) -> BoardSquare<W, H> {
    let mut board = BoardSquare::<W, H>::default();
    for player in [White, Black] {
        add_piece_to_rand_coord(rng, &mut board, Piece::new(player, King));
    }

    if rng.gen() {
        for y in 0..H as i8 {
            for x in 0..W as i8 {
                let coord = Coord::new(x, y);
                if board.get(coord).is_some() {
                    continue;
                }
                if rng.gen_bool(1.0 / 8.0) {
                    let piece = Piece::new(
                        rng.gen::<Player>(),
                        rand_non_king_type_for_coord(rng, coord, H as i8),
                    );
                    board.add_piece(coord, piece);
                }
            }
        }
    } else {
        for player in [White, Black] {
            for _ in 0..(rng.gen_range(5..10)) {
                let piece = Piece::new(player, rand_non_king_type(rng));
                add_piece_to_rand_coord(rng, &mut board, piece);
            }
        }
    }
    board
}

fn check_player_is_in_check<T: Board>(board: &T, player: Player) {
    let king_coord = board.king_coord(player);
    let is_check = all_moves(board, next_player(player))
        .into_iter()
        .any(|om| om.to == king_coord);
    if is_in_check(board, player) != is_check {
        println!("{:?}", board);
        println!("{:?}", player);
        panic!("is_in_check mismatch");
    }
}

fn check_is_in_check<T: Board>(board: &T) {
    check_player_is_in_check(board, White);
    check_player_is_in_check(board, Black);
}

#[test]
fn fuzz_is_in_check() {
    fn test_square<const W: usize, const H: usize>() {
        let mut rng = thread_rng();
        let board = rand_board_square::<W, H, _>(&mut rng);
        check_is_in_check(&board);
    }
    fn test_piece<const W: i8, const H: i8>() {
        let mut rng = thread_rng();
        let board = rand_board_piece::<W, H, _>(&mut rng);
        check_is_in_check(&board);
    }
    for _ in 0..1000 {
        test_square::<7, 7>();
        test_square::<7, 8>();
        test_square::<8, 7>();
        test_square::<8, 8>();

        test_piece::<7, 7>();
        test_piece::<7, 6>();
        test_piece::<6, 7>();
        test_piece::<6, 6>();
    }
}
