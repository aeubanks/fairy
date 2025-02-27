use crate::board::{Board, CastleSide, ExistingPieceResult, Move};
use crate::coord::Coord;
use crate::piece::{Piece, Type, Type::*};
use crate::player::{Player, Player::*};
use derive_enum::EnumFrom;

fn add_move_if_result<T: Board>(
    moves: &mut Vec<Coord>,
    board: &T,
    coord: Coord,
    player: Player,
    result: ExistingPieceResult,
) -> bool {
    if board.existing_piece_result(coord, player) == result {
        moves.push(coord);
        return true;
    }
    false
}

fn add_move_if_in_bounds_and_result<T: Board>(
    moves: &mut Vec<Coord>,
    board: &T,
    coord: Coord,
    player: Player,
    result: ExistingPieceResult,
) {
    if board.in_bounds(coord) {
        add_move_if_result(moves, board, coord, player, result);
    }
}

fn add_moves_for_rider<T: Board>(
    moves: &mut Vec<Coord>,
    board: &T,
    coord: Coord,
    player: Player,
    offset: Coord,
) {
    let mut try_coord = coord + offset;
    while board.in_bounds(try_coord) {
        match board.existing_piece_result(try_coord, player) {
            ExistingPieceResult::Empty => {
                moves.push(try_coord);
            }
            ExistingPieceResult::Friend => {
                break;
            }
            ExistingPieceResult::Opponent => {
                moves.push(try_coord);
                break;
            }
        }
        try_coord = try_coord + offset;
    }
}

fn add_moves_for_leaper<T: Board>(
    moves: &mut Vec<Coord>,
    board: &T,
    coord: Coord,
    player: Player,
    offset: Coord,
) {
    let try_coord = coord + offset;
    if board.in_bounds(try_coord) {
        match board.existing_piece_result(try_coord, player) {
            ExistingPieceResult::Empty | ExistingPieceResult::Opponent => {
                moves.push(try_coord);
            }
            ExistingPieceResult::Friend => {}
        }
    }
}

fn add_castling_moves<T: Board>(moves: &mut Vec<Coord>, board: &T, coord: Coord, player: Player) {
    // castling

    fn any_pieces_between_inc<T: Board>(
        board: &T,
        player: Player,
        y: i8,
        x1: i8,
        x2: i8,
        ignore_x_1: i8,
        ignore_x_2: i8,
    ) -> bool {
        let min_x = x1.min(x2);
        let max_x = x1.max(x2);
        (min_x..=max_x).any(|x| {
            board.existing_piece_result(Coord::new(x, y), player) != ExistingPieceResult::Empty
                && ignore_x_1 != x
                && ignore_x_2 != x
        })
    }

    fn any_checks_between_inc<T: Board>(board: &T, player: Player, y: i8, x1: i8, x2: i8) -> bool {
        let min_x = x1.min(x2);
        let max_x = x1.max(x2);
        (min_x..=max_x).any(|x| is_under_attack(board, Coord::new(x, y), player))
    }

    // make sure the first value is the left rook and the second value is the right rook
    #[cfg(debug_assertions)]
    if let Some(a) = board.get_castling_rights(player, CastleSide::Left) {
        if let Some(b) = board.get_castling_rights(player, CastleSide::Right) {
            debug_assert!(a.x < b.x);
        }
    }
    for (rook_coord, king_dest_x, rook_dest_x) in [
        (board.get_castling_rights(player, CastleSide::Left), 2, 3),
        (
            board.get_castling_rights(player, CastleSide::Right),
            board.width() - 2,
            board.width() - 3,
        ),
    ] {
        if let Some(rook_coord) = rook_coord {
            {
                assert_eq!(coord.y, rook_coord.y);
                let piece = board.get(rook_coord).unwrap();
                assert_eq!(piece.ty(), Rook);
                assert_eq!(piece.player(), player);
            }

            if !any_pieces_between_inc(
                board,
                player,
                coord.y,
                coord.x,
                king_dest_x,
                coord.x,
                rook_coord.x,
            ) && !any_pieces_between_inc(
                board,
                player,
                coord.y,
                rook_coord.x,
                rook_dest_x,
                coord.x,
                rook_coord.x,
            ) && !any_checks_between_inc(board, player, coord.y, king_dest_x, coord.x)
            {
                moves.push(rook_coord);
            }
        }
    }
}

fn add_pawn_moves<T: Board>(moves: &mut Vec<Coord>, board: &T, coord: Coord, player: Player) {
    let dy = match player {
        White => 1,
        Black => -1,
    };

    let left = coord + Coord::new(-1, dy);
    add_move_if_in_bounds_and_result(moves, board, left, player, ExistingPieceResult::Opponent);
    let right = coord + Coord::new(1, dy);
    add_move_if_in_bounds_and_result(moves, board, right, player, ExistingPieceResult::Opponent);

    let front = coord + Coord::new(0, dy);
    let front_empty = add_move_if_result(moves, board, front, player, ExistingPieceResult::Empty);

    if front_empty {
        let initial_y = match player {
            White => 1,
            Black => board.height() - 2,
        };
        if coord.y == initial_y {
            let max_forward = ((board.height() - 1) / 2 - 1).max(1);
            let mut front2 = front;
            for _ in 1..max_forward {
                front2 = front2 + Coord::new(0, dy);
                let empty =
                    add_move_if_result(moves, board, front2, player, ExistingPieceResult::Empty);
                if !empty {
                    break;
                }
            }
        }
    }

    if let Some(p) = board.get_last_pawn_double_move() {
        if p.y == coord.y && (p.x - coord.x).abs() == 1 {
            moves.push(p + Coord::new(0, dy));
        }
    }
}

fn add_moves_for_piece<T: Board>(moves: &mut Vec<Coord>, board: &T, piece: Piece, coord: Coord) {
    match piece.ty() {
        Pawn => add_pawn_moves(moves, board, coord, piece.player()),
        ty => {
            for l in ty.leaper_offsets() {
                add_moves_for_leaper(moves, board, coord, piece.player(), l);
            }
            for r in ty.rider_offsets() {
                add_moves_for_rider(moves, board, coord, piece.player(), r);
            }
        }
    }
    if piece.ty() == King {
        add_castling_moves(moves, board, coord, piece.player());
    }
}

#[must_use]
pub fn all_moves_for_piece<T: Board>(board: &T, piece: Piece, coord: Coord) -> Vec<Coord> {
    let mut moves = Vec::with_capacity(16);
    add_moves_for_piece(&mut moves, board, piece, coord);
    moves
}

#[must_use]
pub fn all_legal_moves<T: Board>(board: &T, player: Player) -> Vec<Move> {
    let mut moves = Vec::with_capacity(32);

    board.foreach_piece(|piece, coord| {
        if piece.player() == player {
            let mut piece_moves = Vec::with_capacity(16);
            add_moves_for_piece(&mut piece_moves, board, piece, coord);
            moves.append(
                &mut piece_moves
                    .iter()
                    .map(|c| Move {
                        from: coord,
                        to: *c,
                    })
                    .collect(),
            );
        }
    });

    moves
}

fn enemy_piece_rider<T: Board>(
    board: &T,
    coord: Coord,
    offset: Coord,
    player: Player,
) -> Option<(Coord, Type)> {
    let mut try_coord = coord + offset;
    while board.in_bounds(try_coord) {
        if let Some(p) = board.get(try_coord) {
            if p.player() != player {
                return Some((try_coord, p.ty()));
            } else {
                return None;
            }
        }
        try_coord = try_coord + offset;
    }
    None
}

fn enemy_piece_leaper<T: Board>(
    board: &T,
    coord: Coord,
    offset: Coord,
    player: Player,
) -> Option<(Coord, Type)> {
    let try_coord = coord + offset;
    if board.in_bounds(try_coord) {
        if let Some(p) = board.get(try_coord) {
            if p.player() != player {
                return Some((try_coord, p.ty()));
            } else {
                return None;
            }
        }
    }
    None
}

pub fn is_under_attack<T: Board>(board: &T, coord: Coord, player: Player) -> bool {
    under_attack_from_coord(board, coord, player).is_some()
}

pub fn under_attack_from_coord<T: Board>(board: &T, coord: Coord, player: Player) -> Option<Coord> {
    if let Some(p) = board.get(coord) {
        assert_eq!(p.player(), player);
    }

    let pts = board.piece_types_of_player(player.next());
    for ty in Type::all() {
        if !pts[ty as usize] {
            continue;
        }
        if ty == Pawn {
            fn has_enemy_pawn<T: Board>(board: &T, coord: Coord, player: Player) -> bool {
                if board.in_bounds(coord) {
                    if let Some(p) = board.get(coord) {
                        return p.player() != player && p.ty() == Pawn;
                    }
                }
                false
            }
            if player == White {
                for try_coord in [coord + Coord::new(1, 1), coord + Coord::new(-1, 1)] {
                    if has_enemy_pawn(board, try_coord, player) {
                        return Some(try_coord);
                    }
                }
            } else {
                for try_coord in [coord + Coord::new(1, -1), coord + Coord::new(-1, -1)] {
                    if has_enemy_pawn(board, try_coord, player) {
                        return Some(try_coord);
                    }
                }
            }
            continue;
        } else {
            for o in ty.rider_offsets() {
                if let Some((c, found_ty)) = enemy_piece_rider(board, coord, o, player) {
                    if ty == found_ty {
                        return Some(c);
                    }
                }
            }
            for o in ty.leaper_offsets() {
                if let Some((c, found_ty)) = enemy_piece_leaper(board, coord, o, player) {
                    if ty == found_ty {
                        return Some(c);
                    }
                }
            }
        }
    }
    None
}

fn add_moves_for_rider_to_end_at_board_no_captures<T: Board>(
    moves: &mut Vec<Coord>,
    board: &T,
    coord: Coord,
    offset: Coord,
) {
    let mut try_coord = coord + offset;
    while board.in_bounds(try_coord) && board.get(try_coord).is_none() {
        moves.push(try_coord);
        try_coord = try_coord + offset;
    }
}

fn add_moves_for_leaper_to_end_at_board_no_captures<T: Board>(
    moves: &mut Vec<Coord>,
    board: &T,
    coord: Coord,
    offset: Coord,
) {
    let try_coord = coord + offset;
    if board.in_bounds(try_coord) && board.get(try_coord).is_none() {
        moves.push(try_coord);
    }
}

fn add_moves_for_non_pawn_piece_end_at_board_no_captures<T: Board>(
    moves: &mut Vec<Coord>,
    board: &T,
    piece: Piece,
    coord: Coord,
) {
    let ty = piece.ty();
    for l in ty.leaper_offsets() {
        add_moves_for_leaper_to_end_at_board_no_captures(moves, board, coord, l);
    }
    for r in ty.rider_offsets() {
        add_moves_for_rider_to_end_at_board_no_captures(moves, board, coord, r);
    }
}

pub fn all_legal_moves_to_end_at_board_captures<T: Board>(
    board: &T,
    piece: Piece,
    coord: Coord,
) -> Vec<Coord> {
    let mut moves = Vec::new();
    if piece.ty() == Pawn {
        let (dy, start_y) = match piece.player() {
            White => (1, 1),
            Black => (-1, board.height() - 2),
        };

        // FIXME: en passant
        for dx in [-1, 1] {
            let try_coord = coord + Coord::new(dx, -dy);
            if coord.y != start_y && board.in_bounds(try_coord) && board.get(try_coord).is_none() {
                moves.push(try_coord);
            }
        }
    } else {
        add_moves_for_non_pawn_piece_end_at_board_no_captures(&mut moves, board, piece, coord);
    }
    #[cfg(debug_assertions)]
    moves.iter().for_each(|&m| {
        assert!(board.get(m).is_none());
    });
    moves
}

#[must_use]
pub fn all_legal_moves_to_end_at_board_no_captures<T: Board>(
    board: &T,
    piece: Piece,
    coord: Coord,
) -> Vec<Coord> {
    let mut moves = Vec::with_capacity(16);
    if piece.ty() == Pawn {
        let (dy, start_y) = match piece.player() {
            White => (1, 1),
            Black => (-1, board.height() - 2),
        };

        let mut try_coord = coord + Coord::new(0, -dy);
        if coord.y != start_y && board.get(try_coord).is_none() {
            moves.push(try_coord);
            if try_coord.y != start_y {
                if match piece.player() {
                    White => coord.y <= (board.height() - 1) / 2,
                    Black => coord.y >= (board.height()) / 2,
                } {
                    let mut blocked = false;
                    while try_coord.y != start_y - dy {
                        if board.get(try_coord).is_some() {
                            blocked = true;
                            break;
                        }
                        try_coord.y -= dy;
                    }
                    if !blocked {
                        moves.push(Coord::new(coord.x, start_y));
                    }
                }
            }
        }
    } else {
        add_moves_for_non_pawn_piece_end_at_board_no_captures(&mut moves, board, piece, coord);
    }
    #[cfg(debug_assertions)]
    moves.iter().for_each(|&m| {
        assert!(board.get(m).is_none());
    });
    moves
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{BoardPiece, BoardSquare};
    use rand::{thread_rng, Rng};

    fn is_in_check<T: Board>(board: &T, player: Player) -> bool {
        is_under_attack(board, board.king_coord(player), player)
    }

    fn assert_moves_eq(expected: &[Coord], moves: Vec<Coord>) {
        use rustc_hash::FxHashSet;

        let mut set = FxHashSet::default();
        for e in expected {
            assert!(set.insert(*e), "duplicate expected");
        }
        let mut found = FxHashSet::default();
        for m in moves {
            assert!(found.insert(m), "duplicate move");
        }
        assert_eq!(set, found);
    }
    #[test]
    fn test_rook() {
        let mut board = BoardSquare::<4, 4>::default();
        {
            let mut moves = Vec::new();
            add_moves_for_piece(
                &mut moves,
                &board,
                Piece::new(White, Rook),
                Coord::new(1, 2),
            );
            assert_moves_eq(
                &[
                    Coord::new(0, 2),
                    Coord::new(1, 1),
                    Coord::new(1, 0),
                    Coord::new(2, 2),
                    Coord::new(3, 2),
                    Coord::new(1, 3),
                ],
                moves,
            );
        }
        board.add_piece(Coord::new(2, 2), Piece::new(White, Rook));
        board.add_piece(Coord::new(1, 1), Piece::new(Black, Rook));
        {
            let mut moves = Vec::new();
            add_moves_for_piece(
                &mut moves,
                &board,
                Piece::new(White, Rook),
                Coord::new(1, 2),
            );
            assert_moves_eq(
                &[Coord::new(0, 2), Coord::new(1, 1), Coord::new(1, 3)],
                moves,
            );
        }
    }

    #[test]
    fn test_bishop() {
        let mut board = BoardSquare::<6, 6>::default();
        {
            let mut moves = Vec::new();
            add_moves_for_piece(
                &mut moves,
                &board,
                Piece::new(White, Bishop),
                Coord::new(1, 2),
            );
            assert_moves_eq(
                &[
                    Coord::new(0, 1),
                    Coord::new(2, 3),
                    Coord::new(3, 4),
                    Coord::new(4, 5),
                    Coord::new(0, 3),
                    Coord::new(2, 1),
                    Coord::new(3, 0),
                ],
                moves,
            );
        }
        board.add_piece(Coord::new(4, 5), Piece::new(White, Bishop));
        board.add_piece(Coord::new(2, 1), Piece::new(Black, Bishop));
        {
            let mut moves = Vec::new();
            add_moves_for_piece(
                &mut moves,
                &board,
                Piece::new(White, Bishop),
                Coord::new(1, 2),
            );
            assert_moves_eq(
                &[
                    Coord::new(0, 1),
                    Coord::new(2, 3),
                    Coord::new(3, 4),
                    Coord::new(0, 3),
                    Coord::new(2, 1),
                ],
                moves,
            );
        }
    }

    #[test]
    fn test_knight() {
        let mut board = BoardSquare::<6, 6>::default();
        {
            let mut moves = Vec::new();
            add_moves_for_piece(
                &mut moves,
                &board,
                Piece::new(White, Knight),
                Coord::new(2, 2),
            );
            assert_moves_eq(
                &[
                    Coord::new(1, 0),
                    Coord::new(1, 4),
                    Coord::new(3, 0),
                    Coord::new(3, 4),
                    Coord::new(0, 1),
                    Coord::new(0, 3),
                    Coord::new(4, 1),
                    Coord::new(4, 3),
                ],
                moves,
            );
        }
        {
            let mut moves = Vec::new();
            add_moves_for_piece(
                &mut moves,
                &board,
                Piece::new(White, Knight),
                Coord::new(0, 0),
            );
            assert_moves_eq(&[Coord::new(1, 2), Coord::new(2, 1)], moves);
        }
        board.add_piece(Coord::new(1, 0), Piece::new(White, Knight));
        board.add_piece(Coord::new(1, 4), Piece::new(Black, Knight));
        {
            let mut moves = Vec::new();
            add_moves_for_piece(
                &mut moves,
                &board,
                Piece::new(White, Knight),
                Coord::new(2, 2),
            );
            assert_moves_eq(
                &[
                    Coord::new(1, 4),
                    Coord::new(3, 0),
                    Coord::new(3, 4),
                    Coord::new(0, 1),
                    Coord::new(0, 3),
                    Coord::new(4, 1),
                    Coord::new(4, 3),
                ],
                moves,
            );
        }
    }
    #[test]
    fn test_king() {
        {
            let board = BoardSquare::<6, 5>::default();
            let mut moves = Vec::new();
            add_moves_for_piece(
                &mut moves,
                &board,
                Piece::new(White, King),
                Coord::new(2, 2),
            );
            assert_moves_eq(
                &[
                    Coord::new(1, 1),
                    Coord::new(1, 2),
                    Coord::new(1, 3),
                    Coord::new(2, 1),
                    Coord::new(2, 3),
                    Coord::new(3, 1),
                    Coord::new(3, 2),
                    Coord::new(3, 3),
                ],
                moves,
            );
        }
        {
            let board = BoardSquare::<6, 5>::default();
            let mut moves = Vec::new();
            add_moves_for_piece(
                &mut moves,
                &board,
                Piece::new(White, King),
                Coord::new(5, 4),
            );
            assert_moves_eq(
                &[Coord::new(4, 3), Coord::new(5, 3), Coord::new(4, 4)],
                moves,
            );
        }
        {
            let mut board = BoardSquare::<6, 5>::default();
            board.add_piece(Coord::new(1, 2), Piece::new(White, Knight));
            board.add_piece(Coord::new(1, 3), Piece::new(Black, Knight));
            let mut moves = Vec::new();
            add_moves_for_piece(
                &mut moves,
                &board,
                Piece::new(White, King),
                Coord::new(2, 2),
            );
            assert_moves_eq(
                &[
                    Coord::new(1, 1),
                    Coord::new(1, 3),
                    Coord::new(2, 1),
                    Coord::new(2, 3),
                    Coord::new(3, 1),
                    Coord::new(3, 2),
                    Coord::new(3, 3),
                ],
                moves,
            );
        }
        {
            let mut board = BoardSquare::<8, 8>::default();
            board.add_piece(Coord::new(0, 0), Piece::new(White, Rook));
            board.add_piece(Coord::new(7, 0), Piece::new(White, Rook));
            {
                let mut moves = Vec::new();
                add_castling_moves(&mut moves, &board, Coord::new(4, 0), White);
                assert_moves_eq(&[], moves);
            }
        }
        {
            let mut board = BoardSquare::<8, 8>::default();
            board.castling_rights = [
                Some(Coord::new(0, 0)),
                Some(Coord::new(7, 0)),
                Some(Coord::new(0, 7)),
                Some(Coord::new(7, 7)),
            ];
            board.add_piece(Coord::new(0, 0), Piece::new(White, Rook));
            board.add_piece(Coord::new(7, 0), Piece::new(White, Rook));
            {
                let mut moves = Vec::new();
                add_castling_moves(&mut moves, &board, Coord::new(4, 0), White);
                assert_moves_eq(&[Coord::new(0, 0), Coord::new(7, 0)], moves);
            }
            {
                let mut board2 = board.clone();
                board2.add_piece(Coord::new(4, 2), Piece::new(Black, Rook));
                let mut moves = Vec::new();
                add_castling_moves(&mut moves, &board2, Coord::new(4, 0), White);
                assert_moves_eq(&[], moves);
            }
            {
                let mut board2 = board.clone();
                board2.add_piece(Coord::new(3, 2), Piece::new(Black, Rook));
                let mut moves = Vec::new();
                add_castling_moves(&mut moves, &board2, Coord::new(4, 0), White);
                assert_moves_eq(&[Coord::new(7, 0)], moves);
            }
            {
                let mut board2 = board.clone();
                board2.add_piece(Coord::new(6, 2), Piece::new(Black, Rook));
                let mut moves = Vec::new();
                add_castling_moves(&mut moves, &board2, Coord::new(4, 0), White);
                assert_moves_eq(&[Coord::new(0, 0)], moves);
            }
            {
                let mut board2 = board.clone();
                let mut moves = Vec::new();
                board2.add_piece(Coord::new(1, 0), Piece::new(White, Knight));
                board2.add_piece(Coord::new(6, 0), Piece::new(Black, Knight));
                add_castling_moves(&mut moves, &board2, Coord::new(4, 0), White);
                assert_moves_eq(&[], moves);
            }
            {
                board.castling_rights[0] = None;
                let mut moves = Vec::new();
                add_castling_moves(&mut moves, &board, Coord::new(4, 0), White);
                assert_moves_eq(&[Coord::new(7, 0)], moves);
            }
            {
                board.castling_rights[1] = None;
                let mut moves = Vec::new();
                add_castling_moves(&mut moves, &board, Coord::new(4, 0), White);
                assert_moves_eq(&[], moves);
            }
        }
        {
            let mut board = BoardSquare::<8, 8>::with_pieces(&[
                (Coord::new(0, 0), Piece::new(White, Rook)),
                (Coord::new(1, 0), Piece::new(White, King)),
                (Coord::new(7, 0), Piece::new(White, Rook)),
            ]);
            board.castling_rights = [Some(Coord::new(0, 0)), Some(Coord::new(7, 0)), None, None];
            {
                let mut moves = Vec::new();
                add_castling_moves(&mut moves, &board, Coord::new(1, 0), White);
                assert_moves_eq(&[Coord::new(0, 0), Coord::new(7, 0)], moves);
            }
        }
        {
            let mut board = BoardSquare::<8, 8>::with_pieces(&[
                (Coord::new(0, 0), Piece::new(White, Rook)),
                (Coord::new(6, 0), Piece::new(White, King)),
                (Coord::new(7, 0), Piece::new(White, Rook)),
            ]);
            board.castling_rights = [Some(Coord::new(0, 0)), Some(Coord::new(7, 0)), None, None];
            {
                let mut moves = Vec::new();
                add_castling_moves(&mut moves, &board, Coord::new(6, 0), White);
                assert_moves_eq(&[Coord::new(0, 0), Coord::new(7, 0)], moves);
            }
        }
        {
            let mut board = BoardSquare::<8, 8>::with_pieces(&[
                (Coord::new(1, 0), Piece::new(White, Rook)),
                (Coord::new(3, 0), Piece::new(White, King)),
                (Coord::new(5, 0), Piece::new(White, Rook)),
            ]);
            board.castling_rights = [Some(Coord::new(1, 0)), Some(Coord::new(5, 0)), None, None];
            {
                let mut moves = Vec::new();
                add_castling_moves(&mut moves, &board, Coord::new(3, 0), White);
                assert_moves_eq(&[Coord::new(1, 0), Coord::new(5, 0)], moves);
            }
        }
        {
            let mut board = BoardSquare::<8, 8>::with_pieces(&[
                (Coord::new(1, 0), Piece::new(White, Rook)),
                (Coord::new(5, 0), Piece::new(White, King)),
                (Coord::new(6, 0), Piece::new(White, Rook)),
            ]);
            board.castling_rights = [Some(Coord::new(1, 0)), Some(Coord::new(6, 0)), None, None];
            {
                let mut moves = Vec::new();
                add_castling_moves(&mut moves, &board, Coord::new(5, 0), White);
                assert_moves_eq(&[Coord::new(1, 0), Coord::new(6, 0)], moves);
            }
        }
        {
            let mut board = BoardSquare::<8, 8>::with_pieces(&[
                (Coord::new(0, 0), Piece::new(White, Rook)),
                (Coord::new(1, 0), Piece::new(White, King)),
                (Coord::new(2, 0), Piece::new(White, Rook)),
            ]);
            board.castling_rights = [Some(Coord::new(0, 0)), Some(Coord::new(2, 0)), None, None];
            {
                let mut moves = Vec::new();
                add_castling_moves(&mut moves, &board, Coord::new(1, 0), White);
                assert_moves_eq(&[Coord::new(2, 0)], moves);
            }
        }
        {
            let mut board = BoardSquare::<8, 8>::with_pieces(&[
                (Coord::new(5, 0), Piece::new(White, Rook)),
                (Coord::new(6, 0), Piece::new(White, King)),
                (Coord::new(7, 0), Piece::new(White, Rook)),
            ]);
            board.castling_rights = [Some(Coord::new(5, 0)), Some(Coord::new(7, 0)), None, None];
            {
                let mut moves = Vec::new();
                add_castling_moves(&mut moves, &board, Coord::new(6, 0), White);
                assert_moves_eq(&[Coord::new(5, 0)], moves);
            }
        }
        {
            let mut board = BoardSquare::<8, 8>::with_pieces(&[
                (Coord::new(3, 0), Piece::new(White, Rook)),
                (Coord::new(4, 0), Piece::new(White, King)),
                (Coord::new(5, 0), Piece::new(White, Rook)),
            ]);
            board.castling_rights = [Some(Coord::new(3, 0)), Some(Coord::new(5, 0)), None, None];
            {
                let mut moves = Vec::new();
                add_castling_moves(&mut moves, &board, Coord::new(4, 0), White);
                assert_moves_eq(&[Coord::new(3, 0), Coord::new(5, 0)], moves);
            }
        }
        {
            let mut board = BoardSquare::<8, 8>::with_pieces(&[
                (Coord::new(0, 0), Piece::new(White, Rook)),
                (Coord::new(6, 0), Piece::new(White, King)),
                (Coord::new(7, 0), Piece::new(White, Rook)),
            ]);
            board.castling_rights = [Some(Coord::new(0, 0)), Some(Coord::new(7, 0)), None, None];
            {
                let mut moves = Vec::new();
                add_castling_moves(&mut moves, &board, Coord::new(6, 0), White);
                assert_moves_eq(&[Coord::new(0, 0), Coord::new(7, 0)], moves);
            }
        }
        {
            let mut board = BoardSquare::<8, 8>::with_pieces(&[
                (Coord::new(0, 0), Piece::new(White, Rook)),
                (Coord::new(2, 0), Piece::new(White, King)),
                (Coord::new(7, 0), Piece::new(White, Rook)),
            ]);
            board.castling_rights = [Some(Coord::new(0, 0)), Some(Coord::new(7, 0)), None, None];
            {
                let mut moves = Vec::new();
                add_castling_moves(&mut moves, &board, Coord::new(2, 0), White);
                assert_moves_eq(&[Coord::new(0, 0), Coord::new(7, 0)], moves);
            }
        }
    }
    #[test]
    fn test_pawn() {
        {
            let board = BoardSquare::<8, 8>::default();
            let mut moves = Vec::new();
            add_pawn_moves(&mut moves, &board, Coord::new(2, 3), White);
            assert_moves_eq(&[Coord::new(2, 4)], moves);
        }
        {
            let mut board = BoardSquare::<8, 8>::default();
            board.add_piece(Coord::new(2, 4), Piece::new(White, Rook));
            let mut moves = Vec::new();
            add_pawn_moves(&mut moves, &board, Coord::new(2, 3), White);
            assert_moves_eq(&[], moves);
        }
        {
            let mut board = BoardSquare::<8, 8>::default();
            board.add_piece(Coord::new(2, 4), Piece::new(Black, Rook));
            let mut moves = Vec::new();
            add_pawn_moves(&mut moves, &board, Coord::new(2, 3), White);
            assert_moves_eq(&[], moves);
        }
        {
            let board = BoardSquare::<8, 8>::default();
            let mut moves = Vec::new();
            add_pawn_moves(&mut moves, &board, Coord::new(2, 1), White);
            assert_moves_eq(&[Coord::new(2, 2), Coord::new(2, 3)], moves);
        }
        {
            let mut board = BoardSquare::<8, 8>::default();
            board.add_piece(Coord::new(2, 3), Piece::new(White, Rook));
            let mut moves = Vec::new();
            add_pawn_moves(&mut moves, &board, Coord::new(2, 1), White);
            assert_moves_eq(&[Coord::new(2, 2)], moves);
        }
        {
            let mut board = BoardSquare::<8, 8>::default();
            board.add_piece(Coord::new(2, 2), Piece::new(Black, Rook));
            let mut moves = Vec::new();
            add_pawn_moves(&mut moves, &board, Coord::new(2, 1), White);
            assert_moves_eq(&[], moves);
        }
        {
            let mut board = BoardSquare::<8, 8>::default();
            board.add_piece(Coord::new(3, 3), Piece::new(White, Knight));
            board.add_piece(Coord::new(2, 3), Piece::new(Black, Knight));
            board.add_piece(Coord::new(1, 3), Piece::new(Black, Knight));
            let mut moves = Vec::new();
            add_pawn_moves(&mut moves, &board, Coord::new(2, 2), White);
            assert_moves_eq(&[Coord::new(1, 3)], moves);
        }
        {
            let board = BoardSquare::<8, 8>::default();
            let mut moves = Vec::new();
            add_pawn_moves(&mut moves, &board, Coord::new(2, 2), Black);
            assert_moves_eq(&[Coord::new(2, 1)], moves);
        }
        {
            let board = BoardSquare::<8, 8>::default();
            let mut moves = Vec::new();
            add_pawn_moves(&mut moves, &board, Coord::new(2, 6), Black);
            assert_moves_eq(&[Coord::new(2, 4), Coord::new(2, 5)], moves);
        }
        {
            let board = BoardSquare::<8, 12>::default();
            let mut moves = Vec::new();
            add_pawn_moves(&mut moves, &board, Coord::new(2, 1), White);
            assert_moves_eq(
                &[
                    Coord::new(2, 2),
                    Coord::new(2, 3),
                    Coord::new(2, 4),
                    Coord::new(2, 5),
                ],
                moves,
            );
        }
        {
            let board = BoardSquare::<8, 13>::default();
            let mut moves = Vec::new();
            add_pawn_moves(&mut moves, &board, Coord::new(2, 1), White);
            assert_moves_eq(
                &[
                    Coord::new(2, 2),
                    Coord::new(2, 3),
                    Coord::new(2, 4),
                    Coord::new(2, 5),
                    Coord::new(2, 6),
                ],
                moves,
            );
        }
        {
            let board = BoardSquare::<8, 12>::default();
            let mut moves = Vec::new();
            add_pawn_moves(&mut moves, &board, Coord::new(3, 10), Black);
            assert_moves_eq(
                &[
                    Coord::new(3, 9),
                    Coord::new(3, 8),
                    Coord::new(3, 7),
                    Coord::new(3, 6),
                ],
                moves,
            );
        }
        {
            let mut board = BoardSquare::<8, 12>::default();
            board.add_piece(Coord::new(3, 7), Piece::new(White, Knight));
            let mut moves = Vec::new();
            add_pawn_moves(&mut moves, &board, Coord::new(3, 10), Black);
            assert_moves_eq(&[Coord::new(3, 9), Coord::new(3, 8)], moves);
        }
    }
    #[test]
    fn test_under_attack() {
        {
            let board = BoardSquare::<8, 8>::with_pieces(&[
                (Coord::new(0, 0), Piece::new(White, Rook)),
                (Coord::new(3, 0), Piece::new(White, Pawn)),
                (Coord::new(0, 3), Piece::new(Black, Pawn)),
                (Coord::new(7, 7), Piece::new(Black, Rook)),
            ]);
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(6, 7), White),
                Some(Coord::new(7, 7))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(5, 7), White),
                Some(Coord::new(7, 7))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(1, 0), White),
                None
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(0, 0), White),
                None
            );

            assert_eq!(
                under_attack_from_coord(&board, Coord::new(1, 0), Black),
                Some(Coord::new(0, 0))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(2, 0), Black),
                Some(Coord::new(0, 0))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(4, 0), Black),
                None
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(0, 1), Black),
                Some(Coord::new(0, 0))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(0, 2), Black),
                Some(Coord::new(0, 0))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(0, 3), Black),
                Some(Coord::new(0, 0))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(0, 4), Black),
                None
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(6, 7), Black),
                None
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(7, 7), Black),
                None
            );
        }
        {
            let board =
                BoardSquare::<8, 8>::with_pieces(&[(Coord::new(1, 1), Piece::new(White, Bishop))]);
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(0, 0), Black),
                Some(Coord::new(1, 1))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(2, 2), Black),
                Some(Coord::new(1, 1))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(3, 3), Black),
                Some(Coord::new(1, 1))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(0, 2), Black),
                Some(Coord::new(1, 1))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(2, 0), Black),
                Some(Coord::new(1, 1))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(1, 0), Black),
                None
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(0, 1), Black),
                None
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(1, 2), Black),
                None
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(2, 1), Black),
                None
            );
        }
        {
            let board = BoardSquare::<8, 8>::with_pieces(&[(
                Coord::new(3, 3),
                Piece::new(White, Cardinal),
            )]);
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(1, 2), Black),
                Some(Coord::new(3, 3))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(1, 4), Black),
                Some(Coord::new(3, 3))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(2, 1), Black),
                Some(Coord::new(3, 3))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(2, 5), Black),
                Some(Coord::new(3, 3))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(4, 1), Black),
                Some(Coord::new(3, 3))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(4, 5), Black),
                Some(Coord::new(3, 3))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(5, 2), Black),
                Some(Coord::new(3, 3))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(5, 4), Black),
                Some(Coord::new(3, 3))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(2, 2), Black),
                Some(Coord::new(3, 3))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(1, 1), Black),
                Some(Coord::new(3, 3))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(2, 4), Black),
                Some(Coord::new(3, 3))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(4, 2), Black),
                Some(Coord::new(3, 3))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(4, 4), Black),
                Some(Coord::new(3, 3))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(2, 3), Black),
                None
            );
        }
        {
            let board =
                BoardSquare::<8, 8>::with_pieces(&[(Coord::new(3, 3), Piece::new(White, Empress))]);
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(1, 2), Black),
                Some(Coord::new(3, 3))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(1, 4), Black),
                Some(Coord::new(3, 3))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(2, 1), Black),
                Some(Coord::new(3, 3))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(2, 5), Black),
                Some(Coord::new(3, 3))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(4, 1), Black),
                Some(Coord::new(3, 3))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(4, 5), Black),
                Some(Coord::new(3, 3))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(5, 2), Black),
                Some(Coord::new(3, 3))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(5, 4), Black),
                Some(Coord::new(3, 3))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(3, 4), Black),
                Some(Coord::new(3, 3))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(3, 5), Black),
                Some(Coord::new(3, 3))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(2, 3), Black),
                Some(Coord::new(3, 3))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(4, 3), Black),
                Some(Coord::new(3, 3))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(3, 2), Black),
                Some(Coord::new(3, 3))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(2, 2), Black),
                None
            );
        }
        {
            let board =
                BoardSquare::<8, 8>::with_pieces(&[(Coord::new(1, 1), Piece::new(White, Queen))]);
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(0, 0), Black),
                Some(Coord::new(1, 1))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(2, 2), Black),
                Some(Coord::new(1, 1))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(3, 3), Black),
                Some(Coord::new(1, 1))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(0, 2), Black),
                Some(Coord::new(1, 1))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(2, 0), Black),
                Some(Coord::new(1, 1))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(1, 0), Black),
                Some(Coord::new(1, 1))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(0, 1), Black),
                Some(Coord::new(1, 1))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(1, 2), Black),
                Some(Coord::new(1, 1))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(1, 3), Black),
                Some(Coord::new(1, 1))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(2, 1), Black),
                Some(Coord::new(1, 1))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(2, 3), Black),
                None
            );
        }
        {
            let board =
                BoardSquare::<8, 8>::with_pieces(&[(Coord::new(1, 1), Piece::new(White, King))]);
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(0, 0), Black),
                Some(Coord::new(1, 1))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(2, 2), Black),
                Some(Coord::new(1, 1))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(3, 3), Black),
                None
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(0, 2), Black),
                Some(Coord::new(1, 1))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(2, 0), Black),
                Some(Coord::new(1, 1))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(1, 0), Black),
                Some(Coord::new(1, 1))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(0, 1), Black),
                Some(Coord::new(1, 1))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(1, 2), Black),
                Some(Coord::new(1, 1))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(1, 3), Black),
                None
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(2, 1), Black),
                Some(Coord::new(1, 1))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(2, 3), Black),
                None
            );
        }
        {
            let board =
                BoardSquare::<8, 8>::with_pieces(&[(Coord::new(1, 1), Piece::new(White, Knight))]);
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(0, 3), Black),
                Some(Coord::new(1, 1))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(2, 3), Black),
                Some(Coord::new(1, 1))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(3, 5), Black),
                None
            );
        }
        {
            let board =
                BoardSquare::<8, 8>::with_pieces(&[(Coord::new(1, 1), Piece::new(White, Ferz))]);
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(0, 0), Black),
                Some(Coord::new(1, 1))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(2, 0), Black),
                Some(Coord::new(1, 1))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(2, 1), Black),
                None
            );
        }
        {
            let board =
                BoardSquare::<8, 8>::with_pieces(&[(Coord::new(1, 1), Piece::new(White, Wazir))]);
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(1, 0), Black),
                Some(Coord::new(1, 1))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(2, 1), Black),
                Some(Coord::new(1, 1))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(2, 0), Black),
                None
            );
        }
        {
            let board = BoardSquare::<8, 8>::with_pieces(&[
                (Coord::new(1, 1), Piece::new(White, Pawn)),
                (Coord::new(6, 6), Piece::new(Black, Pawn)),
            ]);
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(5, 5), White),
                Some(Coord::new(6, 6))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(7, 5), White),
                Some(Coord::new(6, 6))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(6, 5), White),
                None
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(6, 4), White),
                None
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(5, 7), White),
                None
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(6, 7), White),
                None
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(7, 7), White),
                None
            );

            assert_eq!(
                under_attack_from_coord(&board, Coord::new(0, 2), Black),
                Some(Coord::new(1, 1))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(2, 2), Black),
                Some(Coord::new(1, 1))
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(1, 2), Black),
                None
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(1, 3), Black),
                None
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(0, 0), Black),
                None
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(1, 0), Black),
                None
            );
            assert_eq!(
                under_attack_from_coord(&board, Coord::new(2, 0), Black),
                None
            );
        }
    }
    #[test]
    fn test_add_moves_for_piece_to_end_at_board_no_captures() {
        {
            let mut board = BoardSquare::<4, 4>::default();
            board.add_piece(Coord::new(3, 1), Piece::new(White, Bishop));
            board.add_piece(Coord::new(1, 3), Piece::new(Black, Bishop));
            let moves = all_legal_moves_to_end_at_board_no_captures(
                &board,
                Piece::new(White, Rook),
                Coord::new(1, 1),
            );
            assert_moves_eq(
                &[
                    Coord::new(1, 0),
                    Coord::new(0, 1),
                    Coord::new(2, 1),
                    Coord::new(1, 2),
                ],
                moves,
            );
        }
        {
            let mut board = BoardSquare::<4, 4>::default();
            board.add_piece(Coord::new(3, 2), Piece::new(White, Bishop));
            board.add_piece(Coord::new(2, 3), Piece::new(Black, Bishop));
            let moves = all_legal_moves_to_end_at_board_no_captures(
                &board,
                Piece::new(White, Knight),
                Coord::new(1, 1),
            );
            assert_moves_eq(&[Coord::new(3, 0), Coord::new(0, 3)], moves);
        }
        {
            let mut board = BoardSquare::<4, 4>::default();
            board.add_piece(Coord::new(1, 1), Piece::new(White, Bishop));
            let mut moves = all_legal_moves_to_end_at_board_no_captures(
                &board,
                Piece::new(White, Pawn),
                Coord::new(1, 2),
            );
            assert_moves_eq(&[], moves);

            moves = all_legal_moves_to_end_at_board_no_captures(
                &board,
                Piece::new(White, Pawn),
                Coord::new(2, 2),
            );
            assert_moves_eq(&[Coord::new(2, 1)], moves);

            moves = all_legal_moves_to_end_at_board_no_captures(
                &board,
                Piece::new(White, Pawn),
                Coord::new(3, 1),
            );
            assert_moves_eq(&[], moves);

            moves = all_legal_moves_to_end_at_board_no_captures(
                &board,
                Piece::new(Black, Pawn),
                Coord::new(3, 1),
            );
            assert_moves_eq(&[Coord::new(3, 2)], moves);

            moves = all_legal_moves_to_end_at_board_no_captures(
                &board,
                Piece::new(Black, Pawn),
                Coord::new(3, 2),
            );
            assert_moves_eq(&[], moves);
        }
    }

    fn valid_piece_for_coord(piece: Piece, coord: Coord, height: i8) -> bool {
        match piece.ty() {
            Pawn => coord.y != 0 && coord.y != height - 1,
            _ => true,
        }
    }

    fn add_piece_to_rand_coord<R: Rng + ?Sized, T: Board>(
        rng: &mut R,
        board: &mut T,
        piece: Piece,
    ) {
        loop {
            let coord = Coord::new(
                rng.gen_range(0..board.width()),
                rng.gen_range(0..board.height()),
            );
            if board.get(coord).is_some() {
                continue;
            }
            if !valid_piece_for_coord(piece, coord, board.height()) {
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

    fn rand_non_king_type_for_coord<R: Rng + ?Sized>(
        rng: &mut R,
        coord: Coord,
        height: i8,
    ) -> Type {
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

    fn rand_board_piece<const W: usize, const H: usize, R: Rng + ?Sized>(
        rng: &mut R,
    ) -> BoardPiece<W, H, 4> {
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
        let is_check = all_legal_moves(board, player.next())
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
        fn test_piece<const W: usize, const H: usize>() {
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
}
