use derive_rand::Rand;
use num_derive::FromPrimitive;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Rand, Hash, FromPrimitive)]
pub enum Player {
    White,
    Black,
}

pub fn next_player(player: Player) -> Player {
    use Player::*;
    match player {
        White => Black,
        Black => White,
    }
}
