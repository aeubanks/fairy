use derive_enum::EnumFrom;
use derive_rand::Rand;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Rand, Hash, EnumFrom)]
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
