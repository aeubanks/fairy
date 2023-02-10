use derive_enum::EnumFrom;
use derive_rand::Rand;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Rand, Hash, EnumFrom)]
pub enum Player {
    White,
    Black,
}

impl Player {
    pub fn next(self) -> Player {
        use Player::*;
        match self {
            White => Black,
            Black => White,
        }
    }
}
