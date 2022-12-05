use derive_rand::Rand;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Rand)]
pub enum Player {
    White,
    Black,
}
