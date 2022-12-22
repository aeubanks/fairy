use crate::player::Player;
use derive_rand::Rand;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Rand, Hash)]
pub enum Type {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
    Chancellor,
    Archbishop,
    Amazon,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Piece {
    pub player: Player,
    pub ty: Type,
}
