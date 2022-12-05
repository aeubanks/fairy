use crate::player::Player;
use derive_rand::Rand;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Rand)]
pub enum Type {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
    Chancellor,
    Archbishop,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Piece {
    pub player: Player,
    pub ty: Type,
}
