use crate::player::Player;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Piece {
    pub player: Player,
    pub ty: Type,
}
