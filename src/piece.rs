#[derive(Debug, Clone, PartialEq, Eq)]
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
    pub player: u8,
    pub ty: Type,
}
