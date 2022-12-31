use crate::coord::Coord;
use crate::player::Player;
use arrayvec::ArrayVec;
use derive_rand::Rand;
use num_derive::FromPrimitive;
use num_traits::cast::FromPrimitive;
use std::num::NonZeroU8;
use strum::EnumCount;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Rand, Hash, FromPrimitive, EnumCount)]
pub enum Type {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
    Empress,
    Cardinal,
    Amazon,
}

impl Type {
    pub fn char(&self) -> char {
        use Type::*;
        let ret = match self {
            Pawn => 'P',
            Knight => 'N',
            Bishop => 'B',
            Rook => 'R',
            Queen => 'Q',
            King => 'K',
            Empress => 'E',
            Cardinal => 'C',
            Amazon => 'A',
        };
        assert!(ret.is_uppercase());
        ret
    }

    pub fn leaper_offsets(&self) -> ArrayVec<Coord, 2> {
        use Type::*;
        let mut ret = ArrayVec::new();
        match self {
            Pawn => panic!(),
            Knight | Empress | Cardinal | Amazon => {
                ret.push(Coord::new(2, 1));
            }
            King => {
                ret.push(Coord::new(1, 1));
                ret.push(Coord::new(1, 0));
            }
            Rook | Bishop | Queen => {}
        }
        ret
    }

    pub fn rider_offsets(&self) -> ArrayVec<Coord, 2> {
        use Type::*;
        let mut ret = ArrayVec::new();
        match self {
            Pawn => panic!(),
            Rook | Empress => {
                ret.push(Coord::new(1, 0));
            }
            Bishop | Cardinal => {
                ret.push(Coord::new(1, 1));
            }
            Queen | Amazon => {
                ret.push(Coord::new(1, 0));
                ret.push(Coord::new(1, 1));
            }
            King | Knight => {}
        }
        ret
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Piece {
    val: NonZeroU8,
}

impl Piece {
    pub fn new(player: Player, ty: Type) -> Self {
        let val = (1 << 7) | ((player as u8) << 6) | (ty as u8);
        Piece {
            val: NonZeroU8::new(val).unwrap(),
        }
    }

    pub fn player(&self) -> Player {
        Player::from_u8((self.val.get() >> 6) & 1).unwrap()
    }

    pub fn ty(&self) -> Type {
        Type::from_u8(self.val.get() & ((1 << 6) - 1)).unwrap()
    }

    pub fn char(&self) -> char {
        let c = self.ty().char();
        match self.player() {
            Player::White => c,
            Player::Black => c.to_lowercase().next().unwrap(),
        }
    }

    pub fn val(&self) -> u8 {
        self.val.get()
    }
}

impl std::fmt::Debug for Piece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use std::fmt::Write;
        f.write_char(self.char())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use static_assertions::const_assert_eq;
    use Player::*;
    use Type::*;

    #[test]
    fn test_piece() {
        let p = Piece::new(White, King);
        assert_eq!(p.player(), White);
        assert_eq!(p.ty(), King);
    }

    #[test]
    fn test_piece_char() {
        assert_eq!(Piece::new(White, King).char(), 'K');
        assert_eq!(Piece::new(Black, Queen).char(), 'q');
    }

    #[test]
    fn test_piece_fmt() {
        assert_eq!(format!("{:?}", Piece::new(White, King)), "K");
    }

    const_assert_eq!(1, std::mem::size_of::<Piece>());
    const_assert_eq!(1, std::mem::size_of::<Option<Piece>>());
}
