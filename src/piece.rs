use crate::player::Player;
use derive_rand::Rand;
use num_derive::FromPrimitive;
use num_traits::cast::FromPrimitive;
use static_assertions::const_assert_eq;
use std::num::NonZeroU8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Rand, Hash, FromPrimitive)]
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
            Chancellor => 'C',
            Archbishop => 'A',
            Amazon => 'Z',
        };
        assert!(ret.is_uppercase());
        ret
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
}

#[test]
fn test_piece() {
    use Player::*;
    use Type::*;
    let p = Piece::new(White, King);
    assert_eq!(p.player(), White);
    assert_eq!(p.ty(), King);
}

const_assert_eq!(1, std::mem::size_of::<Piece>());
const_assert_eq!(1, std::mem::size_of::<Option<Piece>>());
