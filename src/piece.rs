use crate::coord::Coord;
use crate::player::Player;
use arrayvec::ArrayVec;
use derive_enum::{EnumCount, EnumFrom};
use derive_rand::Rand;
use std::num::NonZeroU8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Rand, Hash, EnumFrom, EnumCount)]
pub enum Type {
    King,
    Amazon,
    Queen,
    Empress,
    Cardinal,
    Rook,
    Nightrider,
    Bishop,
    Knight,
    Bojack,
    Ferz,
    Wazir,
    Dabbaba,
    Alfil,
    Pawn,
}

fn offsets(offset: Coord) -> ArrayVec<Coord, 8> {
    assert!(offset.x >= 0);
    assert!(offset.y >= 0);
    assert!(offset.x > 0 || offset.y > 0);
    let mut ret = ArrayVec::new();
    ret.push(Coord::new(offset.x, offset.y));
    ret.push(Coord::new(-offset.y, offset.x));
    ret.push(Coord::new(-offset.x, -offset.y));
    ret.push(Coord::new(offset.y, -offset.x));
    if offset.x != offset.y && offset.x != 0 && offset.y != 0 {
        ret.push(Coord::new(offset.y, offset.x));
        ret.push(Coord::new(-offset.x, offset.y));
        ret.push(Coord::new(-offset.y, -offset.x));
        ret.push(Coord::new(offset.x, -offset.y));
    }
    ret
}

impl Type {
    fn char(&self) -> char {
        use Type::*;
        match self {
            King => 'K',
            Amazon => 'A',
            Queen => 'Q',
            Empress => 'E',
            Cardinal => 'C',
            Rook => 'R',
            Nightrider => 'T',
            Bishop => 'B',
            Knight => 'N',
            Bojack => 'J',
            Ferz => 'F',
            Wazir => 'W',
            Dabbaba => 'D',
            Alfil => 'L',
            Pawn => 'P',
        }
    }

    pub fn leaper_offsets(&self) -> ArrayVec<Coord, 8> {
        use Type::*;
        let mut ret = ArrayVec::new();
        match self {
            Pawn => panic!(),
            Knight | Empress | Cardinal | Amazon => {
                ret.extend(offsets(Coord::new(2, 1)));
            }
            Bojack => {
                ret.extend(offsets(Coord::new(2, 2)));
                ret.extend(offsets(Coord::new(2, 0)));
            }
            King => {
                ret.extend(offsets(Coord::new(1, 1)));
                ret.extend(offsets(Coord::new(1, 0)));
            }
            Ferz => {
                ret.extend(offsets(Coord::new(1, 1)));
            }
            Wazir => {
                ret.extend(offsets(Coord::new(1, 0)));
            }
            Dabbaba => {
                ret.extend(offsets(Coord::new(2, 0)));
            }
            Alfil => {
                ret.extend(offsets(Coord::new(2, 2)));
            }
            Rook | Bishop | Queen | Nightrider => {}
        }
        ret
    }

    pub fn rider_offsets(&self) -> ArrayVec<Coord, 8> {
        use Type::*;
        let mut ret = ArrayVec::new();
        match self {
            Pawn => panic!(),
            Rook | Empress => {
                ret.extend(offsets(Coord::new(1, 0)));
            }
            Bishop | Cardinal => {
                ret.extend(offsets(Coord::new(1, 1)));
            }
            Nightrider => {
                ret.extend(offsets(Coord::new(2, 1)));
            }
            Queen | Amazon => {
                ret.extend(offsets(Coord::new(1, 0)));
                ret.extend(offsets(Coord::new(1, 1)));
            }
            King | Knight | Bojack | Wazir | Ferz | Alfil | Dabbaba => {}
        }
        ret
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Piece {
    val: NonZeroU8,
}

impl Piece {
    pub const fn new(player: Player, ty: Type) -> Self {
        let val = (1 << 7) | ((player as u8) << 6) | (ty as u8);
        Piece {
            val: match NonZeroU8::new(val) {
                Some(v) => v,
                None => panic!(),
            },
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

    pub fn from_val(val: u8) -> Self {
        Self {
            val: match NonZeroU8::new(val) {
                Some(v) => v,
                None => panic!(),
            },
        }
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
    use std::collections::HashSet;
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

    #[test]
    fn test_type_char() {
        let mut set = HashSet::new();
        for ty in Type::all() {
            let c = ty.char();
            assert!(!set.contains(&c));
            assert!(c.is_uppercase());
            set.insert(c);
        }
    }

    const_assert_eq!(1, std::mem::size_of::<Piece>());
    const_assert_eq!(1, std::mem::size_of::<Option<Piece>>());
}
