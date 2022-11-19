#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Coord {
    pub x: i8,
    pub y: i8,
}

impl std::ops::Add for Coord {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl std::convert::From<(i8, i8)> for Coord {
    fn from((x, y): (i8, i8)) -> Self {
        Self { x, y }
    }
}

#[test]
fn test_coord_add() {
    assert_eq!(
        Coord { x: 3, y: 4 } + Coord { x: 5, y: 6 },
        Coord { x: 8, y: 10 }
    );
}
