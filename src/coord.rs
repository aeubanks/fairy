#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default)]
pub struct Coord {
    pub x: i8,
    pub y: i8,
}

impl Coord {
    pub fn new(x: i8, y: i8) -> Self {
        Self { x, y }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coord_add() {
        assert_eq!(
            Coord { x: 3, y: 4 } + Coord { x: 5, y: 6 },
            Coord { x: 8, y: 10 }
        );
    }
}
