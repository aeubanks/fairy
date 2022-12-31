pub use derive_enum_impl::{EnumCount, EnumFrom};

pub trait EnumCount {
    const COUNT: usize;
}

pub trait EnumFrom: Sized {
    fn from_u8(v: u8) -> Option<Self>;
}
