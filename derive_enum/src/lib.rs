use std::marker::PhantomData;

pub use derive_enum_impl::{EnumCount, EnumFrom};

pub trait EnumCount {
    const COUNT: usize;
}

pub trait EnumFrom: Sized {
    fn from_u8(v: u8) -> Option<Self>;
    fn all() -> EnumFromIter<Self> {
        EnumFromIter(0, PhantomData)
    }
}

pub struct EnumFromIter<T: EnumFrom>(u8, PhantomData<T>);

impl<T: EnumFrom> Iterator for EnumFromIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        let ret = T::from_u8(self.0);
        self.0 += 1;
        ret
    }
}
