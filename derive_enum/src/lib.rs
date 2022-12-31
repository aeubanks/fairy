pub use derive_enum_impl::EnumCount;

pub trait EnumCount {
    const COUNT: usize;
}
