use derive_enum::{EnumCount, EnumFrom};

#[derive(EnumCount)]
enum Struct0 {}

#[derive(EnumCount)]
enum Struct1 {
    One,
}

#[derive(EnumCount)]
enum Struct2 {
    One,
    Two,
}

#[derive(Debug, PartialEq, Eq, EnumFrom)]
enum Struct3 {}

#[derive(Debug, PartialEq, Eq, EnumFrom)]
enum Struct4 {
    One,
}

#[derive(Debug, PartialEq, Eq, EnumFrom)]
enum Struct5 {
    One,
    Two,
}

#[test]
fn test_derive() {
    assert_eq!(Struct0::COUNT, 0);
    assert_eq!(Struct1::COUNT, 1);
    assert_eq!(Struct2::COUNT, 2);

    assert_eq!(Struct3::from_u8(0), None);
    assert_eq!(Struct3::from_u8(1), None);
    assert_eq!(Struct4::from_u8(0), Some(Struct4::One));
    assert_eq!(Struct4::from_u8(1), None);
    assert_eq!(Struct5::from_u8(0), Some(Struct5::One));
    assert_eq!(Struct5::from_u8(1), Some(Struct5::Two));
    assert_eq!(Struct5::from_u8(2), None);

    let mut it3 = Struct3::all();
    assert_eq!(it3.next(), None);

    let mut it4 = Struct4::all();
    assert_eq!(it4.next(), Some(Struct4::One));
    assert_eq!(it4.next(), None);

    let mut it5 = Struct5::all();
    assert_eq!(it5.next(), Some(Struct5::One));
    assert_eq!(it5.next(), Some(Struct5::Two));
    assert_eq!(it5.next(), None);
}
