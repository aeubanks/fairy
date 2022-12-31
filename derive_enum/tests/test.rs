use derive_enum::EnumCount;

#[derive(Copy, Clone, EnumCount)]
enum Struct0 {}

#[derive(Copy, Clone, EnumCount)]
enum Struct1 {
    One,
}

#[derive(Copy, Clone, EnumCount)]
enum Struct2 {
    One,
    Two,
}

#[derive(Copy, Clone, EnumCount)]
enum Struct3 {
    One,
    Two,
    Three,
}

#[test]
fn test_derive() {
    assert_eq!(Struct0::COUNT, 0);
    assert_eq!(Struct1::COUNT, 1);
    assert_eq!(Struct2::COUNT, 2);
    assert_eq!(Struct3::COUNT, 3);
}
