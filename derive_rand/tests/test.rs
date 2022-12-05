use derive_rand::Rand;
use rand::{thread_rng, Rng};

#[derive(PartialEq, Eq, Debug, Rand)]
enum Struct1 {
    One,
}

#[derive(PartialEq, Eq, Debug, Rand)]
enum Struct2 {
    One,
    Two,
}

#[derive(PartialEq, Eq, Debug, Rand)]
enum Struct3 {
    One,
    Two,
    Three,
}

#[test]
fn test_derive() {
    let mut rng = thread_rng();
    rng.gen::<Struct1>();
    rng.gen::<Struct2>();
    rng.gen::<Struct3>();
}
