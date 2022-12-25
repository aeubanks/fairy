use fairy::board::Board;
use fairy::perft::{fen, perft, Position};
use fairy::player::Player::White;

#[test]
fn classical_1() {
    let pos = Position {
        board: Board::classical(),
        player: White,
    };
    assert_eq!(perft(&pos, 1), 20);
    assert_eq!(perft(&pos, 2), 400);
    assert_eq!(perft(&pos, 3), 8902);
    assert_eq!(perft(&pos, 4), 197281);
}

#[test]
fn classical_2() {
    let pos = fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -");
    assert_eq!(perft(&pos, 1), 48);
    assert_eq!(perft(&pos, 2), 2039);
    assert_eq!(perft(&pos, 3), 97862);
}

#[test]
fn classical_3() {
    let pos = fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -");
    assert_eq!(perft(&pos, 1), 14);
    assert_eq!(perft(&pos, 2), 191);
    assert_eq!(perft(&pos, 3), 2812);
    assert_eq!(perft(&pos, 4), 43238);
}

#[test]
fn classical_4() {
    let pos = fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
    assert_eq!(perft(&pos, 1), 44);
    assert_eq!(perft(&pos, 2), 1486);
    assert_eq!(perft(&pos, 3), 62379);
}

#[test]
fn chess960_1() {
    let pos = fen("bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9");
    assert_eq!(perft(&pos, 1), 21);
    assert_eq!(perft(&pos, 2), 528);
    assert_eq!(perft(&pos, 3), 12189);
}

#[test]
fn chess960_2() {
    let pos = fen("2nnrbkr/p1qppppp/8/1ppb4/6PP/3PP3/PPP2P2/BQNNRBKR w HEhe - 1 9");
    assert_eq!(perft(&pos, 1), 21);
    assert_eq!(perft(&pos, 2), 807);
    assert_eq!(perft(&pos, 3), 18002);
}

#[test]
fn chess960_3() {
    let pos = fen("b1q1rrkb/pppppppp/3nn3/8/P7/1PPP4/4PPPP/BQNNRKRB w GE - 1 9");
    assert_eq!(perft(&pos, 1), 20);
    assert_eq!(perft(&pos, 2), 479);
    assert_eq!(perft(&pos, 3), 10471);
}

#[test]
fn chess960_4() {
    let pos = fen("qbbnnrkr/2pp2pp/p7/1p2pp2/8/P3PP2/1PPP1KPP/QBBNNR1R w hf - 0 9");
    assert_eq!(perft(&pos, 1), 22);
    assert_eq!(perft(&pos, 2), 593);
    assert_eq!(perft(&pos, 3), 13440);
}
