use std::collections::HashSet;

use crate::board::{Board, Move, Presets};
use crate::coord::Coord;
use crate::moves;
use crate::piece::Type;
use crate::player::Player;
use tch::{nn::*, *};

fn piece_type_to_index(ty: Type) -> i64 {
    use Type::*;
    match ty {
        King => 0,
        Queen => 1,
        Rook => 2,
        Knight => 3,
        Pawn => 4,
        _ => unreachable!(),
    }
}

fn board_to_tensor<B: Board>(board: &B, dev: Device) -> Tensor {
    let t = Tensor::zeros(
        [2, 5, board.height() as i64, board.width() as i64],
        (Kind::Float, dev),
    );
    board.foreach_piece(|p, c| {
        let _ = t
            .i((
                p.player() as i64,
                piece_type_to_index(p.ty()),
                c.y as i64,
                c.x as i64,
            ))
            .fill_(1);
    });
    t.reshape([10, board.height() as i64, board.width() as i64])
}

fn boards_to_tensor<B: Board>(boards: &[B], dev: Device) -> Tensor {
    let ts = boards
        .iter()
        .map(|b| board_to_tensor(b, dev))
        .collect::<Vec<_>>();
    Tensor::stack(&ts, 0)
}

fn all_moves(width: i8, height: i8) -> Vec<Move> {
    let mut v = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let from = Coord::new(x, y);
            for d in [
                (1, 0),
                (0, 1),
                (-1, 0),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
                (1, 2),
                (-1, 2),
                (1, -2),
                (-1, -2),
                (2, 1),
                (-2, 1),
                (2, -1),
                (-2, -1),
            ] {
                let d = Coord::new(d.0, d.1);
                let mut try_to = d + from;
                let in_bounds = |c: Coord| c.x >= 0 && c.x < width && c.y >= 0 && c.y < height;
                while in_bounds(try_to) {
                    v.push(Move { from, to: try_to });
                    try_to = try_to + d;
                }
            }
        }
    }
    v
}

fn move_tensor_to_vec(t: &Tensor) -> Vec<Vec<f32>> {
    let size = t.size2().unwrap();
    let mut ret = Vec::new();
    ret.reserve(size.0 as usize);
    for i in 0..size.0 {
        let mut v: Vec<f32> = vec![0.0; size.1 as usize];
        t.get(i).copy_data(&mut v, size.1 as usize);
        ret.push(v);
    }
    ret
}

fn move_probabilities<B: Board>(v: &[f32], all_moves: &[Move], board: &B) -> Vec<(Move, f32)> {
    assert_eq!(v.len(), all_moves.len());
    let legal_moves = moves::all_moves(board, Player::White)
        .into_iter()
        .collect::<HashSet<_>>();
    let mut ret = Vec::new();
    let mut total_prob = 0.0;
    for (&prob, &m) in v.iter().zip(all_moves.iter()) {
        if legal_moves.contains(&m) {
            total_prob += prob;
            ret.push((m, prob));
        }
    }
    for (_, p) in &mut ret {
        *p /= total_prob;
    }
    ret
}

struct NN {
    seqs: Vec<SequentialT>,
    lin: Linear,
}

impl NN {
    fn new(vs: &VarStore, input_width: i64, input_height: i64, output_size: i64) -> Self {
        let mut conv_config = ConvConfigND::<[i64; 2]>::default();
        conv_config.padding = [1, 1];

        let mut seqs = Vec::new();

        for i in 0..4 {
            let conv = conv(
                vs.root().sub(format!("conv{}", i)),
                if i == 0 { 10 } else { 32 },
                32,
                [3, 3],
                conv_config,
            );
            let batch = batch_norm2d(
                vs.root().sub(format!("batchnorm{}", i)),
                32,
                Default::default(),
            );
            seqs.push(seq_t().add(conv).add(batch).add_fn(|t| t.relu()));
        }

        let lin = linear(
            vs.root().sub("linear"),
            32 * input_width * input_height,
            output_size,
            Default::default(),
        );

        Self { seqs, lin }
    }
}

impl std::fmt::Debug for NN {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("NN")
    }
}

impl tch::nn::ModuleT for NN {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut t = self.seqs[0].forward_t(xs, train);
        t = self
            .seqs
            .iter()
            .skip(1)
            .fold(t, |t, seq| seq.forward_t(&t, train) + t);
        self.lin.forward_t(&t.flat_view(), train)
    }
}

fn move_probabilities_for_boards<B: Board>(boards: &[B]) -> Vec<Vec<(Move, f32)>> {
    assert!(!boards.is_empty());
    for b in boards {
        assert_eq!(b.width(), boards[0].width());
        assert_eq!(b.height(), boards[0].height());
    }

    let dev = Device::cuda_if_available();
    let mut vs = VarStore::new(dev);
    vs.set_kind(Kind::Float);

    let width = boards[0].width();
    let height = boards[0].height();
    let all_moves = all_moves(width, height);

    let t = boards_to_tensor(boards, dev);
    let n = NN::new(&vs, width as i64, height as i64, all_moves.len() as i64);
    let out = n.forward_t(&t, false);
    let v = move_tensor_to_vec(&out);
    boards
        .iter()
        .zip(v)
        .map(|(b, v)| move_probabilities(&v, &all_moves, b))
        .collect::<Vec<_>>()
}

pub fn nn() {
    let boards = vec![Presets::los_alamos(); 7];
    let probs = move_probabilities_for_boards(&boards);
    dbg!(&probs[0]);
}

#[cfg(test)]
mod tests {
    use crate::board::BoardSquare;

    use super::*;

    #[test]
    fn test_board_to_tensor() {
        let dev = Device::Cpu;
        let board = Presets::los_alamos();
        let t = board_to_tensor(&board, dev);
        assert_eq!(t.size(), vec![10, 6, 6]);
        assert_eq!(
            t.int64_value(&[0 + piece_type_to_index(Type::King), 0, 3]),
            1
        );
        assert_eq!(
            t.int64_value(&[0 + piece_type_to_index(Type::Queen), 0, 2]),
            1
        );
        assert_eq!(
            t.int64_value(&[5 + piece_type_to_index(Type::King), 0, 3]),
            0
        );
        assert_eq!(
            t.int64_value(&[0 + piece_type_to_index(Type::King), 0, 2]),
            0
        );
        assert_eq!(
            t.int64_value(&[0 + piece_type_to_index(Type::Queen), 0, 3]),
            0
        );
        assert_eq!(
            t.int64_value(&[5 + piece_type_to_index(Type::King), 5, 3]),
            1
        );
    }

    #[test]
    fn test_boards_to_tensor() {
        let dev = Device::Cpu;
        let board1 = Presets::los_alamos();
        let board2 = BoardSquare::<6, 6>::default();
        let board3 = Presets::los_alamos();

        let t = boards_to_tensor(&[board1, board2, board3], dev);
        assert_eq!(t.size(), vec![3, 10, 6, 6]);
        assert_eq!(
            t.int64_value(&[0, 0 + piece_type_to_index(Type::King), 0, 3]),
            1
        );
        assert_eq!(
            t.int64_value(&[1, 0 + piece_type_to_index(Type::King), 0, 3]),
            0
        );
        assert_eq!(
            t.int64_value(&[2, 0 + piece_type_to_index(Type::King), 0, 3]),
            1
        );
    }

    #[test]
    fn test_all_moves() {
        let mut moves = HashSet::new();
        for m in all_moves(6, 6) {
            assert!(moves.insert(m));
        }
    }

    #[test]
    fn test_end_to_end() {
        let probs =
            move_probabilities_for_boards(&[Presets::los_alamos(), BoardSquare::<6, 6>::default()]);
        assert_eq!(probs.len(), 2);
        assert_eq!(probs[0].len(), 10);
        assert_eq!(probs[1].len(), 0);
    }
}
