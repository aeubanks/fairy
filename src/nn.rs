use std::collections::HashSet;

use crate::board::{board_piece_to_square, Board, Move};
use crate::coord::Coord;
use crate::moves;
use crate::piece::{Piece, Type};
use crate::player::Player;
use crate::tablebase::{generate_tablebase, PieceSet, TBBoard, TBMoveType, Tablebase};
use crate::timer::Timer;
use tch::{nn::*, *};

const NUM_PLAYERS: i64 = 2;
const NUM_PIECE_TYPES: i64 = 5;

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
        [
            NUM_PLAYERS,
            NUM_PIECE_TYPES,
            board.height() as i64,
            board.width() as i64,
        ],
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
    t.reshape([
        NUM_PLAYERS * NUM_PIECE_TYPES,
        board.height() as i64,
        board.width() as i64,
    ])
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
    let mut ret = Vec::with_capacity(size.0 as usize);
    for i in 0..size.0 {
        let mut v: Vec<f32> = vec![0.0; size.1 as usize];
        t.get(i)
            .log_softmax(0, Kind::Float)
            .copy_data(&mut v, size.1 as usize);
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

struct NNBody {
    seqs: Vec<SequentialT>,
}

const BODY_NUM_FILTERS: i64 = 32;

impl NNBody {
    fn new(vs: &VarStore) -> Self {
        let mut conv_config = ConvConfigND::<[i64; 2]> {
            padding: [1, 1],
            ..Default::default()
        };
        conv_config.padding = [1, 1];

        let mut seqs = Vec::new();

        for i in 0..4 {
            let conv = conv(
                vs.root().sub(format!("body-conv-{}", i)),
                if i == 0 {
                    NUM_PLAYERS * NUM_PIECE_TYPES
                } else {
                    BODY_NUM_FILTERS
                },
                BODY_NUM_FILTERS,
                [3, 3],
                conv_config,
            );
            let batch = batch_norm2d(
                vs.root().sub(format!("body-batchnorm-{}", i)),
                BODY_NUM_FILTERS,
                Default::default(),
            );
            seqs.push(seq_t().add(conv).add(batch).add_fn(|t| t.relu()));
        }

        Self { seqs }
    }
}

impl std::fmt::Debug for NNBody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("NNBody")
    }
}

impl tch::nn::ModuleT for NNBody {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.seqs
            .iter()
            .skip(1)
            .fold(self.seqs[0].forward_t(xs, train), |t, seq| {
                seq.forward_t(&t, train) + t
            })
    }
}

struct NNValueHead {
    seq: SequentialT,
}

const VALUE_HEAD_NUM_FILTERS: i64 = 1;
const VALUE_HEAD_HIDDEN_LAYER_SIZE: i64 = 64;

impl NNValueHead {
    fn new(vs: &VarStore, input_width: i64, input_height: i64) -> Self {
        let conv = conv(
            vs.root().sub("value-head-conv"),
            BODY_NUM_FILTERS,
            VALUE_HEAD_NUM_FILTERS,
            [1, 1],
            Default::default(),
        );

        let batch = batch_norm2d(
            vs.root().sub("value-head-batchnorm"),
            VALUE_HEAD_NUM_FILTERS,
            Default::default(),
        );

        let lin1 = linear(
            vs.root().sub("value-head-linear-1"),
            VALUE_HEAD_NUM_FILTERS * input_width * input_height,
            VALUE_HEAD_HIDDEN_LAYER_SIZE,
            Default::default(),
        );
        let lin2 = linear(
            vs.root().sub("value-head-linear-2"),
            VALUE_HEAD_HIDDEN_LAYER_SIZE,
            1,
            Default::default(),
        );

        Self {
            seq: seq_t()
                .add(conv)
                .add(batch)
                .add_fn(|t| t.relu())
                .add_fn(|t| t.flat_view())
                .add(lin1)
                .add_fn(|t| t.relu())
                .add(lin2)
                .add_fn(|t| t.tanh())
                .add_fn(|t| t.reshape(-1)),
        }
    }
}

impl std::fmt::Debug for NNValueHead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("NNValueHead")
    }
}

impl tch::nn::ModuleT for NNValueHead {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.seq.forward_t(xs, train)
    }
}

struct NNPolicyHead {
    seq: SequentialT,
}

const POLICY_HEAD_NUM_FILTERS: i64 = 2;

impl NNPolicyHead {
    fn new(vs: &VarStore, input_width: i64, input_height: i64, num_outputs: i64) -> Self {
        let conv = conv(
            vs.root().sub("policy-head-conv"),
            BODY_NUM_FILTERS,
            POLICY_HEAD_NUM_FILTERS,
            [1, 1],
            Default::default(),
        );

        let batch = batch_norm2d(
            vs.root().sub("policy-head-batchnorm"),
            2,
            Default::default(),
        );

        let lin = linear(
            vs.root().sub("policy-head-linear"),
            POLICY_HEAD_NUM_FILTERS * input_width * input_height,
            num_outputs,
            Default::default(),
        );

        Self {
            seq: seq_t()
                .add(conv)
                .add(batch)
                .add_fn(|t| t.relu())
                .add_fn(|t| t.flat_view())
                .add(lin),
        }
    }
}

impl std::fmt::Debug for NNPolicyHead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("NNPolicyHead")
    }
}

impl tch::nn::ModuleT for NNPolicyHead {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.seq.forward_t(xs, train)
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
    let body = NNBody::new(&vs);
    let policy_head = NNPolicyHead::new(&vs, width as i64, height as i64, all_moves.len() as i64);
    let out = policy_head.forward_t(&body.forward_t(&t, false), false);
    let v = move_tensor_to_vec(&out);
    boards
        .iter()
        .zip(v)
        .map(|(b, v)| move_probabilities(&v, &all_moves, b))
        .collect::<Vec<_>>()
}

fn tablebase<const W: usize, const H: usize>() -> Tablebase<W, H> {
    use Player::*;
    use Type::*;

    let mut pieces = Vec::new();

    for p in [White, Black] {
        for ty in [Queen, Rook, Knight] {
            pieces.push(Piece::new(p, ty));
        }
    }

    let mut sets = Vec::new();

    for p1 in &pieces {
        for p2 in &pieces {
            sets.push(PieceSet::new(&[
                Piece::new(White, King),
                Piece::new(Black, King),
                *p1,
                *p2,
            ]));
        }
    }

    generate_tablebase(&sets)
}

fn rand_board_for_tablebase<const W: usize, const H: usize>() -> TBBoard<W, H> {
    use rand::{thread_rng, Rng};
    use Player::*;
    use Type::*;

    fn add_piece<const W: usize, const H: usize>(b: &mut TBBoard<W, H>, p: Piece) {
        let mut rng = thread_rng();
        loop {
            let try_coord = Coord::new(rng.gen_range(0..(W as i8)), rng.gen_range(0..(H as i8)));
            if b.get(try_coord).is_some() {
                continue;
            }
            b.add_piece(try_coord, p);
            break;
        }
    }
    let mut b = TBBoard::default();

    add_piece(&mut b, Piece::new(White, King));
    add_piece(&mut b, Piece::new(Black, King));
    let mut rng = thread_rng();
    let mut add_rand_piece = || {
        add_piece(
            &mut b,
            Piece::new(rng.gen(), [Queen, Rook, Knight][rng.gen_range(0..3)]),
        );
    };
    add_rand_piece();
    add_rand_piece();

    b
}

fn tb_result<const W: usize, const H: usize>(b: &TBBoard<W, H>, tb: &Tablebase<W, H>) -> i64 {
    // FIXME: this makes a bunch of unnecessary clones of boards, maybe result_for_real_board should work with all types of boards
    tb.result_for_real_board(Player::White, &board_piece_to_square(b))
        .2 as i64
}

fn tb_results_to_tensor_policy<const W: usize, const H: usize>(
    boards: &[TBBoard<W, H>],
    tb: &Tablebase<W, H>,
) -> Tensor {
    let ts = boards.iter().map(|b| tb_result(b, tb)).collect::<Vec<_>>();
    Tensor::from_slice(&ts)
}

fn tb_result2<const W: usize, const H: usize>(b: &TBBoard<W, H>, tb: &Tablebase<W, H>) -> f32 {
    // FIXME: this makes a bunch of unnecessary clones of boards, maybe result_for_real_board should work with all types of boards
    match tb
        .result_for_real_board(Player::White, &board_piece_to_square(b))
        .2
    {
        TBMoveType::Lose => -1.0,
        TBMoveType::Draw => 0.0,
        TBMoveType::Win => 1.0,
    }
}

fn tb_results_to_tensor_value<const W: usize, const H: usize>(
    boards: &[TBBoard<W, H>],
    tb: &Tablebase<W, H>,
) -> Tensor {
    let ts = boards.iter().map(|b| tb_result2(b, tb)).collect::<Vec<_>>();
    Tensor::from_slice(&ts)
}

pub fn train_nn_tablebase_policy<const W: usize, const H: usize>(
    num_epochs: usize,
    num_boards_per_epoch: usize,
    num_boards_to_evaluate: usize,
) {
    let dev = Device::cuda_if_available();
    let mut vs = VarStore::new(dev);
    vs.set_kind(Kind::Float);

    let tb = tablebase::<W, H>();

    let body = NNBody::new(&vs);
    let policy_head = NNPolicyHead::new(&vs, W as i64, H as i64, 3);
    let mut opt = nn::Adam::default().build(&vs, 0.001).unwrap();

    let (evaluate_xs, evaluate_targets) = {
        let mut evaluate_boards = Vec::new();
        for _ in 0..num_boards_to_evaluate {
            evaluate_boards.push(rand_board_for_tablebase());
        }
        (
            boards_to_tensor(&evaluate_boards, dev),
            tb_results_to_tensor_policy(&evaluate_boards, &tb),
        )
    };

    for epoch in 0..num_epochs {
        let timer = Timer::new();
        // train
        {
            let mut boards = Vec::new();
            for _ in 0..num_boards_per_epoch {
                boards.push(rand_board_for_tablebase());
            }
            let xs = boards_to_tensor(&boards, dev);
            let targets = tb_results_to_tensor_policy(&boards, &tb);

            let loss = policy_head
                .forward_t(&body.forward_t(&xs, true), true)
                .cross_entropy_for_logits(&targets);
            opt.backward_step(&loss);
        }

        // evaluate
        {
            let acc = policy_head.batch_accuracy_for_logits(
                &body.forward_t(&evaluate_xs, false),
                &evaluate_targets,
                dev,
                evaluate_targets.size1().unwrap(),
            );
            let elapsed = timer.elapsed();
            println!("epoch {epoch}: accuracy {acc:.3} ({elapsed:?})");
        }
    }
}

pub fn train_nn_tablebase_value<const W: usize, const H: usize>(
    num_epochs: usize,
    num_boards_per_epoch: usize,
    num_boards_to_evaluate: usize,
) {
    let dev = Device::cuda_if_available();
    let mut vs = VarStore::new(dev);
    vs.set_kind(Kind::Float);

    let tb = tablebase::<W, H>();

    let body = NNBody::new(&vs);
    let value_head = NNValueHead::new(&vs, W as i64, H as i64);
    let mut opt = nn::Adam::default().build(&vs, 0.001).unwrap();

    let (evaluate_xs, evaluate_targets) = {
        let mut evaluate_boards = Vec::new();
        for _ in 0..num_boards_to_evaluate {
            evaluate_boards.push(rand_board_for_tablebase());
        }
        (
            boards_to_tensor(&evaluate_boards, dev),
            tb_results_to_tensor_value(&evaluate_boards, &tb),
        )
    };

    for epoch in 0..num_epochs {
        let timer = Timer::new();
        // train
        {
            let mut boards = Vec::new();
            for _ in 0..num_boards_per_epoch {
                boards.push(rand_board_for_tablebase());
            }
            let xs = boards_to_tensor(&boards, dev);
            let targets = tb_results_to_tensor_value(&boards, &tb);

            let loss = value_head
                .forward_t(&body.forward_t(&xs, true), true)
                .mse_loss(&targets, Reduction::Mean);

            opt.backward_step(&loss);
        }

        // evaluate
        {
            let error = value_head
                .forward_t(&body.forward_t(&evaluate_xs, false), false)
                .mse_loss(&evaluate_targets, Reduction::Mean)
                .double_value(&[]);
            let elapsed = timer.elapsed();
            println!("epoch {epoch:3}: error {error:.3} ({elapsed:?})");
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::board::{presets, BoardSquare};

    use super::*;

    fn board_tensor_val(t: &Tensor, player: Player, ty: Type, x: i64, y: i64) -> i64 {
        t.int64_value(&[
            match player {
                Player::White => 0,
                Player::Black => NUM_PIECE_TYPES,
            } + piece_type_to_index(ty),
            y,
            x,
        ])
    }

    #[test]
    fn test_board_to_tensor() {
        let dev = Device::Cpu;
        let board = presets::los_alamos();
        let t = board_to_tensor(&board, dev);
        assert_eq!(t.size(), vec![10, 6, 6]);
        assert_eq!(board_tensor_val(&t, Player::White, Type::King, 3, 0), 1);
        assert_eq!(board_tensor_val(&t, Player::White, Type::Queen, 2, 0), 1);
        assert_eq!(board_tensor_val(&t, Player::Black, Type::King, 3, 0), 0);
        assert_eq!(board_tensor_val(&t, Player::White, Type::King, 2, 0), 0);
        assert_eq!(board_tensor_val(&t, Player::White, Type::Queen, 3, 0), 0);
        assert_eq!(board_tensor_val(&t, Player::Black, Type::King, 3, 5), 1);
    }

    #[test]
    fn test_board_to_tensor_size() {
        let dev = Device::Cpu;
        let board = BoardSquare::<6, 7>::default();
        let t = board_to_tensor(&board, dev);
        assert_eq!(t.size(), vec![10, 7, 6]);
    }

    #[test]
    fn test_boards_to_tensor() {
        let dev = Device::Cpu;
        let board1 = presets::los_alamos();
        let board2 = BoardSquare::<6, 6>::default();
        let board3 = presets::los_alamos();

        let t = boards_to_tensor(&[board1, board2, board3], dev);
        assert_eq!(t.size(), vec![3, 10, 6, 6]);
        assert_eq!(
            board_tensor_val(&t.i(0), Player::White, Type::King, 3, 0),
            1
        );
        assert_eq!(
            board_tensor_val(&t.i(1), Player::White, Type::King, 3, 0),
            0
        );
        assert_eq!(
            board_tensor_val(&t.i(2), Player::White, Type::King, 3, 0),
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
    fn test_policy_head() {
        let dev = Device::cuda_if_available();
        let vs = VarStore::new(dev);
        let boards = [presets::los_alamos(), Default::default()];
        let xs = boards_to_tensor(&boards, dev);
        let body = NNBody::new(&vs);
        let policy_head =
            NNPolicyHead::new(&vs, boards[0].width() as i64, boards[0].height() as i64, 8);
        let t = policy_head.forward_t(&body.forward_t(&xs, false), false);
        assert_eq!(t.size2().unwrap(), (2, 8));
    }

    #[test]
    fn test_value_head() {
        let dev = Device::cuda_if_available();
        let vs = VarStore::new(dev);
        let boards = [presets::los_alamos(), Default::default()];
        let xs = boards_to_tensor(&boards, dev);
        let body = NNBody::new(&vs);
        let value_head = NNValueHead::new(&vs, boards[0].width() as i64, boards[0].height() as i64);
        let t = value_head.forward_t(&body.forward_t(&xs, false), false);
        assert_eq!(t.size1().unwrap(), 2);
    }

    #[test]
    fn test_end_to_end() {
        let probs =
            move_probabilities_for_boards(&[presets::los_alamos(), BoardSquare::<6, 6>::default()]);
        assert_eq!(probs.len(), 2);
        assert_eq!(probs[0].len(), 10);
        assert_eq!(probs[1].len(), 0);
    }

    #[test]
    fn test_train_nn_tablebase_policy() {
        train_nn_tablebase_policy::<4, 4>(1, 3, 1);
    }

    #[test]
    fn test_train_nn_tablebase_value() {
        train_nn_tablebase_value::<4, 4>(1, 3, 1);
    }
}
