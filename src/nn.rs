use crate::board::Board;
use crate::piece::Type;
use crate::player::Player;
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

pub fn board_to_tensor<B: Board>(original_board: &B, player: Player, dev: Device) -> Tensor {
    let board = original_board.make_player_white(player);
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

pub fn boards_to_tensor<B: Board>(boards: &[B], player: Player, dev: Device) -> Tensor {
    let ts = boards
        .iter()
        .map(|b| board_to_tensor(b, player, dev))
        .collect::<Vec<_>>();
    Tensor::stack(&ts, 0)
}

pub struct NNBody {
    seqs: Vec<SequentialT>,
}

const BODY_NUM_FILTERS: i64 = 32;
const BODY_NUM_LAYERS: i64 = 6;

impl NNBody {
    pub fn new(vs: &VarStore) -> Self {
        let mut conv_config = ConvConfigND::<[i64; 2]> {
            padding: [1, 1],
            ..Default::default()
        };
        conv_config.padding = [1, 1];

        let mut seqs = Vec::new();

        for i in 0..BODY_NUM_LAYERS {
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

pub struct NNValueHead {
    seq: SequentialT,
}

const VALUE_HEAD_NUM_FILTERS: i64 = 1;
const VALUE_HEAD_HIDDEN_LAYER_SIZE: i64 = 64;

impl NNValueHead {
    pub fn new(vs: &VarStore, input_width: i64, input_height: i64) -> Self {
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

pub struct NNPolicyHead {
    seq: SequentialT,
}

const POLICY_HEAD_NUM_FILTERS: i64 = 2;

impl NNPolicyHead {
    pub fn new(vs: &VarStore, input_width: i64, input_height: i64, num_outputs: i64) -> Self {
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

#[cfg(test)]
mod tests {
    use crate::board::presets;
    use crate::board::BoardSquare;
    use crate::coord::Coord;
    use crate::piece::Piece;

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
        let t = board_to_tensor(&board, Player::White, dev);
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
        let t = board_to_tensor(&board, Player::White, dev);
        assert_eq!(t.size(), vec![10, 7, 6]);
    }

    #[test]
    fn test_board_to_tensor_black() {
        let dev = Device::Cpu;
        let board = BoardSquare::<6, 6>::with_pieces(&[(
            Coord::new(1, 2),
            Piece::new(Player::Black, Type::King),
        )]);
        let t = board_to_tensor(&board, Player::Black, dev);
        assert_eq!(t.size(), vec![10, 6, 6]);
        assert_eq!(board_tensor_val(&t, Player::White, Type::King, 1, 3), 1);
        assert_eq!(board_tensor_val(&t, Player::White, Type::King, 1, 2), 0);
        assert_eq!(board_tensor_val(&t, Player::Black, Type::King, 1, 2), 0);
        assert_eq!(board_tensor_val(&t, Player::Black, Type::King, 1, 3), 0);
    }

    #[test]
    fn test_boards_to_tensor() {
        let dev = Device::Cpu;
        let board1 = presets::los_alamos();
        let board2 = BoardSquare::<6, 6>::default();
        let board3 = presets::los_alamos();

        let t = boards_to_tensor(&[board1, board2, board3], Player::White, dev);
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
    fn test_policy_head() {
        let dev = Device::cuda_if_available();
        let vs = VarStore::new(dev);
        let boards = [presets::los_alamos(), Default::default()];
        let xs = boards_to_tensor(&boards, Player::White, dev);
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
        let xs = boards_to_tensor(&boards, Player::White, dev);
        let body = NNBody::new(&vs);
        let value_head = NNValueHead::new(&vs, boards[0].width() as i64, boards[0].height() as i64);
        let t = value_head.forward_t(&body.forward_t(&xs, false), false);
        assert_eq!(t.size1().unwrap(), 2);
    }
}
