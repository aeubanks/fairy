use crate::board::{board_piece_to_square, Board};
use crate::coord::Coord;
use crate::nn::{boards_to_tensor, NNBody, NNPolicyHead, NNValueHead};
use crate::piece::{Piece, Type};
use crate::player::Player;
use crate::tablebase::{generate_tablebase, PieceSet, TBBoard, TBMoveType, Tablebase};
use crate::timer::Timer;
use log::info;
use rand::{thread_rng, Rng};
use tch::{nn::*, *};

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
    dev: Device,
) -> Tensor {
    let ts = boards.iter().map(|b| tb_result(b, tb)).collect::<Vec<_>>();
    Tensor::from_slice(&ts).to_device(dev)
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
    dev: Device,
) -> Tensor {
    let ts = boards.iter().map(|b| tb_result2(b, tb)).collect::<Vec<_>>();
    Tensor::from_slice(&ts).to_device(dev)
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
            boards_to_tensor(&evaluate_boards, Player::White, dev),
            tb_results_to_tensor_policy(&evaluate_boards, &tb, dev),
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
            let xs = boards_to_tensor(&boards, Player::White, dev);
            let targets = tb_results_to_tensor_policy(&boards, &tb, dev);

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
            info!("epoch {epoch}: accuracy {acc:.3} ({elapsed:?})");
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
            boards_to_tensor(&evaluate_boards, Player::White, dev),
            tb_results_to_tensor_value(&evaluate_boards, &tb, dev),
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
            let xs = boards_to_tensor(&boards, Player::White, dev);
            let targets = tb_results_to_tensor_value(&boards, &tb, dev);

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
            info!("epoch {epoch:3}: error {error:.3} ({elapsed:?})");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_nn_tablebase_policy() {
        train_nn_tablebase_policy::<4, 4>(1, 3, 1);
    }

    #[test]
    fn test_train_nn_tablebase_value() {
        train_nn_tablebase_value::<4, 4>(1, 3, 1);
    }
}
