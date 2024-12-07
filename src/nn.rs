use crate::board::{board_piece_to_square, presets, Board, BoardSquare, Move};
use crate::coord::Coord;
use crate::moves;
use crate::piece::{Piece, Type};
use crate::player::Player;
use crate::tablebase::{generate_tablebase, PieceSet, TBBoard, TBMoveType, Tablebase};
use crate::timer::Timer;
use log::{info, warn};
use rand::{thread_rng, Rng};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
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

fn board_to_tensor<B: Board>(original_board: &B, player: Player, dev: Device) -> Tensor {
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

fn boards_to_tensor<B: Board>(boards: &[B], player: Player, dev: Device) -> Tensor {
    let ts = boards
        .iter()
        .map(|b| board_to_tensor(b, player, dev))
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

fn move_probabilities<const W: usize, const H: usize>(
    v: &[f32],
    all_possible_moves: &[Move],
    state: &BoardState<W, H>,
) -> FxHashMap<Move, f32> {
    assert_eq!(v.len(), all_possible_moves.len());
    let legal_moves = moves::all_moves(&state.board, state.player)
        .into_iter()
        .collect::<HashSet<_>>();
    let mut ret = FxHashMap::default();
    let mut total_prob = 0.0;
    for (&prob, &m) in v.iter().zip(all_possible_moves.iter()) {
        // all_possible_moves is always indexed as if the current player is White, so flip if current player is Black
        let actual_move = match state.player {
            Player::White => m,
            Player::Black => Move {
                from: Coord::new(m.from.x, H as i8 - 1 - m.from.y),
                to: Coord::new(m.to.x, H as i8 - 1 - m.to.y),
            },
        };
        if legal_moves.contains(&actual_move) {
            total_prob += prob;
            ret.insert(actual_move, prob);
        }
    }
    for p in ret.values_mut() {
        *p /= total_prob;
    }
    ret
}

struct NNBody {
    seqs: Vec<SequentialT>,
}

const BODY_NUM_FILTERS: i64 = 32;
const BODY_NUM_LAYERS: i64 = 6;

impl NNBody {
    fn new(vs: &VarStore) -> Self {
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct BoardState<const W: usize, const H: usize> {
    // this is always the unflipped board
    board: BoardSquare<W, H>,
    player: Player,
}

impl<const W: usize, const H: usize> BoardState<W, H> {
    fn make_move(&self, m: Move) -> Self {
        let mut clone = self.board.clone();
        clone.make_move(m);
        Self {
            board: clone,
            player: self.player.next(),
        }
    }
}

enum MctsTreeNode {
    Unexplored,
    Parent(MctsParent),
    Leaf(f32),
}

struct MctsParent {
    policy: FxHashMap<Move, f32>,
    // a -> (q, n)
    children: FxHashMap<Move, MctsChild>,
}

struct MctsChild {
    q: f32,
    n: f32,
    node: MctsTreeNode,
}

#[derive(Debug)]
struct MctsExample<const W: usize, const H: usize> {
    state: BoardState<W, H>,
    policy: FxHashMap<Move, f32>,
    reward: f32,
}

struct Mcts<const W: usize, const H: usize> {
    all_possible_moves: Vec<Move>,
    all_possible_moves_idx: FxHashMap<Move, usize>,
    exploration_factor: f32,
    learning_rate: f64,
    num_rollouts_per_game: usize,
    num_games_per_epoch: usize,
    nn_body: NNBody,
    nn_policy_head: NNPolicyHead,
    nn_value_head: NNValueHead,
    dev: Device,
    var_store: VarStore,
}

const MAX_DEPTH: usize = 50;

impl<const W: usize, const H: usize> Mcts<W, H> {
    fn new(dev: Device) -> Self {
        let all_possible_moves = all_moves(W as i8, H as i8);
        let mut all_possible_moves_idx = FxHashMap::default();
        for (i, &m) in all_possible_moves.iter().enumerate() {
            all_possible_moves_idx.insert(m, i);
        }
        let num_moves = all_possible_moves.len();

        let var_store = VarStore::new(dev);

        Self {
            all_possible_moves,
            all_possible_moves_idx,
            exploration_factor: 0.1,
            learning_rate: 0.4,
            num_games_per_epoch: 50,
            num_rollouts_per_game: 100,
            nn_body: NNBody::new(&var_store),
            nn_policy_head: NNPolicyHead::new(&var_store, W as i64, H as i64, num_moves as i64),
            nn_value_head: NNValueHead::new(&var_store, W as i64, H as i64),
            dev,
            var_store,
        }
    }

    fn save_vars<T: AsRef<Path>>(&self, p: T) {
        self.var_store.save(p).unwrap()
    }

    fn load_vars<T: AsRef<Path>>(&mut self, p: T) {
        self.var_store.load(p).unwrap()
    }
}

impl<const W: usize, const H: usize> Clone for Mcts<W, H> {
    fn clone(&self) -> Self {
        let mut ret = Self::new(self.dev);
        ret.var_store.copy(&self.var_store).unwrap();

        ret
    }
}

struct TrainingStats {
    num_zero_examples: usize,
    num_pos_one_examples: usize,
    num_neg_one_examples: usize,
    value_loss: f64,
    policy_loss: f64,
}

impl<const W: usize, const H: usize> Mcts<W, H> {
    fn board_value(
        &self,
        state: &BoardState<W, H>,
        depth: usize,
        visited: &FxHashSet<BoardState<W, H>>,
    ) -> Option<f32> {
        // if game is too long, consider it a draw
        if depth >= MAX_DEPTH {
            return Some(0.0);
        }

        // if king is captured, player loses
        if state.board.maybe_king_coord(state.player).is_none() {
            return Some(-1.0);
        }

        // if state has been seen this game before, consider it a draw
        if visited.contains(state) {
            return Some(0.0);
        }

        // TODO: use tablebase
        None
    }

    fn value_and_policy(&self, state: &BoardState<W, H>) -> (f32, FxHashMap<Move, f32>) {
        let xs = boards_to_tensor(&[state.board.clone()], state.player, self.dev);

        let bt = self.nn_body.forward_t(&xs, false);
        let pt = self.nn_policy_head.forward_t(&bt, false);
        let vt = self.nn_value_head.forward_t(&bt, false);

        let len = pt.size2().unwrap().1 as usize;
        let mut probs: Vec<f32> = vec![0.0; len];
        pt.softmax(-1, Kind::Float).i(0).copy_data(&mut probs, len);
        let policy = move_probabilities(&probs, &self.all_possible_moves, state);

        (vt.double_value(&[0]) as f32, policy)
    }

    fn policy_to_training_data(&self, policy: &FxHashMap<Move, f32>) -> Tensor {
        let t = Tensor::zeros(
            self.all_possible_moves.len() as i64,
            (Kind::Float, self.dev),
        );
        for (m, &p) in policy {
            let _ = t.i(self.all_possible_moves_idx[m] as i64).fill_(p as f64);
        }
        t
    }

    // (board, vs, ps)
    fn examples_to_training_data(
        &self,
        examples: &[MctsExample<W, H>],
    ) -> (Tensor, Tensor, Tensor) {
        #[cfg(debug_assertions)]
        for e in examples {
            assert!(e.reward.is_finite());
            for p in e.policy.values() {
                assert!(p.is_finite());
            }
        }
        let boards = examples
            .iter()
            .map(|e| board_to_tensor(&e.state.board, e.state.player, self.dev))
            .collect::<Vec<_>>();

        let vs = examples.iter().map(|e| e.reward).collect::<Vec<_>>();

        let ps = examples
            .iter()
            .map(|e| self.policy_to_training_data(&e.policy))
            .collect::<Vec<_>>();

        (
            Tensor::stack(&boards, 0),
            Tensor::from_slice(&vs).to_device(self.dev),
            Tensor::stack(&ps, 0),
        )
    }

    fn explore_impl(
        &self,
        node: &mut MctsTreeNode,
        state: &BoardState<W, H>,
        visited: &mut FxHashSet<BoardState<W, H>>,
        depth: usize,
    ) -> f32 {
        match node {
            MctsTreeNode::Unexplored => {
                if let Some(v) = self.board_value(state, depth, visited) {
                    *node = MctsTreeNode::Leaf(v);
                    v
                } else {
                    let (v, p) = self.value_and_policy(state);
                    *node = MctsTreeNode::Parent(MctsParent {
                        policy: p,
                        children: Default::default(),
                    });
                    v
                }
            }
            MctsTreeNode::Leaf(v) => *v,
            MctsTreeNode::Parent(node) => {
                let parent_n_sqrt = node.children.values().map(|c| c.n).sum::<f32>().sqrt();
                let u = |m: Move| {
                    let child = node.children.get(&m);
                    let q = child.map_or(0.0, |child| child.q / child.n);
                    let n = child.map_or(0.0, |child| child.n);
                    let p = node.policy[&m];
                    q + self.exploration_factor * p * parent_n_sqrt / (n + 1.0)
                };
                let all_legal_moves = moves::all_moves(&state.board, state.player);
                let best_move = all_legal_moves
                    .iter()
                    .copied()
                    .max_by(|a, b| u(*a).total_cmp(&u(*b)))
                    .unwrap();

                let next_state = state.make_move(best_move);
                let child = node.children.entry(best_move).or_insert(MctsChild {
                    q: 0.0,
                    n: 0.0,
                    node: MctsTreeNode::Unexplored,
                });
                visited.insert(state.clone());
                let v = -self.explore_impl(&mut child.node, &next_state, visited, depth + 1);
                visited.remove(state);

                child.q += v;
                child.n += 1.0;
                v
            }
        }
    }

    fn node_improved_policy(p: &MctsParent, temperature: f32) -> FxHashMap<Move, f32> {
        assert!(temperature >= 0.0);
        if temperature == 0.0 {
            let argmax = p
                .children
                .iter()
                .max_by(|a, b| a.1.n.total_cmp(&b.1.n))
                .unwrap()
                .0;
            let mut ret = FxHashMap::<Move, f32>::default();
            ret.insert(*argmax, 1.0);
            ret
        } else {
            let inv_temp = 1.0;
            // let inv_temp = 1.0 / temperature;
            let adjusted_children = p
                .children
                .iter()
                .map(|(m, c)| (*m, c.n.powf(inv_temp)))
                .collect::<Vec<_>>();
            let total_n = adjusted_children.iter().fold(0.0, |a, b| a + b.1);
            adjusted_children
                .into_iter()
                .map(|(m, n)| (m, n / total_n))
                .collect()
        }
    }

    fn random_move(policy: &FxHashMap<Move, f32>) -> Move {
        let mut rand: f32 = thread_rng().gen_range(0.0..1.0);
        let mut last_move = None;
        for (&m, p) in policy {
            rand -= p;
            if rand <= 0.0 {
                return m;
            }
            last_move = Some(m);
        }
        dbg!(policy);
        warn!("random_move: probabilities didn't sum to 1.0?");
        last_move.unwrap()
    }

    fn get_examples(&self, board: &BoardSquare<W, H>) -> Vec<MctsExample<W, H>> {
        let mut examples = Vec::<MctsExample<W, H>>::default();

        let mut tree = MctsTreeNode::Unexplored;
        let mut state = BoardState {
            board: board.clone(),
            player: Player::White,
        };
        let mut depth = 0;
        loop {
            // perform some rollouts from current depth
            for _ in 0..self.num_rollouts_per_game {
                self.explore_impl(&mut tree, &state, &mut Default::default(), depth);
            }
            match tree {
                MctsTreeNode::Unexplored => panic!("unexplored node?"),
                MctsTreeNode::Leaf(v) => {
                    for e in &mut examples {
                        e.reward = v;
                        if e.state.player != state.player {
                            e.reward *= -1.0;
                        }
                    }
                    break;
                }
                MctsTreeNode::Parent(mut p) => {
                    let improved_policy = Self::node_improved_policy(&p, 1.0);
                    let m = Self::random_move(&improved_policy);
                    examples.push(MctsExample {
                        state: state.clone(),
                        policy: improved_policy,
                        reward: f32::NAN,
                    });
                    tree = p.children.remove(&m).unwrap().node;
                    state = state.make_move(m);
                    depth += 1;
                }
            }
        }
        examples
    }

    fn train(&mut self, board: &BoardSquare<W, H>) -> TrainingStats {
        let mut examples = Vec::default();
        for _ in 0..self.num_games_per_epoch {
            examples.append(&mut self.get_examples(board));
        }
        let mut num_zero_examples = 0;
        let mut num_pos_one_examples = 0;
        let mut num_neg_one_examples = 0;
        for e in &examples {
            if e.reward == 0.0 {
                num_zero_examples += 1
            } else if e.reward == 1.0 {
                num_pos_one_examples += 1
            } else if e.reward == -1.0 {
                num_neg_one_examples += 1
            } else {
                panic!();
            }
        }
        let (xs, target_vs, target_ps) = self.examples_to_training_data(&examples);

        let bt = self.nn_body.forward_t(&xs, true);
        let ps = self.nn_policy_head.forward_t(&bt, true);
        let vs = self.nn_value_head.forward_t(&bt, true);
        let vs_loss = vs.mse_loss(&target_vs, Reduction::Mean);
        let ps_loss = ps.cross_entropy_loss::<Tensor>(&target_ps, None, Reduction::Mean, -100, 0.0);
        let value_loss = vs_loss.double_value(&[]);
        let policy_loss = ps_loss.double_value(&[]);

        let loss = vs_loss + ps_loss;
        let mut opt = nn::Sgd::default()
            .build(&self.var_store, self.learning_rate)
            .unwrap();
        opt.backward_step(&loss);

        TrainingStats {
            num_zero_examples,
            num_pos_one_examples,
            num_neg_one_examples,
            value_loss,
            policy_loss,
        }
    }
}

fn should_take_new_model<const W: usize, const H: usize>(
    _board: &BoardSquare<W, H>,
    _old: &Mcts<W, H>,
    _new: &Mcts<W, H>,
) -> bool {
    true
}

pub fn train_ai(
    load_vars_path: Option<PathBuf>,
    save_vars_path: Option<PathBuf>,
    stats_path: Option<PathBuf>,
    epochs: usize,
    learning_rate: f64,
    exploration_factor: f32,
    num_games_per_epoch: usize,
    num_rollouts_per_game: usize,
) {
    let mut stats_file = stats_path.map(|p| File::create(p).unwrap());
    let board = presets::mini();

    let dev = Device::cuda_if_available();
    let mut mcts = Mcts::new(dev);
    if let Some(load) = &load_vars_path {
        info!("loading weights from {load:?}");
        mcts.load_vars(load);
    }
    mcts.exploration_factor = exploration_factor;
    mcts.learning_rate = learning_rate;
    mcts.num_games_per_epoch = num_games_per_epoch;
    mcts.num_rollouts_per_game = num_rollouts_per_game;
    for epoch in 0..epochs {
        info!("-- begin epoch {epoch}");

        let timer = Timer::new();
        let prev = mcts.clone();
        let stats = mcts.train(&board);

        info!("num examples with reward  0.0: {}", stats.num_zero_examples);
        info!(
            "num examples with reward  1.0: {}",
            stats.num_pos_one_examples
        );
        info!(
            "num examples with reward -1.0: {}",
            stats.num_neg_one_examples
        );
        info!("value loss:  {}", stats.value_loss);
        info!("policy loss: {}", stats.policy_loss);
        info!("total loss:  {}", stats.value_loss + stats.policy_loss);

        if let Some(s) = &mut stats_file {
            writeln!(
                s,
                "{},{},{}",
                stats.value_loss,
                stats.policy_loss,
                stats.value_loss + stats.policy_loss
            )
            .unwrap();
        }

        info!("pitting old vs new model...");
        if should_take_new_model(&board, &prev, &mcts) {
            info!("using new model");
            if let Some(save) = &save_vars_path {
                info!("saving weights to {save:?}");
                mcts.save_vars(save);
            }
        } else {
            info!("reverting to old model");
            mcts = prev;
        }
        let elapsed = timer.elapsed();
        info!("epoch {epoch} took {elapsed:?}");
    }

    println!("---------------------------");
    println!("playing game...");
    let mut state = BoardState {
        board: board.clone(),
        player: Player::White,
    };
    let mut visited = FxHashSet::default();
    loop {
        println!("move {}", visited.len());
        println!("{:?}", &state.board);
        if !visited.insert(state.clone()) {
            println!("draw by repetition");
            break;
        }
        if state.board.maybe_king_coord(Player::White).is_none() {
            println!("black wins");
            break;
        }
        if state.board.maybe_king_coord(Player::Black).is_none() {
            println!("white wins");
            break;
        }

        let (_, policy) = mcts.value_and_policy(&state);
        let mut best_p = 0.0;
        let mut best_m = Move {
            from: Coord::new(0, 0),
            to: Coord::new(0, 0),
        };
        dbg!(&policy);
        for (m, p) in policy {
            if p > best_p {
                best_p = p;
                best_m = m;
            }
        }
        state = state.make_move(best_m);
    }
}

#[cfg(test)]
mod tests {
    use crate::board::presets;

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

    #[test]
    fn test_train_nn_tablebase_policy() {
        train_nn_tablebase_policy::<4, 4>(1, 3, 1);
    }

    #[test]
    fn test_train_nn_tablebase_value() {
        train_nn_tablebase_value::<4, 4>(1, 3, 1);
    }

    #[test]
    fn test_get_examples_immediate_win_white() {
        let board = BoardSquare::<1, 2>::with_pieces(&[
            (Coord::new(0, 0), Piece::new(Player::White, Type::King)),
            (Coord::new(0, 1), Piece::new(Player::Black, Type::King)),
        ]);

        let dev = Device::Cpu;
        let mcts = Mcts::new(dev);
        let examples = mcts.get_examples(&board);
        assert_eq!(examples.len(), 1);
        assert_eq!(examples[0].reward, 1.0);
        assert_eq!(examples[0].state.player, Player::White);
        assert_eq!(examples[0].policy.len(), 1);
        assert_eq!(
            examples[0].policy[&Move {
                from: Coord::new(0, 0),
                to: Coord::new(0, 1)
            }],
            1.0
        );
    }

    #[test]
    fn test_get_examples_immediate_win_black() {
        let board = BoardSquare::<1, 3>::with_pieces(&[
            (Coord::new(0, 0), Piece::new(Player::White, Type::King)),
            (Coord::new(0, 2), Piece::new(Player::Black, Type::King)),
        ]);

        let dev = Device::Cpu;
        let mcts = Mcts::new(dev);
        let examples = mcts.get_examples(&board);
        assert_eq!(examples.len(), 2);
        assert_eq!(examples[0].reward, -1.0);
        assert_eq!(examples[0].state.player, Player::White);
        assert_eq!(examples[0].policy.len(), 1);
        assert_eq!(
            examples[0].policy[&Move {
                from: Coord::new(0, 0),
                to: Coord::new(0, 1)
            }],
            1.0
        );
        assert_eq!(examples[1].reward, 1.0);
        assert_eq!(examples[1].state.player, Player::Black);
        assert_eq!(examples[1].policy.len(), 1);
        assert_eq!(
            examples[1].policy[&Move {
                from: Coord::new(0, 2),
                to: Coord::new(0, 1)
            }],
            1.0
        );
    }

    #[test]
    fn test_get_examples_immediate_draw_repetition() {
        let board = BoardSquare::<1, 6>::with_pieces(&[
            (Coord::new(0, 0), Piece::new(Player::White, Type::King)),
            (Coord::new(0, 2), Piece::new(Player::White, Type::Knight)),
            (Coord::new(0, 3), Piece::new(Player::Black, Type::Knight)),
            (Coord::new(0, 5), Piece::new(Player::Black, Type::King)),
        ]);

        let dev = Device::Cpu;
        let mcts = Mcts::new(dev);
        let examples = mcts.get_examples(&board);
        assert_eq!(examples.len(), 4);
        for e in examples {
            assert_eq!(e.reward, 0.0);
        }
    }

    #[test]
    fn test_get_examples() {
        let board = presets::los_alamos();

        let dev = Device::Cpu;
        let mcts = Mcts::new(dev);
        let examples = mcts.get_examples(&board);
        assert!(examples.len() > 4);
        assert!(examples.len() <= MAX_DEPTH);
        let first_player = examples[0].state.player;
        let first_reward = examples[0].reward;
        for e in examples {
            let prob_sum = e.policy.iter().fold(0.0, |total, (_, &p)| total + p);
            assert!(prob_sum > 0.99 && prob_sum < 1.01);
            assert_eq!(
                e.reward,
                first_reward
                    * if e.state.player == first_player {
                        1.0
                    } else {
                        -1.0
                    }
            );
        }
    }

    #[test]
    fn test_value_policy() {
        let board = presets::los_alamos();

        let dev = Device::Cpu;
        let mcts = Mcts::new(dev);
        let (v, p) = mcts.value_and_policy(&BoardState {
            board,
            player: Player::White,
        });
        assert!(v >= -1.0 && v <= 1.0);
        let prob_sum = p.iter().fold(0.0, |total, (_, &p)| total + p);
        assert!(prob_sum > 0.99 && prob_sum < 1.01);
    }

    #[test]
    fn test_mcts_clone() {
        let dev = Device::Cpu;
        let state = BoardState {
            board: presets::mini(),
            player: Player::White,
        };
        let mcts1 = Mcts::new(dev);
        let mcts2 = Mcts::new(dev);
        let mcts_clone = mcts1.clone();
        let (v1, _) = mcts1.value_and_policy(&state);
        let (v2, _) = mcts2.value_and_policy(&state);
        let (v3, _) = mcts_clone.value_and_policy(&state);
        assert_ne!(v1, v2);
        assert_eq!(v1, v3);
    }

    #[test]
    fn test_mcts_white_black() {
        let dev = Device::Cpu;
        let mcts = Mcts::new(dev);
        let (v1, _) = mcts.value_and_policy(&BoardState {
            board: presets::mini(),
            player: Player::White,
        });
        let (v2, _) = mcts.value_and_policy(&BoardState {
            board: presets::mini(),
            player: Player::Black,
        });
        assert_eq!(v1, v2);
    }
}
