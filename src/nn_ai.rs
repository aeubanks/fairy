use crate::board::{presets, Board, BoardSquare, Move};
use crate::coord::Coord;
use crate::moves;
use crate::nn::{board_to_tensor, boards_to_tensor, NNBody, NNPolicyHead, NNValueHead};
use crate::piece::Type;
use crate::player::Player;
use crate::timer::Timer;
use derive_enum::EnumFrom;
use log::info;
use rand::{thread_rng, Rng};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use tch::{nn::*, *};

fn all_possible_moves(width: i8, height: i8) -> Vec<Move> {
    let offsets = {
        let mut offsets = HashSet::new();
        for ty in Type::all() {
            if ty == Type::Pawn {
                continue;
            }
            offsets.extend(ty.leaper_offsets());
            offsets.extend(ty.rider_offsets());
        }
        offsets.into_iter().collect::<Vec<_>>()
    };
    let mut moves = HashSet::new();
    for y in 0..height {
        for x in 0..width {
            let in_bounds = |c: Coord| c.x >= 0 && c.x < width && c.y >= 0 && c.y < height;
            let from = Coord::new(x, y);
            for &d in &offsets {
                let mut try_to = d + from;
                while in_bounds(try_to) {
                    moves.insert(Move { from, to: try_to });
                    try_to = try_to + d;
                }
            }
        }
    }
    let mut v = moves.into_iter().collect::<Vec<_>>();
    v.sort_by_key(|c| (c.from.x, c.from.y, c.to.x, c.to.y));
    v
}

fn move_probabilities<const W: usize, const H: usize>(
    v: &[f32],
    all_possible_moves: &[Move],
    state: &BoardState<W, H>,
) -> FxHashMap<Move, f32> {
    assert_eq!(v.len(), all_possible_moves.len());
    let legal_moves = moves::all_legal_moves(&state.board, state.player)
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

struct MctsNode {
    n: f32,
    ty: MctsNodeType,
}

impl MctsNode {
    fn v(&self) -> f32 {
        let ret = match &self.ty {
            MctsNodeType::Leaf(v) => *v,
            MctsNodeType::Parent(c) => c.q / self.n,
        };
        if ret < -1.0 {
            println!("{}", ret);
        }
        assert!(ret >= -1.0);
        assert!(ret <= 1.0);
        ret
    }
}

enum MctsNodeType {
    Leaf(f32),
    Parent(MctsParent),
}

struct MctsParent {
    q: f32,
    visited: bool,
    policy: FxHashMap<Move, f32>,
    // None -> unexpanded
    // Some -> expanded
    children: FxHashMap<Move, MctsNode>,
}

#[derive(Debug)]
struct MctsExample<const W: usize, const H: usize> {
    state: BoardState<W, H>,
    policy: FxHashMap<Move, f32>,
    reward: f32,
}

struct MctsParams {
    exploration_factor: f32,
    learning_rate: f64,
    num_rollouts_per_state: usize,
    num_games_per_epoch: usize,
}

impl Default for MctsParams {
    fn default() -> Self {
        Self {
            exploration_factor: 0.1,
            learning_rate: 0.1,
            num_games_per_epoch: 20,
            num_rollouts_per_state: 50,
        }
    }
}

#[cfg(test)]
impl MctsParams {
    fn for_testing() -> Self {
        Self {
            exploration_factor: 0.1,
            learning_rate: 0.1,
            num_games_per_epoch: 4,
            num_rollouts_per_state: 10,
        }
    }
}

struct Mcts<const W: usize, const H: usize> {
    all_possible_moves: Vec<Move>,
    all_possible_moves_idx: FxHashMap<Move, usize>,
    exploration_factor: f32,
    learning_rate: f64,
    num_rollouts_per_state: usize,
    num_games_per_epoch: usize,
    nn_body: NNBody,
    nn_policy_head: NNPolicyHead,
    nn_value_head: NNValueHead,
    dev: Device,
    var_store: VarStore,
}

const MAX_DEPTH: usize = 200;

impl<const W: usize, const H: usize> Mcts<W, H> {
    fn new(dev: Device) -> Self {
        Self::with_params(dev, MctsParams::default())
    }

    fn with_params(dev: Device, params: MctsParams) -> Self {
        let all_possible_moves = all_possible_moves(W as i8, H as i8);
        let mut all_possible_moves_idx = FxHashMap::default();
        for (i, &m) in all_possible_moves.iter().enumerate() {
            all_possible_moves_idx.insert(m, i);
        }
        let num_moves = all_possible_moves.len();

        let var_store = VarStore::new(dev);

        Self {
            all_possible_moves,
            all_possible_moves_idx,
            exploration_factor: params.exploration_factor,
            learning_rate: params.learning_rate,
            num_games_per_epoch: params.num_games_per_epoch,
            num_rollouts_per_state: params.num_rollouts_per_state,
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

    #[must_use]
    fn expand_node_children(
        &self,
        state: &BoardState<W, H>,
        all_legal_moves: &[Move],
        depth: usize,
        visited: &FxHashSet<BoardState<W, H>>,
    ) -> FxHashMap<Move, MctsNode> {
        let mut children = FxHashMap::default();
        for &m in all_legal_moves {
            let next_state = state.make_move(m);
            let next_node = if let Some(v) = self.board_value(&next_state, depth + 1, visited) {
                MctsNode {
                    n: 1.0,
                    ty: MctsNodeType::Leaf(v),
                }
            } else {
                let (v, policy) = self.value_and_policy(&next_state);
                MctsNode {
                    n: 1.0,
                    ty: MctsNodeType::Parent(MctsParent {
                        visited: false,
                        q: v,
                        policy,
                        children: Default::default(),
                    }),
                }
            };
            children.insert(m, next_node);
        }
        children
    }

    fn perform_one_rollout(
        &self,
        node: &mut MctsNode,
        state: &BoardState<W, H>,
        visited: &mut FxHashSet<BoardState<W, H>>,
        depth: usize,
    ) -> f32 {
        let q = match &mut node.ty {
            MctsNodeType::Leaf(v) => *v,
            MctsNodeType::Parent(parent) => {
                let all_legal_moves = moves::all_legal_moves(&state.board, state.player);
                let q = if parent.visited {
                    if parent.children.is_empty() {
                        parent.children =
                            self.expand_node_children(state, &all_legal_moves, depth, visited);
                    };
                    let n_sqrt = node.n.sqrt();
                    let u = |m: Move| -> f32 {
                        let child = parent.children.get(&m).unwrap();
                        child.v()
                            + self.exploration_factor * parent.policy[&m] * n_sqrt / (child.n + 1.0)
                    };
                    let best_move = all_legal_moves
                        .iter()
                        .copied()
                        .max_by(|a, b| u(*a).total_cmp(&u(*b)))
                        .unwrap();

                    let next_state = state.make_move(best_move);
                    visited.insert(state.clone());
                    let q = -self.perform_one_rollout(
                        parent.children.get_mut(&best_move).unwrap(),
                        &next_state,
                        visited,
                        depth + 1,
                    );
                    visited.remove(state);
                    q
                } else {
                    parent.visited = true;
                    parent.q / node.n
                    // TODO: node.v()
                };
                parent.q += q;
                q
            }
        };

        node.n += 1.0;
        q
    }

    fn node_improved_policy(node: &MctsParent) -> FxHashMap<Move, f32> {
        let adjusted_children = node
            .children
            .iter()
            .map(|(m, c)| (*m, c.n))
            .collect::<Vec<_>>();
        let total_n = adjusted_children.iter().fold(0.0, |a, b| a + b.1);
        adjusted_children
            .into_iter()
            .map(|(m, n)| (m, n / total_n))
            .collect()
    }

    fn random_move(policy: &FxHashMap<Move, f32>, temperature: f32) -> Move {
        let inv_temp = 1.0 / temperature;
        let mut temp_modified_policy = policy
            .iter()
            .map(|(&m, &p)| (m, p.powf(inv_temp)))
            .collect::<Vec<_>>();
        let sum_prob: f32 = temp_modified_policy.iter().map(|&(_, p)| p).sum();
        for (_, p) in &mut temp_modified_policy {
            *p /= sum_prob;
        }
        let mut rand: f32 = thread_rng().gen_range(0.0..1.0);
        for &(m, p) in &temp_modified_policy {
            rand -= p;
            if rand <= 0.0 {
                return m;
            }
        }
        dbg!(&temp_modified_policy);
        panic!("random_move: probabilities didn't sum to 1.0?");
    }

    fn create_root_node(&self, state: &BoardState<W, H>) -> MctsNode {
        let (v, policy) = self.value_and_policy(state);
        MctsNode {
            n: 1.0,
            ty: MctsNodeType::Parent(MctsParent {
                visited: false,
                q: v,
                policy,
                children: Default::default(),
            }),
        }
    }

    fn get_examples(&self, state: &BoardState<W, H>) -> Vec<MctsExample<W, H>> {
        let mut state = state.clone();
        let mut examples = Vec::<MctsExample<W, H>>::default();

        let mut visited = FxHashSet::default();
        let mut node = self.create_root_node(&state);
        let mut depth = 0;
        loop {
            visited.insert(state.clone());
            // perform some rollouts from current depth
            for _ in 0..self.num_rollouts_per_state {
                self.perform_one_rollout(&mut node, &state, &mut visited, depth);
            }
            match node.ty {
                MctsNodeType::Leaf(v) => {
                    for e in &mut examples {
                        assert_ne!(node.n, 0.0);
                        e.reward = v;
                        if e.state.player != state.player {
                            e.reward *= -1.0;
                        }
                    }
                    break;
                }
                MctsNodeType::Parent(mut parent) => {
                    let improved_policy = Self::node_improved_policy(&parent);
                    let m = Self::random_move(&improved_policy, 0.5);
                    examples.push(MctsExample {
                        state: state.clone(),
                        policy: improved_policy,
                        reward: f32::NAN,
                    });
                    node = parent.children.remove(&m).unwrap();
                    state = state.make_move(m);
                    depth += 1;
                }
            }
        }
        examples
    }

    fn train(&mut self, state: &BoardState<W, H>) -> TrainingStats {
        let mut examples = Vec::default();
        for g in 0..self.num_games_per_epoch {
            info!("game {g}/{}", self.num_games_per_epoch);
            examples.append(&mut self.get_examples(state));
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
                panic!("reward is not -1/0/1");
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

        // weight value loss more
        // value loss tends to be order of 0.1
        // policy loss tends to be order of 5
        let loss = 10 * vs_loss + ps_loss;
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
    _state: &BoardState<W, H>,
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
    num_rollouts_per_state: usize,
) {
    let mut stats_file = stats_path.map(|p| File::create(p).unwrap());
    let mut state = BoardState {
        board: presets::mini(),
        player: Player::White,
    };

    let dev = Device::cuda_if_available();
    let params = MctsParams {
        exploration_factor,
        learning_rate,
        num_games_per_epoch,
        num_rollouts_per_state,
    };
    let mut mcts = Mcts::with_params(dev, params);
    if let Some(load) = &load_vars_path {
        info!("loading weights from {load:?}");
        mcts.load_vars(load);
    }
    for epoch in 0..epochs {
        info!("-- begin epoch {epoch}/{epochs}");

        let timer = Timer::new();
        let prev = mcts.clone();
        let stats = mcts.train(&state);

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
        if should_take_new_model(&state, &prev, &mcts) {
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
    let mut visited = FxHashSet::default();
    let mut depth = 0;
    let mut node = mcts.create_root_node(&state);
    loop {
        println!("move {depth}");
        println!("{:?}", &state.board);
        if let MctsNodeType::Leaf(v) = &node.ty {
            println!("mcts says game is finished with value {v}");
        }
        if visited.contains(&state) {
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
        for _ in 0..num_rollouts_per_state {
            mcts.perform_one_rollout(&mut node, &state, &mut visited, depth);
        }
        // get child with highest v()
        let m = *match &node.ty {
            MctsNodeType::Leaf(_) => panic!("evaluating leaf node?"),
            MctsNodeType::Parent(parent) => {
                parent
                    .children
                    .iter()
                    .max_by(|(_, c1), (_, c2)| c1.v().total_cmp(&c2.v()))
                    .unwrap()
                    .0
            }
        };
        visited.insert(state.clone());
        state = state.make_move(m);
        node = match node.ty {
            MctsNodeType::Leaf(_) => panic!("evaluating leaf node?"),
            MctsNodeType::Parent(mut parent) => parent.children.remove(&m).unwrap(),
        };
        depth += 1;
    }
}

#[cfg(test)]
mod tests {
    use crate::board::presets;
    use crate::piece::Piece;

    use super::*;

    #[test]
    fn test_get_examples_immediate_win_white() {
        let board = BoardSquare::<1, 2>::with_pieces(&[
            (Coord::new(0, 0), Piece::new(Player::White, Type::King)),
            (Coord::new(0, 1), Piece::new(Player::Black, Type::King)),
        ]);
        let state = BoardState {
            board,
            player: Player::White,
        };

        let dev = Device::Cpu;
        let mcts = Mcts::new(dev);
        let examples = mcts.get_examples(&state);
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
        let state = BoardState {
            board,
            player: Player::White,
        };

        let dev = Device::Cpu;
        let mcts = Mcts::new(dev);
        let examples = mcts.get_examples(&state);
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
        let state = BoardState {
            board,
            player: Player::White,
        };

        let dev = Device::Cpu;
        let mcts = Mcts::new(dev);
        let examples = mcts.get_examples(&state);
        assert_eq!(examples.len(), 4);
        for e in examples {
            assert_eq!(e.reward, 0.0);
        }
    }

    #[test]
    fn test_get_examples_smoke() {
        let state = BoardState {
            board: presets::mini(),
            player: Player::White,
        };

        let dev = Device::Cpu;
        let mcts = Mcts::with_params(dev, MctsParams::for_testing());
        let examples = mcts.get_examples(&state);
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
