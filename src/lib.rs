#![allow(
    clippy::new_without_default,
    clippy::comparison_chain,
    clippy::bool_to_int_with_if,
    clippy::single_match,
    clippy::collapsible_match,
    clippy::collapsible_if,
    clippy::too_many_arguments
)]
pub mod board;
pub mod coord;
pub mod moves;
pub mod perft;
pub mod piece;
pub mod player;
pub mod tablebase;
pub mod timer;
