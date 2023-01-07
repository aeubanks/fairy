use log::{max_level, LevelFilter};
use std::time::{Duration, Instant};

pub struct Timer(Option<Instant>);

fn enabled() -> bool {
    max_level() >= LevelFilter::Info
}

impl Timer {
    pub fn new() -> Self {
        Self(if enabled() {
            Some(Instant::now())
        } else {
            None
        })
    }

    pub fn elapsed(self) -> Duration {
        if let Some(i) = self.0 {
            i.elapsed()
        } else {
            Duration::ZERO
        }
    }
}
