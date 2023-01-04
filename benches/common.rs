use criterion::measurement::{Measurement, ValueFormatter};
use perf_event::{Builder, Counter};

pub struct Perf;

struct PerfFormatter;

impl ValueFormatter for PerfFormatter {
    fn scale_throughputs(
        &self,
        _typical_value: f64,
        _throughput: &criterion::Throughput,
        _values: &mut [f64],
    ) -> &'static str {
        "insts"
    }
    fn scale_values(&self, mut typical_value: f64, values: &mut [f64]) -> &'static str {
        let mut divisions = 0;
        let mut dividend = 1.0;
        while typical_value > 1000.0 {
            typical_value /= 1000.0;
            divisions += 1;
            dividend *= 1000.0;
            if divisions == 3 {
                break;
            }
        }
        for v in values {
            *v /= dividend;
        }
        match divisions {
            0 => "insts",
            1 => "K insts",
            2 => "M insts",
            3 => "B insts",
            _ => panic!(),
        }
    }
    fn scale_for_machines(&self, _values: &mut [f64]) -> &'static str {
        "insts"
    }
}

impl Measurement for Perf {
    type Intermediate = Counter;
    type Value = u64;
    fn start(&self) -> Self::Intermediate {
        let mut c = Builder::new().inherit(true).build().unwrap();
        c.enable().unwrap();
        c
    }
    fn end(&self, mut i: Self::Intermediate) -> Self::Value {
        i.disable().unwrap();
        i.read().unwrap()
    }
    fn add(&self, v1: &Self::Value, v2: &Self::Value) -> Self::Value {
        v1 + v2
    }
    fn zero(&self) -> Self::Value {
        0
    }
    fn to_f64(&self, value: &Self::Value) -> f64 {
        *value as f64
    }
    fn formatter(&self) -> &dyn ValueFormatter {
        &PerfFormatter
    }
}
