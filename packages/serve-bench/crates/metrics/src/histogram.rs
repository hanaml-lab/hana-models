//! A log-linear histogram in the style of HdrHistogram: values are kept to a fixed number of
//! significant binary digits, so memory is bounded and any percentile is accurate to within
//! 1 / 2^precision of the true value, whatever the range.

#[derive(Clone, Debug)]
pub struct Histogram {
    precision: u32, // significant bits kept below the leading one
    counts: Vec<u64>,
    total: u64,
    min: u64,
    max: u64,
    sum: u128,
}

impl Histogram {
    /// `precision` of 7 keeps values within 1/128 (under 1%) of their true value.
    pub fn new(precision: u32) -> Self {
        assert!(
            (1..=12).contains(&precision),
            "precision must be between 1 and 12 bits"
        );
        Histogram {
            precision,
            counts: vec![0; (64 - precision as usize + 1) << precision],
            total: 0,
            min: u64::MAX,
            max: 0,
            sum: 0,
        }
    }

    fn index(&self, v: u64) -> usize {
        let sub = 1u64 << self.precision;
        if v < sub {
            return v as usize;
        }
        let msb = 63 - v.leading_zeros(); // position of the leading one
        let shift = msb - self.precision;
        let bucket = (shift + 1) as usize;
        let offset = ((v >> shift) - sub) as usize;
        (bucket << self.precision) + offset
    }

    /// The lowest value that falls in bucket `i`.
    fn value_at(&self, i: usize) -> u64 {
        let sub = 1usize << self.precision;
        if i < sub {
            return i as u64;
        }
        let bucket = i >> self.precision;
        let offset = (i & (sub - 1)) as u64;
        ((1u64 << self.precision) + offset) << (bucket - 1)
    }

    pub fn record(&mut self, v: u64) {
        let i = self.index(v);
        self.counts[i] += 1;
        self.total += 1;
        self.min = self.min.min(v);
        self.max = self.max.max(v);
        self.sum += v as u128;
    }

    pub fn merge(&mut self, other: &Histogram) {
        assert_eq!(
            self.precision, other.precision,
            "histograms must share a precision to merge"
        );
        for (a, b) in self.counts.iter_mut().zip(&other.counts) {
            *a += b;
        }
        self.total += other.total;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        self.sum += other.sum;
    }

    pub fn len(&self) -> u64 {
        self.total
    }

    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    pub fn mean(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.sum as f64 / self.total as f64
        }
    }

    pub fn max(&self) -> u64 {
        self.max
    }

    /// The value at percentile `p` (0..=100): the smallest recorded bucket value such that at
    /// least p% of samples are at or below it, clamped to the observed min and max.
    pub fn percentile(&self, p: f64) -> u64 {
        if self.total == 0 {
            return 0;
        }
        let rank = ((p / 100.0) * self.total as f64).ceil().max(1.0) as u64;
        if rank >= self.total {
            return self.max; // the top sample is known exactly
        }
        let mut seen = 0;
        for (i, &c) in self.counts.iter().enumerate() {
            seen += c;
            if seen >= rank {
                return self.value_at(i).clamp(self.min, self.max);
            }
        }
        self.max
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_within_precision() {
        let mut h = Histogram::new(7);
        for v in 1..=100_000u64 {
            h.record(v);
        }
        for (p, want) in [(50.0, 50_000.0), (95.0, 95_000.0), (99.0, 99_000.0)] {
            let got = h.percentile(p) as f64;
            assert!(
                (got - want).abs() / want < 1.0 / 128.0,
                "p{p}: {got} vs {want}"
            );
        }
        assert_eq!(h.percentile(100.0), 100_000);
        assert_eq!(h.len(), 100_000);
    }

    #[test]
    fn small_values_are_exact_and_merge_adds() {
        let mut a = Histogram::new(7);
        let mut b = Histogram::new(7);
        a.record(3);
        b.record(5);
        b.record(5);
        a.merge(&b);
        assert_eq!(a.percentile(50.0), 5);
        assert_eq!(a.percentile(1.0), 3);
        assert!((a.mean() - 13.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn huge_values_fit() {
        let mut h = Histogram::new(7);
        h.record(u64::MAX / 2);
        assert!(h.percentile(50.0) > u64::MAX / 4);
    }
}
