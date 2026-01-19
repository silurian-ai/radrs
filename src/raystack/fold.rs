//! Range folding algorithm for raystack format
//!
//! The range folding algorithm converts variable-length radar gate data
//! into fixed-size chunks for efficient ML training.
//!
//! Uses bucket averaging to preserve energy/power which is important for
//! radar reflectivity (dBZ) values.

/// Fold radar range data to a fixed size using bucket averaging
///
/// # Arguments
/// * `data` - Input data array (n_gates,)
/// * `fold_size` - Target output size
///
/// # Returns
/// Folded data array (fold_size,)
pub fn fold_ranges(data: &[f32], fold_size: usize) -> Vec<f32> {
    let mut result = vec![f32::NAN; fold_size];
    fold_ranges_into(data, &mut result);
    result
}

/// Fold radar range data directly into a preallocated buffer using bucket averaging
///
/// When n_gates > fold_size, divides into buckets and averages values within each.
/// When n_gates <= fold_size, copies data and pads with NaN.
///
/// # Arguments
/// * `data` - Input data array (n_gates,)
/// * `dest` - Destination buffer (fold_size,) - must be preallocated with correct size
#[inline]
pub fn fold_ranges_into(data: &[f32], dest: &mut [f32]) {
    let fold_size = dest.len();

    if data.is_empty() {
        dest.fill(f32::NAN);
        return;
    }

    let n_gates = data.len();

    if n_gates <= fold_size {
        // Copy data and fill rest with NaN
        for (i, &v) in data.iter().enumerate() {
            dest[i] = v;
        }
        dest[n_gates..].fill(f32::NAN);
    } else {
        // Bucket averaging: divide into buckets and average values within each
        let bucket_size = n_gates as f32 / fold_size as f32;

        for i in 0..fold_size {
            let start = (i as f32 * bucket_size) as usize;
            let end = ((i + 1) as f32 * bucket_size) as usize;
            let end = end.min(n_gates);

            let mut sum = 0.0f32;
            let mut count = 0u32;

            for j in start..end {
                let v = data[j];
                if !v.is_nan() {
                    sum += v;
                    count += 1;
                }
            }

            dest[i] = if count > 0 {
                sum / count as f32
            } else {
                f32::NAN
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fold_shorter_than_target() {
        let data = vec![1.0, 2.0, 3.0];
        let result = fold_ranges(&data, 5);
        assert_eq!(result.len(), 5);
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 3.0);
        assert!(result[3].is_nan());
        assert!(result[4].is_nan());
    }

    #[test]
    fn test_fold_longer_than_target() {
        // 10 values [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] folded into 5 buckets
        // Bucket 0: [0, 1] -> avg = 0.5
        // Bucket 1: [2, 3] -> avg = 2.5
        // Bucket 2: [4, 5] -> avg = 4.5
        // Bucket 3: [6, 7] -> avg = 6.5
        // Bucket 4: [8, 9] -> avg = 8.5
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let result = fold_ranges(&data, 5);
        assert_eq!(result.len(), 5);
        assert!((result[0] - 0.5).abs() < 0.01);
        assert!((result[1] - 2.5).abs() < 0.01);
        assert!((result[2] - 4.5).abs() < 0.01);
        assert!((result[3] - 6.5).abs() < 0.01);
        assert!((result[4] - 8.5).abs() < 0.01);
    }

    #[test]
    fn test_fold_empty() {
        let data: Vec<f32> = vec![];
        let result = fold_ranges(&data, 5);
        assert_eq!(result.len(), 5);
        assert!(result.iter().all(|x| x.is_nan()));
    }

    #[test]
    fn test_fold_with_nan() {
        // Test that NaN values are skipped in averaging
        let data = vec![1.0, f32::NAN, 3.0, f32::NAN, 5.0];
        let result = fold_ranges(&data, 2);
        // Bucket 0: [1.0, NaN] -> avg = 1.0 (NaN skipped)
        // Bucket 1: [3.0, NaN, 5.0] -> avg = 4.0
        assert_eq!(result.len(), 2);
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_fold_all_nan_bucket() {
        let data = vec![f32::NAN, f32::NAN, 1.0, 2.0];
        let result = fold_ranges(&data, 2);
        // Bucket 0: [NaN, NaN] -> NaN
        // Bucket 1: [1.0, 2.0] -> 1.5
        assert!(result[0].is_nan());
        assert!((result[1] - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_fold_into_consistency() {
        // Verify fold_ranges and fold_ranges_into produce same results
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();

        let result1 = fold_ranges(&data, 10);

        let mut result2 = vec![0.0; 10];
        fold_ranges_into(&data, &mut result2);

        for i in 0..10 {
            assert!((result1[i] - result2[i]).abs() < 0.01);
        }
    }
}
