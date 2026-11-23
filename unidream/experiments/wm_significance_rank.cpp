// Exact frequency-weighted Spearman ranks for block-bootstrap day counts.
// A row with integer weight w is equivalent to w identical expanded rows.
// The Python wrapper validates shapes, integer weights and tie boundaries.
#include <cmath>
#include <cstdint>
#include <vector>

extern "C" int unidream_weighted_spearman(
    int64_t n, int64_t days, int64_t reps,
    const int64_t *x_order, const int64_t *x_ends,
    const int64_t *y_order, const int64_t *y_ends,
    const int32_t *row_day, const int32_t *weights, double *out) {
  if (n <= 0 || days <= 0 || reps <= 0) return 1;
  std::vector<double> ranks(static_cast<size_t>(n));
  for (int64_t rep = 0; rep < reps; ++rep) {
    const int32_t *w = weights + rep * days;
    double count = 0., xss = 0.;
    for (int64_t start = 0; start < n;) {
      const int64_t end = x_ends[start];
      double group = 0.;
      for (int64_t j = start; j < end; ++j) group += w[row_day[x_order[j]]];
      const double rank = count + (group + 1.) / 2.;
      for (int64_t j = start; j < end; ++j) ranks[x_order[j]] = rank;
      xss += group * rank * rank;
      count += group;
      start = end;
    }
    double ycount = 0., yss = 0., cross = 0.;
    for (int64_t start = 0; start < n;) {
      const int64_t end = y_ends[start];
      double group = 0.;
      for (int64_t j = start; j < end; ++j) group += w[row_day[y_order[j]]];
      const double rank = ycount + (group + 1.) / 2.;
      for (int64_t j = start; j < end; ++j) {
        const int64_t row = y_order[j];
        cross += w[row_day[row]] * ranks[row] * rank;
      }
      yss += group * rank * rank;
      ycount += group;
      start = end;
    }
    const double center = count * (count + 1.) * (count + 1.) / 4.;
    const double vx = xss - center, vy = yss - center;
    out[rep] = count >= 2. && vx > 0. && vy > 0.
        ? (cross - center) / std::sqrt(vx * vy) : NAN;
  }
  return 0;
}
