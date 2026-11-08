import Foundation

@main
struct WeightTrendAnalysisChecks {
  static var calendar: Calendar {
    var calendar = Calendar(identifier: .gregorian)
    calendar.timeZone = TimeZone(identifier: "America/Los_Angeles")!
    return calendar
  }
  static let formatter = ISO8601DateFormatter()

  static func reading(_ id: Int, _ kg: Double, day: Int = 1, hour: Int = 12, cat: String = "cat1") -> CatWeight {
    let date = calendar.date(from: DateComponents(year: 2026, month: 11, day: day, hour: hour))!
    return CatWeight(id: "\(cat)-\(id)", time: formatter.string(from: date),
      weight_lbs: kg / 0.45359237, weight_kg: kg, cat: cat, attribution: "test")
  }
  static func close(_ actual: Double?, _ expected: Double, _ message: String) {
    guard let actual, abs(actual - expected) < 0.0000001 else {
      fatalError("\(message): expected \(expected), got \(String(describing: actual))")
    }
  }

  static func main() {
    let gradual = (0..<15).map { reading($0, 6 + Double($0) * 0.01, day: $0 + 1) }
    let trend = WeightTrendAnalysis(readings: gradual.reversed(), cat: "cat1", calendar: calendar)
    close(trend.latest?.averageKg, 6.095, "Rolling mean uses the latest ten chronological readings")
    assert(trend.latest?.sampleCount == 10 && trend.excludedIDs.isEmpty)

    let spike = (0..<21).map { reading($0, $0 == 10 ? 25 : 6, day: $0 + 1) }
    let filtered = WeightTrendAnalysis(readings: spike, cat: "cat1", calendar: calendar)
    assert(filtered.excludedIDs == ["cat1-10"] && filtered.points.count == 20)
    close(filtered.latest?.averageKg, 6, "A spike must not contaminate the trend, even with zero MAD")

    let badFirst = (0..<10).map { reading($0, $0 == 0 ? 30 : 6, day: $0 + 1) }
    let bootstrap = WeightTrendAnalysis(readings: badFirst, cat: "cat1", calendar: calendar)
    assert(bootstrap.excludedIDs == ["cat1-0"])
    close(bootstrap.points.first?.averageKg, 6, "An initial spike must not establish the baseline")

    var invalid = [reading(0, 6), reading(1, -1), reading(2, .nan), reading(3, .infinity)]
    invalid.append(CatWeight(id: "bad-date", time: "invalid", weight_lbs: 13, weight_kg: 6, cat: "cat1", attribution: "test"))
    let validOnly = WeightTrendAnalysis(readings: invalid, cat: "cat1", calendar: calendar)
    assert(validOnly.points.count == 1 && validOnly.excludedIDs.count == 4)
    close(validOnly.latest?.averageKg, 6, "Invalid weights and dates are excluded")

    let daily = (0..<12).map { reading($0, 6 + Double($0) * 0.01, day: $0 < 6 ? 1 : 2, hour: 10 + $0 % 6) }
    let days = WeightTrendAnalysis(readings: daily, cat: "cat1", calendar: calendar)
    close(days.dailyChanges.first?.averageKg, 6.025, "Daily value uses the final average of that day")
    close(days.dailyChanges.last?.averageKg, 6.065, "Daily average retains the trailing ten readings")
    close(days.dailyChanges.last?.differenceKg, 0.04, "Daily delta compares smoothed values, not raw weights")
    assert(days.dailyChanges.first?.differenceKg == nil)

    let gaps = WeightTrendAnalysis(readings: [reading(1, 6, day: 1), reading(2, 6.1, day: 3)], cat: "cat1", calendar: calendar)
    assert(gaps.dailyChanges.last?.differenceKg == nil, "A missing calendar day must not produce a daily change")

    let cats = WeightTrendAnalysis(readings: gradual + [reading(99, 3, cat: "cat2")], cat: "cat1", calendar: calendar)
    close(cats.latest?.averageKg, 6.095, "Each cat has an independent baseline")
    assert(!cats.excludedIDs.contains("cat2-99"))

    let sparse = WeightTrendAnalysis(readings: [reading(1, 6), reading(2, 6.1, day: 2)], cat: "cat1", calendar: calendar)
    assert(sparse.latest?.sampleCount == 2)
    close(sparse.latest?.averageKg, 6.05, "Warm-up uses the available accepted readings")
    assert(WeightTrendAnalysis(readings: [], cat: "cat1", calendar: calendar).latest == nil)
    print("Passed weight analysis checks: rolling window, outliers, bootstrap, invalid data, daily deltas, missing days, cat isolation, and warm-up.")
  }
}
