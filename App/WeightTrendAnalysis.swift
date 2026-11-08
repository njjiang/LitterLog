import Foundation

struct WeightTrendAnalysis {
  var points: [WeightTrendPoint] = []
  var dailyChanges: [DailyWeightChange] = []
  var excludedIDs: Set<String> = []

  var latest: WeightTrendPoint? { points.last }

  init(readings: [CatWeight], cat: String, calendar: Calendar = .current) {
    let catReadings = readings.filter { canonicalCatID($0.cat) == canonicalCatID(cat) }
    let valid = catReadings.filter { reading in
      let valid = reading.weight_kg.isFinite && reading.weight_kg > 0
        && reading.date != .distantPast && reading.date.timeIntervalSinceReferenceDate.isFinite
      if !valid { excludedIDs.insert(reading.id) }
      return valid
    }.sorted {
      $0.date == $1.date ? $0.id < $1.id : $0.date < $1.date
    }

    var accepted: [CatWeight] = []
    for (index, reading) in valid.enumerated() {
      // Use a local median/MAD filter. Bootstrap the first window together so
      // an isolated bad first reading cannot establish the baseline.
      let context = index < 10 ? Array(valid.prefix(10)) : Array(valid[(index - 9)...index])
      if context.count >= 3 {
        let values = context.map(\.weight_kg)
        let center = Self.median(values)
        let deviation = Self.median(values.map { abs($0 - center) })
        // A 3% floor handles repeated/rounded weights where MAD is zero.
        let threshold = max(center * 0.03, 3 * 1.4826 * deviation)
        if abs(reading.weight_kg - center) > threshold {
          excludedIDs.insert(reading.id)
          continue
        }
      }
      accepted.append(reading)
      let window = accepted.suffix(10)
      points.append(WeightTrendPoint(reading: reading,
        averageKg: window.reduce(0) { $0 + $1.weight_kg } / Double(window.count),
        sampleCount: window.count))
    }

    // Each day's final rolling average is compared only with the immediately
    // preceding calendar day. Missing days never become zero or a daily delta.
    var endOfDay: [Date: WeightTrendPoint] = [:]
    for point in points { endOfDay[calendar.startOfDay(for: point.date)] = point }
    dailyChanges = endOfDay.keys.sorted().compactMap { day in
      guard let point = endOfDay[day] else { return nil }
      let previous = calendar.date(byAdding: .day, value: -1, to: day).flatMap { endOfDay[$0] }
      return DailyWeightChange(day: day, averageKg: point.averageKg,
        sampleCount: point.sampleCount,
        differenceKg: previous.map { point.averageKg - $0.averageKg })
    }
  }

  private static func median(_ values: [Double]) -> Double {
    let sorted = values.sorted()
    let middle = sorted.count / 2
    return sorted.count.isMultiple(of: 2)
      ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
  }
}

struct WeightTrendPoint: Identifiable {
  var reading: CatWeight
  var averageKg: Double
  var sampleCount: Int
  var id: String { reading.id }
  var date: Date { reading.date }
}

struct DailyWeightChange: Identifiable {
  var day: Date
  var averageKg: Double
  var sampleCount: Int
  var differenceKg: Double?
  var id: Date { day }
}
