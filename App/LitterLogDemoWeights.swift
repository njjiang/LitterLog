import Foundation

enum LitterLogDemoWeights {
  static let readings: [CatWeight] = {
    let calendar = Calendar.current
    let anchor = calendar.startOfDay(for: Date())
    let samples: [(String, [Double])] = [
      ("cat1", [6.12, 6.15, 6.13, 6.18, 6.16, 6.20, 6.17, 6.21, 6.19, 6.24, 6.22, 6.25, 6.23, 6.26]),
      ("cat2", [5.81, 5.84, 5.82, 5.86, 5.85, 5.83, 5.88, 5.86, 5.89, 5.87, 5.91, 5.88, 5.92, 5.90])
    ]
    let formatter = ISO8601DateFormatter()
    return samples.flatMap { cat, values in
      values.enumerated().map { index, kilograms in
        let day = calendar.date(byAdding: .day, value: index - 14, to: anchor)!
        let date = calendar.date(byAdding: .hour, value: cat == "cat1" ? 8 : 10, to: day)!
        return CatWeight(id: "demo-\(cat)-\(index)", time: formatter.string(from: date),
          weight_lbs: kilograms / 0.45359237, weight_kg: kilograms, cat: cat, attribution: "sample")
      }
    }
  }()
}
