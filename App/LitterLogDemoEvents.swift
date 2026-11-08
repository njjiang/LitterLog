import Foundation

enum LitterLogDemoEvents {
  static let events: [Detection] = {
    let formatter = ISO8601DateFormatter()
    return [
      Detection(id: 1, time: formatter.string(from: Date().addingTimeInterval(-360)), label: "pee", confidence: 0.96, cat: "cat1"),
      Detection(id: 6, time: formatter.string(from: Date().addingTimeInterval(-600)), label: "none", confidence: 0.94, cat: "cat2"),
      Detection(id: 2, time: formatter.string(from: Date().addingTimeInterval(-3600)), label: "poo", confidence: 0.93, cat: "cat2"),
      Detection(id: 3, time: formatter.string(from: Date().addingTimeInterval(-7200)), label: "visit", confidence: 0.91, cat: "cat1"),
      Detection(id: 4, time: formatter.string(from: Date().addingTimeInterval(-10800)), label: "cat1", confidence: 0.98, cat: "cat1"),
      Detection(id: 5, time: formatter.string(from: Date().addingTimeInterval(-14400)), label: "cat2", confidence: 0.97, cat: "cat2")
    ]
  }()
}
