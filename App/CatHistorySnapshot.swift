import Foundation

struct CatHistorySnapshot {
  struct History {
    let readings: [CatWeight]
    let analysis: WeightTrendAnalysis
  }

  let histories: [String: History]
  let unassigned: [CatWeight]

  init(readings: [CatWeight]) {
    // Parse each timestamp once before sorting, including histories larger
    // than the date parser's shared cache.
    var dated: [(reading: CatWeight, date: Date)] = readings.map { (reading: $0, date: $0.date) }
    dated.sort {
      if $0.date == $1.date { return $0.reading.id < $1.reading.id }
      return $0.date > $1.date
    }
    let groups = Dictionary(grouping: dated.map { $0.reading }, by: { canonicalCatID($0.cat) })
    var histories: [String: History] = [:]
    for cat in ["cat1", "cat2"] {
      let catReadings = groups[cat] ?? []
      histories[cat] = History(readings: catReadings,
        analysis: WeightTrendAnalysis(readings: catReadings, cat: cat))
    }
    self.histories = histories
    unassigned = groups[""] ?? []
  }
}
