import Foundation

@main
struct CatHistorySnapshotChecks {
  static func reading(_ id: String, _ kg: Double, time: String, cat: String = "cat1") -> CatWeight {
    CatWeight(id: id, time: time, weight_lbs: kg / 0.45359237,
      weight_kg: kg, cat: cat, attribution: "test")
  }

  static func main() {
    let cat1Readings = [
      reading("normal-1", 6, time: "2026-11-01T12:00:00Z"),
      reading("normal-2", 6, time: "2026-11-02T12:00:00Z"),
      reading("tie-b", 6, time: "2026-11-03T12:00:00Z"),
      reading("tie-a", 6, time: "2026-11-03T12:00:00Z", cat: "CAT1"),
      reading("outlier", 25, time: "2026-11-04T12:00:00Z"),
      reading("normal-5", 6, time: "2026-11-05T12:00:00Z"),
      reading("invalid-weight", -1, time: "2026-11-06T12:00:00Z"),
      reading("not-a-number", .nan, time: "2026-11-07T12:00:00Z"),
      reading("infinite", .infinity, time: "2026-11-08T12:00:00Z"),
      reading("invalid-date", 6, time: "invalid")
    ]
    let cat2Readings = [
      reading("cat2-old", 4, time: "2026-11-01T12:00:00Z", cat: "cat2"),
      reading("cat2-new", 4.1, time: "2026-11-03T12:00:00Z", cat: "CAT2")
    ]
    let unassigned = [
      reading("unassigned-old", 5, time: "2026-11-01T12:00:00Z", cat: ""),
      reading("unassigned-new", 5.1, time: "2026-11-05T12:00:00Z", cat: "")
    ]
    let input = cat1Readings + cat2Readings + unassigned
    let snapshot = CatHistorySnapshot(readings: input)
    guard let cat1History = snapshot.histories["cat1"], let cat2History = snapshot.histories["cat2"] else {
      fatalError("Both cats must have a history, including when one has no readings")
    }

    let expectedCat1IDs = ["infinite", "not-a-number", "invalid-weight", "normal-5", "outlier",
      "tie-a", "tie-b", "normal-2", "normal-1", "invalid-date"]
    assert(cat1History.readings.map(\.id) == expectedCat1IDs,
      "Raw readings must be newest first, with deterministic ID order for tied dates")
    assert(Set(cat1History.readings.map(\.id)) == Set(cat1Readings.map(\.id)),
      "Invalid weights and outliers must remain available for raw-history review")
    assert(cat1History.readings.first { $0.id == "outlier" }?.weight_kg == 25)
    assert(cat1History.readings.first { $0.id == "not-a-number" }?.weight_kg.isNaN == true)
    assert(cat1History.readings.first { $0.id == "infinite" }?.weight_kg == .infinity)
    assert(cat1History.readings.last?.time == "invalid")
    assert(cat1History.analysis.excludedIDs == ["outlier", "invalid-weight", "not-a-number", "infinite", "invalid-date"])
    assert(cat1History.analysis.points.count == 5 && cat1History.analysis.latest?.averageKg == 6,
      "The cached trend must exclude bad raw data without removing it from the history")

    assert(cat2History.readings.map(\.id) == ["cat2-new", "cat2-old"])
    assert(cat2History.analysis.points.map(\.id) == ["cat2-old", "cat2-new"])
    assert(abs((cat2History.analysis.latest?.averageKg ?? 0) - 4.05) < 0.0000001,
      "Each cat needs an independent trend, including case-normalized cat IDs")
    assert(cat2History.analysis.excludedIDs.isEmpty)
    assert(snapshot.unassigned.map(\.id) == ["unassigned-new", "unassigned-old"],
      "Unassigned readings must be preserved separately in newest-first order")

    let reversed = CatHistorySnapshot(readings: input.reversed())
    assert(reversed.histories["cat1"]?.readings.map(\.id) == expectedCat1IDs,
      "Tied-date ordering must not change when the server changes input order")
    assert(reversed.unassigned.map(\.id) == snapshot.unassigned.map(\.id))

    let empty = CatHistorySnapshot(readings: [])
    assert(Set(empty.histories.keys) == ["cat1", "cat2"])
    assert(empty.histories.values.allSatisfy { $0.readings.isEmpty && $0.analysis.points.isEmpty && $0.analysis.latest == nil })
    assert(empty.unassigned.isEmpty)
    let onlyUnassigned = CatHistorySnapshot(readings: unassigned)
    assert(onlyUnassigned.histories.values.allSatisfy { $0.readings.isEmpty && $0.analysis.latest == nil })
    assert(onlyUnassigned.unassigned.count == 2)

    print("Passed cat history snapshot checks: raw-data preservation, newest-first order, stable ties, cat isolation, unassigned readings, cached trends, and empty input.")
  }
}
