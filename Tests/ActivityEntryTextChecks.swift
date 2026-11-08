import Foundation

@main
struct ActivityEntryTextChecks {
  static func main() {
    func text(cat: String, label: String, corrected: Bool = false) -> ActivityEntryText {
      ActivityEntryText(event: Detection(id: 1, time: "2026-11-02T12:00:00Z",
        label: label, confidence: 0.95, cat: cat, manual_label: corrected ? label : nil))
    }
    let sighting = text(cat: "cat1", label: "cat1")
    assert(sighting.title == "Cat 1" && sighting.detail == nil && sighting.isCatSighting)
    let alias = text(cat: " Cat 1 ", label: " CAT1 ")
    assert(alias.title == "Cat 1" && alias.detail == nil && alias.isCatSighting)
    let cat2 = text(cat: "Cat 2", label: "cat2")
    assert(cat2.title == "Cat 2" && cat2.detail == nil && cat2.isCatSighting)
    let bathroom = text(cat: "cat1", label: "pee")
    assert(bathroom.title == "Pee" && bathroom.detail == "Cat 1" && !bathroom.isCatSighting)
    let poo = text(cat: "cat2", label: "poo")
    assert(poo.title == "Poo" && poo.detail == "Cat 2" && !poo.isCatSighting)
    let both = text(cat: "cat1", label: "both")
    assert(both.title == "Pee + Poo" && both.detail == "Cat 1" && !both.isCatSighting)
    let normalizedWaste = text(cat: " cat2 ", label: " PEE \n")
    assert(normalizedWaste.title == "Pee" && normalizedWaste.detail == "Cat 2")
    let visit = text(cat: "cat1", label: "visit")
    assert(visit.title == "Visit" && visit.detail == "Cat 1")
    let noWaste = text(cat: "cat2", label: "none")
    assert(noWaste.title == "No activity" && noWaste.detail == "Cat 2")
    let corrected = text(cat: "cat2", label: "pee", corrected: true)
    assert(corrected.title == "Pee" && corrected.detail == "Cat 2 · Corrected")
    let correctedUnassigned = text(cat: "", label: "poo", corrected: true)
    assert(correctedUnassigned.title == "Poo" && correctedUnassigned.detail == "Corrected")
    let labelOnly = text(cat: "", label: "cat1")
    assert(labelOnly.title == "Cat 1" && labelOnly.detail == nil && labelOnly.isCatSighting)
    let correctedSighting = text(cat: "cat2", label: "cat2", corrected: true)
    assert(correctedSighting.title == "Cat 2" && correctedSighting.detail == "Corrected" && correctedSighting.isCatSighting)
    let manuallyUnassigned = text(cat: "", label: "cat1", corrected: true)
    assert(manuallyUnassigned.title == "Unassigned cat" && manuallyUnassigned.detail == "Corrected" && manuallyUnassigned.isCatSighting)
    let duplicate = text(cat: "visit", label: "visit")
    assert(duplicate.title == "Visit" && duplicate.detail == nil)
    let unassigned = text(cat: "", label: "recording")
    assert(unassigned.title == "Unclassified" && unassigned.detail == nil)
    let empty = text(cat: "", label: "")
    assert(empty.title == "Activity" && empty.detail == nil)
    let noneEvent = Detection(id: 2, time: "2026-11-02T12:05:00Z", label: "none", confidence: 0.90, cat: "cat2")
    let peeEvent = Detection(id: 3, time: "2026-11-02T12:10:00Z", label: "pee", confidence: 0.95, cat: "cat2")
    assert(!ActivityFilter.all.matches(noneEvent))
    assert(ActivityFilter.all.matches(noneEvent, includeNone: true))
    assert(!ActivityFilter.cat2.matches(noneEvent))
    assert(ActivityFilter.cat2.matches(noneEvent, includeNone: true))
    assert(ActivityFilter.none.matches(noneEvent))
    assert(ActivityFilter.all.matches(peeEvent))
    print("Passed activity text checks.")
  }
}
