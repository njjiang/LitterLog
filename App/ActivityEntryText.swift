import Foundation

struct ActivityEntryText {
  var title: String
  var detail: String?
  var isCatSighting: Bool

  init(event: Detection) {
    let isCorrected = event.manual_label != nil
    let labelIsCat = ["cat1", "cat 1", "cat2", "cat 2"].contains(Self.normalized(event.label))
    isCatSighting = labelIsCat
    var cat = event.cat.trimmingCharacters(in: .whitespacesAndNewlines)
    if cat.isEmpty, labelIsCat, !isCorrected { cat = event.label }
    let activity = activityLabel(event.label)
    title = labelIsCat ? (cat.isEmpty ? "Unassigned cat" : catName(cat)) : activity
    if title.isEmpty { title = "Activity" }
    if cat.isEmpty || Self.normalized(catName(cat)) == Self.normalized(title) {
      detail = isCorrected ? "Corrected" : nil
    } else {
      if isCorrected {
        detail = "\(catName(cat)) · Corrected"
      } else {
        detail = catName(cat)
      }
    }
  }

  private static func normalized(_ text: String) -> String {
    text.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }.joined(separator: " ")
      .folding(options: [.caseInsensitive, .diacriticInsensitive], locale: .current)
  }
}

func canonicalCatID(_ value: String) -> String {
  let cleaned = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
  switch cleaned {
  case "cat1", "cat 1": return "cat1"
  case "cat2", "cat 2": return "cat2"
  default: return cleaned
  }
}

func catName(_ value: String) -> String {
  let cleaned = value.trimmingCharacters(in: .whitespacesAndNewlines)
  switch canonicalCatID(cleaned) {
  case "cat1": return "Cat 1"
  case "cat2": return "Cat 2"
  default: return cleaned.capitalized
  }
}

func activityLabel(_ label: String) -> String {
  let label = label.trimmingCharacters(in: .whitespacesAndNewlines)
  switch label.lowercased() {
  case "all": return "All activity"
  case "cat1", "cat 1", "cat2", "cat 2": return catName(label)
  case "both": return "Pee + Poo"
  case "none": return "No activity"
  case "recording": return "Unclassified"
  default: return label.capitalized
  }
}

enum ActivityFilter: String, CaseIterable, Identifiable {
  case all, sightings, cat1, cat2, pee, poo, both, none, recording
  static let pickerCases: [Self] = [.all, .sightings, .cat1, .cat2, .pee, .poo, .both, .recording]
  var id: Self { self }
  var title: String {
    switch self {
    case .all: return "All"
    case .sightings: return "Sightings"
    case .both: return "Bathroom"
    case .none: return "None"
    default: return activityLabel(rawValue)
    }
  }

  func matches(_ event: Detection, includeNone: Bool = false) -> Bool {
    let label = canonicalCatID(event.label)
    let cat = canonicalCatID(event.cat)
    if self != .none, !includeNone, label == "none" {
      return false
    }
    switch self {
    case .all: return true
    case .sightings: return ["cat1", "cat2"].contains(label)
    case .cat1, .cat2: return cat == rawValue || label == rawValue
    case .both: return ["pee", "poo", "both"].contains(label)
    default: return label == rawValue
    }
  }
}
