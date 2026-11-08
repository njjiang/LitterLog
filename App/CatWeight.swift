import Foundation

struct CatWeight: Codable, Identifiable, Equatable {
  let id: String
  let time: String
  let weight_lbs: Double
  let weight_kg: Double
  let cat: String
  let attribution: String
  var identity_event_id: Int? = nil
  var identity_event: Detection? = nil
  var video_event: Detection? = nil
  var video_match: String? = nil
  var date: Date { EventDateParser.shared.date(from: time) ?? .distantPast }
}
