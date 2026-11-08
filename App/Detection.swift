import Foundation

struct Detection: Codable, Identifiable, Equatable {
  let id: Int
  let time: String
  let label: String
  let confidence: Double
  let cat: String
  var event_id: String? = nil
  var has_video: Bool? = nil
  var attribution: String? = nil
  var original_label: String? = nil
  var manual_label: String? = nil
  var attention_thumbnail: String? = nil
  var attention_seconds: Double? = nil
  var date: Date? { EventDateParser.shared.date(from: time) }
}
