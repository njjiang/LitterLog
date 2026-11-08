import Foundation

struct CatVisit: Decodable, Identifiable, Equatable {
  let id: Int
  let cat: String
  let time: String
  let pee: Bool
  let poo: Bool
  var date: Date? { EventDateParser.shared.date(from: time) }
}
