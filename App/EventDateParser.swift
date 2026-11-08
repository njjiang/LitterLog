import Foundation

final class EventDateParser: @unchecked Sendable {
  static let shared = EventDateParser()
  private let lock = NSLock()
  private let cache = NSCache<NSString, NSDate>()
  private let fractional = ISO8601DateFormatter()
  private let wholeSeconds = ISO8601DateFormatter()
  private init() {
    cache.countLimit = 4096
    fractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
  }
  func date(from value: String) -> Date? {
    guard !value.isEmpty else { return nil }
    lock.lock()
    defer { lock.unlock() }
    if let date = cache.object(forKey: value as NSString) { return date as Date }
    guard let date = fractional.date(from: value) ?? wholeSeconds.date(from: value) else { return nil }
    cache.setObject(date as NSDate, forKey: value as NSString)
    return date
  }
}
