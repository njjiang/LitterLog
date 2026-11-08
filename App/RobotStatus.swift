import Foundation

struct RobotStatus: Decodable, Equatable {
  let connected: Bool
  let state: String?
  let waste_percent: Double?
  let litter_level: String?
  let supported_commands: [String]?
  let last_weight_lbs: Double?
  let cycle_count: Int?
  let wait_time_minutes: Int?
  let hopper_status: String?
  let hopper_enabled: Bool?
}
