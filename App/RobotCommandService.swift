import Foundation

enum RobotCommandOutcome {
  case accepted
  case alreadyCleaning
}

enum RobotCommandService {
  static func send(_ command: String, base: String) async throws -> RobotCommandOutcome {
    var request = URLRequest(url: try commandURL(base: base))
    request.httpMethod = "POST"
    request.setValue("Bearer " + PairingKey.read(), forHTTPHeaderField: "Authorization")
    request.timeoutInterval = 45
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.httpBody = try JSONEncoder().encode(["command": command])
    let (data, response) = try await HomeTLS.session.data(for: request)
    guard let response = response as? HTTPURLResponse, (200..<300).contains(response.statusCode) else {
      let detail = (try? JSONDecoder().decode([String: String].self, from: data))?["error"]
      throw NSError(domain: "LitterLog", code: (response as? HTTPURLResponse)?.statusCode ?? 0,
        userInfo: [NSLocalizedDescriptionKey: detail ?? "The robot did not confirm the request."])
    }
    let result = try JSONDecoder().decode([String: Bool].self, from: data)
    guard result["accepted"] == true else { throw URLError(.badServerResponse) }
    return result["already_cleaning"] == true ? .alreadyCleaning : .accepted
  }

  private static func commandURL(base: String) throws -> URL {
    guard var parts = URLComponents(string: base),
      ["http", "https"].contains(parts.scheme?.lowercased() ?? ""),
      parts.host != nil, parts.user == nil, parts.password == nil,
      parts.query == nil, parts.fragment == nil else { throw URLError(.badURL) }
    parts.path = parts.path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    parts.path = (parts.path.isEmpty ? "" : "/" + parts.path) + "/api/robot/command"
    guard let url = parts.url else { throw URLError(.badURL) }
    return url
  }
}
