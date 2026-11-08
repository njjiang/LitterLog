import SwiftUI

@MainActor final class HomeModel: ObservableObject {
  @Published var weights: [CatWeight] = []
  @Published var catVisits: [CatVisit] = []
  @Published var activityAvailable = false
  @Published var reviewingWeight: CatWeight?
  @Published var correctingEvent: Detection?
  @Published var selectedCat = "cat1"
  @Published var liveSource: CameraSource?
  @Published var clip: ClipPresentation?
  @Published var preparingClip = false
  @Published var preparingClipID: Int?
  @Published var importingHistory = false
  @Published var cameraNote = "Camera not connected"
  @Published var events: [Detection] = []
  @Published var robot: RobotStatus?
  @Published var message: String?
  @Published var busy = false
  @Published var commandNote: String?
  @Published var commandFailed = false
  @Published var hasOlderEvents = true
  @Published var loadingOlder = false
  @Published var connectionNote = "Not connected"
  @AppStorage("autoCycleOnNone") var autoCycleOnNone = true
  private var activeBase = ""
  private var revision = 0
  private var initializedEventBaseline = false
  private var handledAutoCycleEventIDs: Set<Int> = []
  private func endpoint(_ base: String, _ path: String) throws -> URL {
    guard var parts = URLComponents(string: base),
       ["http", "https"].contains(parts.scheme?.lowercased() ?? ""),
       parts.host != nil, parts.user == nil, parts.password == nil,
       parts.query == nil, parts.fragment == nil else { throw URLError(.badURL) }
    parts.path = parts.path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    let route = path.split(separator: "?", maxSplits: 1, omittingEmptySubsequences: false)
    parts.path = (parts.path.isEmpty ? "" : "/" + parts.path) + String(route[0])
    parts.percentEncodedQuery = route.count > 1 ? String(route[1]) : nil
    guard let url = parts.url else { throw URLError(.badURL) }
    return url
  }
  func reset() {
    revision += 1
    activeBase = ""
    robot = nil
    weights = []
    catVisits = []; activityAvailable = false
    reviewingWeight = nil
    events = []
    hasOlderEvents = true
    connectionNote = "Not connected"
    commandNote = nil
    commandFailed = false
    initializedEventBaseline = false
    handledAutoCycleEventIDs = []
  }
  private func get<T: Decodable>(_ base: String, _ path: String) async throws -> T {
    var request = URLRequest(url: try endpoint(base, path))
    request.setValue("Bearer " + PairingKey.read(), forHTTPHeaderField: "Authorization")
    request.timeoutInterval = 180
    let (data, response) = try await HomeTLS.session.data(for: request)
    guard let response = response as? HTTPURLResponse else { throw URLError(.badServerResponse) }
    guard (200..<300).contains(response.statusCode) else {
      let detail = response.statusCode == 401 ? "Pairing token rejected. Enter the server’s LITTERLOG_APP_TOKEN and tap Save pairing token." : "Server returned HTTP \(response.statusCode)."
      throw NSError(domain: "LitterLog", code: response.statusCode, userInfo: [NSLocalizedDescriptionKey: detail])
    }
    return try JSONDecoder().decode(T.self, from: data)
  }
  func refresh(base: String) async {
    if activeBase != base { reset(); activeBase = base }
    let generation = revision
    do {
      let latest: [Detection] = try await get(base, "/api/events")
      guard generation == revision else { return }
      if events.isEmpty { hasOlderEvents = latest.count == 100 }
      let latestIDs = Set(latest.map(\.id))
      let merged = (latest + events.filter { !latestIDs.contains($0.id) }).sorted { $0.time > $1.time }
      if events != merged { events = merged }
      if connectionNote != "Activity connected" { connectionNote = "Activity connected" }
    } catch {
      guard generation == revision else { return }
      let note = "Activity unavailable · \(ConnectionDiagnostic.describe(error))"
      if connectionNote != note { connectionNote = note }
    }
    do {
      let status: RobotStatus = try await get(base, "/api/robot")
      guard generation == revision else { return }
      if robot != status { robot = status }
    } catch {
      guard generation == revision else { return }
      if robot != nil { robot = nil }
    }
    await evaluateAutoCycleOnNone(base: base)
    do {
      let readings: [CatWeight] = try await get(base, "/api/weights")
      if generation == revision, weights != readings { weights = readings }
    } catch { }
  }
  func refreshCatActivity(base: String) async {
    let generation = revision
    do {
      let visits: [CatVisit] = try await get(base, "/api/cat-activity")
      guard generation == revision else { return }
      if catVisits != visits { catVisits = visits }
      activityAvailable = true
    } catch { if generation == revision { activityAvailable = false } }
  }
  func correct(_ event: Detection, base: String, cat: String = "", label: String = "recording", reset: Bool = false, reportError: Bool = true) async -> Bool {
    do {
      let body: [String: Any] = reset ? ["reset": true] : ["cat": cat, "label": label]
      let _: [String: Bool] = try await post(base, "/api/events/\(event.id)/override", body: body)
      events = []
      await refresh(base: base)
      await refreshCatActivity(base: base)
      return true
    } catch {
      if reportError { message = "Could not save correction. Check the server connection." }
      return false
    }
  }
  func loadOlder(base: String) async {
    guard !loadingOlder else { return }
    loadingOlder = true
    let generation = revision
    defer { loadingOlder = false }
    do {
      let older: [Detection] = try await get(base, "/api/events?offset=\(events.count)")
      guard generation == revision else { return }
      hasOlderEvents = older.count == 100
      let ids = Set(events.map(\.id))
      events.append(contentsOf: older.filter { !ids.contains($0.id) })
    } catch { message = "Could not load older events: \(error.localizedDescription)" }
  }
  func connectCamera(base: String, quiet: Bool = false) async {
    do {
      let result: [String: String] = try await get(base, "/api/camera")
      guard let path = result["url"], let baseURL = URL(string: base), let url = URL(string: path, relativeTo: baseURL)?.absoluteURL else { throw URLError(.badURL) }
      liveSource = CameraSource(mode: result["mode"] ?? "video", url: url)
      cameraNote = "Connected to Protect"
    } catch {
      liveSource = nil; cameraNote = "Camera unavailable"
      if !quiet { message = error.localizedDescription }
    }
  }
  private func post<T: Decodable>(_ base: String, _ path: String, body: [String: Any]) async throws -> T {
    var request = URLRequest(url: try endpoint(base, path))
    request.httpMethod = "POST"
    request.setValue("Bearer " + PairingKey.read(), forHTTPHeaderField: "Authorization")
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.httpBody = try JSONSerialization.data(withJSONObject: body)
    request.timeoutInterval = 180
    let (data,response) = try await HomeTLS.session.data(for: request)
    guard let response = response as? HTTPURLResponse, (200..<300).contains(response.statusCode) else { throw URLError(.badServerResponse) }
    return try JSONDecoder().decode(T.self, from: data)
  }
  func importHistory(base: String) async {
    guard !importingHistory else { return }
    importingHistory = true
    defer { importingHistory = false }
    do {
      let result: [String: Int] = try await post(base, "/api/recordings/import", body: ["days": 7])
      await refresh(base: base)
      message = "Loaded \(result["imported"] ?? 0) Protect recordings. Classification appears separately when your model processes them."
    } catch { message = "Could not list Protect recordings. Check the server's Protect connection." }
  }
  func playRecording(_ event: Detection, base: String) async {
    guard event.event_id != nil, !preparingClip else { return }
    let generation = revision
    preparingClip = true; preparingClipID = event.id
    defer { preparingClip = false; preparingClipID = nil }
    do {
      let url = try await recordingURL(event, base: base)
      guard generation == revision else { return }
      clip = ClipPresentation(url: url)
    } catch {
      guard generation == revision else { return }
      message = "Could not prepare this recording. Check the home server connection or try another clip."
    }
  }

  func recordingURL(_ event: Detection, base: String) async throws -> URL {
    guard let id = event.event_id else { throw URLError(.resourceUnavailable) }
    let generation = revision
    var allowed = CharacterSet.urlPathAllowed
    allowed.remove(charactersIn: "/?#")
    guard let encodedID = id.addingPercentEncoding(withAllowedCharacters: allowed) else { throw URLError(.badURL) }
    let result: [String: String] = try await get(base, "/api/recordings/" + encodedID + "/play")
    try Task.checkCancellation()
    guard generation == revision else { throw CancellationError() }
    guard let path = result["url"], let baseURL = URL(string: base), let url = URL(string: path, relativeTo: baseURL)?.absoluteURL,
       url.host == baseURL.host, url.port == baseURL.port, url.scheme == baseURL.scheme,
       url.user == nil, url.password == nil else { throw URLError(.badURL) }
    return url
  }

  func identityEvent(for reading: CatWeight) -> Detection? {
    if let video = reading.video_event { return video }
    guard let id = reading.identity_event_id else { return nil }
    if let embedded = reading.identity_event, embedded.id == id { return embedded }
    // Older servers can still resolve an exact loaded event; never choose
    // a nearby clip just because it falls near the measurement timestamp.
    return events.first { $0.id == id }
  }

  func cameraURL(base: String) async -> URL? {
    do {
      let result: [String: String] = try await get(base, "/api/camera")
      guard let path = result["url"], let baseURL = URL(string: base), let url = URL(string:path, relativeTo:baseURL)?.absoluteURL else { return nil }
      return url
    } catch { message = "Camera unavailable: \(error.localizedDescription)"; return nil }
  }
  var isCatDetectedInRobot: Bool {
    guard let raw = robot?.state?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
          !raw.isEmpty else { return false }
    return raw.contains("cat") || raw.contains("sensor") || raw.contains("interrupted") || raw.contains("timing") || raw.contains("countdown")
  }

  private func evaluateAutoCycleOnNone(base: String) async {
    guard !events.isEmpty else { return }
    if !initializedEventBaseline {
      initializedEventBaseline = true
      // Mark older historical events as already handled, keeping only a fresh latest event eligible.
      for event in events.dropFirst() {
        handledAutoCycleEventIDs.insert(event.id)
      }
      if let latest = events.first,
         let eventDate = latest.date,
         Date().timeIntervalSince(eventDate) > 300 {
        handledAutoCycleEventIDs.insert(latest.id)
      }
    }
    guard autoCycleOnNone,
          isCatDetectedInRobot,
          let latest = events.first,
          !handledAutoCycleEventIDs.contains(latest.id),
          latest.label.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "none" else {
      return
    }
    if let eventDate = latest.date, Date().timeIntervalSince(eventDate) > 600 {
      handledAutoCycleEventIDs.insert(latest.id)
      return
    }
    handledAutoCycleEventIDs.insert(latest.id)
    await command("reset_and_cycle", base: base)
  }

  func command(_ command: String, base: String) async {
    guard !busy, robot?.connected == true else { return }
    busy = true
    commandNote = nil
    commandFailed = false
    defer { busy = false }
    do {
      if command == "reset_and_cycle" {
        if robot?.supported_commands?.contains("reset") == true {
          _ = try? await RobotCommandService.send("reset", base: base)
        }
        switch try await RobotCommandService.send("cycle", base: base) {
        case .accepted:
          commandNote = "None detected · Reset and cycle started."
        case .alreadyCleaning:
          commandNote = "The robot is already cleaning."
        }
      } else if command == "reset" {
        _ = try await RobotCommandService.send("reset", base: base)
        commandNote = "Sensor reset requested."
      } else if command == "pause" {
        _ = try await RobotCommandService.send("pause", base: base)
        commandNote = "Pause requested."
      } else {
        switch try await RobotCommandService.send(command, base: base) {
        case .accepted:
          commandNote = "Clean cycle requested."
        case .alreadyCleaning:
          commandNote = "The robot is already cleaning."
        }
      }
      await refresh(base: base)
    } catch {
      commandFailed = true
      commandNote = "Request not confirmed: \(error.localizedDescription) Check the robot before retrying."
    }
  }
}

