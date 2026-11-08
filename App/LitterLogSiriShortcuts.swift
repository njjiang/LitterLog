import AppIntents
import Foundation

struct StartCleanCycleIntent: AppIntent {
  static let title: LocalizedStringResource = "Start Clean Cycle"
  static let description = IntentDescription("Starts a clean cycle on your connected Litter-Robot.")
  static let supportedModes: IntentModes = .background
  static let authenticationPolicy: IntentAuthenticationPolicy = .requiresAuthentication

  func perform() async throws -> some IntentResult & ProvidesDialog {
    let defaults = UserDefaults.standard
    let demoMode = defaults.object(forKey: "demo") == nil || defaults.bool(forKey: "demo")
    guard !demoMode else {
      return .result(dialog: "Turn off Demo mode in Connections before starting a real clean cycle.")
    }
    let server = defaults.string(forKey: "server")?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    guard !server.isEmpty, !PairingKey.read().isEmpty else {
      return .result(dialog: "Connect your home server and save its pairing token in LitterLog first.")
    }

    try await requestConfirmation(actionName: .start,
      dialog: "Start the Litter-Robot clean cycle?")
    do {
      switch try await RobotCommandService.send("cycle", base: server) {
      case .accepted:
        return .result(dialog: "The clean cycle has started.")
      case .alreadyCleaning:
        return .result(dialog: "The Litter-Robot is already cleaning.")
      }
    } catch {
      return .result(dialog: "I couldn’t confirm the clean cycle. Check the robot and its connection before trying again.")
    }
  }
}

struct LitterLogShortcuts: AppShortcutsProvider {
  static var appShortcuts: [AppShortcut] {
    AppShortcut(intent: StartCleanCycleIntent(), phrases: [
      "Start a clean cycle with \(.applicationName)",
      "Cycle the litter box with \(.applicationName)",
      "Clean the litter box with \(.applicationName)",
      "Cycle the Litter-Robot with \(.applicationName)"
    ], shortTitle: "Start clean cycle", systemImageName: "arrow.clockwise")
  }

  static let shortcutTileColor: ShortcutTileColor = .grayGreen
}
