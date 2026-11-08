import SwiftUI
import AVKit

@MainActor final class PlaybackState: ObservableObject {
  @Published var failed = false
  private var observation: NSKeyValueObservation?
  func watch(_ item: AVPlayerItem?) {
    observation = nil
    failed = false
    observation = item?.observe(\.status, options: [.initial, .new]) { [weak self] item, _ in
      let failed = item.status == .failed
      DispatchQueue.main.async { self?.failed = failed }
    }
  }
}

