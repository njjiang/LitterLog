import SwiftUI
import AVKit

@MainActor final class RecordedPlayback: ObservableObject {
  let player = AVPlayer()
  @Published var loading = true
  @Published var errorText: String?
  private let url: URL
  private var loader: RecordedResourceLoader?
  private var itemObservation: NSKeyValueObservation?
  private var playbackObservation: NSKeyValueObservation?
  init(url: URL) { self.url = url }
  func start() {
    stop()
    loading = true; errorText = nil
    let item: AVPlayerItem
    if url.isFileURL { item = AVPlayerItem(url: url) }
    else {
      let loader = RecordedResourceLoader(url: url)
      self.loader = loader
      item = AVPlayerItem(asset: loader.asset())
    }
    item.preferredForwardBufferDuration = 3
    player.replaceCurrentItem(with: item)
    itemObservation = item.observe(\.status, options: [.initial, .new]) { [weak self] _, _ in
      Task { @MainActor [weak self] in self?.updateState() }
    }
    playbackObservation = player.observe(\.timeControlStatus, options: [.initial, .new]) { [weak self] _, _ in
      Task { @MainActor [weak self] in self?.updateState() }
    }
    player.play()
  }
  private func updateState() {
    if player.currentItem?.status == .failed {
      loading = false
      errorText = "This recording couldn’t load. Check the home server connection and try again."
    } else {
      loading = player.currentItem?.status != .readyToPlay || player.timeControlStatus == .waitingToPlayAtSpecifiedRate
    }
  }
  func stop() {
    itemObservation = nil; playbackObservation = nil
    player.pause(); player.replaceCurrentItem(with: nil)
    loader?.stop(); loader = nil
  }
}
