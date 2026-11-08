import SwiftUI
import AVKit

struct ContentView: View {
  @StateObject private var model = HomeModel()
  @StateObject private var playback = PlaybackState()
  @AppStorage("demo") private var demo = true
  @AppStorage("server") private var server = ""
  @AppStorage("stream") private var stream = ""
  @State private var tab = CommandLine.arguments.contains("--activity-tab") ? LitterLogTab.activity : (CommandLine.arguments.contains("--cats-tab") ? LitterLogTab.cats : LitterLogTab.home)
  @State private var showConnections = false
  @State private var player: AVPlayer?
  @State private var demoState = "Ready"

  private var events: [Detection] { demo ? LitterLogDemoEvents.events : model.events }

  var body: some View {
    TabView(selection: $tab) {
      Tab("Home", systemImage: "house.fill", value: LitterLogTab.home) {
        NavigationStack {
          HomeOverviewView(model: model, demo: demo, server: server, events: events, player: player,
            playbackFailed: playback.failed, demoState: demoState,
            openConnections: { showConnections = true },
            openActivity: { tab = .activity },
            openCat: { model.selectedCat = $0; tab = .cats },
            clean: { send("cycle") }, send: send)
            .toolbar {
              connectionsButton
            }
        }
      }
      Tab("Activity", systemImage: "clock.fill", value: LitterLogTab.activity) {
        NavigationStack {
          ActivityFeedView(model: model, demo: demo, server: server, events: events)
            .toolbar { connectionsButton }
        }
      }
      Tab("Cats", systemImage: "pawprint.fill", value: LitterLogTab.cats) {
        NavigationStack {
          CatsView(model: model, demo: demo, server: server)
            .toolbar { connectionsButton }
        }
      }
    }
    .tint(appAccent)
    .sheet(isPresented: $showConnections) {
      ConnectionsView(model: model, demo: $demo, server: $server, stream: $stream,
        reloadVideo: configurePlayer, connectCamera: connectCamera)
    }
    .task(id: "\(demo)-\(server)") {
      model.reset()
      guard !demo, !server.isEmpty else { return }
      while !Task.isCancelled {
        await model.refresh(base: server)
        do { try await Task.sleep(for: .seconds(5)) } catch { break }
      }
    }
    .task(id: "activity-\(demo)-\(server)-\(tab)") {
      guard !demo, !server.isEmpty, tab == .cats else { return }
      while !Task.isCancelled {
        await model.refreshCatActivity(base: server)
        do { try await Task.sleep(for: .seconds(30)) } catch { break }
      }
    }
    .task(id: "camera-\(demo)-\(server)-\(tab)") {
      guard !demo, !server.isEmpty, tab == .home else { return }
      await connectCamera()
    }
    .sheet(item: $model.correctingEvent) { event in
      CorrectionView(event: event, model: model, server: server)
    }
    .sheet(item: $model.reviewingWeight) { reading in
      WeightReviewView(reading: reading, model: model, server: server, isDemo: demo) {
        model.reviewingWeight = nil
        tab = .activity
      }
    }
    .sheet(item: $model.clip) { clip in
      NavigationStack {
        RecordedVideo(url: clip.url)
          .navigationTitle("Recording")
          .navigationBarTitleDisplayMode(.inline)
          .toolbar {
            ToolbarItem(placement: .confirmationAction) {
              Button("Done") { model.clip = nil }
            }
          }
      }.tint(appAccent)
    }
    .onChange(of: stream) { _, _ in configurePlayer() }
    .onChange(of: demo) { _, _ in configurePlayer() }
    .onAppear { configurePlayer() }
    .onDisappear { player?.pause() }
    .alert("LitterLog", isPresented: Binding(get: { model.message != nil }, set: { if !$0 { model.message = nil } })) {
      Button("OK") { model.message = nil }
    } message: { Text(model.message ?? "") }
  }

  private var connectionsButton: some ToolbarContent {
    ToolbarItem(placement: .topBarTrailing) {
      Button("Connections", systemImage: "gearshape") { showConnections = true }
        .labelStyle(.iconOnly)
    }
  }

  private func send(_ command: String) {
    if demo {
      demoState = ["cycle": "Cleaning", "pause": "Paused", "resume": "Ready", "reset": "Ready"][command] ?? "Ready"
      model.commandFailed = false
      model.commandNote = command == "reset" ? "Sample sensor reset" : "Sample cycle · \(demoState.lowercased())"
    } else {
      Task { await model.command(command, base: server) }
    }
  }

  private func configurePlayer() {
    player?.pause()
    player = nil
    playback.watch(nil)
    guard !demo, let url = URL(string: stream),
      ["http", "https"].contains(url.scheme?.lowercased() ?? ""),
      url.host != nil, url.user == nil, url.password == nil,
      !url.path.isEmpty, url.path != "/" else { return }
    player = AVPlayer(url: url)
    player?.isMuted = true
    playback.watch(player?.currentItem)
    player?.play()
  }

  private func connectCamera() async {
    await model.connectCamera(base: server, quiet: true)
    if let source = model.liveSource, source.mode == "video" {
      player?.pause()
      player = AVPlayer(url: source.url)
      playback.watch(player?.currentItem)
      player?.isMuted = true
      player?.play()
    }
  }
}

private enum LitterLogTab: Hashable {
  case home, activity, cats
}
