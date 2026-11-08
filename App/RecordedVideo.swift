import SwiftUI
import AVKit

struct RecordedVideo: View {
  let url: URL
  @StateObject private var playback: RecordedPlayback
  init(url: URL) { self.url = url; _playback = StateObject(wrappedValue: RecordedPlayback(url: url)) }
  var body: some View {
    ZStack {
      VideoPlayer(player: playback.player)
      if let error = playback.errorText {
        VStack(spacing: 14) {
          Image(systemName: "wifi.exclamationmark").font(.title)
          Text(error).multilineTextAlignment(.center)
          Button("Try again") { playback.start() }
            .buttonStyle(.borderedProminent).tint(filledAccent).foregroundStyle(filledButtonText)
        }.padding(24).background(.regularMaterial, in: RoundedRectangle(cornerRadius: 18)).padding()
      } else if playback.loading {
        ProgressView("Loading video…").padding(20).background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
      }
    }.onAppear { playback.start() }.onDisappear { playback.stop() }
  }
}

