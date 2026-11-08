import SwiftUI
import AVKit

struct CameraCardView: View {
  var demo: Bool
  var source: CameraSource?
  var player: AVPlayer?
  var failed: Bool
  var openConnections: () -> Void

  private var isConnected: Bool {
    !demo && source != nil && (source?.mode == "snapshots" || !failed)
  }

  var body: some View {
    VStack(spacing: 0) {
      if demo {
        Rectangle().fill(LitterLogStyle.iconBackground)
          .aspectRatio(1.65, contentMode: .fit)
          .overlay {
            Image("DemoCorner").resizable().scaledToFill()
          }
          .clipped()
          .accessibilityLabel("Illustration of a sleeping cat in a sunny litter corner")
      } else if let source, source.mode == "snapshots" {
        LiveSnapshotFeed(url: source.url)
      } else if let player, !failed {
        VideoPlayer(player: player).aspectRatio(1.65, contentMode: .fit)
      } else {
        VStack(spacing: 12) {
          Image(systemName: "web.camera.fill").font(.largeTitle).foregroundStyle(appAccent)
          Text(failed ? "Camera unavailable" : "Camera not connected")
            .font(.headline).multilineTextAlignment(.center)
          Button("Connect camera", action: openConnections).buttonStyle(.bordered)
        }
        .frame(maxWidth: .infinity).padding(.vertical, 30)
        .background(LitterLogStyle.iconBackground)
      }
      HStack(spacing: 12) {
        Image(systemName: "web.camera.fill").font(.body).foregroundStyle(appAccent)
        VStack(alignment: .leading, spacing: 3) {
          Text("Camera").font(LitterLogStyle.sectionHeading)
          if !demo {
            Text(source == nil ? "Connect your home camera" : "UniFi Protect")
              .font(.caption).foregroundStyle(.secondary)
          }
        }
        Spacer(minLength: 0)
        StatusPill(title: demo ? "Demo" : isConnected ? "Connected" : "Offline",
          symbol: demo ? "info.circle" : isConnected ? "circle.fill" : "wifi",
          isPositive: isConnected)
      }
      .padding(16)
      .background(cardSurface)
    }
    .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
  }
}
