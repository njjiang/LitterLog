import SwiftUI

struct LiveSnapshotFeed: View {
  let url: URL
  @StateObject private var feed = SnapshotModel()
  var body: some View {
    ZStack(alignment: .bottomLeading) {
      if let image = feed.image {
        Image(uiImage: image).resizable().scaledToFit()
      } else { ProgressView("Connecting to Protect…").frame(maxWidth: .infinity, minHeight: 150) }
      if feed.unavailable {
        Text("Reconnecting…").font(.caption2.bold()).padding(8).background(.ultraThinMaterial, in: Capsule()).padding(10)
      }
    }.clipShape(RoundedRectangle(cornerRadius: 14)).task(id: url) { await feed.run(url: url) }
  }
}
// AVPlayer requests only the byte ranges it needs; URLSession validates the home CA
// and delivers each network chunk immediately instead of downloading the whole clip.
