import SwiftUI
import AVKit

struct HomeOverviewView: View {
  @ObservedObject var model: HomeModel
  var demo: Bool
  var server: String
  var events: [Detection]
  var player: AVPlayer?
  var playbackFailed: Bool
  var demoState: String
  var openConnections: () -> Void
  var openActivity: () -> Void
  var openCat: (String) -> Void
  var clean: () -> Void
  var send: (String) -> Void
  @Environment(\.dynamicTypeSize) private var typeSize
  @Environment(\.horizontalSizeClass) private var horizontalSizeClass

  private var isWide: Bool {
    horizontalSizeClass == .regular && !typeSize.isAccessibilitySize
  }

  private var cameraAndCatsColumn: some View {
    VStack(alignment: .leading, spacing: 20) {
      CameraCardView(demo: demo, source: model.liveSource, player: player,
        failed: playbackFailed, openConnections: openConnections)
      VStack(alignment: .leading, spacing: 12) {
        HStack {
          Text("Cats").font(LitterLogStyle.sectionHeading)
          Spacer()
          Button("View cats", systemImage: "arrow.up.right") { openCat(model.selectedCat) }
            .font(.caption.weight(.medium))
        }
        LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 12), count: typeSize.isAccessibilitySize ? 1 : 2), spacing: 12) {
          ForEach(["cat1", "cat2"], id: \.self) { cat in
            Button { openCat(cat) } label: {
              CatSummaryCard(cat: cat, latest: (demo ? LitterLogDemoWeights.readings : model.weights)
                .filter { canonicalCatID($0.cat) == cat }.max { $0.date < $1.date }, demo: demo)
            }.buttonStyle(.plain)
          }
        }
      }
    }
  }

  private var robotAndActivityColumn: some View {
    VStack(alignment: .leading, spacing: 20) {
      RobotCareCard(model: model, demo: demo, demoState: demoState, clean: clean, send: send)
      VStack(alignment: .leading, spacing: 12) {
        HStack {
          Text("Recent activity").font(LitterLogStyle.sectionHeading)
          Spacer()
          Button("See all", systemImage: "arrow.up.right", action: openActivity)
            .font(.caption.weight(.medium))
        }
        if events.isEmpty {
          ContentUnavailableView("No activity yet", systemImage: "pawprint.fill",
            description: Text("Your cats’ activity will appear here when your home server is connected."))
            .card()
        } else {
          VStack(spacing: 16) {
            ForEach(events.prefix(3)) { event in
              if event.id != events.first?.id { Divider() }
              EventRowView(event: event, model: model, demo: demo, server: server)
            }
          }.card()
        }
      }
    }
  }

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 20) {
        HomeGreetingView(demo: demo)
        if isWide {
          HStack(alignment: .top, spacing: 20) {
            cameraAndCatsColumn.frame(maxWidth: .infinity)
            robotAndActivityColumn.frame(maxWidth: .infinity)
          }
        } else {
          cameraAndCatsColumn
          robotAndActivityColumn
        }
        if demo {
          Button(action: openConnections) {
            Label("Demo data · Connections", systemImage: "info.circle")
              .font(.caption).frame(maxWidth: .infinity).padding(.vertical, 8)
          }.buttonStyle(.plain).foregroundStyle(.secondary)
        }
      }
      .foregroundStyle(LitterLogStyle.ink)
      .padding(.horizontal, 20)
      .padding(.top, 8)
      .padding(.bottom, 24)
      .frame(maxWidth: isWide ? 1120 : 680)
      .frame(maxWidth: .infinity)
    }
    .background(canvas)
    .navigationTitle("LitterLog")
    .navigationBarTitleDisplayMode(.large)
  }
}

private struct HomeGreetingView: View {
  var demo: Bool

  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      HStack {
        Text(Date.now, format: .dateTime.weekday(.wide).month(.abbreviated).day())
          .font(.caption.weight(.medium)).foregroundStyle(.secondary)
        Spacer()
        if demo { StatusPill(title: "Demo mode", symbol: "leaf.fill") }
      }
    }
  }
}
