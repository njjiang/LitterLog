import SwiftUI

struct EventRowView: View {
  static let thumbnailWidth: CGFloat = 56
  static let columnSpacing: CGFloat = 12

  var event: Detection
  @ObservedObject var model: HomeModel
  var demo: Bool
  var server: String

  private var activityEmoji: String? {
    switch event.label.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
    case "pee": return "💧"
    case "poo": return "💩"
    case "both": return "💧💩"
    default: return nil
    }
  }

  private var thumbnailURL: URL? {
    guard !demo, let path = event.attention_thumbnail, let base = URL(string: server),
      let url = URL(string: path, relativeTo: base)?.absoluteURL,
      url.host == base.host, url.port == base.port, url.scheme == base.scheme,
      var components = URLComponents(url: url, resolvingAgainstBaseURL: false) else { return nil }
    components.queryItems = [URLQueryItem(name: "token", value: PairingKey.read())]
    return components.url
  }

  var body: some View {
    let text = ActivityEntryText(event: event)
    return HStack(spacing: Self.columnSpacing) {
      Group {
        if let thumbnailURL {
          HomeThumbnail(url: thumbnailURL)
        } else {
          Image(systemName: "photo")
            .font(.title3).foregroundStyle(LitterLogStyle.secondaryIcon)
        }
      }
      .frame(width: Self.thumbnailWidth, height: 48)
      .background(LitterLogStyle.iconBackground)
      .clipShape(RoundedRectangle(cornerRadius: 9))
      .accessibilityHidden(true)
      VStack(alignment: .leading, spacing: 5) {
        HStack(spacing: 6) {
          if text.isCatSighting {
            Image(systemName: "pawprint.fill")
              .font(.subheadline).foregroundStyle(.secondary).accessibilityHidden(true)
          } else if let activityEmoji {
            Text(activityEmoji).font(.body).accessibilityHidden(true)
          }
          Text(text.title)
        }.font(.body.weight(.semibold)).foregroundStyle(.primary)
        if let detail = text.detail {
          Text(detail).font(.subheadline).foregroundStyle(.secondary)
        }
      }
      .fixedSize(horizontal: false, vertical: true)
      Spacer(minLength: 4)
      VStack(alignment: .trailing, spacing: 4) {
        if let date = event.date {
          Text(date, format: .dateTime.hour().minute()).font(.subheadline).monospacedDigit().foregroundStyle(.secondary)
        }
        if event.has_video == true || !demo {
          HStack(spacing: 0) {
            if event.has_video == true {
              Button {
                Task { await model.playRecording(event, base: server) }
              } label: {
                if model.preparingClipID == event.id {
                  ProgressView().frame(width: 44, height: 44)
                } else {
                  Label("Play recording", systemImage: "play.fill").labelStyle(.iconOnly)
                    .font(.subheadline).frame(width: 44, height: 44)
                    .background(appAccent.opacity(0.08), in: Circle())
                }
              }.buttonStyle(.plain).foregroundStyle(appAccent).disabled(model.preparingClip)
            }
            if !demo {
              Menu {
                Button("Edit prediction", systemImage: "pencil") { model.correctingEvent = event }
              } label: {
                Label("Activity options", systemImage: "ellipsis").labelStyle(.iconOnly)
                  .frame(width: 44, height: 44)
              }.foregroundStyle(.secondary)
            }
          }
        }
      }
    }
    .accessibilityElement(children: .contain)
    .contextMenu {
      if !demo {
        Button("Edit prediction", systemImage: "pencil") { model.correctingEvent = event }
      }
    }
  }
}
