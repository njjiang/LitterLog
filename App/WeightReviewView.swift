import SwiftUI

struct WeightReviewView: View {
  let reading: CatWeight
  let linkedEvent: Detection?
  let isDemo: Bool
  let server: String
  let openActivity: () -> Void
  @ObservedObject var model: HomeModel
  @StateObject private var form: WeightReviewForm
  @Environment(\.dismiss) private var dismiss

  init(reading: CatWeight, model: HomeModel, server: String, isDemo: Bool, openActivity: @escaping () -> Void) {
    self.reading = reading; self.model = model; self.server = server; self.isDemo = isDemo
    self.linkedEvent = isDemo ? nil : model.identityEvent(for: reading)
    self.openActivity = openActivity
    _form = StateObject(wrappedValue: WeightReviewForm(reading: reading))
  }
  private var currentReading: CatWeight { model.weights.first { $0.id == reading.id } ?? reading }
  private var assignment: String {
    currentReading.cat.isEmpty ? "Unassigned" : catName(currentReading.cat)
  }
  private func thumbnail(_ event: Detection) -> URL? {
    guard let path = event.attention_thumbnail, let base = URL(string: server),
       let url = URL(string: path, relativeTo: base)?.absoluteURL,
       url.host == base.host, url.port == base.port, url.scheme == base.scheme,
       var parts = URLComponents(url: url, resolvingAgainstBaseURL: false) else { return nil }
    parts.queryItems = [URLQueryItem(name: "token", value: PairingKey.read())]
    return parts.url
  }
  var body: some View {
    NavigationStack {
      ScrollView {
        VStack(alignment: .leading, spacing: 20) {
          VStack(alignment: .leading, spacing: 12) {
            Text(reading.date.formatted(date: .abbreviated, time: .shortened)).font(.subheadline).foregroundStyle(.secondary)
            HStack(alignment: .firstTextBaseline, spacing: 8) {
              Text(String(format: "%.2f", reading.weight_kg)).font(LitterLogStyle.metric).monospacedDigit()
              Text("kg").font(.subheadline).foregroundStyle(.secondary)
              Spacer()
              Text(String(format: "%.2f lb", reading.weight_lbs)).foregroundStyle(.secondary)
            }
            Label(assignment, systemImage: currentReading.cat.isEmpty ? "questionmark.circle" : "cat.fill").foregroundStyle(appAccent)
            if currentReading.attribution == "ambiguous video identities" {
              Text("The nearby clips do not agree on one cat.").font(.subheadline).foregroundStyle(.secondary)
            }
          }.card()

          if isDemo {
            ContentUnavailableView("Sample weigh-in", systemImage: "scalemass", description: Text("Demo weights have no real recording. Turn off Demo mode to review your cats’ videos."))
          } else if let event = linkedEvent {
            VStack(alignment: .leading, spacing: 14) {
              Text(form.note == nil ? (reading.video_match == "near weigh-in" ? "Nearby video" : "Matched video") : "Reviewed video").font(.headline)
              if let url = thumbnail(event) {
                HomeThumbnail(url: url).frame(maxWidth: .infinity).frame(height: 180)
                  .clipped().clipShape(RoundedRectangle(cornerRadius: 14)).accessibilityLabel("Matched recording preview")
              }
              if let date = event.date { Text(date.formatted(date: .abbreviated, time: .shortened)).font(.subheadline).foregroundStyle(.secondary) }
              if event.has_video == true {
                Button { form.watch(event, model: model, server: server) } label: {
                  HStack {
                    if form.preparing { ProgressView().tint(filledButtonText) }
                    else { Image(systemName: "play.fill") }
                    Text(form.preparing ? "Preparing video…" : "Watch video")
                  }.frame(maxWidth: .infinity).padding(.vertical, 5).foregroundStyle(filledButtonText)
                }.buttonStyle(.borderedProminent).tint(filledAccent).disabled(form.preparing || form.saving)
              } else {
                Label("The recording link is not available yet.", systemImage: "web.camera").font(.subheadline).foregroundStyle(.secondary)
              }
              Divider()
              Picker("Cat in this video", selection: $form.cat) {
                Text("Unassigned").tag("")
                Text("Cat 1").tag("cat1")
                Text("Cat 2").tag("cat2")
              }.disabled(form.saving || form.preparing)
              Text("Choose the cat you see in the video. The correction also updates weight matches that use this clip.")
                .font(.subheadline).foregroundStyle(.secondary)
              Button { Task { await form.save(event, model: model, server: server) } } label: {
                HStack {
                  if form.saving { ProgressView() }
                  Text(form.saving ? "Saving…" : "Save cat correction")
                }.frame(maxWidth: .infinity)
              }.buttonStyle(.bordered).disabled(form.saving || form.preparing)
            }.card()
          } else {
            VStack(alignment: .leading, spacing: 14) {
              Label("No matched video", systemImage: "web.camera").font(.headline)
              Text("No recording was found at or within three minutes of this weigh-in. You can look for other clips in Activity.")
                .font(.subheadline).foregroundStyle(.secondary)
              Button("Review Activity") { dismiss(); openActivity() }
            }.card()
          }
          if let note = form.note {
            Label(note, systemImage: "checkmark.circle.fill").font(.subheadline).foregroundStyle(appAccent)
            Text("Current weight assignment: \(assignment)").font(.subheadline).foregroundStyle(.secondary)
          }
          if let error = form.error { Label(error, systemImage: "exclamationmark.circle").font(.subheadline).foregroundStyle(warningText) }
        }.padding(20)
      }.background(canvas).navigationTitle("Review weigh-in").navigationBarTitleDisplayMode(.inline)
        .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { form.cancelPlaybackPreparation(); dismiss() }.disabled(form.saving) } }
        .interactiveDismissDisabled(form.saving)
        .onChange(of: form.cat) { _, _ in if !form.saving { form.note = nil; form.error = nil } }
        .onDisappear { form.cancelPlaybackPreparation() }
        .sheet(item: $form.clip) { clip in
          NavigationStack {
            RecordedVideo(url: clip.url).navigationTitle("Weigh-in video").navigationBarTitleDisplayMode(.inline)
              .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { form.clip = nil } } }
          }
        }
    }.tint(appAccent)

  }
}

@MainActor private final class WeightReviewForm: ObservableObject {
  @Published var cat: String
  @Published var saving = false
  @Published var preparing = false
  @Published var clip: ClipPresentation?
  @Published var note: String?
  @Published var error: String?
  private var preparation: Task<Void, Never>?
  init(reading: CatWeight) { cat = reading.cat }

  func watch(_ event: Detection, model: HomeModel, server: String) {
    guard !preparing, !saving else { return }
    preparing = true; error = nil
    preparation = Task { [weak self] in
      guard let self else { return }
      defer { self.preparing = false; self.preparation = nil }
      do {
        let url = try await model.recordingURL(event, base: server)
        try Task.checkCancellation()
        self.clip = ClipPresentation(url: url)
      } catch {
        if !Task.isCancelled { self.error = "Couldn’t load this recording. Check the home server connection and try again." }
      }
    }
  }

  func save(_ event: Detection, model: HomeModel, server: String) async {
    guard !saving, !preparing else { return }
    saving = true; note = nil; error = nil
    defer { saving = false }
    let label = ["cat1", "cat2"].contains(event.label.lowercased()) ? (cat.isEmpty ? "recording" : cat) : event.label
    if await model.correct(event, base: server, cat: cat, label: label, reportError: false) {
      note = "Video correction saved."
    } else {
      error = "Couldn’t save the cat correction. Check the connection and try again."
    }
  }

  func cancelPlaybackPreparation() { preparation?.cancel() }
}

