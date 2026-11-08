import SwiftUI

struct CatsView: View {
  @Environment(\.dynamicTypeSize) private var dynamicTypeSize
  @Environment(\.horizontalSizeClass) private var horizontalSizeClass
  @ObservedObject var model: HomeModel
  var demo: Bool
  var server: String
  @State private var snapshot: CatHistorySnapshot?
  @State private var preparedRequest: HistoryRequest?
  @State private var visibleHistoryCount = 5
  @State private var visibleUnassignedCount = 5

  private var isWide: Bool {
    horizontalSizeClass == .regular && !dynamicTypeSize.isAccessibilitySize
  }

  private struct HistoryRequest: Equatable {
    let readings: [CatWeight]
    let demo: Bool
    let server: String
  }

  var body: some View {
    let source = demo ? LitterLogDemoWeights.readings : model.weights
    let request = HistoryRequest(readings: source, demo: demo, server: server)
    let current = preparedRequest?.demo == demo && preparedRequest?.server == server ? snapshot : nil
    let history = current?.histories[model.selectedCat]
    let readings = history?.readings ?? []
    let analysis = history?.analysis ?? WeightTrendAnalysis(readings: [], cat: model.selectedCat)
    let unassigned = current?.unassigned ?? []
    return ScrollView {
      LazyVStack(alignment: .leading, spacing: 20) {
        Picker("Cat", selection: $model.selectedCat) { Text("Cat 1").tag("cat1"); Text("Cat 2").tag("cat2") }.pickerStyle(.segmented)
        if demo {
          StatusPill(title: "Sample health data", symbol: "leaf.fill")
        }
        if isWide, let latest = readings.first {
          HStack(alignment: .top, spacing: 20) {
            CatPhotoHeader(cat: model.selectedCat, isDemo: demo).id(model.selectedCat).card()
              .frame(maxWidth: .infinity)
            WeightTrendSummary(analysis: analysis, latestReading: latest) { model.reviewingWeight = latest }
              .frame(maxWidth: .infinity)
          }
          HStack(alignment: .top, spacing: 20) {
            LitterLogWeightChart(analysis: analysis, isDemo: demo) { model.reviewingWeight = $0 }
              .id("\(model.selectedCat)-\(demo)")
              .frame(maxWidth: .infinity)
            RawWeightChart(readings: readings, excludedIDs: analysis.excludedIDs) { model.reviewingWeight = $0 }
              .id("raw-\(model.selectedCat)-\(demo)")
              .frame(maxWidth: .infinity)
          }
          if !analysis.dailyChanges.isEmpty {
            DailyWeightChangesView(changes: analysis.dailyChanges)
          }
        } else {
          CatPhotoHeader(cat: model.selectedCat, isDemo: demo).id(model.selectedCat).card()
          if let latest = readings.first {
            WeightTrendSummary(analysis: analysis, latestReading: latest) { model.reviewingWeight = latest }
            LitterLogWeightChart(analysis: analysis, isDemo: demo) { model.reviewingWeight = $0 }
              .id("\(model.selectedCat)-\(demo)")
            RawWeightChart(readings: readings, excludedIDs: analysis.excludedIDs) { model.reviewingWeight = $0 }
              .id("raw-\(model.selectedCat)-\(demo)")
            if !analysis.dailyChanges.isEmpty {
              DailyWeightChangesView(changes: analysis.dailyChanges)
            }
          } else if current == nil {
            ProgressView("Loading weights…")
              .font(.body).frame(maxWidth: .infinity, minHeight: 100).card()
          } else {
            VStack(spacing: 14) {
              Image(systemName: "chart.xyaxis.line").font(.system(size: 42, weight: .light)).foregroundStyle(appAccent)
                .padding(20).background(appAccent.opacity(0.10), in: Circle())
              Text("No weigh-ins yet").font(.headline).multilineTextAlignment(.center)
              Text("When a Whisker weight matches a video of \(catName(model.selectedCat)), it will appear here. You can correct a clip’s cat to help match it.")
                .font(.subheadline).foregroundStyle(.secondary).multilineTextAlignment(.center)
            }.frame(maxWidth: .infinity).padding(.vertical, 18).card()
          }
        }
        CatActivityCharts(visits: model.catVisits, cat: model.selectedCat, available: model.activityAvailable, demo: demo).id(model.selectedCat)
        if !unassigned.isEmpty {
          DisclosureGroup {
            VStack(spacing: 16) {
              Text("Review nearby clips in Activity. Saved cat corrections update the weight matches.")
                .font(.subheadline).foregroundStyle(.secondary).frame(maxWidth: .infinity, alignment: .leading)
              ForEach(unassigned.prefix(visibleUnassignedCount)) { reading in
                Divider()
                weightHistoryRow(reading)
              }
              if unassigned.count > visibleUnassignedCount {
                Button("Show more weigh-ins") { visibleUnassignedCount += 5 }
                  .font(.body).frame(maxWidth: .infinity, minHeight: 44)
              }
            }.padding(.top, 12)
          } label: {
            Label("\(unassigned.count) weigh-ins still need a cat", systemImage: "web.camera.fill")
              .font(.body)
            }
          .card()
        }
        if !readings.isEmpty {
          VStack(alignment: .leading, spacing: 16) {
            HStack {
              Label("Raw weigh-ins", systemImage: "clock").font(LitterLogStyle.sectionHeading)
              Spacer()
              Text("\(readings.count)").font(.subheadline.weight(.semibold)).foregroundStyle(appAccent)
                .padding(.horizontal, 9).padding(.vertical, 4).background(appAccent.opacity(0.12), in: Capsule())
            }
            ForEach(Array(readings.prefix(visibleHistoryCount).enumerated()), id: \.element.id) { index, reading in
              if index > 0 { Divider() }
              weightHistoryRow(reading, excluded: analysis.excludedIDs.contains(reading.id))
            }
            if readings.count > visibleHistoryCount {
              Button("Show more weigh-ins") { visibleHistoryCount += 5 }
                .font(.body).frame(maxWidth: .infinity, minHeight: 44)
            }
          }.card()
        }
      }.padding(20).frame(maxWidth: isWide ? 1120 : 680).frame(maxWidth: .infinity)
    }.background(canvas).navigationTitle("Cats")
      .task(id: request) {
        guard preparedRequest != request else { return }
        let preparation = Task.detached(priority: .userInitiated) {
          CatHistorySnapshot(readings: request.readings)
        }
        let result = await withTaskCancellationHandler {
          await preparation.value
        } onCancel: {
          preparation.cancel()
        }
        guard !Task.isCancelled else { return }
        snapshot = result
        preparedRequest = request
      }
      .onChange(of: model.selectedCat) { _, _ in visibleHistoryCount = 5 }
      .onChange(of: demo) { _, _ in visibleHistoryCount = 5; visibleUnassignedCount = 5 }
      .onChange(of: server) { _, _ in visibleHistoryCount = 5; visibleUnassignedCount = 5 }
  }
  private func weightHistoryRow(_ reading: CatWeight, excluded: Bool = false) -> some View {
    Button { model.reviewingWeight = reading } label: {
     HStack(spacing: 12) {
      Group {
        if !demo, let event = model.identityEvent(for: reading), let url = thumbnailURL(event) {
          HomeThumbnail(url: url)
        } else {
          Image(systemName: "photo").foregroundStyle(.secondary)
        }
      }
      .frame(width: 60, height: 48)
      .background(appAccent.opacity(0.10))
      .clipped().clipShape(RoundedRectangle(cornerRadius: 9))
      .accessibilityHidden(true)
      let detailsLayout = dynamicTypeSize.isAccessibilitySize
        ? AnyLayout(VStackLayout(alignment: .leading, spacing: 8))
        : AnyLayout(HStackLayout(alignment: .top, spacing: 12))
      detailsLayout {
        VStack(alignment: .leading, spacing: 4) {
          Text(reading.date, format: .dateTime.month(.abbreviated).day().year()).font(.body)
          Text(reading.date, format: .dateTime.hour().minute()).font(.subheadline).foregroundStyle(.secondary)
          if excluded {
            Text("Excluded from trend").font(.subheadline).foregroundStyle(.secondary)
          } else if demo || reading.cat.isEmpty {
            Text(demo ? "Sample reading" : "Confirm cat")
              .font(.subheadline).foregroundStyle(.secondary)
          }
        }.frame(maxWidth: .infinity, alignment: .leading)
        VStack(alignment: dynamicTypeSize.isAccessibilitySize ? .leading : .trailing, spacing: 4) {
          Text(String(format: "%.2f kg", reading.weight_kg)).font(LitterLogStyle.rowHeading).monospacedDigit()
          Text(String(format: "%.2f lb", reading.weight_lbs)).font(.subheadline).foregroundStyle(.secondary)
        }
      }
      Image(systemName: "chevron.right").font(.body).foregroundStyle(.secondary)
     }.contentShape(Rectangle())
    }
    .buttonStyle(.plain).accessibilityHint("Review this weigh-in and its matched video")
  }
  private func thumbnailURL(_ event: Detection) -> URL? {
    guard !demo, let path = event.attention_thumbnail, let base = URL(string: server),
      let url = URL(string: path, relativeTo: base)?.absoluteURL,
      url.host == base.host, url.port == base.port, url.scheme == base.scheme,
      var parts = URLComponents(url: url, resolvingAgainstBaseURL: false) else { return nil }
    parts.queryItems = [URLQueryItem(name: "token", value: PairingKey.read())]
    return parts.url
  }
}
