import SwiftUI

struct ActivityFeedView: View {
  @ObservedObject var model: HomeModel
  var demo: Bool
  var server: String
  var events: [Detection]
  @State private var filter = ActivityFilter.all
  @State private var includeNone = CommandLine.arguments.contains("--show-none")
  @State private var showFilters = CommandLine.arguments.contains("--show-filters")

  private var sections: [ActivityDaySection] {
    let shown = events.filter { filter.matches($0, includeNone: includeNone) }
    let grouped = Dictionary(grouping: shown) { event in
      event.date.map { Calendar.current.startOfDay(for: $0) } ?? .distantPast
    }
    return grouped.keys.sorted(by: >).map { ActivityDaySection(date: $0, events: grouped[$0] ?? []) }
  }

  var body: some View {
    List {
      Section {
        Button {
          showFilters = true
        } label: {
          Label(filter.title, systemImage: "line.3.horizontal.decrease")
            .font(.subheadline.weight(.medium))
            .padding(.horizontal, 14).padding(.vertical, 10)
            .background(appAccent.opacity(0.08), in: Capsule())
        }
        .buttonStyle(.plain).foregroundStyle(appAccent)
        .accessibilityLabel("Filter: \(filter.title)")
        .padding(.vertical, 8)
      }.listRowBackground(Color.clear).listRowInsets(EdgeInsets(top: 0, leading: 0, bottom: 8, trailing: 0))

      ForEach(sections) { section in
        Section {
          ForEach(section.events) { event in
            EventRowView(event: event, model: model, demo: demo, server: server)
              .padding(.vertical, 7)
              .alignmentGuide(.listRowSeparatorLeading) { _ in 0 }
              .alignmentGuide(.listRowSeparatorTrailing) { $0.width }
              .listRowBackground(cardSurface)
          }
        } header: {
          if Calendar.current.isDateInToday(section.date) {
            Text("Today")
          } else if section.date == .distantPast {
            Text("Earlier activity")
          } else {
            Text(section.date, format: .dateTime.month(.abbreviated).day())
          }
        }
      }
      if sections.isEmpty {
        ContentUnavailableView("No activity yet", systemImage: "pawprint.fill",
          description: Text(emptyDescription))
          .listRowBackground(Color.clear)
      }
      if !demo {
        Section {
          if model.hasOlderEvents {
            Button(model.loadingOlder ? "Loading…" : "Load older events") {
              Task { await model.loadOlder(base: server) }
            }.disabled(model.loadingOlder)
          }
          Button(model.importingHistory ? "Loading recordings…" : "Import past 7 days from Protect") {
            Task { await model.importHistory(base: server) }
          }.disabled(model.importingHistory)
        } footer: {
          Text("Recordings are available to watch before classification finishes.")
        }.listRowBackground(cardSurface)
      }
    }
    .listStyle(.insetGrouped)
    .scrollContentBackground(.hidden)
    .background(canvas)
    .navigationTitle("Activity")
    .refreshable { if !demo { await model.refresh(base: server) } }
    .sheet(isPresented: $showFilters) {
      NavigationStack {
        Form {
          Section {
            Picker("Filter", selection: $filter) {
              ForEach(ActivityFilter.pickerCases) { item in
                Text(item.title).tag(item)
              }
            }
            .pickerStyle(.inline)
            .labelsHidden()
          }
          Section {
            Toggle("None", isOn: $includeNone)
          }
        }
        .scrollContentBackground(.hidden).background(canvas)
        .navigationTitle("Filter").navigationBarTitleDisplayMode(.inline)
        .toolbar {
          ToolbarItem(placement: .confirmationAction) {
            Button("Done") { showFilters = false }
          }
        }
      }.tint(appAccent)
    }
  }

  private var emptyDescription: String {
    if !includeNone, filter != .none, events.contains(where: { ActivityFilter.none.matches($0) }) {
      return "No-activity detections are hidden by default. Turn on Show no activity to view them."
    }
    return filter == .all
      ? "Activity will appear when your home server receives a detection."
      : "Try a different activity filter."
  }
}

private struct ActivityDaySection: Identifiable {
  var date: Date
  var events: [Detection]
  var id: Date { date }
}
