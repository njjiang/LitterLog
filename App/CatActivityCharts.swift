import SwiftUI
import Charts

struct CatActivityCharts: View {
  let visits: [CatVisit]
  let cat: String
  let available: Bool
  let demo: Bool
  private var _days = SwiftUI.State<Int>(initialValue: 7)
  private var _selectedDay = SwiftUI.State<Date?>(initialValue: nil)
  private var _peeHour = SwiftUI.State<Int?>(initialValue: nil)
  private var _pooHour = SwiftUI.State<Int?>(initialValue: nil)
  private var days: Int { _days.wrappedValue }
  private let pooColor = Color.orange
  private struct Count: Identifiable {
    let date: Date
    let kind: String
    let count: Int
    var id: String { "\(date.timeIntervalSince1970)-\(kind)" }
  }
  var body: some View {
    let calendar = Calendar.current
    let now = Date()
    let today = calendar.startOfDay(for: now)
    let start = calendar.date(byAdding: .day, value: 1-days, to: today)!
    let dated = visits.compactMap { visit -> (visit: CatVisit, date: Date)? in
      guard visit.cat == cat, let date = visit.date, date >= start, date <= now else { return nil }
      return (visit, date)
    }
    let filtered = dated.map(\.visit)
    let dates = (0..<days).compactMap { calendar.date(byAdding: .day, value: $0, to: start) }
    var peeByDay: [Date: Int] = [:]
    var pooByDay: [Date: Int] = [:]
    for item in dated {
      let day = calendar.startOfDay(for: item.date)
      if item.visit.pee { peeByDay[day, default: 0] += 1 }
      if item.visit.poo { pooByDay[day, default: 0] += 1 }
    }
    let counts = dates.flatMap { date in
      [
        Count(date: date, kind: "Pee", count: peeByDay[date, default: 0]),
        Count(date: date, kind: "Poo", count: pooByDay[date, default: 0])
      ]
    }
    return VStack(alignment: .leading, spacing: 18) {
      HStack {
        Text("Bathroom activity").font(LitterLogStyle.sectionHeading)
        Spacer()
        Picker("Activity period", selection: _days.projectedValue) {
          Text("7 days").tag(7); Text("30 days").tag(30)
        }.pickerStyle(.menu)
      }
      if demo || !available {
        Text(demo ? "Bathroom charts appear when connected to your cats’ recorded activity." : "Activity charts unavailable. Check the home server connection.")
          .font(.subheadline).foregroundStyle(.secondary)
      } else {
        HStack(spacing: 32) {
          total("Pee", count: filtered.filter(\.pee).count, color: appAccent)
          total("Poo", count: filtered.filter(\.poo).count, color: pooColor)
        }
        if filtered.isEmpty {
          Text("No attributed pee or poo detections in this period.").font(.subheadline).foregroundStyle(.secondary)
        } else {
          Text("Daily counts").font(LitterLogStyle.sectionHeading)
          Chart {
            if let selected = _selectedDay.wrappedValue {
              RuleMark(x: .value("Selected day", selected, unit: .day)).foregroundStyle(.secondary.opacity(0.4))
            }
            ForEach(counts) { item in
            LineMark(x: .value("Day", item.date, unit: .day), y: .value("Visits", item.count))
              .foregroundStyle(by: .value("Activity", item.kind))
              .symbol(by: .value("Activity", item.kind))
              .lineStyle(StrokeStyle(lineWidth: 2))
              .accessibilityLabel("\(item.date.formatted(date: .abbreviated, time: .omitted)), \(item.kind)")
              .accessibilityValue("\(item.count) estimated visits")
          }
          }
          .chartForegroundStyleScale(["Pee": appAccent, "Poo": pooColor])
          .chartYScale(domain: 0...max(1, counts.map(\.count).max() ?? 1))
          .chartXAxis {
            AxisMarks(values: .automatic(desiredCount: 3)) {
              AxisValueLabel(format: .dateTime.month(.abbreviated).day()).font(.subheadline)
            }
          }
          .chartYAxis {
            AxisMarks(values: .automatic(desiredCount: 3)) {
              AxisGridLine()
              AxisValueLabel().font(.subheadline)
            }
          }
          .chartLegend {
            HStack(spacing: 20) {
              Text("Pee").foregroundStyle(appAccent)
              Text("Poo").foregroundStyle(pooColor)
            }.font(.subheadline)
          }
          .chartOverlay { proxy in
            GeometryReader { geometry in
              ChartTouchSurface { point in
                guard let frame = proxy.plotFrame.map({ geometry[$0] }), frame.contains(point),
                   let date: Date = proxy.value(atX: point.x - frame.minX) else { return }
                _selectedDay.wrappedValue = calendar.startOfDay(for: min(today, max(start, date)))
              }
            }
          }
          .onChange(of: days) { _, _ in _selectedDay.wrappedValue = nil }
          .frame(height: 190)
          let selected = _selectedDay.wrappedValue ?? today
          let dayCounts = counts.filter { calendar.isDate($0.date, inSameDayAs: selected) }
          Text("\(selected.formatted(date: .abbreviated, time: .omitted)) · " + dayCounts.map { "\($0.kind): \($0.count)" }.joined(separator: " · "))
            .font(.subheadline).foregroundStyle(.secondary).frame(minHeight: 32, alignment: .leading)
          timing("Pee", filtered: filtered.filter(\.pee), color: appAccent)
          timing("Poo", filtered: filtered.filter(\.poo), color: pooColor)
        }
      }
    }.card()
  }
  private func total(_ title: String, count: Int, color: Color) -> some View {
    VStack(alignment: .leading, spacing: 4) {
      Text(title).font(.subheadline).foregroundStyle(color)
      Text("\(count)").font(.title2.weight(.semibold)).monospacedDigit()
      Text("estimated visits").font(.subheadline).foregroundStyle(.secondary)
    }.frame(maxWidth: .infinity, alignment: .leading)
  }
  private func timing(_ title: String, filtered: [CatVisit], color: Color) -> some View {
    var hours = Array(repeating: 0, count: 24)
    for visit in filtered {
      if let date = visit.date {
        let hour = Calendar.current.component(.hour, from: date)
        if (0..<24).contains(hour) { hours[hour] += 1 }
      }
    }
    let selection = title == "Pee" ? _peeHour.projectedValue : _pooHour.projectedValue
    let maximum = hours.max() ?? 0
    let peaks = hours.indices.filter { hours[$0] == maximum && maximum > 0 }
    return VStack(alignment: .leading, spacing: 8) {
      Text("\(title) by time of day").font(LitterLogStyle.sectionHeading)
      if filtered.isEmpty {
        Text("No \(title.lowercased()) detections yet.").font(.subheadline).foregroundStyle(.secondary)
      } else {
        Chart {
          if let hour = selection.wrappedValue {
            RuleMark(x: .value("Selected hour", hour)).foregroundStyle(.secondary.opacity(0.4))
          }
          ForEach(0..<24, id: \.self) { hour in
          LineMark(x: .value("Hour", hour), y: .value("Visits", hours[hour]))
            .foregroundStyle(color)
            .symbol(.circle)
            .lineStyle(StrokeStyle(lineWidth: 2))
            .accessibilityLabel("\(hour):00–\(hour + 1):00")
            .accessibilityValue("\(hours[hour]) estimated visits")
        }
        }
        .chartOverlay { proxy in
          GeometryReader { geometry in
            ChartTouchSurface { point in
              guard let frame = proxy.plotFrame.map({ geometry[$0] }), frame.contains(point),
                 let hour: Double = proxy.value(atX: point.x - frame.minX) else { return }
              selection.wrappedValue = min(23, max(0, Int(hour.rounded())))
            }
          }
        }
        .chartXScale(domain: -1...24)
        .chartXAxis {
          AxisMarks(values: [0, 6, 12, 18]) { value in
            AxisValueLabel { if let hour = value.as(Int.self) { Text(String(format: "%02d:00", hour)).font(.subheadline) } }
          }
        }
        .chartYAxis {
          AxisMarks(values: .automatic(desiredCount: 3)) {
            AxisGridLine()
            AxisValueLabel().font(.subheadline)
          }
        }
        .chartYScale(domain: 0...max(1, maximum))
        .frame(height: 120)
        Text(selection.wrappedValue.map { hour in String(format: "%02d:00–%02d:00 · %d visits", hour, hour+1, hours[hour]) } ?? (filtered.count < 3 ? "More visits needed to show a usual time." : peaks.count == 1 ? "Most recorded around \(String(format: "%02d:00–%02d:00", peaks[0], peaks[0]+1))" : "No single busiest hour yet."))
          .font(.subheadline).foregroundStyle(.secondary).frame(minHeight: 32, alignment: .leading)
      }
    }
  }
}

