import SwiftUI
import Charts

struct RawWeightChart: View {
  var readings: [CatWeight]
  var excludedIDs: Set<String>
  var reviewReading: (CatWeight) -> Void
  @State private var selectedDate: Date?

  private var chronological: [CatWeight] {
    readings.reversed().filter { $0.weight_kg.isFinite && $0.weight_kg > 0 && $0.date != .distantPast }
  }
  private func selected(in chronological: [CatWeight]) -> CatWeight? {
    guard let selectedDate else { return nil }
    return chronological.min { abs($0.date.timeIntervalSince(selectedDate)) < abs($1.date.timeIntervalSince(selectedDate)) }
  }

  var body: some View {
    let chronological = self.chronological
    let selected = self.selected(in: chronological)
    return VStack(alignment: .leading, spacing: 16) {
      Text("Raw weights").font(LitterLogStyle.sectionHeading)
      Text("All weigh-ins, including outliers").font(.subheadline).foregroundStyle(.secondary)
      if let reading = selected {
        Button { reviewReading(reading) } label: {
          Label("\(reading.weight_kg.formatted(.number.precision(.fractionLength(2)))) kg · \(reading.date.formatted(date: .abbreviated, time: .shortened))", systemImage: "play.circle")
            .font(.subheadline.weight(.medium))
        }.buttonStyle(.plain).foregroundStyle(appAccent)
      }
      Chart {
        ForEach(chronological) { reading in
          LineMark(x: .value("Date", reading.date), y: .value("Raw weight (kg)", reading.weight_kg))
            .foregroundStyle(appAccent.opacity(0.3))
            .accessibilityHidden(true)
          PointMark(x: .value("Date", reading.date), y: .value("Raw weight (kg)", reading.weight_kg))
            .symbol(excludedIDs.contains(reading.id) ? .diamond : .circle)
            .symbolSize(excludedIDs.contains(reading.id) ? 65 : 35)
            .foregroundStyle(excludedIDs.contains(reading.id) ? Color.secondary : appAccent)
            .accessibilityLabel(reading.date.formatted(date: .abbreviated, time: .shortened))
            .accessibilityValue("\(reading.weight_kg.formatted()) kilograms\(excludedIDs.contains(reading.id) ? ", excluded from the rolling average" : "")")
        }
        if let reading = selected {
          RuleMark(x: .value("Selected date", reading.date)).foregroundStyle(.secondary.opacity(0.5))
        }
      }
      .frame(height: 180)
      .chartYAxisLabel(position: .topLeading) { Text("kg").font(.subheadline) }
      .chartYAxis {
        AxisMarks(position: .leading, values: .automatic(desiredCount: 4)) {
          AxisGridLine()
          AxisValueLabel().font(.subheadline)
        }
      }
      .chartXAxis {
        AxisMarks(values: .automatic(desiredCount: 3)) {
          AxisValueLabel(format: .dateTime.month(.abbreviated).day()).font(.subheadline)
        }
      }
      .chartOverlay { proxy in
        GeometryReader { geometry in
          ChartTouchSurface { point in
            guard let anchor = proxy.plotFrame else { return }
            let frame = geometry[anchor]
            selectedDate = proxy.value(atX: min(max(0, point.x - frame.minX), frame.width), as: Date.self)
          }
        }
      }
      if !excludedIDs.isEmpty {
        Label("Excluded", systemImage: "diamond.fill")
          .font(.subheadline).foregroundStyle(.secondary)
      }
    }.card()
  }
}
