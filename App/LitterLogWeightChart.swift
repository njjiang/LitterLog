import SwiftUI
import Charts

struct LitterLogWeightChart: View {
  var analysis: WeightTrendAnalysis
  let isDemo: Bool
  let reviewReading: (CatWeight) -> Void
  private var _selectedDate = SwiftUI.State<Date?>(initialValue: nil)
  private func selected(in chronological: [WeightTrendPoint]) -> WeightTrendPoint? {
    guard let date = _selectedDate.wrappedValue else { return nil }
    return chronological.min { abs($0.date.timeIntervalSince(date)) < abs($1.date.timeIntervalSince(date)) }
  }
  private var weightDomain: ClosedRange<Double> {
    let low = analysis.points.map(\.averageKg).min() ?? 0
    let high = analysis.points.map(\.averageKg).max() ?? 1
    let padding = max(0.15, (high - low) * 0.25)
    return max(0, low - padding)...(high + padding)
  }
  var body: some View {
    let chronological = analysis.points
    let selected = self.selected(in: chronological)
    let domain = weightDomain
    return VStack(alignment: .leading, spacing: 16) {
      HStack {
        Text("Weight trend").font(LitterLogStyle.sectionHeading)
        Spacer()
        Button("Latest") { _selectedDate.wrappedValue = nil }.font(.body)
          .opacity(selected == nil ? 0 : 1).disabled(selected == nil)
      }
      if let reading = selected {
        HStack(alignment: .firstTextBaseline) {
          Text(reading.date.formatted(date: .abbreviated, time: .shortened))
            .font(.subheadline).foregroundStyle(.secondary)
          Spacer(minLength: 4)
          VStack(alignment: .trailing, spacing: 3) {
            Text(String(format: "%.2f kg", reading.averageKg)).font(LitterLogStyle.rowHeading).monospacedDigit()
            Text(String(format: "%.2f lb", (reading.averageKg / 0.45359237))).font(.subheadline).foregroundStyle(.secondary)
          }
        }.accessibilityElement(children: .combine)
        Button { reviewReading(reading.reading) } label: {
          Label("Raw weigh-in: \(reading.reading.weight_kg.formatted(.number.precision(.fractionLength(2)))) kg", systemImage: "play.circle")
            .font(.body)
        }.buttonStyle(.plain).foregroundStyle(appAccent)
      }
      Text("10-reading rolling average").font(.subheadline).foregroundStyle(.secondary)
      if chronological.isEmpty {
        Text("No weights to chart.").font(.subheadline).foregroundStyle(.secondary)
      }
      Chart {
        ForEach(chronological) { reading in
          if chronological.count > 1 {
            AreaMark(x: .value("Date", reading.date),
              yStart: .value("Baseline", domain.lowerBound),
              yEnd: .value("Weight (kg)", reading.averageKg))
              .foregroundStyle(LinearGradient(colors: [appAccent.opacity(0.15), appAccent.opacity(0.01)], startPoint: .top, endPoint: .bottom))
              .accessibilityHidden(true)
            LineMark(x: .value("Date", reading.date), y: .value("Weight (kg)", reading.averageKg))
              .interpolationMethod(.linear).lineStyle(StrokeStyle(lineWidth: 2))
              .foregroundStyle(appAccent.opacity(0.65))
          }
          if chronological.count == 1 {
            PointMark(x: .value("Date", reading.date), y: .value("Weight (kg)", reading.averageKg))
              .symbolSize(42).foregroundStyle(appAccent)
              .accessibilityLabel(reading.date.formatted(date: .abbreviated, time: .shortened))
              .accessibilityValue(String(format: "%.2f kilograms", reading.averageKg))
          }
        }
        if let reading = selected {
          RuleMark(x: .value("Selected date", reading.date))
            .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 4])).foregroundStyle(appAccent.opacity(0.6))
          PointMark(x: .value("Selected date", reading.date), y: .value("Selected weight", reading.averageKg))
            .symbolSize(110).foregroundStyle(appAccent)
        }
      }
      .frame(height: 210)
      .chartYScale(domain: domain)
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
            select(at: point, proxy: proxy, geometry: geometry)
          }
        }
      }
    }.card()
  }
  private func select(at point: CGPoint, proxy: ChartProxy, geometry: GeometryProxy) {
    guard let anchor = proxy.plotFrame else { return }
    let frame = geometry[anchor]
    let position = min(max(0, point.x - frame.minX), frame.width)
    _selectedDate.wrappedValue = proxy.value(atX: position, as: Date.self)
  }
}
