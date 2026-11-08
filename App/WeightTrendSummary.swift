import SwiftUI

struct WeightTrendSummary: View {
  var analysis: WeightTrendAnalysis
  var latestReading: CatWeight
  var reviewReading: () -> Void

  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      if let point = analysis.latest {
        Text("\(point.sampleCount)-reading average").font(LitterLogStyle.sectionHeading)
        HStack(alignment: .firstTextBaseline, spacing: 6) {
          Text(point.averageKg, format: .number.precision(.fractionLength(2)))
            .font(LitterLogStyle.metric).monospacedDigit()
          Text("kg").font(.body).foregroundStyle(.secondary)
          Spacer(minLength: 0)
          Text("\((point.averageKg / 0.45359237).formatted(.number.precision(.fractionLength(2)))) lb")
            .font(.body).foregroundStyle(.secondary)
        }
        Text(point.date.formatted(date: .abbreviated, time: .shortened))
          .font(.subheadline).foregroundStyle(.secondary)
        if let day = analysis.dailyChanges.last {
          Divider()
          HStack(alignment: .firstTextBaseline) {
            Text("Daily change").font(.body)
            Spacer()
            Text(day.differenceKg.map { String(format: "%+.3f kg", abs($0) < 0.0005 ? 0 : $0) } ?? "—")
              .font(LitterLogStyle.rowHeading).monospacedDigit()
          }
          Text("Outliers excluded").font(.subheadline).foregroundStyle(.secondary)
        }
      } else {
        Text("No valid weights for a trend").font(.headline)
      }
      Button(action: reviewReading) {
        Label("Latest weigh-in: \(latestReading.weight_kg.formatted(.number.precision(.fractionLength(2)))) kg", systemImage: "play.circle")
          .font(.body)
      }.buttonStyle(.plain).foregroundStyle(appAccent)
        .frame(minHeight: 44, alignment: .leading)
    }.card()
  }
}
