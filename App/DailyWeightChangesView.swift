import SwiftUI

struct DailyWeightChangesView: View {
  var changes: [DailyWeightChange]

  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      Text("Daily weight changes").font(LitterLogStyle.sectionHeading)
      ForEach(Array(changes.suffix(7).reversed())) { day in
        HStack(alignment: .firstTextBaseline) {
          Text(day.day, format: .dateTime.month(.abbreviated).day()).font(.body)
          Spacer(minLength: 8)
          Text(day.differenceKg.map { String(format: "%+.3f kg", abs($0) < 0.0005 ? 0 : $0) } ?? "—")
            .font(LitterLogStyle.rowHeading).monospacedDigit()
            .accessibilityLabel(day.differenceKg == nil ? Text("No previous-day data") : Text("Weight change"))
            .accessibilityValue(day.differenceKg.map { String(format: "%+.3f kilograms", $0) } ?? "")
        }.accessibilityElement(children: .combine)
        if day.id != changes.suffix(7).first?.id { Divider() }
      }
    }.card()
  }
}
