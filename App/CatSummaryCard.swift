import SwiftUI

struct CatSummaryCard: View {
  var cat: String
  var latest: CatWeight?
  var demo: Bool

  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      HStack {
        Text(catName(cat)).font(.subheadline.weight(.semibold))
        Spacer(minLength: 4)
        Image(systemName: "chevron.right").font(.caption2.weight(.semibold)).foregroundStyle(.tertiary)
      }
      VStack(alignment: .leading, spacing: 5) {
        if let latest {
          HStack(alignment: .firstTextBaseline, spacing: 3) {
            Text(latest.weight_kg, format: .number.precision(.fractionLength(2)))
              .font(LitterLogStyle.metric).monospacedDigit()
            Text("kg").font(.caption).foregroundStyle(.secondary)
          }
        } else {
          Text("No weigh-ins yet").font(.caption).foregroundStyle(.secondary)
        }
        Text(demo ? "Sample weight" : "Latest weight")
          .font(.caption).foregroundStyle(.secondary)
      }
    }
    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
    .padding(16)
    .background(cardSurface, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
    .contentShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
    .accessibilityElement(children: .combine)
    .accessibilityHint("View cat health and weight history")
  }
}
