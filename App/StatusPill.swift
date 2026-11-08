import SwiftUI

struct StatusPill: View {
  var title: String
  var symbol = "circle.fill"
  var color: Color = .secondary
  var isPositive = false

  var body: some View {
    Label(title, systemImage: symbol)
      .font(.subheadline.weight(.medium))
      .foregroundStyle(isPositive ? successText : color)
      .padding(.horizontal, 9)
      .padding(.vertical, 5)
      .background(isPositive ? successText.opacity(0.12) : LitterLogStyle.iconBackground, in: Capsule())
      .fixedSize(horizontal: false, vertical: true)
  }
}
