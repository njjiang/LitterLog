import SwiftUI
import UIKit

enum LitterLogStyle {
  static let ink = Color.primary
  static let sectionHeading = Font.headline
  static let rowHeading = Font.body.weight(.semibold)
  static let metric = Font.largeTitle.weight(.semibold)
  static let iconBackground = Color(uiColor: .tertiarySystemFill)
  static let secondaryIcon = Color.secondary
}

let appAccent = Color("AccentColor")
// Keep white labels readable while links brighten for dark backgrounds.
let filledAccent = Color("FilledAccentColor")
let canvas = Color(uiColor: .systemGroupedBackground)
let cardSurface = Color(uiColor: .secondarySystemGroupedBackground)
let cardOutline = Color(uiColor: .separator).opacity(0.18)
let filledButtonText = Color.white
let successText = Color("SuccessColor")
let warningText = Color(uiColor: .systemOrange)

extension View {
  func card() -> some View {
    padding(20)
      .background(cardSurface, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
  }
}
