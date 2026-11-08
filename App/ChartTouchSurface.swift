import SwiftUI

struct ChartTouchSurface: View {
  let select: (CGPoint) -> Void
  var body: some View {
    Rectangle().fill(.clear).contentShape(Rectangle())
      .gesture(SpatialTapGesture().onEnded { select($0.location) })
      .simultaneousGesture(DragGesture(minimumDistance: 12).onChanged { value in
        guard abs(value.translation.width) > abs(value.translation.height) else { return }
        select(value.location)
      })
      .onContinuousHover { phase in
        if case .active(let point) = phase { select(point) }
      }
      .accessibilityHidden(true)
  }
}

