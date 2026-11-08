import SwiftUI
import UIKit

struct CatPhotoCropView: View {
  let image: UIImage
  let catName: String
  var cancel: () -> Void
  var usePhoto: (UIImage) -> Void
  @State private var zoom: CGFloat = 1
  @State private var offset = CGSize.zero
  @State private var viewportSide: CGFloat = 1
  @GestureState private var gestureZoom: CGFloat = 1
  @GestureState private var gestureOffset = CGSize.zero

  var body: some View {
    NavigationStack {
      VStack(spacing: 24) {
        GeometryReader { proxy in
          let side = min(proxy.size.width, proxy.size.height)
          let totalZoom = zoom * gestureZoom
          let totalOffset = CGSize(width: offset.width + gestureOffset.width,
            height: offset.height + gestureOffset.height)
          ZStack {
            Color.black
            Image(uiImage: image)
              .resizable()
              .scaledToFill()
              .frame(width: side, height: side)
              .scaleEffect(totalZoom)
              .offset(totalOffset)
          }
          .frame(width: side, height: side)
          .clipShape(Circle())
          .overlay(Circle().stroke(.white.opacity(0.8), lineWidth: 2))
          .frame(maxWidth: .infinity, maxHeight: .infinity)
          .contentShape(Rectangle())
          .gesture(dragGesture(side: side).simultaneously(with: zoomGesture(side: side)))
          .onAppear { viewportSide = side }
          .onChange(of: side) { _, newValue in
            viewportSide = newValue
            offset = clamped(offset, zoom: zoom, side: newValue)
          }
          .accessibilityLabel("Crop preview for \(catName)")
          .accessibilityHint("Drag the photo to position it. Pinch to zoom.")
        }
        .aspectRatio(1, contentMode: .fit)
        Text("Pinch to zoom and drag to position")
          .font(.subheadline)
          .foregroundStyle(.secondary)
      }
      .padding(20)
      .background(canvas)
      .navigationTitle("Crop photo")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        ToolbarItem(placement: .cancellationAction) {
          Button("Cancel", action: cancel)
        }
        ToolbarItem(placement: .confirmationAction) {
          Button("Use Photo") {
            if let cropped = croppedImage() { usePhoto(cropped) }
          }
        }
      }
    }
    .tint(appAccent)
    .interactiveDismissDisabled()
  }

  private func dragGesture(side: CGFloat) -> some Gesture {
    DragGesture()
      .updating($gestureOffset) { value, state, _ in state = value.translation }
      .onEnded { value in
        let proposed = CGSize(width: offset.width + value.translation.width,
          height: offset.height + value.translation.height)
        offset = clamped(proposed, zoom: zoom, side: side)
      }
  }

  private func zoomGesture(side: CGFloat) -> some Gesture {
    MagnificationGesture()
      .updating($gestureZoom) { value, state, _ in state = value }
      .onEnded { value in
        zoom = min(max(zoom * value, 1), 6)
        offset = clamped(offset, zoom: zoom, side: side)
      }
  }

  private func clamped(_ proposed: CGSize, zoom: CGFloat, side: CGFloat) -> CGSize {
    let base = baseDisplaySize(side: side)
    let limitX = max(0, (base.width * zoom - side) / 2)
    let limitY = max(0, (base.height * zoom - side) / 2)
    return CGSize(width: min(max(proposed.width, -limitX), limitX),
      height: min(max(proposed.height, -limitY), limitY))
  }

  private func baseDisplaySize(side: CGFloat) -> CGSize {
    let scale = max(side / image.size.width, side / image.size.height)
    return CGSize(width: image.size.width * scale, height: image.size.height * scale)
  }

  private func croppedImage() -> UIImage? {
    guard viewportSide > 0, let cgImage = image.cgImage else { return nil }
    let base = baseDisplaySize(side: viewportSide)
    let displayed = CGSize(width: base.width * zoom, height: base.height * zoom)
    let originX = (displayed.width - viewportSide) / 2 - offset.width
    let originY = (displayed.height - viewportSide) / 2 - offset.height
    let xScale = CGFloat(cgImage.width) / displayed.width
    let yScale = CGFloat(cgImage.height) / displayed.height
    let crop = CGRect(x: originX * xScale, y: originY * yScale,
      width: viewportSide * xScale, height: viewportSide * yScale).integral
      .intersection(CGRect(x: 0, y: 0, width: cgImage.width, height: cgImage.height))
    guard crop.width > 0, crop.height > 0, let result = cgImage.cropping(to: crop) else { return nil }
    return UIImage(cgImage: result)
  }
}
