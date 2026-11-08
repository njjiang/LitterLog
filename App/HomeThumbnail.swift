import SwiftUI
import UIKit

struct HomeThumbnail: View {
  let url: URL
  private var _image = SwiftUI.State<UIImage?>(initialValue: nil)
  var body: some View {
    Group {
      if let image = _image.wrappedValue { Image(uiImage: image).resizable().scaledToFill() }
      else { Image(systemName: "photo").foregroundStyle(.secondary) }
    }.task(id: url) {
      _image.wrappedValue = nil
      if let (data, response) = try? await HomeTLS.session.data(from: url),
       (response as? HTTPURLResponse)?.statusCode == 200, !Task.isCancelled {
        _image.wrappedValue = UIImage(data: data)
      }
    }
  }
}
