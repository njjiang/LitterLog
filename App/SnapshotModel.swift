import SwiftUI
import UIKit

@MainActor final class SnapshotModel: ObservableObject {
  @Published var image: UIImage?
  @Published var unavailable = false
  func run(url: URL) async {
    while !Task.isCancelled {
      do {
        var request = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData)
        request.timeoutInterval = 10
        let (data,response) = try await HomeTLS.session.data(for: request)
        guard (response as? HTTPURLResponse)?.statusCode == 200, let frame = UIImage(data: data) else { throw URLError(.cannotDecodeContentData) }
        image = frame; unavailable = false
      } catch { if Task.isCancelled { return }; unavailable = true }
      do { try await Task.sleep(for: .seconds(1)) } catch { return }
    }
  }
}
