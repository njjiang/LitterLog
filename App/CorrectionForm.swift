import SwiftUI

@MainActor final class CorrectionForm: ObservableObject {
  @Published var cat: String
  @Published var label: String
  @Published var saving = false
  init(event: Detection) { cat = event.cat; label = event.label }
}
