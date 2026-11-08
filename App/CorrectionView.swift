import SwiftUI

struct CorrectionView: View {
  let event: Detection
  @ObservedObject var model: HomeModel
  let server: String
  @StateObject private var form: CorrectionForm
  @Environment(\.dismiss) private var dismiss
  init(event: Detection, model: HomeModel, server: String) {
    self.event = event; self.model = model; self.server = server
    _form = StateObject(wrappedValue: CorrectionForm(event: event))
  }
  var body: some View {
    NavigationStack {
      Form {
        Section("Confirmed from video") {
          Picker("Cat", selection: $form.cat) { Text("Unassigned").tag(""); Text("Cat 1").tag("cat1"); Text("Cat 2").tag("cat2") }
          Picker("Activity", selection: $form.label) {
            ForEach(["recording","cat1","cat2","none","pee","poo","both"], id: \.self) { Text(activityLabel($0)).tag($0) }
          }
          Text("Your correction takes priority for waste and weight attribution. The original prediction is preserved.").font(.caption)
        }
        Section("Original model output") { Text(catName(event.original_label ?? event.label)) }
        Button("Save correction") { Task { form.saving = true; if await model.correct(event, base: server, cat: form.cat, label: form.label) { dismiss() }; form.saving = false } }.disabled(form.saving)
        Button("Remove correction", role: .destructive) { Task { if await model.correct(event, base: server, reset: true) { dismiss() } } }
      }.scrollContentBackground(.hidden).background(canvas).tint(appAccent).navigationTitle("Edit prediction").navigationBarTitleDisplayMode(.inline)
        .toolbar { ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } } }
    }
  }
}
