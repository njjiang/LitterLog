import SwiftUI

struct ConnectionsView: View {
  @ObservedObject var model: HomeModel
  @Binding var demo: Bool
  @Binding var server: String
  @Binding var stream: String
  var reloadVideo: () -> Void
  var connectCamera: () async -> Void
  @State private var pairingToken = PairingKey.read()
  @State private var connecting = false
  @Environment(\.dismiss) private var dismiss

  var body: some View {
    NavigationStack {
      Form {
        Section {
          Toggle("Demo mode", systemImage: "leaf.fill", isOn: $demo)
        } footer: {
          Text("Preview sample activity and controls.")
        }.listRowBackground(cardSurface)
        Section {
          TextField("Server URL", text: $server, prompt: Text("https://your-mac.local:8765"))
            .textInputAutocapitalization(.never).autocorrectionDisabled().keyboardType(.URL)
          SecureField("App pairing token", text: $pairingToken)
            .textInputAutocapitalization(.never).autocorrectionDisabled()
          Button("Save pairing token", systemImage: "checkmark") {
            model.message = PairingKey.save(pairingToken) ? "Pairing token saved in Keychain." : "Could not save pairing token."
          }
          Button(connecting ? "Connecting…" : "Test connection", systemImage: "wifi") {
            connecting = true
            Task {
              await model.refresh(base: server)
              connecting = false
            }
          }.disabled(demo || server.isEmpty || connecting)
          Text(model.connectionNote).font(.subheadline)
            .foregroundStyle(!demo && model.connectionNote == "Activity connected" ? successText : .secondary)
        } header: {
          Text("Home server")
        } footer: {
          Text("Use your Mac’s network address and the server’s LITTERLOG_APP_TOKEN. Your iPhone must be able to reach your Mac.")
        }.listRowBackground(cardSurface)
        Section {
          TextField("HLS or MP4 stream URL", text: $stream)
            .textInputAutocapitalization(.never).autocorrectionDisabled().keyboardType(.URL)
          Button("Reload video", systemImage: "arrow.clockwise", action: reloadVideo)
          Button("Connect Protect relay", systemImage: "web.camera.fill") {
            demo = false
            Task { await connectCamera() }
          }.disabled(server.isEmpty)
        } header: {
          Text("Camera · UniFi Protect")
        } footer: {
          Text("The home server handles Protect credentials. The relay refreshes snapshots every second. Use an HLS or MP4 bridge URL for continuous video; direct RTSP is not supported.")
        }.listRowBackground(cardSurface)
        Section("Connected care") {
          Label("Litter-Robot 4", systemImage: "washer.fill")
          Text("Your home server connects to Whisker. Cleaning is supported; pause and resume are available in demo mode.")
            .font(.subheadline).foregroundStyle(.secondary)
          Label("Video classifier", systemImage: "pawprint.fill")
          Text("Run your classifier on your Mac and send predictions to the home server. LitterLog checks for new activity every five seconds.")
            .font(.subheadline).foregroundStyle(.secondary)
        }.listRowBackground(cardSurface)
      }
      .contentMargins(.top, 12, for: .scrollContent)
      .listSectionSpacing(.compact)
      .scrollContentBackground(.hidden)
      .background(canvas)
      .navigationTitle("Connections")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        ToolbarItem(placement: .confirmationAction) {
          Button("Done") { dismiss() }
        }
      }
    }.tint(appAccent)
  }
}
