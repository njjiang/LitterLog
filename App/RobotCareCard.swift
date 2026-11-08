import SwiftUI

struct RobotCareCard: View {
  @Environment(\.dynamicTypeSize) private var dynamicTypeSize
  @ObservedObject var model: HomeModel
  var demo: Bool
  var demoState: String
  var clean: () -> Void
  var send: (String) -> Void
  @State private var confirmCycle = false

  private var waste: Double? { demo ? 42 : model.robot?.waste_percent }
  private var state: String { demo ? demoState : model.robot?.state ?? "Offline" }
  private var hopperStatus: String? {
    guard !demo, let status = model.robot?.hopper_status?.trimmingCharacters(in: .whitespacesAndNewlines),
      !status.isEmpty else { return nil }
    return status.uppercased()
  }
  private var canControl: Bool {
    demo || (model.robot?.connected == true && model.robot?.supported_commands?.contains("cycle") == true && !model.busy)
  }
  private var canPause: Bool {
    demo || (model.robot?.connected == true && !model.busy)
  }
  private var canReset: Bool {
    demo || (model.robot?.connected == true && !model.busy)
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 20) {
      HStack(spacing: 12) {
        Image(systemName: "washer.fill").font(.title3).foregroundStyle(appAccent)
          .frame(width: 44, height: 44).background(LitterLogStyle.iconBackground, in: RoundedRectangle(cornerRadius: 14))
        VStack(alignment: .leading, spacing: 3) {
          Text("Litter-Robot 4").font(LitterLogStyle.sectionHeading)
        }
        Spacer(minLength: 0)
        StatusPill(title: state, symbol: state == "Ready" ? "checkmark" : "circle.fill")
      }
      let metricLayout = dynamicTypeSize.isAccessibilitySize
        ? AnyLayout(VStackLayout(alignment: .leading, spacing: 16))
        : AnyLayout(HStackLayout(alignment: .top, spacing: 24))
      metricLayout {
        metric("Waste drawer", value: waste.map { "\(Int($0))%" } ?? "—")
        metric("Litter level", value: demo ? "Optimal" : model.robot?.litter_level?.capitalized ?? "—")
      }
      .frame(maxWidth: .infinity, alignment: .leading)
      ProgressView(value: min(1, max(0, (waste ?? 0) / 100))) {
        Text("Waste drawer capacity").font(.subheadline).foregroundStyle(.secondary)
      }
      .labelsHidden().tint(appAccent)
      .accessibilityValue(waste.map { "\(Int($0)) percent full" } ?? "Unavailable")
      if let hopper = hopperStatus {
        VStack(alignment: .leading, spacing: 6) {
          if hopper == "EMPTY" {
            Label("LitterHopper empty", systemImage: "exclamationmark.triangle.fill")
              .font(.body.weight(.semibold)).foregroundStyle(warningText)
            Text("Refill the hopper.").font(.subheadline).foregroundStyle(.secondary)
          } else {
            LabeledContent("LitterHopper", value: hopper.replacingOccurrences(of: "_", with: " ").capitalized)
              .font(.body)
            if ["MOTOR_FAULT_SHORT", "MOTOR_OT_AMPS", "MOTOR_DISCONNECTED", "JAMMED", "OFFLINE"].contains(hopper) {
              Label("LitterHopper needs attention", systemImage: "exclamationmark.triangle.fill")
                .font(.subheadline).foregroundStyle(warningText)
            }
          }
        }
      }
      Button {
        confirmCycle = true
      } label: {
        HStack(spacing: 9) {
          if model.busy { ProgressView().tint(filledButtonText) }
          Text(model.busy ? "Requesting clean…" : "Start clean cycle")
            .font(.body.weight(.semibold)).foregroundStyle(filledButtonText)
        }
        .frame(maxWidth: .infinity, minHeight: 24)
      }
      .buttonStyle(.borderedProminent).controlSize(.large).tint(filledAccent)
      .buttonBorderShape(.roundedRectangle(radius: 14)).disabled(!canControl)
      .confirmationDialog("Start a clean cycle?", isPresented: $confirmCycle, titleVisibility: .visible) {
        Button(demo ? "Simulate cleaning" : "Start cleaning") { clean() }
      } message: {
        Text(demo ? "Try a sample clean cycle." : "Start cleaning your Litter-Robot 4? An active countdown will be reset first.")
      }
      HStack(spacing: 12) {
        Button {
          send("pause")
        } label: {
          Label("Pause", systemImage: "pause.fill")
            .font(.body.weight(.semibold))
            .frame(maxWidth: .infinity, minHeight: 24)
        }
        .disabled(!canPause)
        Button {
          send("reset")
        } label: {
          Label("Reset", systemImage: "arrow.counterclockwise")
            .font(.body.weight(.semibold))
            .frame(maxWidth: .infinity, minHeight: 24)
        }
        .disabled(!canReset)
      }
      .buttonStyle(.bordered).controlSize(.large).tint(appAccent)
      .buttonBorderShape(.roundedRectangle(radius: 14))
      if let note = model.commandNote {
        Label(note, systemImage: model.commandFailed ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
          .font(.subheadline).foregroundStyle(model.commandFailed ? warningText : appAccent)
      }
      if !demo, let weight = model.robot?.last_weight_lbs, weight > 0 {
        LabeledContent("Latest device weight", value: String(format: "%.2f lb", weight)).font(.subheadline)
      }
      if !demo, let cycles = model.robot?.cycle_count {
        LabeledContent("Device cycles", value: "\(cycles)").font(.subheadline)
      }
      if demo {
        Text("Demo mode · Simulated controls")
          .font(.subheadline).foregroundStyle(.secondary)
      }
    }
    .card()
  }

  private func metric(_ title: String, value: String) -> some View {
    VStack(alignment: .leading, spacing: 8) {
      Text(title).font(.subheadline).foregroundStyle(.secondary)
      Text(value).font(.title2.weight(.semibold)).monospacedDigit().foregroundStyle(.primary)
        .fixedSize(horizontal: false, vertical: true)
    }
    .frame(maxWidth: .infinity, alignment: .leading)
    .accessibilityElement(children: .combine)
  }
}
