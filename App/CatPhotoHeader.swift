import SwiftUI
import PhotosUI
import ImageIO

struct CatPhotoHeader: View {
  let cat: String
  let isDemo: Bool
  @StateObject private var photo: CatPhotoModel
  @State private var cropImage: UIImage?
  @State private var showingCrop = false
  private var _selection = SwiftUI.State<PhotosPickerItem?>(initialValue: nil)
  init(cat: String, isDemo: Bool) {
    self.cat = cat; self.isDemo = isDemo
    _photo = StateObject(wrappedValue: CatPhotoModel(cat: cat))
  }
  var body: some View {
    let portrait = photo.image
    let loading = photo.loading
    return VStack(alignment: .leading, spacing: 10) {
      HStack(spacing: 10) {
        PhotosPicker(selection: _selection.projectedValue, matching: .images, preferredItemEncoding: .compatible) {
          HStack(spacing: 14) {
            ZStack {
              Circle().fill(appAccent.opacity(0.12))
              if let portrait { Image(uiImage: portrait).resizable().scaledToFill() }
              else { Image(systemName: "cat.fill").font(.system(size: 32, weight: .medium)).foregroundStyle(appAccent) }
            }
            .frame(width: 64, height: 64).clipShape(Circle())
            .overlay { if loading { ProgressView().padding(12).background(.regularMaterial, in: Circle()) } }
            VStack(alignment: .leading, spacing: 5) {
              Text(catName(cat)).font(LitterLogStyle.sectionHeading).foregroundStyle(.primary)
              Label(loading ? "Loading photo…" : portrait == nil ? "Add photo" : "Change photo",
                systemImage: "camera.fill")
                .font(.subheadline).foregroundStyle(appAccent)
            }
          }
        }
        .buttonStyle(.plain).disabled(loading)
        .accessibilityLabel("\(portrait == nil ? "Add" : "Change") photo for \(catName(cat))")
        Spacer(minLength: 0)
        if portrait != nil {
          Menu {
            Button("Remove photo", role: .destructive) { photo.remove() }
          } label: {
            Image(systemName: "ellipsis").frame(width: 44, height: 44).contentShape(Rectangle())
          }.disabled(loading).accessibilityLabel("Photo options for \(catName(cat))")
        }
      }
      if let error = photo.errorText {
        Label(error, systemImage: "exclamationmark.circle").font(.subheadline).foregroundStyle(warningText)
      }
    }
    .task(id: _selection.wrappedValue) {
      guard let item = _selection.wrappedValue else { return }
      if let image = await photo.load(item) {
        cropImage = image
        showingCrop = true
      }
      guard !Task.isCancelled, _selection.wrappedValue == item else { return }
      _selection.wrappedValue = nil
    }
    .fullScreenCover(isPresented: $showingCrop) {
      if let cropImage {
        CatPhotoCropView(image: cropImage, catName: catName(cat)) {
          showingCrop = false
          self.cropImage = nil
        } usePhoto: { cropped in
          photo.save(cropped)
          showingCrop = false
          self.cropImage = nil
        }
      }
    }
  }
}


private enum CatPhotoStorage {
  static func file(for cat: String) throws -> URL {
    guard ["cat1", "cat2"].contains(cat) else { throw CocoaError(.fileWriteInvalidFileName) }
    return try FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask,
                     appropriateFor: nil, create: true)
      .appendingPathComponent("CatPhotos", isDirectory: true).appendingPathComponent(cat + ".jpg")
  }
  static func preparedImage(from data: Data) throws -> UIImage {
    guard let source = CGImageSourceCreateWithData(data as CFData, [kCGImageSourceShouldCache: false] as CFDictionary),
       let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
        kCGImageSourceCreateThumbnailFromImageAlways: true,
        kCGImageSourceCreateThumbnailWithTransform: true,
        kCGImageSourceThumbnailMaxPixelSize: 2048,
        kCGImageSourceShouldCacheImmediately: true
       ] as CFDictionary) else { throw CocoaError(.fileReadCorruptFile) }
    return UIImage(cgImage: image)
  }
  static func jpeg(from image: UIImage) throws -> Data {
    let side: CGFloat = 640
    let renderer = UIGraphicsImageRenderer(size: CGSize(width: side, height: side))
    let resized = renderer.image { _ in
      image.draw(in: CGRect(x: 0, y: 0, width: side, height: side))
    }
    let output = NSMutableData()
    guard let destination = CGImageDestinationCreateWithData(output, "public.jpeg" as CFString, 1, nil) else {
      throw CocoaError(.fileWriteUnknown)
    }
    // Re-encode only the resized pixels; original location/EXIF data is not copied.
    guard let cgImage = resized.cgImage else { throw CocoaError(.fileReadCorruptFile) }
    CGImageDestinationAddImage(destination, cgImage, [kCGImageDestinationLossyCompressionQuality: 0.88] as CFDictionary)
    guard CGImageDestinationFinalize(destination) else { throw CocoaError(.fileWriteUnknown) }
    return output as Data
  }
}

@MainActor private final class CatPhotoModel: ObservableObject {
  @Published var image: UIImage?
  @Published var loading = false
  @Published var errorText: String?
  private let cat: String
  private var requestID = UUID()
  init(cat: String) {
    self.cat = cat
    if let url = try? CatPhotoStorage.file(for: cat), FileManager.default.fileExists(atPath: url.path) {
      image = UIImage(contentsOfFile: url.path)
      if image == nil { errorText = "The saved photo couldn’t load. Please choose it again." }
    }
  }
  func load(_ item: PhotosPickerItem) async -> UIImage? {
    let request = UUID()
    requestID = request
    loading = true; errorText = nil
    defer { if requestID == request { loading = false } }
    do {
      guard let data = try await item.loadTransferable(type: Data.self) else { throw CocoaError(.fileReadCorruptFile) }
      try Task.checkCancellation()
      let image = try await Task.detached(priority: .userInitiated) {
        try CatPhotoStorage.preparedImage(from: data)
      }.value
      try Task.checkCancellation()
      guard requestID == request else { return nil }
      return image
    } catch {
      if !Task.isCancelled, requestID == request {
        errorText = "Couldn’t save that photo. Please try again or choose another photo."
      }
      return nil
    }
  }
  func save(_ cropped: UIImage) {
    do {
      let data = try CatPhotoStorage.jpeg(from: cropped)
      let url = try CatPhotoStorage.file(for: cat)
      try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
      try data.write(to: url, options: .atomic)
      guard let saved = UIImage(data: data) else { throw CocoaError(.fileReadCorruptFile) }
      image = saved
      errorText = nil
    } catch { errorText = "Couldn’t save that photo. Please try again or choose another photo." }
  }
  func remove() {
    do {
      let url = try CatPhotoStorage.file(for: cat)
      if FileManager.default.fileExists(atPath: url.path) { try FileManager.default.removeItem(at: url) }
      image = nil; errorText = nil
    } catch { errorText = "Couldn’t remove the photo. Please try again." }
  }
}

