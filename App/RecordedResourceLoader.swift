import AVKit

final class RecordedResourceLoader: NSObject, AVAssetResourceLoaderDelegate, URLSessionDataDelegate, @unchecked Sendable {
  private final class Transfer {
    let request: AVAssetResourceLoadingRequest
    let offset: Int64
    var remaining: Int64?
    var expectedBytes: Int64?
    var received: Int64 = 0
    init(_ request: AVAssetResourceLoadingRequest, offset: Int64, remaining: Int64?) {
      self.request = request; self.offset = offset; self.remaining = remaining
    }
  }
  private let remoteURL: URL
  private let queue = DispatchQueue(label: "com.litterlog.recorded-video")
  private var transfers: [Int: Transfer] = [:]
  private var requestTasks: [ObjectIdentifier: URLSessionDataTask] = [:]
  private lazy var session: URLSession = {
    let configuration = URLSessionConfiguration.default
    configuration.timeoutIntervalForRequest = 30
    configuration.timeoutIntervalForResource = 180
    let callbacks = OperationQueue()
    callbacks.maxConcurrentOperationCount = 1
    callbacks.underlyingQueue = queue
    return URLSession(configuration: configuration, delegate: self, delegateQueue: callbacks)
  }()
  init(url: URL) { remoteURL = url; super.init() }
  func asset() -> AVURLAsset {
    var components = URLComponents(url: remoteURL, resolvingAgainstBaseURL: false)!
    components.scheme = "litterlog-video"
    let asset = AVURLAsset(url: components.url!)
    asset.resourceLoader.setDelegate(self, queue: queue)
    return asset
  }
  func stop() {
    queue.async { [self] in
      for task in requestTasks.values { task.cancel() }
      transfers.removeAll(); requestTasks.removeAll()
      session.invalidateAndCancel()
    }
  }
  func resourceLoader(_ resourceLoader: AVAssetResourceLoader,
            shouldWaitForLoadingOfRequestedResource loadingRequest: AVAssetResourceLoadingRequest) -> Bool {
    guard ["http", "https"].contains(remoteURL.scheme?.lowercased() ?? "") else { return false }
    let data = loadingRequest.dataRequest
    let start = data.map { max($0.requestedOffset, $0.currentOffset) } ?? 0
    guard start >= 0 else { loadingRequest.finishLoading(with: URLError(.badServerResponse)); return true }
    let remaining: Int64?
    if let data, !data.requestsAllDataToEndOfResource {
      let (end, overflow) = data.requestedOffset.addingReportingOverflow(Int64(data.requestedLength))
      guard !overflow, end > start else { loadingRequest.finishLoading(with: URLError(.badServerResponse)); return true }
      remaining = end - start
    } else { remaining = nil }
    var request = URLRequest(url: remoteURL)
    request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
    if data == nil { request.setValue("bytes=0-1", forHTTPHeaderField: "Range") }
    else if let remaining { request.setValue("bytes=\(start)-\(start + remaining - 1)", forHTTPHeaderField: "Range") }
    else { request.setValue("bytes=\(start)-", forHTTPHeaderField: "Range") }
    let task = session.dataTask(with: request)
    transfers[task.taskIdentifier] = Transfer(loadingRequest, offset: start, remaining: remaining)
    requestTasks[ObjectIdentifier(loadingRequest)] = task
    task.resume()
    return true
  }
  func resourceLoader(_ resourceLoader: AVAssetResourceLoader, didCancel loadingRequest: AVAssetResourceLoadingRequest) {
    guard let task = requestTasks.removeValue(forKey: ObjectIdentifier(loadingRequest)) else { return }
    transfers.removeValue(forKey: task.taskIdentifier)
    task.cancel()
  }
  private func finish(_ task: URLSessionTask, error: Error? = nil) {
    guard let transfer = transfers.removeValue(forKey: task.taskIdentifier) else { return }
    requestTasks.removeValue(forKey: ObjectIdentifier(transfer.request))
    guard !transfer.request.isCancelled else { return }
    if let error { transfer.request.finishLoading(with: error) }
    else { transfer.request.finishLoading() }
  }
  func urlSession(_ session: URLSession, didReceive challenge: URLAuthenticationChallenge,
          completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void) {
    HomeTLS.delegate.urlSession(session, didReceive: challenge, completionHandler: completionHandler)
  }
  func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
          newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
    // Recording URLs are local media files. Never forward their token through a redirect.
    completionHandler(nil)
  }
  func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
          completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
    guard let transfer = transfers[dataTask.taskIdentifier], let response = response as? HTTPURLResponse else {
      completionHandler(.cancel); return
    }
    let total: Int64
    if response.statusCode == 206,
     let value = response.value(forHTTPHeaderField: "Content-Range"), value.hasPrefix("bytes ") {
      let parts = value.dropFirst(6).split(separator: "/")
      let range = parts.first?.split(separator: "-") ?? []
      guard parts.count == 2, range.count == 2,
         let start = Int64(range[0]), let end = Int64(range[1]), let length = Int64(parts[1]),
         start == transfer.offset, end >= start, length > end else {
        finish(dataTask, error: URLError(.badServerResponse)); completionHandler(.cancel); return
      }
      total = length
      transfer.expectedBytes = end - start + 1
    } else if response.statusCode == 200, transfer.offset == 0, response.expectedContentLength > 0 {
      total = response.expectedContentLength
      transfer.expectedBytes = total
    } else {
      finish(dataTask, error: URLError(.badServerResponse)); completionHandler(.cancel); return
    }
    if let info = transfer.request.contentInformationRequest {
      info.contentType = "public.mpeg-4"
      info.contentLength = total
      info.isByteRangeAccessSupported = response.statusCode == 206 || response.value(forHTTPHeaderField: "Accept-Ranges") == "bytes"
    }
    if transfer.request.dataRequest == nil {
      finish(dataTask); completionHandler(.cancel); return
    }
    completionHandler(.allow)
  }
  func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
    guard let transfer = transfers[dataTask.taskIdentifier], !transfer.request.isCancelled,
       let request = transfer.request.dataRequest else { return }
    let size = transfer.remaining.map { min(data.count, Int($0)) } ?? data.count
    request.respond(with: Data(data.prefix(size)))
    transfer.received += Int64(size)
    if let remaining = transfer.remaining {
      transfer.remaining = remaining - Int64(size)
      if transfer.remaining == 0 { finish(dataTask); dataTask.cancel() }
    }
  }
  func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
    guard let transfer = transfers[task.taskIdentifier] else { return }
    if let error { finish(task, error: error) }
    else if let expected = transfer.expectedBytes, transfer.received < expected {
      finish(task, error: URLError(.networkConnectionLost))
    } else { finish(task) }
  }
}

