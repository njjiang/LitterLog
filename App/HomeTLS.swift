import Foundation
import Security

final class HomeTLS: NSObject, URLSessionDelegate, @unchecked Sendable {
  static let delegate = HomeTLS()
  static let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)

  /// Optional mapping of lowercase hostnames/IPs to base64-encoded DER root CA certificates,
  /// loaded from `App/LocalTLS.plist` when present in the app bundle.
  private static let pinnedCertificates: [String: String] = {
    guard let url = Bundle.main.url(forResource: "LocalTLS", withExtension: "plist"),
          let dict = NSDictionary(contentsOf: url) as? [String: String] else {
      return [:]
    }
    return Dictionary(uniqueKeysWithValues: dict.map { ($0.key.lowercased(), $0.value) })
  }()

  private static func certificate(for host: String) -> String? {
    pinnedCertificates[host.lowercased()]
  }

  func urlSession(_ session: URLSession, didReceive challenge: URLAuthenticationChallenge,
          completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void) {
    let host = challenge.protectionSpace.host.lowercased()
    guard let caDER = Self.certificate(for: host),
       challenge.protectionSpace.authenticationMethod == NSURLAuthenticationMethodServerTrust,
       let trust = challenge.protectionSpace.serverTrust else {
      completionHandler(.performDefaultHandling, nil); return
    }
    guard let data = Data(base64Encoded: caDER),
       let root = SecCertificateCreateWithData(nil, data as CFData),
       SecTrustSetPolicies(trust, SecPolicyCreateSSL(true, host as CFString)) == errSecSuccess,
       SecTrustSetAnchorCertificates(trust, [root] as CFArray) == errSecSuccess,
       SecTrustSetAnchorCertificatesOnly(trust, true) == errSecSuccess,
       SecTrustEvaluateWithError(trust, nil) else {
      completionHandler(.cancelAuthenticationChallenge, nil); return
    }
    completionHandler(.useCredential, URLCredential(trust: trust))
  }
}
