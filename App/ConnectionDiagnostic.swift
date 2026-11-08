import Foundation
import Security
import CryptoKit

enum ConnectionDiagnostic {
  static func describe(_ error: Error) -> String {
    let failure = error as NSError
    var details = [failure.localizedDescription, "Error \(failure.code)"]
    if let value = failure.userInfo[NSURLErrorFailingURLPeerTrustErrorKey],
     CFGetTypeID(value as CFTypeRef) == SecTrustGetTypeID() {
      let trust = value as! SecTrust
      var reason: CFError?
      _ = SecTrustEvaluateWithError(trust, &reason)
      if let reason { details.append(CFErrorCopyDescription(reason) as String) }
      if let chain = SecTrustCopyCertificateChain(trust) as? [SecCertificate] {
        for certificate in chain {
          let data = SecCertificateCopyData(certificate) as Data
          let fingerprint = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
          details.append("\(SecCertificateCopySubjectSummary(certificate) as String? ?? "Certificate"): \(fingerprint)")
        }
      }
    }
    return details.joined(separator: "\n")
  }
}


