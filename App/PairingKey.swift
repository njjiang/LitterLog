import Foundation
import Security

enum PairingKey {
  static let service = "com.litterlog.pairing"
  private static let lock = NSLock()
  private static var cachedValue: String?
  static func read() -> String {
    lock.lock()
    defer { lock.unlock() }
    if let cachedValue { return cachedValue }
    let query: [String: Any] = [kSecClass as String:kSecClassGenericPassword, kSecAttrService as String:service, kSecAttrAccount as String:"server", kSecReturnData as String:true, kSecMatchLimit as String:kSecMatchLimitOne]
    var result: CFTypeRef?
    guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess, let data = result as? Data else { return "" }
    let value = String(data:data,encoding:.utf8) ?? ""
    cachedValue = value
    return value
  }
  static func save(_ value: String) -> Bool {
    lock.lock()
    defer { lock.unlock() }
    cachedValue = nil
    let query: [String: Any] = [kSecClass as String:kSecClassGenericPassword, kSecAttrService as String:service, kSecAttrAccount as String:"server"]
    SecItemDelete(query as CFDictionary)
    if value.isEmpty { cachedValue = ""; return true }
    var item=query
    item[kSecValueData as String]=Data(value.utf8)
    item[kSecAttrAccessible as String]=kSecAttrAccessibleWhenUnlockedThisDeviceOnly
    let saved = SecItemAdd(item as CFDictionary,nil) == errSecSuccess
    if saved { cachedValue = value }
    return saved
  }
}
