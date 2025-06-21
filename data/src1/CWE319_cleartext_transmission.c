// 不安全：通过明文传输敏感信息
void send_credentials_unsafe(const char* username, const char* password) {
    // 创建未加密的HTTP连接
    HTTPConnection* conn = http_connect("http://example.com/login");
    
    // 明文发送凭证
    http_send(conn, "username=%s&password=%s", username, password);
} 