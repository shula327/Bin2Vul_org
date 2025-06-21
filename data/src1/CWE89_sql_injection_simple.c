// SQL注入漏洞 (CWE-89)
void login_unsafe(const char* username, const char* password) {
    char query[1024];
    
    // 不安全：SQL注入风险
    snprintf(query, sizeof(query), 
             "SELECT * FROM users WHERE username = '%s' AND password = '%s'", 
             username, password);
    
    // 执行查询
    // 攻击者可以提供: admin' --
    // 结果查询将变成: SELECT * FROM users WHERE username = 'admin' -- ' AND password = '任何字符'
} 