// 命令注入漏洞 (CWE-78)
void ping_host_unsafe(const char* hostname) {
    char command[256];
    
    // 不安全：直接拼接用户输入到命令字符串
    snprintf(command, sizeof(command), "ping -c 4 %s", hostname);
    
    // 执行命令
    system(command);
    
    // 攻击者可以输入: localhost; rm -rf /important_files
    // 结果命令: ping -c 4 localhost; rm -rf /important_files
} 