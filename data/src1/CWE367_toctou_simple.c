// 检查时间与使用时间漏洞 (CWE-367)
void process_file_unsafe(const char* filename) {
    // 检查时间 (Time of Check)
    if (access(filename, R_OK) == 0) {
        // 此处存在竞争条件窗口
        // 攻击者可能在access()和fopen()之间修改文件（如创建符号链接）
        
        // 使用时间 (Time of Use)
        FILE* file = fopen(filename, "r");
        
        // 处理文件...
        // 如果文件被替换为符号链接，可能访问到攻击者指定的文件
    }
} 