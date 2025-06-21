// 错误的文件权限检查 (CWE-706)
int is_secure_file_unsafe(const char* filename) {
    struct stat file_stat;
    
    // 获取文件状态
    if (stat(filename, &file_stat) != 0) {
        return 0;
    }
    
    // 不安全：仅检查世界可写权限，忽略了组权限
    // 即使文件对组成员是可写的，也会返回"安全"的结果
    if (file_stat.st_mode & S_IWOTH) {
        return 0; // 文件不安全
    }
    
    return 1; // 不安全：返回"安全"状态，尽管可能对组可写
} 