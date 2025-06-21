// 不安全：详细错误信息泄露
void handle_error_unsafe(const char* function, const char* error_details) {
    printf("错误: %s 在 %s 中发生，详细信息: %s\n", 
        "数据库连接失败", function, error_details);
} 