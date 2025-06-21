// 不安全：直接执行命令
void execute_command_unsafe(const char* command) {
    char full_command[256];
    snprintf(full_command, sizeof(full_command), "ls %s", command);
    printf("[不安全] 执行命令: %s\n", full_command);
    system(full_command);
} 